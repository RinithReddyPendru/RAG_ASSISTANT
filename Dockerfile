# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install build dependencies for compiling chroma/hnswlib and networkx if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir langgraph langchain-ollama networkx langchain-experimental

# Copy application files
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/

# Set working directory to backend where app.py lives
WORKDIR /app/backend

# Expose port 8000
EXPOSE 8000

# Set environment variable so Ollama python client knows where the host's Ollama runs 
# (assuming Ollama is running on the host machine and accessible via docker bridge)
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
