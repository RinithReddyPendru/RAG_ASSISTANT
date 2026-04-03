from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
from rag_core import ingest_documents, DATA_DIR
from rag_agent import agentic_rag

app = FastAPI(title="RAG Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    api_key: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Use LangGraph agent
        model_name = request.api_key if request.api_key else "llama3"
        
        # We pass recursion_limit to prevent infinite loops (like rewrite -> retrieve -> grade -> rewrite)
        result = agentic_rag.invoke(
            {"question": request.message, "model_name": model_name},
            {"recursion_limit": 15}
        )
        
        response = result.get("generation", "Sorry, I could not process your request.")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import BackgroundTasks

@app.post("/api/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), model_name: str = Form("llama3")):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Trigger ingestion after upload in the background
    background_tasks.add_task(ingest_documents, model_name, file_path)
    
    return {"message": f"Successfully uploaded {file.filename}, ingestion processing in background...", "chunks": "..."}

# Mount frontend static files at the root route
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend"), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
