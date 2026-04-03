@echo off
echo Starting RAG Assistant Server and Interface...
cd /d "%~dp0\backend"
call ..\venv\Scripts\activate.bat

echo Opening browser...
start http://localhost:8000

uvicorn app:app --host 0.0.0.0 --port 8000
pause
