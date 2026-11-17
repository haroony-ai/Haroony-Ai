@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment for Haroony AI...
    python -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
)
start "" "http://127.0.0.1:8000/setup"
.venv\Scripts\uvicorn backend.app:app --host 0.0.0.0 --port 8000
pause
