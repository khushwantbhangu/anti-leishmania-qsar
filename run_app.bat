@echo off
cd /d "%~dp0"

set PYTHON_EXE=%~dp0\.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    python -m venv .venv
    set PYTHON_EXE=%~dp0\.venv\Scripts\python.exe
)

"%PYTHON_EXE%" -m pip install -r requirements.txt
"%PYTHON_EXE%" -m streamlit run app/streamlit_app.py --server.headless true
