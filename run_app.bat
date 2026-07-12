@echo off
cd /d "%~dp0"

set PYTHON_EXE=%~dp0\.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

"%PYTHON_EXE%" -m pip install -r requirements.txt
"%PYTHON_EXE%" -m streamlit run app/streamlit_app.py --server.headless true
