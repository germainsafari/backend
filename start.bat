@echo off
echo Setting up Python environment...

@REM REM Remove existing venv if it exists
@REM if exist venv (
@REM     echo Removing existing virtual environment...
@REM     rmdir /s /q venv
@REM )

@REM REM Create new virtual environment
@REM echo Creating new virtual environment...
@REM python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install pandas first with binary
echo Installing pandas...
python -m pip install pandas==2.1.1 --only-binary=:all:

REM Install other requirements
echo Installing requirements...
python -m pip install -r requirements.txt

@REM REM Install additional dependencies
@REM echo Installing additional dependencies...
@REM python -m pip install uvicorn[standard]

REM Start the FastAPI server
echo Starting FastAPI server...
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
