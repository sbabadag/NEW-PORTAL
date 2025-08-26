@echo off
echo Portal Frame Analysis - Environment Setup
echo ========================================
echo.

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment!
        echo Make sure Python is installed and available.
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created.
) else (
    echo ✓ Virtual environment already exists.
)

REM Activate and install requirements
echo.
echo Installing required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install packages!
    pause
    exit /b 1
)

echo.
echo ✓ Environment setup complete!
echo.
echo You can now run the portal analysis tool using:
echo   run_portal.bat
echo.
echo Or manually with:
echo   .venv\Scripts\python.exe portal.py
echo.
pause
