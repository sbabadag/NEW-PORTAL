@echo off
echo Portal Frame Analysis Tool - Modern UI
echo =====================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo Using Python environment: %cd%\.venv
echo.

REM Test environment first
echo Testing environment...
".venv\Scripts\python.exe" test_environment.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Environment test failed!
    pause
    exit /b 1
)

echo.
echo Starting Modern Portal Analysis Tool...
echo This will open a modern GUI with CustomTkinter.
echo.
".venv\Scripts\python.exe" portal_modern_ui.py

echo.
echo Application closed. Press any key to exit.
pause >nul
