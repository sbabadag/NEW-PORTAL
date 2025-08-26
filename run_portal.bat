@echo off
echo Portal Frame Analysis Tool
echo ========================
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
echo Choose how to run the Portal Analysis Tool:
echo.
echo 1. Modern UI (CustomTkinter - Recommended)
echo 2. Console Version (works in terminal)
echo 3. Interactive Jupyter Notebook
echo 4. Exit
echo.

:choice
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting Modern UI...
    echo This will open a modern GUI with CustomTkinter.
    echo.
    ".venv\Scripts\python.exe" portal_modern_ui.py
    goto end
) else if "%choice%"=="2" (
    echo.
    echo Starting Console Version...
    echo.
    ".venv\Scripts\python.exe" portal.py
    goto end
) else if "%choice%"=="3" (
    echo.
    echo Starting Jupyter Notebook...
    echo Your browser will open with the interactive interface.
    echo Close the browser tab and press Ctrl+C in this window when done.
    echo.
    ".venv\Scripts\jupyter" notebook portal_interactive.ipynb
    goto end
) else if "%choice%"=="4" (
    goto end
) else (
    echo Invalid choice. Please enter 1, 2, 3, or 4.
    goto choice
)

:end
echo.
echo Thank you for using Portal Frame Analysis Tool!
pause >nul
