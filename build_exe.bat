@echo off
echo Building Portal Frame Analysis Standalone Executable...
echo.

echo Step 1: Installing PyInstaller if not already installed...
pip install pyinstaller

echo.
echo Step 2: Building executable with PyInstaller...
python -m PyInstaller --onefile ^
    --windowed ^
    --name "PortalFrameAnalysis" ^
    --add-data "steel_check.py;." ^
    --hidden-import "numpy" ^
    --hidden-import "matplotlib" ^
    --hidden-import "matplotlib.backends.backend_tkagg" ^
    --hidden-import "customtkinter" ^
    --hidden-import "tkinter" ^
    --hidden-import "tkinter.messagebox" ^
    --hidden-import "tkinter.ttk" ^
    --hidden-import "threading" ^
    --hidden-import "steel_check" ^
    --hidden-import "PIL" ^
    --hidden-import "PIL.Image" ^
    --hidden-import "PIL.ImageTk" ^
    --exclude-module "jupyter" ^
    --exclude-module "notebook" ^
    --exclude-module "ipython" ^
    --exclude-module "ipywidgets" ^
    portal_modern_ui.py

echo.
echo Step 3: Build completed!
echo The executable is located in: dist\PortalFrameAnalysis.exe
echo.
echo You can now distribute this single .exe file to run the application anywhere!
pause
