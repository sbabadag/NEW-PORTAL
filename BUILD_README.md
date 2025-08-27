# Portal Frame Analysis - Standalone Executable Build

## Overview
This directory contains scripts to build a standalone executable of the Portal Frame Analysis application using PyInstaller.

## Requirements
- Python 3.7 or higher
- All dependencies from requirements.txt
- PyInstaller (will be installed automatically)

## Building the Executable

### For Windows:
1. Open Command Prompt or PowerShell
2. Navigate to this directory
3. Run: `build_exe.bat`

### For Linux/macOS:
1. Open Terminal
2. Navigate to this directory
3. Run: `chmod +x build_exe.sh && ./build_exe.sh`

### Manual Build (Alternative):
```bash
# Install PyInstaller
pip install pyinstaller

# Build the executable
pyinstaller portal_analysis.spec
```

## Output
The standalone executable will be created in the `dist/` folder:
- Windows: `dist/PortalFrameAnalysis.exe`
- Linux/macOS: `dist/PortalFrameAnalysis`

## Features of the Standalone Executable
- ✅ No Python installation required on target machines
- ✅ All dependencies bundled (NumPy, Matplotlib, CustomTkinter)
- ✅ Complete portal frame analysis functionality
- ✅ Steel section optimization
- ✅ Modern UI with graphs and diagrams
- ✅ Single file distribution

## Distribution
The executable can be distributed as a single file that will run on any compatible system without requiring Python or any additional installations.

## File Structure After Build
```
dist/
├── PortalFrameAnalysis.exe    # Main executable (Windows)
└── PortalFrameAnalysis        # Main executable (Linux/macOS)

build/                         # Temporary build files (can be deleted)
├── PortalFrameAnalysis/
└── ...

portal_analysis.spec           # PyInstaller specification file
build_exe.bat                  # Windows build script
build_exe.sh                   # Linux/macOS build script
```

## Troubleshooting

### Common Issues:
1. **Import errors**: All required modules are included in the spec file
2. **Missing files**: The steel_check.py module is automatically included
3. **Size optimization**: Jupyter/notebook modules are excluded to reduce size

### If build fails:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Try building with the spec file: `pyinstaller portal_analysis.spec`
3. Check the console output for specific error messages

## Performance Notes
- First startup may take a few seconds as the executable extracts files
- Subsequent startups will be faster
- The executable size will be approximately 50-100 MB due to bundled libraries
