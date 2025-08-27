# Portal Frame Analysis - Standalone Executable

## ✅ Build Complete!

Your Portal Frame Analysis application has been successfully built into a standalone executable!

### 📁 File Location
```
C:\Users\AVM1\PORTAL\dist\PortalFrameAnalysis.exe
```

### 📊 File Details
- **Size**: ~25.8 MB
- **Type**: Windows 64-bit executable
- **Dependencies**: All bundled (no Python installation required)

## 🚀 Features Included

✅ **Complete Portal Frame Analysis**
- Portal frame geometry input
- Load combinations (ULS/SLS)
- N-V-M diagram generation
- Real-time analysis calculations

✅ **Steel Section Database**
- 43 steel sections (24 HEA + 19 IPE)
- Automatic section property lookup
- Custom section support

✅ **Steel Verification**
- Eurocode 3 compliance
- TS 648, ÇYTHYE support
- Utilization ratio calculations
- Safety factor verification

✅ **Total Weight Optimization**
- Frame geometry calculations
- Multi-section combination testing
- Weight minimization algorithms
- Alternative solution display

✅ **Modern User Interface**
- CustomTkinter dark theme
- Interactive plots with Matplotlib
- Tabbed results display
- Progress tracking

## 📦 Distribution

### Single File Distribution
The `PortalFrameAnalysis.exe` file is completely self-contained and can be:
- Copied to any Windows 10/11 (64-bit) computer
- Run without installing Python or any dependencies
- Distributed via email, USB, or network share
- Used immediately after download

### System Requirements
- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 50 MB free space
- **Display**: 1200x800 minimum resolution

## 🔧 Usage Instructions

1. **First Run**: Double-click `PortalFrameAnalysis.exe`
   - First startup may take 10-15 seconds (extraction)
   - Subsequent starts will be faster

2. **Basic Workflow**:
   - Enter geometry parameters
   - Select steel sections from dropdowns
   - Click "Hesapla ve Çiz" to analyze
   - Click "Tahkik Et" to verify sections
   - Click "Optimize Et" for optimization

3. **Results**:
   - View N-V-M diagrams in tabs
   - Check verification results
   - Apply optimized sections automatically

## 🛠️ Build Information

- **Built with**: PyInstaller 6.15.0
- **Python Version**: 3.9.13
- **Build Date**: August 26, 2025
- **Bundled Libraries**: NumPy, Matplotlib, CustomTkinter, PIL

## 📝 Build Files Created

```
build/                     # Temporary build files (can be deleted)
dist/
  └── PortalFrameAnalysis.exe  # Main executable (DISTRIBUTE THIS)
build_exe.bat             # Build script for Windows
build_exe.sh              # Build script for Linux/macOS
portal_analysis.spec      # PyInstaller specification
BUILD_README.md           # Build documentation
test_executable.py        # Test instructions
```

## 🔄 Rebuilding

To rebuild the executable with changes:
```bash
cd "C:\Users\AVM1\PORTAL"
.\build_exe.bat
```

Or manually:
```bash
python -m PyInstaller PortalFrameAnalysis.spec
```

## ✨ Success!

Your Portal Frame Analysis application is now a standalone executable ready for distribution! 🎉

**Distribution file**: `C:\Users\AVM1\PORTAL\dist\PortalFrameAnalysis.exe`
