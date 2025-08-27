# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['portal_modern_ui.py'],
    pathex=[],
    binaries=[],
    datas=[('steel_check.py', '.')],
    hiddenimports=['numpy', 'matplotlib', 'matplotlib.backends.backend_tkagg', 'customtkinter', 'tkinter', 'tkinter.messagebox', 'tkinter.ttk', 'threading', 'steel_check', 'PIL', 'PIL.Image', 'PIL.ImageTk'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['jupyter', 'notebook', 'ipython', 'ipywidgets'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PortalFrameAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
