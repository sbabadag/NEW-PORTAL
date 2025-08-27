# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['portal_modern_ui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('steel_check.py', '.'),
    ],
    hiddenimports=[
        'numpy',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'customtkinter',
        'tkinter',
        'tkinter.messagebox',
        'tkinter.ttk',
        'threading',
        'math',
        'steel_check',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'jupyter',
        'notebook',
        'ipython',
        'ipywidgets'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
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
    icon='icon.ico'  # Add an icon if you have one
)
