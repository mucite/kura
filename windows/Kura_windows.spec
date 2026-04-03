# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [
    ('../models', 'models'),
    ('../shared', 'shared'),
    ('../.env.example', '.'),
    ('physio_scribe_crossplatform.py', '.'),
]
binaries = []
hiddenimports = [
    'llama_cpp',
    'faster_whisper',
    'PySimpleGUI',
    'sounddevice',
    'fpdf',
    'requests',
    'dotenv',
    'psutil',
    'numpy',
    'wave',
    'physio_scribe_crossplatform',
    'shared.config_manager',
    'shared.license_manager',
    'shared.practice_config',
    'shared.learning_manager',
]

# Collect all faster_whisper and ctranslate2 dependencies
tmp = collect_all('faster_whisper')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]
tmp = collect_all('ctranslate2')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

a = Analysis(
    ['main_windows.py', 'physio_scribe_crossplatform.py'],
    pathex=['.', '..'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Kura',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # No console window (windowed app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Kura',
)