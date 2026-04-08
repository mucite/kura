# -*- mode: python ; coding: utf-8 -*-
import shutil
import os
from PyInstaller.utils.hooks import collect_all

# Build datas list - models are optional (downloaded on first launch)
datas = [
    ('../core', 'core'),  # Model downloader and utilities
    ('../shared', 'shared'),  # Shared configuration and utilities
    ('../.env.dist', '.'),
    ('../.env.example', '.'),
    ('physio_scribe.py', '.'),
    ('Info.plist', '.'),
    ('assets', 'assets'),
]

# Only include models if they exist (for development builds)
# Production builds download models on first launch to keep installer small
if os.path.exists('../models') and os.path.isdir('../models'):
    print("📦 Including models directory in bundle (development build)")
    datas.append(('../models', 'models'))
else:
    print("⚠️  Models directory not found - will be downloaded on first launch (production build)")

# Bundle ffmpeg so customers don't need it installed
_ffmpeg = shutil.which('ffmpeg')
if not _ffmpeg:
    import os
    for _p in ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg']:
        if os.path.exists(_p):
            _ffmpeg = _p
            break
if not _ffmpeg:
    raise RuntimeError("ffmpeg not found — run: brew install ffmpeg")
binaries = [(_ffmpeg, '.')]
hiddenimports = [
    'mlx', 'mlx.core', 'mlx.nn', 'mlx.optimizers', 'mlx._reprlib_fix',
    'mlx_lm', 'mlx_lm.models', 'mlx_lm.models.llama',
    'mlx_whisper',
    'fpdf', 'requests', 'dotenv', 'rumps', 'sounddevice', 'numpy', 'tkinter',
    'physio_scribe',
    'shared.config_manager', 'shared.license_manager', 'shared.practice_config',
    'core.model_downloader', 'core.model_download_dialog',
    'huggingface_hub', 'huggingface_hub.hf_api', 'huggingface_hub.file_download',
]
tmp_ret = collect_all('mlx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mlx_lm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mlx_whisper')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py', 'physio_scribe.py'],
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file='entitlements.plist',
    icon=['../assets/stethoscope.icns'],
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
app = BUNDLE(
    coll,
    name='Kura.app',
    icon='../assets/stethoscope.icns',
    bundle_identifier='de.kura-medical.kura',
)
