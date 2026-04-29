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

# Models are never bundled — they are downloaded on first launch.
# Set BUNDLE_MODELS=1 only for internal dev builds that need offline testing.
if os.environ.get('BUNDLE_MODELS') == '1' and os.path.exists('../models') and os.path.isdir('../models'):
    print("📦 Including models directory in bundle (BUNDLE_MODELS=1)")
    datas.append(('../models', 'models'))
else:
    print("ℹ️  Models excluded from bundle — will be downloaded on first launch (production build)")

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
    codesign_identity='Developer ID Application: Musie Kebede Gizaw (NY589846RW)',
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
    info_plist={
        'CFBundleName': 'Kura Medical',
        'CFBundleDisplayName': 'Kura',
        'CFBundleVersion': '2026.1.0',
        'CFBundleShortVersionString': '2026.1',
        'NSHighResolutionCapable': True,
        'LSUIElement': True,
        'LSMinimumSystemVersion': '11.0',
        'NSHumanReadableCopyright': '© 2026 Kura Medical. All rights reserved.',
        'NSMicrophoneUsageDescription': (
            'Kura benötigt Zugriff auf das Mikrofon, um Ihre Therapiesitzungen '
            'aufzuzeichnen und SOAP-Befunde zu erstellen. Alle Aufnahmen werden '
            'lokal verarbeitet und automatisch gelöscht (DSGVO-konform).'
        ),
        'NSSpeechRecognitionUsageDescription': (
            'Kura verwendet Spracherkennung, um Ihre Therapiesitzungen in '
            'SOAP-Befunde umzuwandeln. Die Verarbeitung erfolgt 100% lokal auf Ihrem Mac.'
        ),
    },
)
