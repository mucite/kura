# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs
import os

# ── Models are NOT bundled - they download on first launch ──────────────────
# This keeps the installer small (~50 MB instead of ~6 GB)
# Models download automatically from HuggingFace on first app launch

datas = [
    ('../core', 'core'),  # Include core (with model_downloader.py)
    ('../shared', 'shared'),
    ('../data', 'data'),  # GKV prices
    ('../.env.dist', '.'),  # Bundled .env with pre-configured HF_TOKEN
    ('../.env.example', '.'),
    ('physio_scribe_crossplatform.py', '.'),
]
binaries = []
hiddenimports = [
    'llama_cpp',
    'whisper',
    'faster_whisper',
    'PySimpleGUI',
    'sounddevice',
    'fpdf',
    'requests',
    'dotenv',
    'psutil',
    'numpy',
    'wave',
    'tiktoken_ext.openai_public',
    'tiktoken_ext',
    'physio_scribe_crossplatform',
    'shared.config_manager',
    'shared.license_manager',
    'shared.practice_config',
    'shared.learning_manager',
    'huggingface_hub',  # For model downloads on first launch
    'core.model_downloader',  # Model download logic
    'core.model_download_dialog',  # GUI dialog for model downloads
]

# Collect llama_cpp with its dynamic libraries
tmp = collect_all('llama_cpp')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# Explicitly add llama_cpp DLLs from lib directory
llama_cpp_libs = collect_dynamic_libs('llama_cpp')
if llama_cpp_libs:
    binaries += llama_cpp_libs
else:
    # Fallback: manually find llama_cpp lib directory
    try:
        import llama_cpp
        llama_cpp_dir = os.path.dirname(llama_cpp.__file__)
        lib_dir = os.path.join(llama_cpp_dir, 'lib')
        if os.path.exists(lib_dir):
            for file in os.listdir(lib_dir):
                if file.endswith(('.dll', '.lib')):
                    binaries.append((os.path.join(lib_dir, file), 'llama_cpp/lib'))
    except:
        pass

# Collect whisper with its data files (mel_filters.npz, tokenizers)
tmp = collect_all('whisper')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# Explicitly add whisper assets directory
try:
    import whisper
    whisper_dir = os.path.dirname(whisper.__file__)
    assets_dir = os.path.join(whisper_dir, 'assets')
    normalizers_dir = os.path.join(whisper_dir, 'normalizers')

    if os.path.exists(assets_dir):
        for file in os.listdir(assets_dir):
            filepath = os.path.join(assets_dir, file)
            if os.path.isfile(filepath):
                datas.append((filepath, 'whisper/assets'))

    if os.path.exists(normalizers_dir):
        for file in os.listdir(normalizers_dir):
            if file.endswith(('.json', '.tiktoken')):
                filepath = os.path.join(normalizers_dir, file)
                if os.path.isfile(filepath):
                    datas.append((filepath, 'whisper/normalizers'))
except Exception as e:
    print(f"Warning: Could not add whisper data files: {e}")

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
    hookspath=['hooks'],
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