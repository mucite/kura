"""
PyInstaller hook for openai-whisper
Ensures asset files (mel_filters.npz, tokenizers) are included in the bundle
"""
from PyInstaller.utils.hooks import collect_data_files, get_package_paths
import os

# Collect all data files from whisper package
datas = collect_data_files('whisper')

# Explicitly add assets directory to ensure mel_filters.npz is included
try:
    _, whisper_dir = get_package_paths('whisper')
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
    print(f"Warning: Could not collect whisper data files: {e}")

# No binaries needed for whisper
binaries = []

# Hidden imports for whisper dependencies
hiddenimports = [
    'tiktoken_ext.openai_public',
    'tiktoken_ext',
]

