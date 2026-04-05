"""
PyInstaller hook for llama-cpp-python
Ensures DLLs from lib/ subdirectory are included in the bundle
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, get_package_paths
import os

# Collect all dynamic libraries (DLLs)
binaries = collect_dynamic_libs('llama_cpp')

# Also explicitly add lib directory contents
try:
    _, llama_cpp_dir = get_package_paths('llama_cpp')
    lib_dir = os.path.join(llama_cpp_dir, 'lib')

    if os.path.exists(lib_dir):
        for file in os.listdir(lib_dir):
            filepath = os.path.join(lib_dir, file)
            if os.path.isfile(filepath) and file.endswith(('.dll', '.lib', '.so', '.dylib')):
                # Add to binaries with correct target path
                binaries.append((filepath, 'llama_cpp/lib'))
except Exception as e:
    print(f"Warning: Could not collect llama_cpp lib directory: {e}")

# Ensure package data is collected
datas = []

