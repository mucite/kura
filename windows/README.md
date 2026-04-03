# Windows Version

GUI application for Windows with CPU-friendly AI inference.

## Quick Start

```batch
copy ..\env.example ..\.env
pip install -r ../requirements-windows.txt
python main_windows.py
```

## Build

```batch
build_windows.bat              REM Create Kura.exe
create_installer_windows.bat   REM Create NSIS installer
```

## Files

- `main_windows.py` - GUI app (PySimpleGUI)
- `physio_scribe_crossplatform.py` - Cross-platform AI engine
- `build_windows.bat` - PyInstaller script
- `create_installer_windows.bat` - Installer script

## Performance

- Startup: 3-5s
- Transcription: 3-8s
- AI Analysis: 15-25s (CPU) or 8-15s (GPU)
- **Total:** 30-45s (CPU) or 25-35s (GPU)

## Model Options

- **Ollama** (recommended): https://ollama.ai
- **Transformers + Torch:** `pip install torch transformers`
- **llama-cpp-python:** Lightweight CPU inference

See `../README.md` and `../INSTALLATION.md` for full documentation.

