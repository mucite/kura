# Kura for Windows - Build & Distribution

**Version:** 2026.4.1  
**Platform:** Windows 10/11 (64-bit)  
**Status:** Production Ready

This directory contains all Windows-specific files for building, packaging, and distributing Kura.

---

## Contents

### Core Application
- **`physio_scribe_crossplatform.py`** - Main application logic (cross-platform compatible)
- **`main_windows.py`** - Windows entry point and launcher

### Build Configuration
- **`Kura_windows.spec`** - PyInstaller specification file
- **`build_msix.ps1`** - Automated MSIX package build script

### MSIX Packaging
- **`msix/AppxManifest.xml`** - Package manifest (generated at build time)
- **`msix/Assets/`** - Store logos and tile images

---

## Quick Start

### Run for Development

```powershell
pip install -r ..\requirements-windows.txt
python main_windows.py
```

### Build MSIX Package

```powershell
# Full build (PyInstaller + MSIX)
.\build_msix.ps1 -Version "2026.4.1"

# Skip PyInstaller if dist\Kura already exists
.\build_msix.ps1 -Version "2026.4.1" -SkipPyInstaller

# With code signing (sideload / enterprise)
.\build_msix.ps1 -Version "2026.4.1" -CertPfx ".\cert.pfx" -CertPassword "pass"
```

Output: `dist\Kura_2026.4.1.msix`

---

## Building from Source

### Prerequisites
- Windows 10/11 (64-bit)
- Python 3.12
- Windows SDK (for `makeappx.exe`) — install via [Windows SDK](https://developer.microsoft.com/windows/downloads/windows-sdk/)
- 10+ GB free disk space
- 8+ GB RAM

### Steps

1. Create a virtual environment
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies
   ```powershell
   pip install -r ..\requirements-windows.txt
   ```

3. Build
   ```powershell
   .\build_msix.ps1 -Version "2026.4.1"
   ```

**Expected build time:** 15–30 minutes  
**Expected MSIX size:** ~2–3 GB (models download on first launch)

---

## MSIX Distribution

The MSIX package can be distributed in three ways:

| Method | Signing required | Notes |
|--------|-----------------|-------|
| Microsoft Store | No (Store signs during ingestion) | Preferred for consumer distribution |
| Enterprise / MDM | Yes (trusted cert or self-signed) | Sideload via Intune / DISM |
| Direct sideload | Yes | Pass `-CertPfx` to `build_msix.ps1` |

For Store submission, upload the unsigned `.msix` directly in Partner Center.

---

## Code Signing (Sideload / Enterprise)

```powershell
signtool sign /fd SHA256 /f cert.pfx /p PASSWORD `
  /tr http://timestamp.digicert.com /td SHA256 `
  dist\Kura_2026.4.1.msix
```

Or pass `-CertPfx` and `-CertPassword` to `build_msix.ps1` to sign during the build.

---

## Key Features

- AI-Powered Transcription (Whisper / faster-whisper)
- Medical Note Generation (Llama 3)
- GKV Billing Automation — §125 SGB V Anlage 2, Optica Tarifcode 22 (01.01.2026)
- PDF Invoice Generation
- System Tray Integration
- Offline Operation (models downloaded once on first launch)

---

## Performance

| Metric | Value |
|--------|-------|
| Startup | 30–60 s (model loading) |
| Transcription (10 min audio) | 1–2 min |
| Note generation | 5–15 s |
| RAM usage | ~3–4 GB |
| Installed size | ~5–6 GB |

---

## Troubleshooting

**Build fails — makeappx not found**  
Install the Windows SDK and ensure `makeappx.exe` is reachable from PATH or a standard SDK path.

**Runtime — app won't start**  
Check `kura_error.log` in `%APPDATA%\Kura\`. Verify microphone permissions in Windows Settings → Privacy → Microphone.

**Models not downloading**  
Ensure outbound HTTPS to `huggingface.co` is not blocked. Check `%APPDATA%\Kura\` for partial downloads.

---

## Support

- **Email**: support@kura-medical.com
- **GitHub Issues**: Bug reports and feature requests

---

**Last Updated:** April 2026