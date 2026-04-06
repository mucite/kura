# Kura for Windows - Build & Distribution

**Version:** 2026.3.0  
**Platform:** Windows 10/11 (64-bit)  
**Status:** ✅ Production Ready

This directory contains all Windows-specific files for building, packaging, and distributing Kura.

---

## 📋 Contents

### Core Application
- **`physio_scribe_crossplatform.py`** - Main application (3,396 lines, cross-platform compatible)
- **`main_windows.py`** - Windows entry point and launcher

### Build Configuration
- **`Kura_windows.spec`** - PyInstaller specification file
- **`build_optimized.ps1`** - Automated build script
- **`requirements-windows.txt`** - Python dependencies (moved to root)

### Installer & Distribution
- **`Kura.iss`** - Inno Setup installer script
- **`Kura_Installer.ps1`** - PowerShell installation helper
- **`Kura_Uninstaller.ps1`** - PowerShell uninstallation helper

### Documentation
- **`QUICK_START.md`** - User guide for end users
- **`README.md`** - This file (developer guide)

---

## 🚀 Quick Start

### For End Users
See **`QUICK_START.md`** for installation and usage instructions.

### For Developers

1. **Install dependencies**
   ```powershell
   pip install -r ..\requirements-windows.txt
   ```

2. **Run directly** (for development/testing)
   ```powershell
   python main_windows.py
   ```

3. **Build executable**
   ```powershell
   .\build_optimized.ps1
   ```

---

## 🔨 Building from Source

### Prerequisites
- Windows 10/11 (64-bit)
- Python 3.10, 3.11, or 3.12
- 10+ GB free disk space
- 8+ GB RAM

### Build Steps

1. **Create virtual environment** (recommended)
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**
   ```powershell
   pip install -r ..\requirements-windows.txt
   ```

3. **Run the build script**
   ```powershell
   .\build_optimized.ps1
   ```

4. **Test the build**
   ```powershell
   .\dist\Kura_windows\Kura.exe
   ```

**Expected build time:** 15-30 minutes  
**Expected size:** ~5-6 GB (includes AI models)

---

## 📦 Creating the Installer

1. **Install Inno Setup**
   ```powershell
   winget install JRSoftware.InnoSetup
   ```

2. **Compile the installer**
   - Open `Kura.iss` in Inno Setup Compiler
   - Press Ctrl+F9 to compile

3. **Output**
   - Installer: `Output/Kura_Setup_2026.3.0.exe`
   - Size: ~2-3 GB (compressed)

---

## 🎯 Key Features

- ✅ **AI-Powered Transcription** (Whisper)
- ✅ **Medical Note Generation** (Llama 3.2)
- ✅ **Billing Automation** (GKV codes)
- ✅ **PDF Invoice Generation**
- ✅ **System Tray Integration**
- ✅ **Cross-Platform Compatible**
- ✅ **Offline Operation** (no internet required)

---

## 📊 Performance

- **Startup**: 30-60 seconds (model loading)
- **Transcription**: 1-2 minutes (10-min recording)
- **Note Generation**: 5-15 seconds
- **Memory Usage**: ~3-4 GB
- **Disk Space**: ~5-6 GB installed

---

## 🔧 Troubleshooting

### Build Issues
- Ensure all dependencies installed: `pip install -r ..\requirements-windows.txt`
- Verify models exist: `..\models\whisper\medium.pt`
- Check disk space: Need 10+ GB free

### Runtime Issues
- Check `kura_error.log` in application directory
- Verify microphone permissions (Windows Settings → Privacy)
- Ensure 8+ GB RAM available

---

## 📚 Documentation

- **User Guide**: `QUICK_START.md`
- **Technical Summary**: `..\WINDOWS_BUILD_SUMMARY.md`
- **Deployment Checklist**: `..\DEPLOYMENT_CHECKLIST.md`
- **Implementation Summary**: `..\IMPLEMENTATION_SUMMARY.md`

---

## 🔐 Code Signing

Sign executables to prevent SmartScreen warnings:

```powershell
signtool sign /f kura_codesign.pfx /p PASSWORD /t http://timestamp.digicert.com .\dist\Kura_windows\Kura.exe
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

GPL-3.0 - See LICENSE file for details

---

## 📞 Support

- **Email**: support@kura-medical.com
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides included

---

**Last Updated:** April 5, 2026  
**Build Status:** ✅ Ready for Production

