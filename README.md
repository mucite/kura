# Kura Medical v2026

AI-powered medical documentation for physiotherapy. Generates SOAP notes with automatic ICD-10 coding (§ 84 SGB V compliant). 100% local processing, GDPR-safe.

---

## 🚀 Quick Start

| Need | Command | Time |
|------|---------|------|
| **Test the app** | See `QUICK_START.md` | 5 min |
| **Build for distribution** | See `BUILD_GUIDE.md` | 10 min |
| **Development setup** | See `INSTALL.md` | 30 min |
| **Release version** | See `RELEASE.md` | varies |

---

## ✨ Key Features

- ⚡ **SOAP-Befunde**: 25-35 seconds
- 🔒 **100% local** (GDPR-compliant, no cloud)
- 🏥 **§ 84 SGB V compliant**
- 🧠 **AI-powered**: Llama 3.2 3B + Whisper STT
- 💰 **Automatic ICD-10 coding**
- 📄 **PDF export**
- 🔐 **License management**

---

## 📦 Available Build Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `build_windows.bat` | Quick test | `dist\Kura\Kura.exe` |
| `build_release.bat v2026` | Professional installer | `dist\Kura_Setup_2026.exe` |
| `verify_build.bat` | Verify success | Report |
| `submit_to_microsoft.bat` | Microsoft whitelist | (approval email) |

See `SCRIPTS.md` for details.

---

## 💻 System Requirements

- **OS**: Windows 10+ or macOS 11+
- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 10GB free for models
- **Python**: 3.12+ (Windows) / 3.10+ (macOS)

---

## 📚 Documentation

- **QUICK_START.md** - Get running in 30 seconds
- **BUILD_GUIDE.md** - Complete build and distribution guide
- **INSTALL.md** - Development environment setup
- **RELEASE.md** - Release version process
- **SCRIPTS.md** - Build script reference

---

## 🔐 Code Signing

Professional code signing certificate included:
- **Valid**: 10 years (2026-2036)
- **Status**: Fully configured
- **Password**: Stored securely in environment variable `KURA_CERT_PASSWORD`
- Configure: Set `KURA_CERT_PASSWORD` in your CI/CD secrets or local `.env`
- Used automatically by build scripts

---

## 📋 Project Structure

```
medic/
├── windows/               # Windows app (our focus)
│   ├── main_windows.py
│   ├── physio_scribe_crossplatform.py
│   ├── build_windows.bat
│   ├── build_release.bat
│   └── kura_codesign.pfx (signing cert)
├── macos/                 # macOS app
├── shared/                # Cross-platform code
├── models/                # AI models (bundled)
│   ├── Llama-3.2-3B-Instruct-4bit-GGUF/
│   └── whisper/
├── INSTALL.md             # Setup guide
├── BUILD_GUIDE.md         # Build guide
├── QUICK_START.md         # 30-sec quick start
├── RELEASE.md             # Release process
└── SCRIPTS.md             # Script reference
```

---

## 🎯 Typical Workflows

### Test in 5 Minutes
```bash
cd windows
build_windows.bat
dist\Kura\Kura.exe
```

### Build Professional Installer (15 minutes)
```bash
REM 1. Download Inno Setup (one-time)
REM    https://jrsoftware.org/isdl.php

REM 2. Build
cd windows
build_release.bat v2026

REM 3. Share installer
REM    dist\Kura_Setup_2026.exe
```

### Remove SmartScreen Warning (1-3 days)
```bash
cd windows
submit_to_microsoft.bat v2026
```

---

## 📞 Support & Information

- **Website**: [kura-medical.de](https://kura-medical.de)
- **Issues**: Contact support
- **License**: Commercial (Free trial: 5 reports, Pro: €49/month)

---

**© 2026 Kura Medical | Made in Germany 🇩🇪**
