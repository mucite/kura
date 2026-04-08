# Kura Medical v2026

AI-powered medical documentation for German physiotherapy. Records sessions via voice, generates SOAP notes with automatic ICD-10 coding, and validates billing compliance (§125 SGB V / HMK 2026). 100% local processing — GDPR-safe, no cloud.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Architecture](#architecture)
8. [Testing](#testing)
9. [Building & Deployment](#building--deployment)
10. [Maintenance](#maintenance)
11. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# macOS
open Kura_macOS_v2026.dmg          # drag to Applications, launch

# Windows
.\Kura_Setup_2026.exe              # run installer, launch from Start Menu
```

Developer setup:

```bash
git clone https://github.com/mucite/kura.git
cd kura
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env              # fill in HF_TOKEN and DS24_API_KEY
```

---

## Features

| Feature | Detail |
|---|---|
| Voice-to-SOAP | Whisper STT → LLaMA 3.1 8B → structured S/O/A/P in ~25-35 s |
| Billing compliance | §106b audit engine, HMK 2026, GKV/PKV/BG dispatcher |
| ICD-10-GM coding | Automatic extraction + Diagnosegruppe mapping |
| 100% local | No data leaves the device — GDPR / DSGVO compliant |
| Multi-platform | macOS (Metal) and Windows (CUDA / CPU) |
| License management | Trial (5 reports) · Pro (€49/month) via Digistore24 |
| PDF export | Compliant SOAP note PDFs ready for file |
| Annual pricing | GKV prices loaded from `data/gkv_prices_YYYY.json` |

---

## System Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | macOS 11+ / Windows 10+ | macOS 13+ / Windows 11 |
| RAM | 16 GB | 16 GB |
| Disk | 8 GB free | 20 GB free |
| GPU | CPU fallback supported | Apple M1+ / NVIDIA 4 GB VRAM |
| Python | 3.10+ (macOS) / 3.12+ (Windows) | 3.12 |

---

## Installation

### End Users

**macOS:**
```bash
# Download, mount DMG, drag Kura.app to /Applications
# First launch: grant microphone permission when prompted
```

**Windows:**
```powershell
# Run Kura_Setup_2026.exe
# First launch: grant microphone permission if prompted
```

### Developers

```bash
git clone https://github.com/mucite/kura.git
cd kura

python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-test.txt   # for running tests

cp .env.template .env
```

---

## Configuration

### Environment Variables

Create `~/Documents/Kura/.env` (or set in your shell):

```bash
# RECOMMENDED: HuggingFace token for faster model downloads (FREE)
# Without this, downloads will be 10x slower with lower rate limits
# Get yours at: https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_token_here

# Required for Pro license: Digistore24 credentials
DS24_API_KEY=your_api_key
DS24_PRODUCT_ID=681469

# Optional: code signing (for builds)
KURA_CERT_PASSWORD=your_cert_password

# Optional: logging verbosity
KURA_LOG_LEVEL=INFO     # DEBUG | INFO | WARNING | ERROR
```

**⚡ Quick HF_TOKEN Setup:**

```bash
# Interactive setup (recommended)
python setup_hf_token.py

# Or direct setup
python setup_hf_token.py hf_your_token_here
```

For detailed instructions, see: **[HF_TOKEN_SETUP.md](HF_TOKEN_SETUP.md)**

> **Why set HF_TOKEN?** Model downloads will be **10x faster** with higher rate limits and better reliability. It's completely FREE and takes 2 minutes. Without it, you'll see warnings and slower downloads.

### Practice Configuration

Edit `~/.kura_practice.json` or use the in-app settings:

```json
{
  "practice": {
    "name": "Physiotherapie Mustermann",
    "license_number": "123456789",
    "location": "Berlin"
  }
}
```

### GKV Pricing

Pricing is read from `data/gkv_prices_YYYY.json`. The current year is loaded automatically; if not found, 2026 prices are used as fallback. To update for a new year, drop in `data/gkv_prices_2027.json`.

---

## Usage

### Typical Session

1. **Open Kura** — macOS menu bar icon or Windows Start Menu
2. **New session** — enter patient name, select insurance type (GKV / PKV / BG)
3. **Record** — click "Start", dictate the session naturally (measurements, tests, findings)
4. **Stop** — AI generates SOAP note in ~30 seconds
5. **Review** — edit if needed, then "Save & Export PDF"
6. **Billing** — §106b audit result shown automatically; missing fields and red flags highlighted

### Trial vs. Pro

| | Trial | Pro |
|---|---|---|
| Reports | 5 free | Unlimited |
| Billing audit | Full | Full |
| Custom ICD-10 rules | — | Yes |
| Practice config | — | Yes |
| Price | Free | €49/month |

---

## Architecture

```
┌─────────────────────────────────────┐
│  Platform Frontends                 │
│  macOS (menu bar)  Windows (GUI)    │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  core/                              │
│  ├── ai/          Whisper + LLaMA   │
│  ├── config/      JSON config load  │
│  ├── errors.py    structured errors │
│  ├── logging.py   rotating logs     │
│  └── health.py    startup checks    │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  shared/                            │
│  ├── billing_engine.py  HMK 2026    │
│  ├── license_manager.py DS24 HMAC   │
│  ├── config_manager.py  3-layer     │
│  └── practice_config.py per-Praxis  │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  data/                              │
│  └── gkv_prices_2026.json           │
└─────────────────────────────────────┘
```

### Billing Engine

`shared/billing_engine.py` implements three deterministic engines dispatched by insurance type:

- **`_GKVEngine`** — Heilmittelkatalog 2026, §106b audit items, Diagnosegruppe matching, fixed GKV prices
- **`_PKVEngine`** — documentation completeness scoring, reimbursement likelihood estimate
- **`_BGEngine`** — BG-specific position codes and documentation requirements

ICD-10 codes are mapped to Diagnosegruppen (e.g. WS1a, EX2, LY1) which determine the billing position, required documentation, and Regelfall unit limits.

### AI Pipeline

1. **Whisper** transcribes the session audio (large-v3-turbo)
2. **LLaMA 3.1 8B** (GGUF 4-bit) generates the SOAP note from the transcript
3. ICD-10 extraction parses the LLM output (pre-differential-diagnosis section preferred)
4. Billing engine evaluates compliance independently of the LLM

---

## Testing

```bash
# All tests
pytest tests/ -v

# Skip slow / model-dependent tests
pytest tests/ -v -m "not slow and not requires_models"

# With coverage report
pytest tests/ -v --cov=core --cov=shared --cov-report=html
open htmlcov/index.html
```

Test modules:

| File | Covers |
|---|---|
| `tests/test_billing_engine.py` | AuditItem, ICD matching, GKV/PKV/BG engines (70+ tests) |
| `tests/test_license_manager.py` | Trial flow, HMAC tamper detection, save/load (30+ tests) |
| `tests/test_health.py` | Python version, license status mapping, check_all (17+ tests) |
| `tests/test_errors.py` | safe_execute, validate_input, ErrorRecovery |
| `tests/test_config.py` | Config loading and merging |

CI runs on every push/PR via `.github/workflows/ci.yml` (ruff, mypy, bandit, pytest+coverage on Python 3.11 + 3.12).

---

## Building & Deployment

### macOS

```bash
cd macos
./build_release.sh v2026.4.1
# Output: dist/Kura_macOS_v2026.4.1.dmg
```

### Windows

```powershell
cd windows
.\build_release.bat v2026.4.1
# Output: dist\Kura_Setup_2026.4.1.exe
```

### Creating a Release

```bash
git tag -a v2026.4.1 -m "Release v2026.4.1"
git push origin v2026.4.1
# GitHub Actions builds executables and attaches them to the GitHub Release
```

---

## Maintenance

### Annually (November–December)

- Update GKV pricing: add `data/gkv_prices_YYYY.json`
- Review §125 SGB V / HMK for rule changes
- Evaluate newer LLM checkpoints
- Renew code signing certificate (valid 2026–2036)

### Dependencies

```bash
pip list --outdated          # check for updates
safety check                 # security scan
ruff check core/ shared/     # lint
```

### Log Files

**macOS:**
```bash
tail -f ~/Library/Logs/Kura/kura.log
```

**Windows:**
```powershell
Get-Content $env:LOCALAPPDATA\Kura\Logs\kura.log -Tail 50 -Wait
```

---

## Troubleshooting

### "Models not found"

```bash
ls -la models/
# Expected: Llama-3.1-8B-Instruct-*/ and whisper-large-v3-turbo/
# If missing: reinstall the app or re-run the model download script
```

### "GPU / Metal Error"

Restart the machine to clear the Metal shader cache. Kura falls back to CPU automatically (slower but functional).

### "License validation failed"

```bash
ping api.digistore24.com          # check connectivity
cat ~/Documents/Kura/.env | grep DS24   # check credentials
# Offline grace period: 3 days without internet before blocking
```

### "Pricing data expired"

The app warns but continues using 2026 prices. Drop the new year's file into `data/` and restart.

### Health Check

```bash
python -m core.health
# ✅ Python Version: 3.12
# ✅ Memory: 12.3GB / 16.0GB available
# ✅ AI Models: All present
# ✅ License: Pro active
# Status: HEALTHY
```

### Support

- Email: support@kura-medical.de
- Website: [kura-medical.de](https://kura-medical.de)
- Include: OS version, Kura version, `kura.log`

---

## Project Structure

```
medic/
├── core/                  # Platform-agnostic core
│   ├── ai/                # Whisper + LLaMA pipeline
│   ├── config/            # Config loading
│   ├── errors.py
│   ├── health.py
│   └── logging.py
├── shared/                # Business logic
│   ├── billing_engine.py  # HMK 2026 compliance
│   ├── license_manager.py
│   ├── config_manager.py
│   └── practice_config.py
├── macos/                 # macOS frontend
├── windows/               # Windows frontend
├── data/                  # GKV pricing JSON
├── tests/                 # pytest suite
├── .github/workflows/     # CI/CD
└── pyproject.toml         # ruff / mypy / pytest config
```

---

## License

Commercial Software — Proprietary License  
© 2026 Kura Medical GmbH. All rights reserved.  
Sales & licensing: sales@kura-medical.de

---

**Version**: v2026.4.1 | **Updated**: 2026-04-05 | Made in Germany