# Kura Release Guide

---

## The Rule

**One file controls everything: `version.json` at project root.**

```
version.json  ← edit this → run release.sh → done
```

Never change `APP_VERSION` manually anywhere. It is read automatically from `version.json`.

---

## Version Format

```
YYYY.N.N   →  2026.4.0
```

- `YYYY` = year (always 2026 for now)
- `N` = release number (increment for each release)
- `.N` = patch/hotfix (increment for small fixes, omit for major releases)

---

## Releasing a New Version

```bash
./scripts/release.sh 2026.4.0 "What changed in this release"
```

That script does:
1. Updates `version.json` (date, changelog, version number)
2. Pushes `version.json` to Cloudflare R2 — **this triggers the in-app update notification for all users**
3. Creates a git commit + tag

Then build and publish:
```bash
cd macos && ./build_release.sh v2026.4.0     # macOS DMG
cd windows && build_release.bat v2026.4.0    # Windows ZIP
git push origin main --tags
gh release create v2026.4.0 --title "Kura v2026.4.0"
```

---

## What Each Version File Does

| File | Purpose | Who reads it |
|---|---|---|
| `version.json` (local) | Source of truth — version number + changelog | `shared/version.py` at startup |
| `version.json` (R2) | Triggers update notification in running apps | Both mains on boot + manual check |
| Gist `"version"` field | Config/billing data version — **not the app version** | `config_manager.py` for cache invalidation |

> The Gist version and app version are **independent**. Update the Gist version when billing codes or config change. Update `version.json` when releasing a new app binary.

---

## R2 Setup (one-time)

Set these environment variables (add to `~/.zshrc`):

```bash
export R2_BUCKET=your-bucket-name
export R2_ENDPOINT=https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com
export AWS_ACCESS_KEY_ID=your-r2-access-key
export AWS_SECRET_ACCESS_KEY=your-r2-secret-key
```

Then `release.sh` will push automatically.

---

## Hotfix (no binary rebuild needed)

If only fixing a bug that doesn't require a new DMG/ZIP:

```bash
./scripts/release.sh 2026.3.3 "Fixed X"
git push origin main --tags
# No binary rebuild needed — just the R2 version.json push
```

---

## 📦 Build Releases

### macOS Build

**Requirements**:
- PyInstaller: `pip install pyinstaller`
- create-dmg: `brew install create-dmg`

**Build**:
```bash
cd macos
./build_release.sh v2026
```

**Output** (in `macos/dist/`):
- `Kura_macOS_v2026.dmg` (~3.2 GB)
- `Kura_macOS_v2026.dmg.sha256`

**Time**: 10-15 minutes

**⚠️ GitHub has a 2 GB limit**: Host the DMG externally (Google Drive, AWS S3, etc.) and link to it in GitHub release notes.

---

### Windows Build

**Requirements**:
- PyInstaller: `pip install pyinstaller`

**Build** (on Windows PC):
```cmd
cd windows
build_release.bat v2026
```

**Output** (in `windows\dist\`):
- `Kura_Windows_v2026.zip` (~2-3 GB)
- `Kura_Windows_v2026.zip.sha256`

**Time**: 10-15 minutes

---

## 📤 Upload to GitHub

### 1. Host the macOS DMG externally

The macOS DMG (~3.2 GB) exceeds GitHub's 2 GB limit, so host it externally:

**Free options**:
- **Google Drive** — create shareable link
- **Dropbox** — public download link
- **MEGA** — 20 GB free storage

**Professional options**:
- **AWS S3** — ~$0.08/GB/month
- **DigitalOcean Spaces** — $5/month (250 GB included)
- **Cloudflare R2** — free egress

**Example with Google Drive**:
1. Upload `Kura_macOS_v2026.dmg` to Google Drive
2. Right-click → Share → Copy link
3. Convert the link to direct-download format
4. Use this link in the GitHub release notes

### 2. Navigate to GitHub Releases

https://github.com/mucite/kura/releases/new

### 3. Create the release

- **Tag**: `v2026`
- **Title**: `Kura Medical v2026`
- **Description**: copy the template below (update with your external DMG link)

### 4. Upload Windows files to GitHub

Drag and drop the Windows files into "Attach binaries":
- ✅ `Kura_Windows_v2026.zip` (~2 GB — fits on GitHub)
- ✅ `Kura_Windows_v2026.zip.sha256`

### 5. Publish

Click "Publish release".

---

## 📄 GitHub Release Description Template (German — customer-facing)

> This block is intentionally kept in German because it is shown to German customers on the GitHub release page. Do not translate it unless you're publishing an English version of the product.

```markdown
## 🏥 Kura Medical v2026

KI-gestützte Dokumentation für Physiotherapie mit automatischer SOAP-Befunderstellung.

### ✨ Features

- ⚡ **Blitzschnell**: SOAP-Befunde in 25-35 Sekunden
- 🔒 **DSGVO-konform**: 100% lokale Verarbeitung, keine Cloud
- 🏥 **§ 84 SGB V konform**: Medizinische Dokumentation nach Standards
- 🧠 **KI-optimiert**: MLX (macOS) / llama.cpp (Windows)
- 💰 **Automatische Abrechnung**: ICD-10 Kodierung inklusive

---

## 📥 Downloads

### 🍎 macOS Version

**Datei**: `Kura_macOS_v2026.dmg`

**Systemanforderungen**:
- macOS 11.0+ (Big Sur oder neuer)
- Apple Silicon (M1/M2/M3) oder Intel
- 8GB RAM (16GB empfohlen)
- 7GB freier Speicherplatz

**Installation**:
1. DMG-Datei herunterladen und öffnen
2. **Kura.app** in den **Programme**-Ordner ziehen (Drag & Drop)
3. DMG auswerfen (Rechtsklick im Finder → Auswerfen)
4. **Wichtig beim ersten Start**:
   - Öffne den **Programme**-Ordner
   - **Rechtsklick** auf Kura.app → **"Öffnen"** wählen
   - Sicherheitswarnung mit "Öffnen" bestätigen
   - (macOS Gatekeeper - nur beim ersten Mal nötig)
5. **Mikrofon-Berechtigung erlauben**:
   - macOS fragt beim ersten Start nach Mikrofon-Zugriff
   - **"OK"** oder **"Erlauben"** klicken
   - Ohne Mikrofon-Zugriff kann Kura nicht aufnehmen

**Nach Installation**:
- Kura erscheint als **🩺 Symbol in der Menüleiste** (oben rechts, neben der Uhr)
- **NICHT im Dock** - es ist ein Menüleisten-Programm
- Klick auf 🩺 öffnet das Menü
- Status sollte zeigen: "✅ Kura Bereit (Lokal & DSGVO)"
- Zum Beenden: Menü → "Beenden"

**Falls das Symbol nicht erscheint**:
- Terminal öffnen und eingeben: `open -a Kura`
- Oder: Kura.app aus Programme-Ordner erneut starten

**Falls "Mikrofon-Zugriff verweigert" Fehler**:
- Systemeinstellungen → Datenschutz & Sicherheit → Mikrofon
- Kura in der Liste suchen und Häkchen setzen
- Kura neu starten

**Deinstallation**:
1. Kura beenden (Menüleiste → Beenden)
2. Programme-Ordner öffnen → Kura.app in Papierkorb
3. Optional: `~/Library/Application Support/Kura/` löschen (Daten)

**Download**: [Link wird in GitHub Release angegeben - extern gehostet wegen 3.2GB Größe]

---

### 🪟 Windows Version

**Datei**: `Kura_Windows_v2026.zip`

**Systemanforderungen**:
- Windows 10/11 (64-bit)
- 8GB RAM (16GB empfohlen)
- 10GB freier Speicherplatz
- CUDA optional (für schnellere Verarbeitung)

**Installation**:
1. ZIP-Datei herunterladen
2. In beliebigen Ordner entpacken
3. Kura.exe ausführen

---

## 🔐 Sicherheit & Checksums

**macOS verifizieren**:
```bash
shasum -a 256 Kura_macOS_v2026.dmg
```

**Windows verifizieren**:
```powershell
Get-FileHash -Algorithm SHA256 Kura_Windows_v2026.zip
```

Vergleichen Sie die Ausgabe mit den `.sha256` Dateien.

---

## 🆓 Kostenlos testen

- **5 kostenlose Befunde** zum Testen
- Alle Funktionen freigeschaltet
- Keine Kreditkarte erforderlich

## 💎 Kura Pro

- **299 € einmalig** – Dauerlizenz, unbegrenzte Befunde, kein Abo
- Alle Updates bis **31.12.2027** inklusive (GKV-Preise, HMK, ICD-10-GM)
- Danach optional **79 €/Jahr** für weitere Regel-Updates
- [Lizenz kaufen →](https://kura.lemonsqueezy.com/checkout/buy/2400563b-a13a-4e42-b734-d79122e7ec92)

---

## 📞 Support

- **Email**: github.com/mucite/kura/issues
- **Website**: https://kura-medical.de
- **Issues**: [GitHub Issues](https://github.com/mucite/kura/issues)

---

**Release Date**: March 29, 2026
**Build**: Manual
**Compliance**: DSGVO/GDPR, § 84 SGB V
```

---

## ✅ Pre-Release Checklist

Before uploading:
- [ ] Both DMG and ZIP built successfully
- [ ] Checksums generated for both
- [ ] Tested DMG on clean macOS
- [ ] Tested ZIP on clean Windows
- [ ] All features working (recording, AI, PDF, license)
- [ ] Version numbers correct

---

## 🔗 Download Links (after publishing)

**macOS**:
```
https://github.com/mucite/kura/releases/download/v2026/Kura_macOS_v2026.dmg
```

**Windows**:
```
https://github.com/mucite/kura/releases/download/v2026/Kura_Windows_v2026.zip
```

---

## 🔄 Future Releases

To release a new version (e.g., v2027):

1. Update the version in scripts if needed
2. Build on both platforms
3. Create a new GitHub release
4. Upload new files
5. Publish

---

## 🛠️ Troubleshooting

### macOS build fails
```bash
pip install pyinstaller
brew install create-dmg
chmod +x macos/*.sh
```

### Windows build fails
```cmd
pip install pyinstaller
pip install -r requirements-windows.txt
```

### DMG creation fails
- Ensure `create-dmg` is installed
- Check that Kura.app was built successfully
- Verify the icon file exists: `assets/stethoscope.icns`

### macOS DMG too large for GitHub
- **Solution**: host externally (Google Drive, AWS S3, etc.)
- Link to it in GitHub release notes
- Upload the checksum file to GitHub for verification
- Windows ZIP usually fits within the 2 GB limit

---

## 🔒 Privacy

**What stays private**:
- ✅ All source code (.py files)
- ✅ `.env` file (secrets, API keys)
- ✅ Development files
- ✅ Build configurations

**What gets published**:
- ✅ Compiled executables only (DMG/ZIP)
- ✅ Checksums (.sha256)
- ✅ Release notes

---

**No CI/CD • No source code upload • Just executables**