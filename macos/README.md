# macOS Version

Menu bar application optimized for Apple Silicon.

## Quick Start

```bash
cp ../env.example ../.env
pip install -r ../requirements.txt
python main.py
```

Alternatively, run the automated installer for a practice setup (creates a venv, installs deps, and sets up user config):

```bash
bash install_for_practice.sh
```

## Build

```bash
bash build_app.sh              # Create Kura.app
bash create_installer.sh       # Create DMG
```

If you're deploying to multiple machines in a practice, the `install_for_practice.sh` script can be used on each Mac to standardize the environment.

## Files

- `main.py` - Menu bar app (rumps)
- `physio_scribe.py` - MLX-optimized AI engine
- `build_app.sh` - PyInstaller script
- `create_installer.sh` - DMG creation

## Performance

- Startup: 3-5s
- Transcription: 2-5s  
- AI Analysis: 8-12s
- **Total:** 25-35s

## Troubleshooting

### "The application 'Kura' can't be opened"

1. **Check if .env is included:**
   ```bash
   ls -la dist/Kura.app/Contents/Resources/.env
   ```

2. **Verify build:**
   ```bash
   bash test_app.sh
   ```

3. **Check crash logs:**
   ```bash
   ls -la ~/Library/Logs/Kura/
   ```

4. **Rebuild:**
   ```bash
   rm -rf build dist
   bash build_app.sh
   ```

5. **Open with right-click:** Right-click → Open (bypasses Gatekeeper)

See `BUILD_VERIFICATION.md` for detailed troubleshooting and `../INSTALLATION.md` for full documentation.

