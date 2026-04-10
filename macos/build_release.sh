#!/bin/bash
# Build Kura macOS release DMG.
# No Apple Developer account required.
#
# Usage:
#   ./build_release.sh [version]
#   ./build_release.sh v2026.1

set -e

VERSION="${1:-v2026}"
DMG_NAME="Kura_macOS_${VERSION}.dmg"
INSTALLER_INTERMEDIATE="Kura_Installer_${VERSION}.dmg"

cd "$(dirname "$0")"

echo "🍎 Building Kura macOS Release ${VERSION}"
echo "════════════════════════════════════════════"
echo ""

# ── Dependency check ──────────────────────────────────────────────────────────
if ! command -v create-dmg &>/dev/null; then
    echo "❌ create-dmg not found. Install with: brew install create-dmg"
    exit 1
fi
# pyinstaller may live in the project venv
VENV="$(dirname "$0")/../.venv"
if command -v pyinstaller &>/dev/null; then
    : # found on PATH — fine
elif [ -x "$VENV/bin/pyinstaller" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "❌ pyinstaller not found. Install with: pip install pyinstaller"
    exit 1
fi

# ── 1. Build Kura.app ─────────────────────────────────────────────────────────
echo "📦 Step 1/3 — Building Kura.app..."
./build_app.sh

if [ ! -d "dist/Kura.app" ]; then
    echo "❌ Build failed: dist/Kura.app not found"
    exit 1
fi

APP_SIZE=$(du -sh dist/Kura.app | cut -f1)
echo "   ✅ Kura.app built (${APP_SIZE})"

# ── 2. Create DMG ─────────────────────────────────────────────────────────────
echo ""
echo "📦 Step 2/3 — Creating DMG installer..."
./create_installer.sh "${VERSION}"

if [ ! -f "dist/${INSTALLER_INTERMEDIATE}" ]; then
    echo "❌ DMG creation failed"
    exit 1
fi

mv "dist/${INSTALLER_INTERMEDIATE}" "dist/${DMG_NAME}"
DMG_SIZE=$(du -sh "dist/${DMG_NAME}" | cut -f1)
echo "   ✅ DMG: ${DMG_NAME} (${DMG_SIZE})"

# ── 3. Create ZIP ─────────────────────────────────────────────────────────────
ZIP_NAME="Kura_macOS_${VERSION}.zip"
echo ""
echo "📦 Step 3/4 — Creating ZIP..."
cd dist
zip -r --quiet "${ZIP_NAME}" Kura.app
ZIP_SIZE=$(du -sh "${ZIP_NAME}" | cut -f1)
echo "   ✅ ZIP: ${ZIP_NAME} (${ZIP_SIZE})"
cd ..

# ── 4. Checksums ──────────────────────────────────────────────────────────────
echo ""
echo "🔐 Step 4/4 — SHA-256 checksums..."
cd dist
shasum -a 256 "${DMG_NAME}" > "${DMG_NAME}.sha256"
shasum -a 256 "${ZIP_NAME}" > "${ZIP_NAME}.sha256"
echo "   ✅ $(cat "${DMG_NAME}.sha256")"
echo "   ✅ $(cat "${ZIP_NAME}.sha256")"
cd ..

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "✅ Build complete!"
echo "════════════════════════════════════════════"
echo ""
echo "Output:"
echo "  📦 dist/${DMG_NAME}  (${DMG_SIZE})"
echo "  🔐 dist/${DMG_NAME}.sha256"
echo "  📦 dist/${ZIP_NAME}  (${ZIP_SIZE})"
echo "  🔐 dist/${ZIP_NAME}.sha256"
echo ""
echo "What customers do:"
echo "  DMG: Open → Double-click 'Install Kura' — done, no warnings"
echo "  ZIP: Extract → drag Kura.app to /Applications"
echo ""
echo "Upload checklist:"
echo "  • Host DMG and ZIP on Google Drive / S3 / DigitalOcean"
echo "    (both likely exceed GitHub's 2 GB limit)"
echo "  • Upload the .sha256 files to the GitHub release"
echo "  • Paste the external download URLs in release notes"
echo ""