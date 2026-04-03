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
if ! command -v pyinstaller &>/dev/null; then
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

# ── 3. Checksum ───────────────────────────────────────────────────────────────
echo ""
echo "🔐 Step 3/3 — SHA-256 checksum..."
cd dist
shasum -a 256 "${DMG_NAME}" > "${DMG_NAME}.sha256"
echo "   ✅ $(cat "${DMG_NAME}.sha256")"
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
echo ""
echo "What customers do:"
echo "  1. Download ${DMG_NAME}"
echo "  2. Open the DMG"
echo "  3. Double-click 'Install Kura' — done, no warnings"
echo ""
echo "Upload checklist:"
echo "  • Host the DMG on Google Drive / S3 / DigitalOcean"
echo "    (exceeds GitHub's 2 GB limit)"
echo "  • Upload the .sha256 file to the GitHub release"
echo "  • Paste the external download URL in release notes"
echo ""