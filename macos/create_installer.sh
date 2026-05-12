#!/bin/bash
# Create branded DMG with Kura.app + double-click installer script.
#
# Usage:
#   ./create_installer.sh [version]
#
# Compatible with bash 3.2 (macOS default).

set -e

VERSION="${1:-v2026}"
DMG_NAME="Kura_Installer_${VERSION}.dmg"
STAGING="dist/dmg_staging"

SIGN_IDENTITY="Developer ID Application: Musie Kebede Gizaw (NY589846RW)"
NOTARY_PROFILE="kura-notary"

cd "$(dirname "$0")"

# ── Cleanup ───────────────────────────────────────────────────────────────────
cleanup() {
    rm -rf "$STAGING"
    # Detach any stale create-dmg volumes
    hdiutil info 2>/dev/null \
        | awk '/\/Volumes\/dmg\./{print $1}' \
        | while IFS= read -r dev; do
            hdiutil detach "$dev" -force 2>/dev/null || true
          done
}
trap cleanup EXIT

# ── Staging ───────────────────────────────────────────────────────────────────
echo "🧹 Cleaning staging area..."
rm -f "dist/${DMG_NAME}"
rm -rf "$STAGING"
mkdir -p "$STAGING"

echo "🎨 Generating DMG background..."
python3 "$(dirname "$0")/create_dmg_background.py"
echo ""

echo "📋 Staging files..."
cp -R dist/Kura.app "$STAGING/"
cp install.command "$STAGING/Install Kura.command"
chmod +x "$STAGING/Install Kura.command"
xattr -rd com.apple.quarantine "$STAGING/Install Kura.command" 2>/dev/null || true
# Do NOT codesign the .command file — ad-hoc signing triggers the "Apple could not
# verify … malware" dialog. Unsigned + quarantine shows only the softer
# "downloaded from internet, are you sure?" prompt with a single [Open] click.
echo "   ✅ Kura.app"
echo "   ✅ Install Kura.command"

echo ""
echo "🔏 Code-signing Kura.app with hardened runtime..."
codesign --deep --force --options runtime --timestamp \
    --entitlements entitlements.plist \
    --sign "$SIGN_IDENTITY" \
    "$STAGING/Kura.app"
codesign --verify --deep --strict --verbose=2 "$STAGING/Kura.app"
echo "   ✅ Signed and verified"

# ── Build DMG ─────────────────────────────────────────────────────────────────
echo ""
echo "📦 Creating DMG (${DMG_NAME})..."

MAX_RETRIES=3
attempt=1
while true; do
    set +e
    create-dmg \
      --volname "Kura Medical ${VERSION}" \
      --volicon "../assets/stethoscope.icns" \
      --background "assets/dmg_background.png" \
      --window-pos 200 120 \
      --window-size 660 420 \
      --icon-size 100 \
      --icon "Kura.app" 160 200 \
      --icon "Install Kura.command" 490 200 \
      --hide-extension "Kura.app" \
      --hide-extension "Install Kura.command" \
      "dist/${DMG_NAME}" \
      "$STAGING"
    CREATE_DMG_EXIT=$?
    set -e

    [ "$CREATE_DMG_EXIT" -eq 0 ] && break

    if [ $attempt -ge $MAX_RETRIES ]; then
        echo "❌ create-dmg failed after ${MAX_RETRIES} attempts (exit ${CREATE_DMG_EXIT})"
        exit "$CREATE_DMG_EXIT"
    fi

    echo "⚠️  create-dmg failed (attempt ${attempt}/${MAX_RETRIES}) — retrying in 3s..."
    rm -f "dist/${DMG_NAME}"
    sleep 3
    attempt=$((attempt + 1))
done

echo "✅ DMG built: dist/${DMG_NAME}"

# ── Notarize ──────────────────────────────────────────────────────────────────
echo ""
echo "📤 Submitting DMG to Apple notary service (typically 2–10 min)..."
xcrun notarytool submit "dist/${DMG_NAME}" \
    --keychain-profile "$NOTARY_PROFILE" \
    --wait

echo "📎 Stapling notarization ticket to DMG..."
xcrun stapler staple "dist/${DMG_NAME}"

echo ""
echo "🔍 Gatekeeper verification:"
spctl -a -vvv -t open --context context:primary-signature "dist/${DMG_NAME}" || true

echo ""
echo "✅ Notarized and stapled: dist/${DMG_NAME}"