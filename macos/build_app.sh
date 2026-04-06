#!/bin/bash
# Build Kura.app with ad-hoc signing.
# No Apple Developer account required.

set -e

cd "$(dirname "$0")"

# Activate virtualenv so pyinstaller and all dependencies are on PATH
VENV="$(dirname "$0")/../.venv"
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

echo "🧹 Cleaning old builds..."
rm -rf build dist

echo "📦 Building Kura.app with PyInstaller..."
pyinstaller Kura.spec --noconfirm

# Inject Info.plist (replace PyInstaller's minimal one)
echo "📝 Injecting Info.plist..."
cp Info.plist dist/Kura.app/Contents/Info.plist

# Ad-hoc sign with proper entitlements (allows microphone + MLX JIT to work)
echo "✍️  Signing app (ad-hoc)..."
codesign --remove-signature dist/Kura.app 2>/dev/null || true

# Sign all .so / .dylib files individually first — codesign --deep skips them
# if they are not inside a proper .framework bundle structure.
find dist/Kura.app/Contents -name "*.so" -o -name "*.dylib" | while read f; do
    codesign --force --sign - "$f" 2>/dev/null || true
done

# Now sign the whole bundle (entitlements go on the main executable)
codesign --force --deep \
         --sign - \
         --entitlements entitlements.plist \
         dist/Kura.app

echo "🔍 Verifying signature..."
codesign --verify --deep --verbose dist/Kura.app 2>&1 || echo "⚠️  Signature warnings above (ad-hoc — expected on Apple Silicon)"

# Set executable permissions
chmod -R 755 dist/Kura.app/Contents/MacOS/
chmod +x dist/Kura.app/Contents/MacOS/Kura

echo ""
echo "✅ Build complete: dist/Kura.app"
