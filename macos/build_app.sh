#!/bin/bash
# Build Kura.app with ad-hoc signing.
# No Apple Developer account required.

set -e

cd "$(dirname "$0")"

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
codesign --force --deep \
         --sign - \
         --entitlements entitlements.plist \
         dist/Kura.app

echo "🔍 Verifying signature..."
codesign --verify --deep --verbose dist/Kura.app

# Set executable permissions
chmod -R 755 dist/Kura.app/Contents/MacOS/
chmod +x dist/Kura.app/Contents/MacOS/Kura

echo ""
echo "✅ Build complete: dist/Kura.app"
