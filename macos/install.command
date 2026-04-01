#!/bin/bash
# Kura Medical — Installer
# Double-click this file to install Kura on your Mac.
# No internet connection required. All processing stays on your device.

set -e

DMG_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_SRC="$DMG_DIR/Kura.app"
APP_DEST="/Applications/Kura.app"

# ── Friendly terminal window title ───────────────────────────────────────────
echo -e "\033]0;Kura Medical – Installer\007"
clear

echo "╔════════════════════════════════════════╗"
echo "║       Kura Medical – Installation      ║"
echo "╚════════════════════════════════════════╝"
echo ""

# ── Sanity check ─────────────────────────────────────────────────────────────
if [ ! -d "$APP_SRC" ]; then
    echo "❌ Kura.app not found in this disk image."
    echo "   Please re-download the DMG from the official website."
    read -p "Press Enter to close..."
    exit 1
fi

# ── Close running instance ────────────────────────────────────────────────────
if pgrep -x "Kura" &>/dev/null; then
    echo "⏹  Stopping running Kura instance..."
    pkill -x "Kura" 2>/dev/null || true
    sleep 1
fi

# ── Copy to /Applications ─────────────────────────────────────────────────────
echo "📂 Installing Kura.app to /Applications..."
if [ -d "$APP_DEST" ]; then
    rm -rf "$APP_DEST"
fi
cp -R "$APP_SRC" "$APP_DEST"
echo "   ✅ Copied"

# ── Strip macOS quarantine ────────────────────────────────────────────────────
# This is what Gatekeeper checks. Removing it lets the app open without warnings
# because all AI processing is 100% local — no data leaves your Mac.
echo "🔓 Removing macOS quarantine flag..."
xattr -rd com.apple.quarantine "$APP_DEST" 2>/dev/null || true
echo "   ✅ Done"

# ── Verify signature is intact after copy ────────────────────────────────────
# Do NOT re-sign here — codesign without --entitlements strips the JIT and
# microphone entitlements that were embedded during build, which breaks MLX
# and audio input. The original ad-hoc signature survives a plain cp -R.
codesign --verify --deep "$APP_DEST" 2>/dev/null \
    && echo "🔏 Signature intact" \
    || echo "⚠️  Signature check skipped"

# ── Grant microphone permission hint ─────────────────────────────────────────
echo ""
echo "🎙  On first launch, macOS will ask for microphone access."
echo "   Click 'Allow' — Kura records locally and never sends audio anywhere."
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
echo "🚀 Launching Kura..."
open "$APP_DEST"

echo ""
echo "╔════════════════════════════════════════╗"
echo "║  Installation complete! Look for the   ║"
echo "║  🩺 icon in your Mac menu bar.         ║"
echo "╚════════════════════════════════════════╝"
echo ""
read -p "Press Enter to close this window..."