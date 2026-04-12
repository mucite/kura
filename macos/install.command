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
    echo "❌ Kura.app nicht gefunden."
    echo "   Bitte laden Sie die DMG-Datei erneut von der offiziellen Website herunter."
    read -p "Enter drücken zum Schließen..."
    exit 1
fi

# ── Close running instance ────────────────────────────────────────────────────
if pgrep -x "Kura" &>/dev/null; then
    echo "⏹  Beende laufende Kura-Instanz..."
    pkill -x "Kura" 2>/dev/null || true
    sleep 1
fi

# ── Copy to /Applications ─────────────────────────────────────────────────────
echo "📂 Installiere Kura.app in /Applications..."
if [ -d "$APP_DEST" ]; then
    rm -rf "$APP_DEST"
fi
cp -R "$APP_SRC" "$APP_DEST"
echo "   ✅ Kopiert"

# ── Strip macOS quarantine ────────────────────────────────────────────────────
# Remove the quarantine extended attribute (first layer of Gatekeeper).
echo "🔓 Entferne macOS-Quarantäne..."
xattr -rd com.apple.quarantine "$APP_DEST" 2>/dev/null || true
echo "   ✅ Erledigt"

# ── Add to Gatekeeper allowlist ───────────────────────────────────────────────
# xattr alone is insufficient on macOS 14+: Gatekeeper also consults a security
# database. spctl --add registers the app as explicitly trusted.
# osascript shows the native macOS password dialog (friendlier than sudo prompt).
echo ""
echo "🔐 macOS-Sicherheitsfreigabe..."
echo "   macOS fragt jetzt nach Ihrem Mac-Passwort."
echo "   Das ist einmalig nötig, damit Kura ohne Warnung öffnet."
echo ""

osascript -e "do shell script \"spctl --add '${APP_DEST}'\" with administrator privileges" 2>/dev/null \
    && echo "   ✅ App freigegeben" \
    || echo "   ⚠️  Freigabe übersprungen – beim ersten Start ggf. Warnung bestätigen"

# ── Verify signature is intact after copy ────────────────────────────────────
# Do NOT re-sign here — codesign without --entitlements strips the JIT and
# microphone entitlements that were embedded during build, which breaks MLX
# and audio input. The original ad-hoc signature survives a plain cp -R.
codesign --verify --deep "$APP_DEST" 2>/dev/null \
    && echo "🔏 Signatur intakt" \
    || echo "⚠️  Signaturprüfung übersprungen"

# ── Grant microphone permission hint ─────────────────────────────────────────
echo ""
echo "🎙  Beim ersten Start fragt macOS nach dem Mikrofonzugang."
echo "   Bitte auf 'Erlauben' klicken – die Aufnahme erfolgt lokal,"
echo "   keine Audiodaten verlassen Ihren Mac."
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
echo "🚀 Starte Kura..."
open "$APP_DEST"

echo ""
echo "╔════════════════════════════════════════╗"
echo "║  Installation abgeschlossen!           ║"
echo "║  Das 🩺-Symbol erscheint in der        ║"
echo "║  Menüleiste Ihres Macs.                ║"
echo "╚════════════════════════════════════════╝"
echo ""
read -p "Enter drücken zum Schließen..."