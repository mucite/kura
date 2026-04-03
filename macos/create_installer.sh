#!/bin/bash
# Create branded DMG with Kura.app + double-click installer script.
#
# Usage:
#   ./create_installer.sh [version]

set -e

# ── Cleanup helpers ───────────────────────────────────────────────────────────

# Tracks temp files/dirs created during this run so EXIT can remove them all.
_CLEANUP_PATHS=()
register_cleanup() { _CLEANUP_PATHS+=("$@"); }

cleanup() {
    # 1. Kill any background jobs still running (watchdog, tee)
    jobs -p | xargs -r kill 2>/dev/null || true

    # 2. Remove every temp file/dir registered during this run
    for path in "${_CLEANUP_PATHS[@]}"; do
        rm -rf "$path"
    done

    # 3. Detach any temp volumes left behind by create-dmg (named dmg.XXXXXXXX)
    local stale
    stale=$(hdiutil info 2>/dev/null | awk '/\/Volumes\/dmg\./{print $1}')
    if [ -n "$stale" ]; then
        echo "⚠️  Cleaning up stale DMG volume(s)..."
        while IFS= read -r dev; do
            hdiutil detach "$dev" -force 2>/dev/null && \
                echo "   ↳ detached $dev" || true
        done <<< "$stale"
    fi
}

trap cleanup EXIT

# Runs in the background alongside create-dmg.
# Monitors create-dmg's output log for the "Resource busy" string — only then
# does it force-detach, so AppleScript and other legitimate disk use is never
# interrupted. Throttled to one detach per 20 s to avoid spamming.
dmg_watchdog() {
    local pid=$1 logfile=$2
    local last_detach=-60
    # Track when each dmg.* volume first appeared so we can timeout on it
    declare -A vol_first_seen

    while kill -0 "$pid" 2>/dev/null; do
        sleep 3
        local now=$SECONDS

        [ $((now - last_detach)) -lt 20 ] && continue

        local stale
        stale=$(hdiutil info 2>/dev/null | awk '/\/Volumes\/dmg\./{print $1}')
        [ -z "$stale" ] && { unset vol_first_seen; declare -A vol_first_seen; continue; }

        local reason=''

        # Trigger 1: "Resource busy" seen in output
        if grep -q "Resource busy" "$logfile" 2>/dev/null; then
            reason='stuck unmount (Resource busy)'
        fi

        # Trigger 2: volume has been mounted for >60 s with no progress
        while IFS= read -r dev; do
            if [ -z "${vol_first_seen[$dev]+x}" ]; then
                vol_first_seen[$dev]=$now
            elif [ $((now - vol_first_seen[$dev])) -gt 60 ]; then
                reason="stuck volume ${dev} mounted for >60 s"
            fi
        done <<< "$stale"

        [ -z "$reason" ] && continue

        printf '\n⚠️  Watchdog: %s — force-detaching...\n' "$reason"
        while IFS= read -r dev; do
            hdiutil detach "$dev" -force 2>/dev/null \
                && printf '   ↳ detached %s\n' "$dev" || true
        done <<< "$stale"
        last_detach=$now
        unset vol_first_seen; declare -A vol_first_seen
    done
}

VERSION="${1:-v2026}"
DMG_NAME="Kura_Installer_${VERSION}.dmg"
STAGING="dist/dmg_staging"
register_cleanup "$STAGING"

cd "$(dirname "$0")"

echo "🧹 Cleaning staging area..."
rm -f "dist/${DMG_NAME}"
rm -rf "$STAGING"
mkdir -p "$STAGING"

# ── Prepare staging folder ────────────────────────────────────────────────────
echo "📋 Staging files..."
cp -R dist/Kura.app "$STAGING/"
cp install.command "$STAGING/Install Kura.command"

# Make the installer executable and strip its own quarantine
chmod +x "$STAGING/Install Kura.command"
xattr -rd com.apple.quarantine "$STAGING/Install Kura.command" 2>/dev/null || true

# Ad-hoc sign the installer script so macOS allows it to run
codesign --force --sign - "$STAGING/Install Kura.command" 2>/dev/null || true

echo "   ✅ Kura.app"
echo "   ✅ Install Kura.command"

# ── Build DMG ─────────────────────────────────────────────────────────────────
echo ""
echo "📦 Creating DMG (${DMG_NAME})..."

run_create_dmg() {
    create-dmg \
      --volname "Kura Medical ${VERSION}" \
      --volicon "../assets/stethoscope.icns" \
      --window-pos 200 120 \
      --window-size 660 420 \
      --icon-size 100 \
      --icon "Kura.app" 160 200 \
      --icon "Install Kura.command" 490 200 \
      --hide-extension "Kura.app" \
      --hide-extension "Install Kura.command" \
      "dist/${DMG_NAME}" \
      "$STAGING"
}

MAX_RETRIES=3
attempt=1
while true; do
    DMGLOG=$(mktemp /tmp/create-dmg-XXXXXX);      register_cleanup "$DMGLOG"
    EXITFILE=$(mktemp /tmp/create-dmg-exit-XXXXXX); register_cleanup "$EXITFILE"

    # Pipe create-dmg output through tee so the user sees it AND the watchdog
    # can grep for "Resource busy" to know when it's safe to force-detach.
    {
        exit_code=0
        run_create_dmg 2>&1 || exit_code=$?
        echo "$exit_code" > "$EXITFILE"
    } | tee "$DMGLOG" &
    TEE_PID=$!

    dmg_watchdog "$TEE_PID" "$DMGLOG" &
    WATCHDOG_PID=$!

    set +e
    wait "$TEE_PID"
    set -e

    kill "$WATCHDOG_PID" 2>/dev/null
    wait "$WATCHDOG_PID" 2>/dev/null

    CREATE_DMG_EXIT=$(cat "$EXITFILE" 2>/dev/null || echo 1)

    [ "$CREATE_DMG_EXIT" -eq 0 ] && break

    if [ $attempt -ge $MAX_RETRIES ]; then
        echo "❌ create-dmg failed after ${MAX_RETRIES} attempts (exit ${CREATE_DMG_EXIT})"
        exit "$CREATE_DMG_EXIT"
    fi

    echo "⚠️  create-dmg failed (attempt ${attempt}/${MAX_RETRIES}, exit ${CREATE_DMG_EXIT}) — retrying..."
    rm -f "dist/${DMG_NAME}"
    sleep 3
    attempt=$((attempt + 1))
done

echo "✅ Done: dist/${DMG_NAME}"
