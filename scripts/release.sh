#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Kura Medical — Release Script
# Usage: ./scripts/release.sh <version> [changelog]
# Example: ./scripts/release.sh 2026.4.0 "Fixed billing codes, improved MLD detection"
# ─────────────────────────────────────────────────────────────────────────────
set -e

VERSION="$1"
CHANGELOG="${2:-}"
TODAY=$(date +%Y-%m-%d)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version> [changelog]"
    echo "Example: $0 2026.4.0 \"Fixed MLD billing\""
    exit 1
fi

# Validate format: YYYY.N or YYYY.N.N
if ! echo "$VERSION" | grep -qE '^20[0-9]{2}\.[0-9]+(\.[0-9]+)?$'; then
    echo "ERROR: Version must be YYYY.N or YYYY.N.N (e.g. 2026.4.0)"
    exit 1
fi

echo "── Kura Release: v$VERSION ─────────────────────────────────"

# 1. Update version.json (single source of truth)
python3 -c "
import json
d = {
    'version': '$VERSION',
    'min_version': '2026.1',
    'release_date': '$TODAY',
    'changelog': '''$CHANGELOG'''
}
print(json.dumps(d, indent=4, ensure_ascii=False))
" > "$ROOT/version.json"
echo "✓ version.json → $VERSION"

# 2. Push version.json to Cloudflare R2
# Requires: aws CLI configured with R2 credentials
# Set env vars: R2_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, R2_ENDPOINT
if [ -n "$R2_BUCKET" ] && [ -n "$R2_ENDPOINT" ]; then
    aws s3 cp "$ROOT/version.json" "s3://$R2_BUCKET/version.json" \
        --endpoint-url "$R2_ENDPOINT" \
        --content-type "application/json" \
        --cache-control "no-cache"
    echo "✓ version.json pushed to R2 → users will see update notification"
else
    echo "⚠ R2 not configured — push manually:"
    echo "  aws s3 cp version.json s3://YOUR_BUCKET/version.json --endpoint-url YOUR_R2_ENDPOINT"
fi

# 3. Git commit + tag
cd "$ROOT"
git add version.json
git commit -m "release: v$VERSION"
git tag "v$VERSION"
echo "✓ git commit + tag v$VERSION"

echo ""
echo "── Done. Next steps: ────────────────────────────────────────"
echo "  1. Build macOS:   cd macos && ./build_release.sh v$VERSION"
echo "  2. Build Windows: cd windows && build_release.bat v$VERSION"
echo "  3. Push:          git push origin main --tags"
echo "  4. GitHub release: gh release create v$VERSION --title 'Kura v$VERSION'"