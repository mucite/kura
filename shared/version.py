"""
Single source of truth for Kura app version.
Both macos/main.py and windows/main_windows.py import from here.
Update APP_VERSION here before each release.
"""

_VERSION_URL = "https://pub-f83ad51a8a6d46859a3b16a78c2b95b3.r2.dev/version.json"

APP_VERSION = "2026.4.0"
VERSION_URL = _VERSION_URL


def version_gt(a: str, b: str) -> bool:
    """Return True if version string a is strictly greater than b."""
    try:
        return tuple(int(x) for x in a.split(".")) > tuple(int(x) for x in b.split("."))
    except Exception:
        return False
