"""
Single source of truth for Kura app version.
Both macos/main.py and windows/main_windows.py import from here.
"""
import json
import os
import sys

_VERSION_URL = "https://pub-f83ad51a8a6d46859a3b16a78c2b95b3.r2.dev/version.json"


def _project_root() -> str:
    """Resolve project root whether running from source or PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        # PyInstaller: _MEIPASS is the unpacked temp dir; version.json is bundled there
        return sys._MEIPASS
    # Running from source: shared/ is one level below root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_app_version() -> str:
    try:
        path = os.path.join(_project_root(), "version.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)["version"]
    except Exception as e:
        raise RuntimeError(
            f"version.json not found or invalid — broken installation: {e}"
        ) from e


def version_gt(a: str, b: str) -> bool:
    """Return True if version string a is strictly greater than b."""
    try:
        return tuple(int(x) for x in a.split(".")) > tuple(int(x) for x in b.split("."))
    except Exception:
        return False


APP_VERSION = get_app_version()
VERSION_URL = _VERSION_URL