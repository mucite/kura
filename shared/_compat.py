"""
Windows stdout/stderr UTF-8 compatibility fix.
Imported by shared modules that may run in a Windows console.
"""
import os
import sys


def fix_windows_encoding() -> None:
    """Reconfigure stdout/stderr to UTF-8 on Windows consoles.

    Windows cmd/PowerShell defaults to cp1252 which cannot encode Unicode
    characters (German umlauts, emoji).  This is a no-op on macOS/Linux.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            setattr(sys, stream_name, open(os.devnull, "w", encoding="utf-8"))
        elif hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
