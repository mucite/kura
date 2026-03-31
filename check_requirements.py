#!/usr/bin/env python3
"""
Kura v2026 - Pre-Installation Requirements Checker
Run this BEFORE installing to ensure your system is compatible
"""
import sys
import platform
import shutil
import os

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (need 3.8+)"

def check_ffmpeg():
    """Check FFmpeg installation"""
    if shutil.which("ffmpeg"):
        return True, "FFmpeg installed"
    return False, "FFmpeg not found (install from ffmpeg.org)"

def check_os():
    """Check supported OS"""
    os_name = platform.system()
    if os_name == "Darwin":
        version = platform.mac_ver()[0]
        return True, f"macOS {version}"
    elif os_name == "Windows":
        version = platform.win32_ver()[1]
        return True, f"Windows {version}"
    else:
        return False, f"Unsupported OS: {os_name}"

def check_disk_space():
    """Check free disk space"""
    try:
        if sys.platform == "win32":
            import ctypes
            free_bytes = ctypes.c_ulonglong()
            ctypes.windll.kernel32.GetDiskFreeSpaceEx(
                ctypes.c_wchar_p(os.getcwd()),
                ctypes.byref(free_bytes),
                None,
                None
            )
            free_gb = free_bytes.value / (1024**3)
        else:
            stat = os.statvfs(os.path.expanduser("~"))
            free_gb = stat.f_bavail * stat.f_frsize / (1024**3)
        
        if free_gb >= 7:
            return True, f"{free_gb:.1f}GB free (need 7GB)"
        else:
            return False, f"Only {free_gb:.1f}GB free (need 7GB)"
    except Exception:
        return None, "Could not determine disk space"

def check_ram():
    """Check available RAM"""
    try:
        import psutil
        mem_gb = psutil.virtual_memory().total / (1024**3)
        if mem_gb >= 8:
            return True, f"{mem_gb:.1f}GB RAM"
        elif mem_gb >= 4:
            return None, f"{mem_gb:.1f}GB RAM (slow, but OK)"
        else:
            return False, f"Only {mem_gb:.1f}GB RAM (need 8GB)"
    except ImportError:
        return None, "psutil not installed (skipping RAM check)"
    except Exception:
        return None, "Could not determine RAM"

def check_microphone():
    """Check if microphone is likely available"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        if devices:
            return True, "Microphone input found"
        return False, "No microphone input detected"
    except ImportError:
        return None, "sounddevice not installed (skipping microphone check)"
    except Exception:
        return None, "Could not check microphone"

def check_internet():
    """Check internet connectivity"""
    try:
        import socket
        socket.create_connection(("google.com", 80), timeout=2)
        return True, "Internet connection OK"
    except Exception:
        return False, "No internet connection"

def main():
    print("\n" + "="*60)
    print("🩺 Kura v2026 - Pre-Installation Checker")
    print("="*60 + "\n")
    
    checks = [
        ("Python Version", check_python),
        ("Operating System", check_os),
        ("FFmpeg", check_ffmpeg),
        ("Free Disk Space", check_disk_space),
        ("RAM", check_ram),
        ("Microphone", check_microphone),
        ("Internet Connection", check_internet),
    ]
    
    results = {}
    critical_pass = 0
    critical_total = 0
    
    for name, check_func in checks:
        result, message = check_func()
        results[name] = (result, message)
        
        # Determine criticality
        critical = name in ["Python Version", "Operating System", "FFmpeg", "Free Disk Space"]
        
        if critical:
            critical_total += 1
            if result:
                critical_pass += 1
        
        # Print with symbol
        if result is True:
            symbol = "✅"
        elif result is False:
            symbol = "❌"
        else:
            symbol = "⚠️ "
        
        print(f"{symbol} {name}: {message}")
    
    print("\n" + "="*60)
    
    if critical_pass == critical_total:
        print("✅ ALL REQUIREMENTS MET - Ready to install Kura!")
        print("\nNext steps:")
        print("1. Create .env file: cp .env.example .env")
        print("2. Add HF_TOKEN to .env")
        print("3. Install: pip install -r requirements.txt (macOS) or")
        print("           pip install -r requirements-windows.txt (Windows)")
        print("4. Run: python app.py")
        return 0
    else:
        print("❌ MISSING REQUIREMENTS - Fix these before installing:")
        print()
        for name, (result, message) in results.items():
            critical = name in ["Python Version", "Operating System", "FFmpeg", "Free Disk Space"]
            if critical and result is False:
                print(f"  • {name}: {message}")
        print("\nFor help, see COMPATIBILITY.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())

