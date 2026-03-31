"""
Kura v2026 - Platform Auto-Selector
Automatically uses the right version for your OS
"""
import platform
import sys
import os

def get_app_module():
    """Automatically select the correct app module based on OS"""
    system = platform.system()
    
    if system == "Darwin":
        # macOS - use rumps (menu bar app)
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'macos'))
            from main import KuraApp
            return KuraApp, "macOS (Menu Bar)"
        except ImportError as e:
            print(f"Error: {e}")
            print("Install with: pip install -r requirements.txt")
            sys.exit(1)
    
    elif system == "Windows":
        # Windows - use PySimpleGUI (GUI app)
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows'))
            from main_windows import KuraApp
            return KuraApp, "Windows (GUI)"
        except ImportError as e:
            print(f"Error: {e}")
            print("Install with: pip install -r requirements-windows.txt")
            sys.exit(1)
    
    elif system == "Linux":
        # Linux - use PySimpleGUI (GUI app)
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows'))
            from main_windows import KuraApp
            return KuraApp, "Linux (GUI)"
        except ImportError as e:
            print(f"Error: {e}")
            print("Install with: pip install -r requirements-windows.txt")
            sys.exit(1)
    
    else:
        print(f"Unsupported OS: {system}")
        sys.exit(1)

if __name__ == "__main__":
    KuraApp, platform_info = get_app_module()
    print(f"🩺 Kura v2026 - Starting on {platform_info}")
    
    app = KuraApp()
    
    # Handle macOS app.run() vs Windows app.run()
    if hasattr(app, 'run'):
        app.run()
    else:
        # Fallback for non-rumps versions
        app.show_main_window()

