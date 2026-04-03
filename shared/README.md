# Shared Modules

Common code used by both macOS and Windows versions.

## Modules

### config_manager.py
Configuration management with remote sync and fallback support.
- Syncs from remote Gist
- Falls back to local defaults
- Billing codes and rules

### license_manager.py
Hardware-locked license verification.
- Lemon Squeezy API integration
- Trial mode (5 free sessions)
- Offline grace period

## Usage

Both platforms import:
```python
from shared.config_manager import ConfigManager
from shared.license_manager import LicenseManager
```

## Adding New Modules

1. Create module in this folder
2. Import in both `macos/main.py` and `windows/main_windows.py`
3. Update `__init__.py`

