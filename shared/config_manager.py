import copy
import json
import logging
import os
import platform

from ._compat import fix_windows_encoding
from .practice_config import PracticeConfig

fix_windows_encoding()

logger = logging.getLogger("kura.config_manager")

# Paths - platform-specific
if platform.system() == "Windows":
    # Try APPDATA first, then LOCALAPPDATA, then user home as fallback
    appdata = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    _DATA_DIR = os.path.join(appdata, "Kura")
else:
    # macOS and Linux
    _DATA_DIR = os.path.expanduser("~/Library/Application Support/Kura")

_GIST_CACHE     = os.path.join(_DATA_DIR, "gist_config_cache.json")
_LOCAL_OVERRIDE = os.path.join(_DATA_DIR, "config_override.json")

_GIST_URL = (
    "https://gist.githubusercontent.com/mucite/"
    "6994897471e0676bbbdd2468002c24fc/raw/physio_config_2026.json"
)


_FALLBACK = {
    "version": "2026.0.0",
    "billing_codes": {
        "KG": "20501", "KG_ZNS": "20710", "KG_Gruppe": "20601",
        "MT": "21201",
        "MLD_30": "20205", "MLD_45": "20201", "MLD_60": "20202",
        "KPE_I": "21110", "KPE_II": "21111",
    },
    "context": {"audit_standard": "§ 106b SGB V", "special_focus": ["Allgemein"]},
    "billing_rules": {},
    "audit_rules": {},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key.startswith("_"):
            continue  # skip comment keys
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


class ConfigManager:
    def __init__(self, practice_name: str = None, license_status=None):
        """
        license_status: Result from LicenseManager.verify_locally()
            - True = licensed (sync Gist)
            - "TRIAL" = trial mode (no Gist, use fallback)
            - False = expired (no Gist, use fallback)
        """
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
        except Exception as dir_err:
            print(f"Warning: Could not create data directory: {dir_err}")

        self.practice_config = PracticeConfig(practice_name=practice_name)
        self.data = copy.deepcopy(_FALLBACK)
        self.license_status = license_status

        # Layer 1: Gist (remote, cached locally) — ONLY for licensed users
        if license_status is True:
            self._sync_gist()
        else:
            # Trial/expired users: use fallback only, but check for revocation list
            self._load_gist_revocation_only()
            
            # Trial comparison will be shown in GUI by main_windows.py, not here


        # Layer 2: local customer override (wins over Gist, never pushed back)
        self._apply_local_override()

        # Layer 3: practice config (BSNR, ICD rules, billing shortcuts)
        self._merge_practice_config()

    # ── Layer 1: Gist ─────────────────────────────────────────────────────────

    def _load_gist_revocation_only(self):
        """
        For trial/expired users: Load Gist ONLY to check revocation list.
        Don't apply config updates (they get fallback only).
        This prevents trial users from getting premium features while still
        allowing remote license revocation.
        """
        try:
            import requests
            r = requests.get(_GIST_URL, timeout=5)
            if r.status_code == 200:
                remote = r.json()
                # Save to cache so LicenseManager can read revocation list
                try:
                    with open(_GIST_CACHE, "w", encoding="utf-8") as f:
                        json.dump(remote, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
                print("Gist revocation check: OK (trial mode, no config applied)")
                return
        except Exception:
            pass
        
        # Offline: trial users just use fallback, no problem
        print("Trial mode: using basic fallback config")

    def _sync_gist(self):
        """Pull latest Gist; fall back to local cache; fall back to hardcoded.
        Only called for licensed users."""
        try:
            import requests
            r = requests.get(_GIST_URL, timeout=5)
            if r.status_code == 200:
                remote = r.json()
                self.data = remote
                # Save as local cache so override can reference real keys
                try:
                    with open(_GIST_CACHE, "w", encoding="utf-8") as f:
                        json.dump(remote, f, indent=2, ensure_ascii=False)
                except Exception as write_err:
                    print(f"Cache-Schreibfehler: {write_err}")
                print(f"✅ Premium Config: v{self.data.get('version')} (Licensed)")
                return
        except Exception as e:
            print(f"Gist-Sync fehlgeschlagen: {e}")

        # Offline: use last cached Gist (licensed users can work offline with grace period)
        if os.path.exists(_GIST_CACHE):
            try:
                with open(_GIST_CACHE, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                print("Premium Config: offline cache (Licensed)")
                return
            except Exception:
                pass

        print("Kein Gist-Cache — nutze hardcoded Fallback (Licensed, offline)")

    # ── Layer 2: local override ───────────────────────────────────────────────

    def _apply_local_override(self):
        if not os.path.exists(_LOCAL_OVERRIDE):
            return
        try:
            with open(_LOCAL_OVERRIDE, "r", encoding="utf-8") as f:
                override = json.load(f)
        except Exception as e:
            print(f"Lokale Konfiguration fehlerhaft (ignoriert): {e}")
            return

        # Strip any top-level keys that do not exist in the Gist base.
        # Customers may edit values or add entries within existing sections,
        # but cannot introduce new top-level sections or rename structure.
        allowed = set(self.data.keys())
        rejected = []
        sanitised = {}
        for key, val in override.items():
            if key.startswith("_"):
                continue  # comment key — always skip
            if key not in allowed:
                rejected.append(key)
            else:
                sanitised[key] = val

        if rejected:
            print(f"Lokale Konfiguration: unbekannte Schluessel ignoriert: {rejected}")

        if sanitised:
            self.data = _deep_merge(self.data, sanitised)
            print(f"Lokale Konfiguration angewendet: {_LOCAL_OVERRIDE}")

    def create_override_template(self):
        """
        Write config_override.json pre-filled with the current Gist values.
        Customer edits only the keys they want to change; the rest stays from Gist.
        Called from the tray menu. Returns the file path.
        """
        if os.path.exists(_LOCAL_OVERRIDE):
            return _LOCAL_OVERRIDE   # already exists — just open it

        template = copy.deepcopy(self.data)
        template["_comment"] = (
            "Lokale Ueberschreibungen der Kura-Konfiguration (ab 01.01.2026). "
            "Aendern Sie nur die Werte, die Sie anpassen moechten. "
            "Alle anderen Werte werden weiterhin automatisch aus der "
            "Kura-Gist-Konfiguration gezogen. Diese Datei wird NICHT "
            "an Kura Medical uebertragen."
        )
        # PKV-Preise: praxiseigene Honorare — nicht im Gist enthalten, hier individuell setzen
        if "pkv_preise" not in template:
            template["pkv_preise"] = {
                "_hinweis": (
                    "Praxiseigene PKV-Honorare in Euro. "
                    "Tragen Sie hier Ihre tatsaechlichen Behandlungspreise ein. "
                    "Diese ueberschreiben die GebueeTh-Orientierungswerte in der Kura-Ausgabe. "
                    "GKV-Festpreise (§125 SGB V) werden hierdurch NICHT veraendert."
                ),
                "20501": 0.0,   # KG Einzelbehandlung 20 min
                "20511": 0.0,   # KG-ZNS 45 min
                "20560": 0.0,   # KG atemtherapeutisch 20 min
                "21200": 0.0,   # MT Erstbefundung 30 min
                "21201": 0.0,   # MT Folgebehandlung 20 min
                "20205": 0.0,   # MLD 30 min
                "20201": 0.0,   # MLD 45 min
                "20202": 0.0,   # MLD 60 min
                "21110": 0.0,   # KPE Phase I 60 min
                "21111": 0.0,   # KPE Phase II 30 min
            }
        try:
            with open(_LOCAL_OVERRIDE, "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
        except Exception as write_err:
            logger.error(f"Override-Template Schreibfehler: {write_err}")
            return None
        return _LOCAL_OVERRIDE

    @property
    def local_override_path(self) -> str:
        return _LOCAL_OVERRIDE

    @property
    def gist_cache_path(self) -> str:
        return _GIST_CACHE

    # ── Layer 3: practice config ──────────────────────────────────────────────

    def _merge_practice_config(self):
        if not (self.practice_config and self.practice_config.config):
            return
        pc = self.practice_config.config
        self.data["practice"]           = pc.get("practice", {})
        self.data["icd10_rules"]        = pc.get("icd10_rules", {})
        self.data["compliance_standard"] = pc.get("compliance_standard", "§ 106b SGB V")
        if "billing_codes" not in self.data:
            self.data["billing_codes"] = {}
        if "billing" in pc:
            self.data["billing_codes"].update(pc["billing"])

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def billing_codes(self):
        return self.data.get("billing_codes", {})

    @property
    def codes(self):
        return self.billing_codes

    @property
    def billing_rules(self):
        return self.data.get("billing_rules", {})

    @property
    def audit_rules(self):
        return self.data.get("audit_rules", {})

    @property
    def compliance_standard(self):
        return self.data.get("compliance_standard", "§ 106b SGB V")

    @property
    def version(self):
        return self.data.get("version", "Fallback")

    @property
    def pkv_preise(self) -> dict:
        """
        Praxiseigene PKV-Preise (Positionsnummer → Betrag in €).
        Aus config_override.json unter dem Schlüssel 'pkv_preise'.
        Beispiel: {"21201": 72.00, "20501": 55.00}
        GKV-Festpreise §125 SGB V werden hierdurch nicht berührt.
        """
        # Nur tatsächlich gesetzte Preise zurückgeben (0.0 = nicht konfiguriert → GebüTh-Fallback)
        return {
            k: float(v) for k, v in self.data.get("pkv_preise", {}).items()
            if not k.startswith("_") and isinstance(v, (int, float)) and float(v) > 0
        }
