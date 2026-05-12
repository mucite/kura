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


class ConfigUnavailableError(RuntimeError):
    """Raised on first launch when neither the Gist nor a local cache is reachable."""


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
        self.data = None
        self.license_status = license_status

        # Layer 1: Gist (remote, cached locally). All users go through this path —
        # the Gist hosts the §125 SGB V code catalogue, fees, and revocation list.
        # If both the live fetch and the local cache fail, we cannot continue.
        self._load_gist_or_cache()

        # Layer 2: local customer override (wins over Gist, never pushed back)
        self._apply_local_override()

        # Layer 3: practice config (BSNR, ICD rules, billing shortcuts)
        self._merge_practice_config()

    # ── Layer 1: Gist ─────────────────────────────────────────────────────────

    def _load_gist_or_cache(self):
        """Pull the live Gist (and refresh the cache). If unreachable, fall back
        to the on-disk cache from a previous successful run. If both fail, raise
        ConfigUnavailableError so the app can surface a clear error and exit.
        """
        try:
            import requests
            r = requests.get(_GIST_URL, timeout=5)
            if r.status_code == 200:
                remote = r.json()
                self.data = remote
                try:
                    with open(_GIST_CACHE, "w", encoding="utf-8") as f:
                        json.dump(remote, f, indent=2, ensure_ascii=False)
                except Exception as write_err:
                    print(f"Cache-Schreibfehler: {write_err}")
                print(f"✅ Config geladen: v{self.data.get('version')} (Gist live)")
                return
        except Exception as e:
            print(f"Gist-Sync fehlgeschlagen: {e}")

        if os.path.exists(_GIST_CACHE):
            try:
                with open(_GIST_CACHE, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                print(f"✅ Config geladen: v{self.data.get('version')} (Offline-Cache)")
                return
            except Exception as e:
                print(f"Cache-Lesefehler: {e}")

        raise ConfigUnavailableError(
            "Kura-Konfiguration konnte weder online (Gist) noch aus dem lokalen "
            "Cache geladen werden. Eine Internetverbindung ist beim ersten Start "
            "erforderlich."
        )

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
                # §125 SGB V — GKV-Festpreise als Orientierung (01.01.2026)
                # PKV-Praxispreise liegen typisch 50-120% über GKV-Satz (GebüTh 2026)
                "20501": 0.0,   # KG Einzelbehandlung 20 min (GKV: €29.63)
                "20507": 0.0,   # KG am Gerät / MTT 45 min (GKV: €55.81)
                "20710": 0.0,   # KG-ZNS Erwachsene (Bobath/PNF/Vojta) 45 min (GKV: €47.06)
                "20708": 0.0,   # KG-ZNS Kinder 45 min (GKV: Tarifcode 22 2026)
                "20560": 0.0,   # KG atemtherapeutisch 20 min (GKV: €29.63)
                "21200": 0.0,   # MT Erstbefundung 30 min (GKV: €42.71)
                "21201": 0.0,   # MT Folgebehandlung 20 min (GKV: €35.59)
                "20205": 0.0,   # MLD 30 min (GKV: €35.97)
                "20201": 0.0,   # MLD 45 min (GKV: €53.94)
                "20202": 0.0,   # MLD 60 min (GKV: €71.94)
                "21110": 0.0,   # KPE Phase I 60 min (GKV: €58.42)
                "21111": 0.0,   # KPE Phase II 30 min (GKV: €46.26)
                "20601": 0.0,   # KG Gruppenbehandlung 45 min (GKV: €13.26)
                "20106": 0.0,   # Klassische Massage (KMT) 20 min (GKV: €21.63)
                "20107": 0.0,   # Bindegewebsmassage (BGM) 20 min (GKV: €25.98)
                "20102": 0.0,   # Unterwasserdruckstrahl 20 min (GKV: €33.75)
                "20902": 0.0,   # KG Bewegungsbad Einzel 30 min (GKV: €33.87)
                "21501": 0.0,   # Fango / Wärmetherapie 20 min (GKV: €16.16)
                "21302": 0.0,   # Elektrotherapie 15 min (GKV: €8.43)
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
