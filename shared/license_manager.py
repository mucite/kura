"""
Kura License Manager — Digistore24 integration
------------------------------------------------
Flow:
  1. First run: verify_locally() → "TRIAL"
  2. User enters key: activate(key) → verifies with Digistore24, stores locally
  3. Subsequent runs: verify_locally() → validate() against DS24 (cached 12h)
  4. Offline grace: 3 days before hard block
  5. Workstation binding: hardware_id = SHA-256(MAC + serial)[:16]
"""

import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import requests

from ._compat import fix_windows_encoding

fix_windows_encoding()

logger = logging.getLogger("kura.license")

# ── Digistore24 REST API ──────────────────────────────────────────────────────
# Docs: https://www.digistore24.com/app/tools.api
# All calls: GET https://api.digistore24.com/api/call/{SELLER_API_KEY}/json/{action}
_DS24_BASE = "https://api.digistore24.com/api/call"  # append /{action}, auth via X-DS-API-KEY header

# Loaded from environment — set in .env or system env vars
_DS24_API_KEY   = os.environ.get("DS24_API_KEY", "")
_DS24_PRODUCT   = os.environ.get("DS24_PRODUCT_ID", "")

_REVALIDATE_HOURS = 12    # re-ping DS24 every 12h
_GRACE_DAYS       = 3     # offline grace period

# Digistore24 license key format: 8 groups of 5 alphanumeric chars
# e.g.  RMEL3-3UDDC-YHJHF-C7TH9-QRYJK-FHZSV-KU26F-NS3CC
_KEY_RE = re.compile(r'^[A-Z0-9]{5}(-[A-Z0-9]{5}){7}$')


class LicenseManager:

    def __init__(self):
        # Platform-specific data directory with robust fallback
        if platform.system() == "Windows":
            # Try APPDATA first, then LOCALAPPDATA, then user home as absolute fallback
            appdata = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
            data_dir = os.path.join(appdata, "Kura")
        else:
            # macOS and Linux
            data_dir = os.path.expanduser("~/Library/Application Support/Kura")

        # Create directory with error handling
        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception as dir_err:
            print(f"Warning: Could not create license data directory {data_dir}: {dir_err}")
            # Fallback to user home if main directory fails
            data_dir = os.path.join(os.path.expanduser("~"), ".kura_data")
            try:
                os.makedirs(data_dir, exist_ok=True)
            except Exception:
                print("Critical: Could not create fallback directory either")

        self.license_file     = os.path.join(data_dir, "license.json")
        self.trial_file       = os.path.join(data_dir, "trial.dat")
        self.hardware_id_file = os.path.expanduser("~/.kura_hardware")

        self.max_trials  = 5
        self.mac_address = self._get_mac()
        self.hardware_id = self._build_hardware_id()
        self._cache: dict | None = None
        self._block_reason: str = ""        # set whenever verify_locally() returns False
        self._grace_days_remaining: int = 0  # days left in offline grace (0 = not in grace)

    @property
    def block_reason(self) -> str:
        """
        Human-readable reason why verify_locally() returned False.
        One of: 'trial_expired' | 'subscription_expired' | 'offline_grace_expired' | ''
        """
        return self._block_reason

    @property
    def grace_days_remaining(self) -> int:
        """
        Days left in offline grace period when verify_locally() returned True.
        0 means not currently in grace period.
        """
        return self._grace_days_remaining

    # ── Hardware fingerprinting ───────────────────────────────────────────────

    def _get_mac(self) -> str:
        try:
            import psutil
            for iface, addrs in sorted(psutil.net_if_addrs().items()):
                if any(iface.startswith(p) for p in ('lo', 'utun', 'bridge', 'p2p', 'awdl')):
                    continue
                for addr in addrs:
                    if hasattr(addr, 'family') and addr.family.name in ('AF_LINK', 'AF_PACKET'):
                        mac = addr.address
                        if mac and mac != '00:00:00:00:00:00':
                            return mac.upper()
        except Exception:
            pass
        import uuid as _u
        raw = f"{_u.getnode():012x}"
        return ":".join(raw[i:i+2] for i in range(0, 12, 2)).upper()

    def _build_hardware_id(self) -> str:
        parts = [self.mac_address]
        try:
            if platform.system() == "Darwin":
                raw = subprocess.check_output(
                    ["ioreg", "-l"], timeout=5, stderr=subprocess.DEVNULL
                ).decode(errors="replace")
                m = re.search(r'"IOPlatformSerialNumber"\s*=\s*"([^"]+)"', raw)
                if m:
                    parts.append(m.group(1))
                uuid_raw = subprocess.check_output(
                    ["ioreg", "-d2", "-c", "IOPlatformExpertDevice"],
                    timeout=5, stderr=subprocess.DEVNULL
                ).decode(errors="replace")
                m2 = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', uuid_raw)
                if m2:
                    parts.append(m2.group(1))
            elif platform.system() == "Windows":
                for cmd in (["wmic", "baseboard", "get", "SerialNumber"],
                             ["wmic", "cpu", "get", "ProcessorId"]):
                    out = subprocess.check_output(
                        cmd, timeout=5, stderr=subprocess.DEVNULL
                    ).decode(errors="replace").strip().splitlines()
                    val = out[-1].strip() if out else ""
                    if val:
                        parts.append(val)
        except Exception:
            pass
        return hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]

    # ── Local cache ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        try:
            with open(self.license_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            self._cache = {}
            return self._cache

        sign = raw.get("_sign")
        if sign is None:
            # Legacy file (pre-HMAC signing) — accept as-is; normal 12h cache logic applies
            self._cache = raw
            return self._cache

        if sign != self._license_hmac(raw):
            # Signature mismatch — file tampered, clear and force re-activation
            print("License: HMAC mismatch — file tampered. Clearing cache.")
            try:
                os.remove(self.license_file)
            except Exception:
                pass
            self._cache = {}
            return self._cache

        self._cache = raw
        return self._cache

    def _save(self, data: dict):
        # Strip any existing signature before recomputing
        to_write = {k: v for k, v in data.items() if k != "_sign"}
        to_write["_sign"] = self._license_hmac(to_write)
        self._cache = to_write
        try:
            with open(self.license_file, "w", encoding="utf-8") as f:
                json.dump(to_write, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"License save error: {e}")

    # ── Digistore24 API call ──────────────────────────────────────────────────

    def _ds24_get(self, action: str, params: dict = None, timeout: int = 10):
        """
        GET https://api.digistore24.com/api/call/{action}
        API key passed via X-DS-API-KEY header.
        Docs: https://dev.digistore24.com/hc/en-us/articles/32479630493585-API-basics
        Raises RuntimeError if DS24_API_KEY is not configured (treated as offline).
        """
        api_key = os.environ.get("DS24_API_KEY", "") or _DS24_API_KEY
        if not api_key:
            raise RuntimeError("DS24_API_KEY not configured — treating as offline")
        url = f"{_DS24_BASE}/{action}"
        headers = {"X-DS-API-KEY": api_key}
        return requests.get(url, params=params or {}, headers=headers, timeout=timeout)

    # ── Activate ──────────────────────────────────────────────────────────────

    def activate(self, license_key: str) -> tuple[bool, str]:
        """Verify a Digistore24 serial/license key and store it locally."""
        key = license_key.strip().upper()
        # Normalize any dash variant (en-dash, em-dash, etc.) to ASCII hyphen
        key = re.sub(r'[\u2010-\u2015\u2212\ufe58\ufe63\uff0d]', '-', key)
        # Remove whitespace that may have been inserted during copy/paste
        key = re.sub(r'\s+', '', key)

        if not _KEY_RE.match(key):
            return False, (
                "Ungültiges Lizenzschlüssel-Format.\n"
                "Bitte kopieren Sie den Schlüssel direkt aus Ihrer Kaufbestätigung."
            )

        try:
            resp = self._ds24_get(
                "validateLicenseKey",
                {"purchase_id": key, "license_key": key},
            )
        except requests.exceptions.ConnectionError:
            return False, "Keine Internetverbindung. Bitte Netzwerk prüfen und erneut versuchen."
        except Exception as e:
            return False, f"Netzwerkfehler: {e}"

        try:
            data = resp.json()
        except Exception:
            return False, f"Ungültige Serverantwort (HTTP {resp.status_code})."

        if resp.status_code == 200 and data.get("result") == "success":
            self._save({
                "key":          key,
                "hardware_id":  self.hardware_id,
                "mac":          self.mac_address,
                "status":       "active",
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "activated_at": datetime.now(timezone.utc).isoformat(),
            })
            return True, "Kura Pro wurde erfolgreich aktiviert."

        error = data.get("message") or data.get("error") or str(data)
        print(f"DS24 activate failed {resp.status_code}: {resp.text}")
        return False, f"Lizenzschlüssel ungültig oder nicht gefunden:\n{error}"

    # ── Validate ──────────────────────────────────────────────────────────────

    def validate(self) -> bool:
        """Re-check stored license against Digistore24. 12h cache, 3-day offline grace."""
        self._grace_days_remaining = 0
        cache = self._load()
        if not cache:
            return False

        if cache.get("hardware_id") != self.hardware_id:
            print("License: hardware mismatch — key belongs to a different workstation.")
            self._block_reason = "subscription_expired"
            return False

        last = cache.get("validated_at")
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                now_dt  = datetime.now(timezone.utc)
                # Clock-rollback attack: validated_at is suspiciously in the future
                if last_dt > now_dt + timedelta(minutes=5):
                    print("License: clock skew detected — forcing revalidation.")
                    last = None  # fall through to DS24 check
                else:
                    age = now_dt - last_dt
                    if age < timedelta(hours=_REVALIDATE_HOURS):
                        if cache.get("status") != "active":
                            self._block_reason = "subscription_expired"
                        return cache.get("status") == "active"
            except Exception:
                last = None  # malformed date — force revalidation

        key = cache.get("key", "")
        try:
            resp = self._ds24_get(
                "validateLicenseKey",
                {"purchase_id": key, "license_key": key},
                timeout=8,
            )
            data = resp.json()
            if resp.status_code == 200 and data.get("result") == "success":
                cache["status"]       = "active"
                cache["validated_at"] = datetime.now(timezone.utc).isoformat()
                self._save(cache)
                return True
            # DS24 explicitly rejected — subscription cancelled/expired
            cache["status"] = "invalid"
            self._save(cache)
            self._block_reason = "subscription_expired"
            return False

        except Exception:
            # Offline — apply grace period
            if last:
                try:
                    last_dt_grace = datetime.fromisoformat(last)
                    if last_dt_grace.tzinfo is None:
                        last_dt_grace = last_dt_grace.replace(tzinfo=timezone.utc)
                    age = datetime.now(timezone.utc) - last_dt_grace
                    if age < timedelta(days=_GRACE_DAYS):
                        days_left = max(0, _GRACE_DAYS - age.days)
                        self._grace_days_remaining = days_left
                        print(f"License: offline grace, {days_left}d remaining.")
                        if cache.get("status") == "active":
                            return True
                except Exception:
                    pass
            print("License: offline and grace period expired.")
            self._block_reason = "offline_grace_expired"
            return False

    # ── Deactivate ────────────────────────────────────────────────────────────

    def deactivate(self) -> tuple[bool, str]:
        """Remove license from this workstation (clears local cache)."""
        cache = self._load()
        if not cache.get("key"):
            return False, "Keine aktive Lizenz auf diesem Gerät gefunden."

        try:
            os.remove(self.license_file)
        except Exception:
            pass
        self._cache = None
        return True, (
            "Lizenz wurde von diesem Gerät entfernt.\n"
            "Sie können den Schlüssel jetzt auf einem anderen Gerät aktivieren."
        )

    # ── Trial management (HMAC-signed) ───────────────────────────────────────

    def _trial_hmac(self, payload: str) -> str:
        import hmac as _hmac
        secret = hashlib.sha256(
            (self.hardware_id + self.mac_address).encode()
        ).digest()
        return _hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()

    def _license_hmac(self, data: dict) -> str:
        """HMAC-SHA256 over canonical JSON of all license fields (excl. '_sign').
        Key is hardware-bound — copy of license.json to another machine won't verify."""
        import hmac as _hmac
        payload = json.dumps(
            {k: v for k, v in sorted(data.items()) if k != "_sign"},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        secret = hashlib.sha256((self.hardware_id + "kura_lic_v1").encode()).digest()
        return _hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()

    def get_trial_count(self) -> int:
        try:
            with open(self.trial_file, "r") as f:
                raw = f.read().strip()
            parts = raw.split(":")
            if len(parts) != 3:
                return self.max_trials
            count_str, hw_id, stored_mac = parts
            payload = f"{count_str}:{hw_id}"
            if hw_id != self.hardware_id:
                return self.max_trials
            if stored_mac != self._trial_hmac(payload):
                return self.max_trials
            return int(count_str)
        except FileNotFoundError:
            return 0
        except Exception:
            return self.max_trials

    def increment_trial(self):
        count = self.get_trial_count()
        if count >= self.max_trials:
            return
        new_count = count + 1
        payload   = f"{new_count}:{self.hardware_id}"
        mac       = self._trial_hmac(payload)
        try:
            with open(self.trial_file, "w") as f:
                f.write(f"{payload}:{mac}")
        except Exception as e:
            print(f"Trial increment error: {e}")

    # ── Unified entry point ───────────────────────────────────────────────────

    def verify_locally(self):
        """
        Returns: True (licensed) | "TRIAL" (in trial) | False (expired/invalid)
        When returning False, block_reason is set to explain why.
        """
        if os.path.exists(self.license_file):
            if self.validate():
                self._block_reason = ""
                return True
            # block_reason already set inside validate()
            return False
        count = self.get_trial_count()
        if count < self.max_trials:
            self._block_reason = ""
            return "TRIAL"
        self._block_reason = "trial_expired"
        return False

    # ── Dev helpers ──────────────────────────────────────────────────────────

    def dev_reset_trial(self):
        """DEV ONLY — delete license.json and trial.dat to restore fresh trial state.
        No-op when running as a compiled/frozen bundle."""
        if getattr(sys, "frozen", False):
            return  # disabled in production builds
        for path in (self.license_file, self.trial_file):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        self._cache = None
        self._block_reason = ""
        self._grace_days_remaining = 0
        logger.debug(f"[DEV] Trial reset. Files removed: {self.license_file}, {self.trial_file}")

    # ── Legacy shims ─────────────────────────────────────────────────────────

    def verify_online(self, license_key: str) -> bool:
        ok, _ = self.activate(license_key)
        return ok

    def save_key(self, key: str):
        pass
