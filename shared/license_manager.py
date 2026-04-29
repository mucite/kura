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

# ── Kura License API ──────────────────────────────────────────────────────────
# The desktop app calls our own backend; the backend holds DS24_API_KEY and
# proxies to Digistore24. Customers never see the Digistore24 API key.
# Override with KURA_LICENSE_API for staging/local testing.
_LICENSE_API = os.environ.get(
    "KURA_LICENSE_API",
    "https://kura-medical.de/api/license/check",
)

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
        self.deactivated_flag = os.path.join(data_dir, "deactivated.flag")
        self.hardware_id_file = os.path.expanduser("~/.kura_hardware")
        self.gist_cache_file  = os.path.join(data_dir, "gist_config_cache.json")

        self.max_trials  = 5
        self.mac_address = self._get_mac()
        self.hardware_id = self._build_hardware_id()
        self._cache: dict | None = None
        self._block_reason: str = ""        # set whenever verify_locally() returns False
        self._grace_days_remaining: int = 0  # days left in offline grace (0 = not in grace)
        self._is_revoked: bool = False      # set if hardware_id in Gist revocation list
        
        # Auto-log hardware_id for support/revocation purposes
        logger.info(f"Kura Hardware ID: {self.hardware_id}")
        print(f"[INFO] Hardware ID: {self.hardware_id}")

        # Log Hardware ID to persistent support log
        self._log_hardware_id_to_file()

    @property
    def block_reason(self) -> str:
        """
        Human-readable reason why verify_locally() returned False.
        One of: 'deactivated' | 'trial_expired' | 'subscription_expired' |
                'offline_grace_expired' | 'license_revoked' | 'payment_failed' | ''
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

    def _log_hardware_id_to_file(self):
        """Write Hardware ID to persistent log file for support purposes"""
        try:
            import platform
            from datetime import datetime

            # Use same data_dir as license files
            if platform.system() == "Windows":
                appdata = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
                log_dir = os.path.join(appdata, "Kura")
            else:
                log_dir = os.path.expanduser("~/Library/Application Support/Kura")

            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "kura_support.log")

            # Append to log with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} | Hardware ID: {self.hardware_id} | MAC: {self.mac_address}\n")
        except Exception:
            pass  # Silent fail - not critical

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

    # ── License API call (proxied through our backend) ────────────────────────

    def _check_license_remote(
        self,
        license_key: str,
        bestellnummer: str | None = None,
        timeout: int = 10,
    ) -> tuple[bool, str]:
        """
        Call our backend, which proxies to Digistore24 server-side.

        Mirrors the working Windows flow: backend forwards both license_key
        and purchase_id (Bestellnummer) plus the X-DS-API-KEY header to
        api.digistore24.com/api/call/validateLicenseKey.

        Returns: (is_valid, reason_or_status)
          valid:   True,  "active"
          invalid: False, "license_invalid" | "subscription_cancelled" |
                          "payment_refunded" | "payment_chargeback" |
                          "invalid_format"
        Raises RuntimeError on network/server failure (caller treats as offline).
        """
        payload = {
            "license_key": license_key,
            "hardware_id": self.hardware_id,
        }
        if bestellnummer:
            payload["bestellnummer"] = bestellnummer

        try:
            resp = requests.post(_LICENSE_API, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"License API unreachable: {e}")

        if resp.status_code >= 500:
            raise RuntimeError(f"License API server error: HTTP {resp.status_code}")

        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"License API invalid response: HTTP {resp.status_code}")

        is_valid = bool(data.get("valid"))
        reason = data.get("status") if is_valid else data.get("reason", "license_invalid")
        return is_valid, reason or ("active" if is_valid else "license_invalid")

    # ── Activate ──────────────────────────────────────────────────────────────

    def activate(self, license_key: str, bestellnummer: str | None = None) -> tuple[bool, str]:
        """Verify a Digistore24 serial/license key and store it locally.
        Optionally pass `bestellnummer` (DS24 order number / purchase_id) for stricter validation."""
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

        order_no = (bestellnummer or "").strip() or None

        try:
            is_valid, reason = self._check_license_remote(key, bestellnummer=order_no)
        except RuntimeError as e:
            return False, "Keine Internetverbindung. Bitte Netzwerk prüfen und erneut versuchen."

        if is_valid:
            entry = {
                "key":          key,
                "hardware_id":  self.hardware_id,
                "mac":          self.mac_address,
                "status":       "active",
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "activated_at": datetime.now(timezone.utc).isoformat(),
            }
            if order_no:
                entry["bestellnummer"] = order_no
            self._save(entry)
            try:
                os.remove(self.deactivated_flag)
            except FileNotFoundError:
                pass
            except Exception:
                pass
            return True, "Kura Pro wurde erfolgreich aktiviert."

        print(f"License activate rejected: {reason}")
        return False, f"Lizenzschlüssel ungültig oder nicht gefunden:\n{reason}"

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
        saved_order = cache.get("bestellnummer")
        try:
            # Re-check license + payment status via our backend (proxies DS24).
            # Auto-detects: cancelled subscriptions, chargebacks, refunds.
            is_valid, reason = self._check_license_remote(key, bestellnummer=saved_order)

            if not is_valid:
                # Payment failed/cancelled/refunded — auto-revoke
                print(f"License: revoked, reason = {reason}")
                cache["status"] = "revoked"
                cache["revoked_reason"] = reason
                self._save(cache)
                self._block_reason = "payment_failed"
                return False

            # Active — update validation timestamp
            cache["status"] = "active"
            cache["validated_at"] = datetime.now(timezone.utc).isoformat()
            cache["last_payment_check"] = reason
            self._save(cache)
            return True

        except RuntimeError:
            # Offline or API unavailable — apply grace period
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

        # Drop a marker so verify_locally() can distinguish deactivation from
        # natural trial expiry, and the trial pool can't be re-tapped.
        try:
            with open(self.deactivated_flag, "w") as f:
                f.write(datetime.now(timezone.utc).isoformat())
        except Exception:
            pass

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
        # No license file: prefer the explicit "deactivated" reason over generic
        # trial expiry so the UI can show "you deactivated this seat" wording.
        if os.path.exists(self.deactivated_flag):
            self._block_reason = "deactivated"
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
