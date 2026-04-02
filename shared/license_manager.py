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
import os
import platform
import re
import subprocess
from datetime import datetime, timedelta

import requests

# ── Digistore24 REST API ──────────────────────────────────────────────────────
# Docs: https://www.digistore24.com/app/tools.api
# All calls: GET https://api.digistore24.com/api/call/{SELLER_API_KEY}/json/{action}
_DS24_BASE = "https://api.digistore24.com/api/call"

# Loaded from environment — set in .env or system env vars
_DS24_API_KEY   = os.environ.get("DS24_API_KEY", "")
_DS24_PRODUCT   = os.environ.get("DS24_PRODUCT_ID", "")

_REVALIDATE_HOURS = 12    # re-ping DS24 every 12h
_GRACE_DAYS       = 3     # offline grace period

# Digistore24 serial/license key format: groups of alphanumeric chars
# e.g.  DS24-XXXX-XXXX-XXXX  or  XXXX-XXXX-XXXX-XXXX
_KEY_RE = re.compile(r'^[A-Z0-9]{4,8}(-[A-Z0-9]{4,8}){2,5}$')


class LicenseManager:

    def __init__(self):
        data_dir = os.path.expanduser("~/Library/Application Support/Kura")
        os.makedirs(data_dir, exist_ok=True)

        self.license_file     = os.path.join(data_dir, "license.json")
        self.trial_file       = os.path.join(data_dir, "trial.dat")
        self.hardware_id_file = os.path.expanduser("~/.kura_hardware")

        self.max_trials  = 5
        self.mac_address = self._get_mac()
        self.hardware_id = self._build_hardware_id()
        self._cache: dict | None = None

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
                self._cache = json.load(f)
        except Exception:
            self._cache = {}
        return self._cache

    def _save(self, data: dict):
        self._cache = data
        try:
            with open(self.license_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"License save error: {e}")

    # ── Digistore24 API call ──────────────────────────────────────────────────

    def _ds24_get(self, action: str, params: dict = None, timeout: int = 10):
        """
        GET https://api.digistore24.com/api/call/{API_KEY}/json/{action}
        Docs: https://www.digistore24.com/app/tools.api
        """
        url = f"{_DS24_BASE}/{_DS24_API_KEY}/json/{action}"
        return requests.get(url, params=params or {}, timeout=timeout)

    # ── Activate ──────────────────────────────────────────────────────────────

    def activate(self, license_key: str) -> tuple[bool, str]:
        """Verify a Digistore24 serial/license key and store it locally."""
        key = license_key.strip().upper()

        if not _KEY_RE.match(key):
            return False, (
                "Ungültiges Lizenzschlüssel-Format.\n"
                "Bitte kopieren Sie den Schlüssel direkt aus Ihrer Kaufbestätigung."
            )

        try:
            resp = self._ds24_get(
                "getOrderBySerialNumber",
                {"serial_number": key, "product_id": _DS24_PRODUCT},
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
            order = data.get("data", {})
            order_status = order.get("order_status", "")
            if order_status in ("complete", "paid"):
                self._save({
                    "key":          key,
                    "hardware_id":  self.hardware_id,
                    "mac":          self.mac_address,
                    "status":       "active",
                    "validated_at": datetime.utcnow().isoformat(),
                    "activated_at": datetime.utcnow().isoformat(),
                    "order_id":     order.get("order_id", ""),
                })
                return True, "Kura Pro wurde erfolgreich aktiviert."
            return False, f"Bestellung nicht abgeschlossen (Status: {order_status})."

        error = data.get("message") or data.get("error") or str(data)
        print(f"DS24 activate failed {resp.status_code}: {resp.text}")
        return False, f"Aktivierung fehlgeschlagen:\n{error}"

    # ── Validate ──────────────────────────────────────────────────────────────

    def validate(self) -> bool:
        """Re-check stored license against Digistore24. 12h cache, 3-day offline grace."""
        cache = self._load()
        if not cache:
            return False

        if cache.get("hardware_id") != self.hardware_id:
            print("License: hardware mismatch — key belongs to a different workstation.")
            return False

        last = cache.get("validated_at")
        if last:
            age = datetime.utcnow() - datetime.fromisoformat(last)
            if age < timedelta(hours=_REVALIDATE_HOURS):
                return cache.get("status") == "active"

        key = cache.get("key", "")
        try:
            resp = self._ds24_get(
                "getOrderBySerialNumber",
                {"serial_number": key, "product_id": _DS24_PRODUCT},
                timeout=8,
            )
            data = resp.json()
            if resp.status_code == 200 and data.get("result") == "success":
                order_status = data.get("data", {}).get("order_status", "")
                if order_status in ("complete", "paid"):
                    cache["status"]       = "active"
                    cache["validated_at"] = datetime.utcnow().isoformat()
                    self._save(cache)
                    return True
            cache["status"] = "invalid"
            self._save(cache)
            return False

        except Exception:
            if last:
                age = datetime.utcnow() - datetime.fromisoformat(last)
                if age < timedelta(days=_GRACE_DAYS):
                    print(f"License: offline grace, {_GRACE_DAYS - age.days}d remaining.")
                    return cache.get("status") == "active"
            print("License: offline and grace period expired.")
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
        """
        if os.path.exists(self.license_file):
            return True if self.validate() else False
        count = self.get_trial_count()
        if count < self.max_trials:
            return "TRIAL"
        return False

    # ── Legacy shims ─────────────────────────────────────────────────────────

    def verify_online(self, license_key: str) -> bool:
        ok, _ = self.activate(license_key)
        return ok

    def save_key(self, key: str):
        pass
