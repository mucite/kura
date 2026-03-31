"""
Kura License Manager — Lemon Squeezy integration
-------------------------------------------------
Flow:
  1. First run: verify_locally() → "TRIAL"
  2. User enters key: activate(key) → stores instance_id + hardware fingerprint
  3. Subsequent runs: verify_locally() → validate() against LS (cached 24h)
  4. Offline grace: 7 days before hard block
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

# ── Lemon Squeezy endpoints ───────────────────────────────────────────────────
_LS_ACTIVATE   = "https://api.lemonsqueezy.com/v1/licenses/activate"
_LS_VALIDATE   = "https://api.lemonsqueezy.com/v1/licenses/validate"
_LS_DEACTIVATE = "https://api.lemonsqueezy.com/v1/licenses/deactivate"

_REVALIDATE_HOURS = 12    # re-ping LS every 12h (monthly sub — catch cancellations fast)
_GRACE_DAYS       = 3     # offline grace; short for subscriptions
_KEY_RE = re.compile(      # LS key format: UUID v4 — 8-4-4-4-12
    r'^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}$'
)


class LicenseManager:

    def __init__(self):
        data_dir = os.path.expanduser("~/Library/Application Support/Kura")
        os.makedirs(data_dir, exist_ok=True)

        self.license_file    = os.path.join(data_dir, "license.json")
        self.trial_file      = os.path.join(data_dir, "trial.dat")
        # legacy paths (kept so old installs still work)
        self.hardware_id_file = os.path.expanduser("~/.kura_hardware")

        self.max_trials  = 5
        self.mac_address = self._get_mac()
        self.hardware_id = self._build_hardware_id()
        self._cache: dict | None = None

    # ── Hardware fingerprinting ───────────────────────────────────────────────

    def _get_mac(self) -> str:
        """Return primary MAC address (uppercase, colon-separated)."""
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
        # fallback — uuid.getnode() is guaranteed to return node/MAC
        import uuid as _u
        raw = f"{_u.getnode():012x}"
        return ":".join(raw[i:i+2] for i in range(0, 12, 2)).upper()

    def _build_hardware_id(self) -> str:
        """16-char fingerprint tied to this physical machine."""
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

    # ── Lemon Squeezy API calls ───────────────────────────────────────────────
    # Docs: https://docs.lemonsqueezy.com/api/license-api
    # All endpoints: POST, Content-Type: application/x-www-form-urlencoded
    # No Authorization header required (license API is public-facing)

    _LS_HEADERS = {
        "Accept": "application/json",
    }

    def _ls_post(self, url: str, payload: dict, timeout: int = 10):
        """Send a form-encoded POST to the Lemon Squeezy license API."""
        return requests.post(url, data=payload, headers=self._LS_HEADERS, timeout=timeout)

    def activate(self, license_key: str) -> tuple[bool, str]:
        """Activate a license key on this workstation."""
        key = license_key.strip().upper()

        if not _KEY_RE.match(key):
            return False, (
                "Ungültiges Lizenzschlüssel-Format.\n"
                "Erwartet: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX\n"
                "Bitte kopieren Sie den Schlüssel direkt aus Ihrer Kaufbestätigung."
            )

        instance_name = f"Kura@{platform.node()} [{self.mac_address}]"

        try:
            resp = self._ls_post(
                _LS_ACTIVATE,
                {"license_key": key, "instance_name": instance_name},
            )
        except requests.exceptions.ConnectionError:
            return False, "Keine Internetverbindung. Bitte Netzwerk prüfen und erneut versuchen."
        except Exception as e:
            return False, f"Netzwerkfehler: {e}"

        try:
            data = resp.json()
        except Exception:
            return False, f"Ungueltige Serverantwort (HTTP {resp.status_code})."

        if resp.status_code == 200 and data.get("activated"):
            lk      = data.get("license_key", {})
            inst    = data.get("instance", {})
            status  = lk.get("status", "")
            inst_id = inst.get("id", "")

            if status != "active":
                return False, f"Lizenz nicht aktiv (Status: {status})."

            self._save({
                "key":          key,
                "instance_id":  inst_id,
                "hardware_id":  self.hardware_id,
                "mac":          self.mac_address,
                "status":       "active",
                "validated_at": datetime.utcnow().isoformat(),
                "activated_at": datetime.utcnow().isoformat(),
            })
            return True, "Kura Pro wurde erfolgreich aktiviert."

        if resp.status_code == 422:
            error = data.get("error", "Unbekannter Fehler")
            if "already" in error.lower():
                return False, (
                    "Dieser Schlüssel ist bereits auf einem anderen Gerät aktiviert.\n\n"
                    "Deaktivieren Sie ihn zuerst auf dem alten Geraet,\n"
                    "oder kaufen Sie eine neue Lizenz."
                )
            return False, f"Aktivierung abgelehnt: {error}"

        error_detail = data.get("error") or data.get("message") or str(data)
        print(f"LS activate failed {resp.status_code}: {resp.text}")
        return False, f"Aktivierung fehlgeschlagen (HTTP {resp.status_code}):\n{error_detail}"

    def validate(self) -> bool:
        """
        Check the stored license against Lemon Squeezy.
        Uses local cache (12h); 3-day offline grace.
        """
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

        key     = cache.get("key", "")
        inst_id = cache.get("instance_id", "")
        try:
            # Validate only requires license_key; instance_id narrows to this seat
            resp = self._ls_post(
                _LS_VALIDATE,
                {"license_key": key, "instance_id": inst_id},
                timeout=8,
            )
            data = resp.json()

            if resp.status_code == 200 and data.get("valid"):
                lk_status = data.get("license_key", {}).get("status", "")
                if lk_status == "active":
                    cache["status"]       = "active"
                    cache["validated_at"] = datetime.utcnow().isoformat()
                    self._save(cache)
                    return True
                cache["status"] = lk_status or "invalid"
                self._save(cache)
                return False

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

    def deactivate(self) -> tuple[bool, str]:
        """Remove license from this workstation (frees the Lemon Squeezy seat)."""
        cache = self._load()
        key     = cache.get("key", "")
        inst_id = cache.get("instance_id", "")

        if not key:
            return False, "Keine aktive Lizenz auf diesem Geraet gefunden."

        try:
            resp = self._ls_post(
                _LS_DEACTIVATE,
                {"license_key": key, "instance_id": inst_id},
                timeout=8,
            )
            data = resp.json()
            if resp.status_code == 200 and data.get("deactivated"):
                try:
                    os.remove(self.license_file)
                except Exception:
                    pass
                self._cache = None
                return True, (
                    "Lizenz wurde von diesem Geraet entfernt.\n"
                    "Sie koennen den Schluessel jetzt auf einem anderen Geraet verwenden."
                )
        except requests.exceptions.ConnectionError:
            return False, "Keine Internetverbindung. Deaktivierung nicht moeglich."
        except Exception as e:
            return False, f"Fehler: {e}"

        return False, "Deaktivierung fehlgeschlagen. Bitte wenden Sie sich an den Support."

    # ── Trial management (HMAC-signed) ───────────────────────────────────────
    # File format:  {count}:{hardware_id}:{hmac}
    # HMAC key   :  SHA-256(hardware_id + mac_address)  — machine-specific secret
    # Any deletion, manual edit, or transplant to another machine → exhausted.

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
                return self.max_trials          # malformed → exhausted
            count_str, hw_id, stored_mac = parts
            payload = f"{count_str}:{hw_id}"
            if hw_id != self.hardware_id:
                return self.max_trials          # different machine → exhausted
            if stored_mac != self._trial_hmac(payload):
                return self.max_trials          # tampered → exhausted
            return int(count_str)
        except FileNotFoundError:
            return 0
        except Exception:
            return self.max_trials              # unreadable → exhausted

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
        Called on every session start and before saving.
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
