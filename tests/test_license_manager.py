"""
Unit tests for LicenseManager.
All tests use a temp directory — no real license files or DS24 calls.
"""
import json
import os
from unittest.mock import patch

import pytest

from shared.license_manager import LicenseManager


@pytest.fixture
def mgr(tmp_path):
    """LicenseManager wired to a temp directory."""
    m = LicenseManager.__new__(LicenseManager)
    m.license_file = str(tmp_path / "license.json")
    m.trial_file = str(tmp_path / "trial.dat")
    m.hardware_id_file = str(tmp_path / ".kura_hardware")
    m.max_trials = 5
    m.mac_address = "AA:BB:CC:DD:EE:FF"
    m.hardware_id = "deadbeef12345678"
    m._cache = None
    m._block_reason = ""
    m._grace_days_remaining = 0
    return m


# ── Trial management ──────────────────────────────────────────────────────────

class TestTrialManagement:
    def test_no_trial_file_returns_zero(self, mgr):
        assert mgr.get_trial_count() == 0

    def test_verify_locally_returns_trial_when_no_license(self, mgr):
        assert mgr.verify_locally() == "TRIAL"

    def test_increment_writes_file(self, mgr):
        mgr.increment_trial()
        assert os.path.exists(mgr.trial_file)

    def test_increment_increases_count(self, mgr):
        mgr.increment_trial()
        assert mgr.get_trial_count() == 1

    def test_increment_multiple(self, mgr):
        for _ in range(3):
            mgr.increment_trial()
        assert mgr.get_trial_count() == 3

    def test_increment_capped_at_max(self, mgr):
        for _ in range(mgr.max_trials + 5):
            mgr.increment_trial()
        assert mgr.get_trial_count() == mgr.max_trials

    def test_trial_expired_returns_false(self, mgr):
        for _ in range(mgr.max_trials):
            mgr.increment_trial()
        assert mgr.verify_locally() is False

    def test_trial_expired_sets_block_reason(self, mgr):
        for _ in range(mgr.max_trials):
            mgr.increment_trial()
        mgr.verify_locally()
        assert mgr.block_reason == "trial_expired"

    def test_tampered_trial_returns_max(self, mgr):
        with open(mgr.trial_file, "w") as f:
            f.write("3:wronghwid:badsig")
        # Wrong hardware_id → returns max_trials (expired)
        assert mgr.get_trial_count() == mgr.max_trials

    def test_malformed_trial_returns_max(self, mgr):
        with open(mgr.trial_file, "w") as f:
            f.write("notvalidformat")
        assert mgr.get_trial_count() == mgr.max_trials


# ── License save / load / HMAC ────────────────────────────────────────────────

class TestLicenseSaveLoad:
    def _active_payload(self, mgr):
        return {"key": "TESTKEY", "status": "active", "hardware_id": mgr.hardware_id}

    def test_save_creates_file(self, mgr):
        mgr._save(self._active_payload(mgr))
        assert os.path.exists(mgr.license_file)

    def test_save_adds_hmac_signature(self, mgr):
        mgr._save(self._active_payload(mgr))
        with open(mgr.license_file) as f:
            data = json.load(f)
        assert "_sign" in data

    def test_load_returns_none_when_no_file(self, mgr):
        assert mgr._load() is None or mgr._load() == {}

    def test_roundtrip(self, mgr):
        mgr._save(self._active_payload(mgr))
        mgr._cache = None  # clear in-memory cache
        loaded = mgr._load()
        assert loaded["key"] == "TESTKEY"
        assert loaded["status"] == "active"

    def test_tampered_file_rejected(self, mgr):
        mgr._save(self._active_payload(mgr))
        # Tamper the file
        with open(mgr.license_file) as f:
            data = json.load(f)
        data["status"] = "active_hacked"
        with open(mgr.license_file, "w") as f:
            json.dump(data, f)
        mgr._cache = None
        loaded = mgr._load()
        # After tamper, load returns empty dict (HMAC mismatch clears the file)
        assert not loaded.get("status") == "active_hacked"

    def test_signature_is_hardware_bound(self, mgr):
        mgr._save(self._active_payload(mgr))
        sig1 = mgr._license_hmac(self._active_payload(mgr))
        # Different hardware_id → different signature
        mgr.hardware_id = "ffffffff99999999"
        sig2 = mgr._license_hmac(self._active_payload(mgr))
        assert sig1 != sig2


# ── verify_locally with active license ───────────────────────────────────────

class TestVerifyLocally:
    def test_trial_mode_before_first_use(self, mgr):
        result = mgr.verify_locally()
        assert result == "TRIAL"

    def test_block_reason_empty_when_trial(self, mgr):
        mgr.verify_locally()
        assert mgr.block_reason == ""

    def test_returns_false_when_license_file_exists_but_validate_fails(self, mgr):
        # Write a license file that fails validation (no DS24 key → offline → invalid)
        with open(mgr.license_file, "w") as f:
            json.dump({}, f)
        # validate() will fail because no DS24 API key and no grace period
        result = mgr.verify_locally()
        assert result is False

    def test_grace_days_initially_zero(self, mgr):
        assert mgr.grace_days_remaining == 0


# ── Hardware ID ───────────────────────────────────────────────────────────────

class TestHardwareId:
    def test_build_returns_16_hex_chars(self, mgr):
        mgr.mac_address = "AA:BB:CC:DD:EE:FF"
        result = mgr._build_hardware_id()
        assert isinstance(result, str)
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_mac_different_id(self, mgr):
        mgr.mac_address = "AA:BB:CC:DD:EE:FF"
        id1 = mgr._build_hardware_id()
        mgr.mac_address = "11:22:33:44:55:66"
        id2 = mgr._build_hardware_id()
        assert id1 != id2


# ── HMAC helpers ──────────────────────────────────────────────────────────────

class TestHmacHelpers:
    def test_trial_hmac_is_deterministic(self, mgr):
        h1 = mgr._trial_hmac("3:deadbeef12345678")
        h2 = mgr._trial_hmac("3:deadbeef12345678")
        assert h1 == h2

    def test_trial_hmac_differs_for_different_payloads(self, mgr):
        h1 = mgr._trial_hmac("3:deadbeef12345678")
        h2 = mgr._trial_hmac("4:deadbeef12345678")
        assert h1 != h2

    def test_license_hmac_is_deterministic(self, mgr):
        data = {"key": "X", "status": "active"}
        assert mgr._license_hmac(data) == mgr._license_hmac(data)

    def test_license_hmac_excludes_sign_field(self, mgr):
        data1 = {"key": "X", "status": "active"}
        data2 = {"key": "X", "status": "active", "_sign": "something"}
        assert mgr._license_hmac(data1) == mgr._license_hmac(data2)


# ── dev_reset_trial ───────────────────────────────────────────────────────────

class TestDevReset:
    def test_reset_clears_trial(self, mgr):
        mgr.increment_trial()
        assert mgr.get_trial_count() > 0
        mgr.dev_reset_trial()
        assert mgr.get_trial_count() == 0

    def test_reset_clears_block_reason(self, mgr):
        mgr._block_reason = "trial_expired"
        mgr.dev_reset_trial()
        assert mgr.block_reason == ""

    def test_reset_no_op_when_frozen(self, mgr):
        mgr.increment_trial()
        import sys
        with patch.object(sys, "frozen", True, create=True):
            mgr.dev_reset_trial()
        # In frozen mode reset is a no-op — file should still exist
        assert os.path.exists(mgr.trial_file)
