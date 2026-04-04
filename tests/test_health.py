"""
Unit tests for SystemHealthChecker.
License checks are mocked — no real files or DS24 calls.
"""
import sys
from unittest.mock import MagicMock, patch

from core.health import HealthCheckResult, SystemHealthChecker

# ── HealthCheckResult dataclass ───────────────────────────────────────────────

class TestHealthCheckResult:
    def test_fields(self):
        r = HealthCheckResult(component="Foo", status="OK", message="good")
        assert r.component == "Foo"
        assert r.status == "OK"
        assert r.message == "good"
        assert r.details is None

    def test_details_stored(self):
        r = HealthCheckResult("X", "ERROR", "bad", details={"code": 1})
        assert r.details["code"] == 1


# ── Python version check ──────────────────────────────────────────────────────

def _version(major, minor, micro=0):
    """Return an object that mimics sys.version_info attributes."""
    from types import SimpleNamespace
    return SimpleNamespace(major=major, minor=minor, micro=micro,
                           releaselevel="final", serial=0)


class TestCheckPythonVersion:
    def test_310_passes(self):
        checker = SystemHealthChecker()
        with patch.object(sys, "version_info", _version(3, 10)):
            checker.check_python_version()
        assert checker.results[0].status == "OK"

    def test_314_passes(self):
        checker = SystemHealthChecker()
        with patch.object(sys, "version_info", _version(3, 14)):
            checker.check_python_version()
        assert checker.results[0].status == "OK"

    def test_39_fails(self):
        checker = SystemHealthChecker()
        with patch.object(sys, "version_info", _version(3, 9)):
            checker.check_python_version()
        assert checker.results[0].status == "ERROR"

    def test_27_fails(self):
        checker = SystemHealthChecker()
        with patch.object(sys, "version_info", _version(2, 7, 18)):
            checker.check_python_version()
        assert checker.results[0].status == "ERROR"


# ── License status mapping (the bug we fixed) ─────────────────────────────────

class TestLicenseStatusMapping:
    def _check_with_mock(self, return_value, trial_count=0):
        checker = SystemHealthChecker()
        mock_mgr = MagicMock()
        mock_mgr.verify_locally.return_value = return_value
        mock_mgr.get_trial_count.return_value = trial_count
        mock_mgr.max_trials = 5
        with patch("shared.license_manager.LicenseManager", return_value=mock_mgr):
            checker.check_license_system()
        return checker.results[0]

    def test_active_license_is_ok(self):
        result = self._check_with_mock(True)
        assert result.status == "OK"

    def test_trial_active_is_ok(self):
        result = self._check_with_mock("TRIAL", trial_count=2)
        assert result.status == "OK"

    def test_trial_message_shows_remaining(self):
        result = self._check_with_mock("TRIAL", trial_count=2)
        assert "3" in result.message  # 5 - 2 = 3 remaining

    def test_expired_license_is_error(self):
        """Critical regression: expired license must be ERROR, not WARNING."""
        result = self._check_with_mock(False)
        assert result.status == "ERROR"

    def test_expired_license_message(self):
        result = self._check_with_mock(False)
        assert "expired" in result.message.lower() or "invalid" in result.message.lower()

    def test_license_system_exception_is_error(self):
        checker = SystemHealthChecker()
        with patch("shared.license_manager.LicenseManager", side_effect=RuntimeError("boom")):
            checker.check_license_system()
        assert checker.results[0].status == "ERROR"


# ── check_all orchestration ───────────────────────────────────────────────────

class TestCheckAll:
    def _patched_checker(self, inject_result=None):
        checker = SystemHealthChecker()

        def side_effect():
            if inject_result:
                checker.results.append(inject_result)

        patches = [
            patch.object(checker, "check_python_version"),
            patch.object(checker, "check_system_resources"),
            patch.object(checker, "check_disk_space"),
            patch.object(checker, "check_models"),
            patch.object(checker, "check_gpu"),
            patch.object(checker, "check_license_system",
                         side_effect=side_effect if inject_result else None),
            patch.object(checker, "check_configuration"),
            patch.object(checker, "check_pricing_data"),
        ]
        return checker, patches

    def test_returns_tuple(self):
        checker, patches = self._patched_checker()
        with self._apply(patches):
            passed, results = checker.check_all()
        assert isinstance(passed, bool)
        assert isinstance(results, list)

    def test_all_ok_returns_true(self):
        checker, patches = self._patched_checker()
        with self._apply(patches):
            passed, _ = checker.check_all()
        assert passed is True

    def test_error_result_returns_false(self):
        err = HealthCheckResult("X", "ERROR", "bad")
        checker, patches = self._patched_checker(inject_result=err)
        with self._apply(patches):
            passed, _ = checker.check_all()
        assert passed is False

    def test_warning_does_not_fail_all(self):
        warn = HealthCheckResult("X", "WARNING", "minor issue")
        checker, patches = self._patched_checker(inject_result=warn)
        with self._apply(patches):
            passed, _ = checker.check_all()
        assert passed is True

    def test_results_reset_each_call(self):
        checker, patches = self._patched_checker()
        with self._apply(patches):
            checker.check_all()
            _, r1 = checker.check_all()
        # Second call resets — results only from the second call
        assert len(r1) == len(checker.results)

    @staticmethod
    def _apply(patches):
        from contextlib import ExitStack
        stack = ExitStack()
        for p in patches:
            stack.enter_context(p)
        return stack
