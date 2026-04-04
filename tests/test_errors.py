"""
Unit tests for error handling and logging.
"""
import logging

import pytest

from core.errors import (
    ErrorRecovery,
    KuraError,
    ModelError,
    safe_execute,
    validate_input,
)
from core.logging import PerformanceLogger, get_logger, setup_logging


class TestErrorHandling:
    """Test error handling utilities."""

    def test_kura_error_basic(self):
        """Test basic KuraError creation."""
        error = KuraError("Technical message", user_message="User-friendly message")

        assert str(error) == "Technical message"
        assert error.user_message == "User-friendly message"
        assert error.timestamp is not None

    def test_kura_error_with_context(self):
        """Test KuraError with context data."""
        error = KuraError(
            "Something failed",
            user_message="Please try again",
            patient_id=123,
            operation="generate_report"
        )

        assert error.context["patient_id"] == 123
        assert error.context["operation"] == "generate_report"

    def test_model_error_inheritance(self):
        """Test ModelError is a KuraError."""
        error = ModelError("Model loading failed")

        assert isinstance(error, KuraError)
        assert isinstance(error, Exception)

    def test_safe_execute_success(self):
        """Test safe_execute wrapper with successful function."""

        @safe_execute
        def working_function(x, y):
            return x + y

        result = working_function(2, 3)
        assert result == 5

    def test_safe_execute_with_error(self):
        """Test safe_execute wrapper with failing function."""

        @safe_execute
        def failing_function():
            raise ValueError("Something broke")

        # Should return None (default) instead of raising
        result = failing_function()
        assert result is None

    def test_safe_execute_custom_default(self):
        """Test safe_execute with custom default value."""

        @safe_execute
        def failing_function():
            raise ValueError("Error")

        wrapped = safe_execute(failing_function, default=42)
        result = wrapped()
        assert result == 42

    def test_safe_execute_reraise(self):
        """Test safe_execute with reraise option."""

        @safe_execute
        def failing_function():
            raise ValueError("Error")

        wrapped = safe_execute(failing_function, reraise=True)

        with pytest.raises(ValueError):
            wrapped()

    def test_validate_input_type(self):
        """Test input validation for type."""
        # Should pass
        validate_input(42, "number", expected_type=int)

        # Should fail
        with pytest.raises(ValueError, match="must be int"):
            validate_input("42", "number", expected_type=int)

    def test_validate_input_min_max(self):
        """Test input validation for range."""
        # Should pass
        validate_input(5, "value", min_value=0, max_value=10)

        # Should fail - too small
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_input(-1, "value", min_value=0)

        # Should fail - too large
        with pytest.raises(ValueError, match="must be <= 10"):
            validate_input(11, "value", max_value=10)

    def test_validate_input_allowed_values(self):
        """Test input validation for allowed values."""
        # Should pass
        validate_input("GKV", "insurance", allowed_values=["GKV", "PKV", "BG"])

        # Should fail
        with pytest.raises(ValueError, match="must be one of"):
            validate_input("INVALID", "insurance", allowed_values=["GKV", "PKV", "BG"])

    def test_validate_input_not_empty(self):
        """Test input validation for empty values."""
        # Should pass
        validate_input("text", "name", not_empty=True)

        # Should fail
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_input("", "name", not_empty=True)

    def test_error_recovery_success_first_try(self):
        """Test error recovery when function succeeds first try."""
        recovery = ErrorRecovery(max_retries=3)

        call_count = [0]

        def working_function():
            call_count[0] += 1
            return "success"

        result = recovery.retry_on_failure(working_function)

        assert result == "success"
        assert call_count[0] == 1  # Only called once

    def test_error_recovery_success_after_retries(self):
        """Test error recovery when function succeeds after retries."""
        recovery = ErrorRecovery(max_retries=3, backoff_factor=0.1)

        call_count = [0]

        def flaky_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary error")
            return "success"

        result = recovery.retry_on_failure(flaky_function)

        assert result == "success"
        assert call_count[0] == 3

    def test_error_recovery_all_retries_fail(self):
        """Test error recovery when all retries fail."""
        recovery = ErrorRecovery(max_retries=3, backoff_factor=0.1)

        def always_failing_function():
            raise ValueError("Permanent error")

        with pytest.raises(ValueError, match="Permanent error"):
            recovery.retry_on_failure(always_failing_function)


class TestLogging:
    """Test logging configuration."""

    def test_setup_logging(self):
        """Test basic logging setup."""
        logger = setup_logging(level="DEBUG", log_to_console=False)

        assert logger is not None
        assert logger.level == logging.DEBUG

    def test_get_logger(self):
        """Test getting module-specific logger."""
        logger = get_logger("test_module")

        assert logger.name == "kura.test_module"

    def test_performance_logger_success(self):
        """Test performance logging for successful operation."""
        logger = get_logger("test")

        with PerformanceLogger("test_operation", logger):
            # Simulate some work
            sum([i for i in range(1000)])

        # Should complete without error

    def test_performance_logger_with_error(self):
        """Test performance logging for failed operation."""
        logger = get_logger("test")

        with pytest.raises(ValueError):
            with PerformanceLogger("failing_operation", logger):
                raise ValueError("Test error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

