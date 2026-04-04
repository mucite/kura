"""
Error Handling Utilities for Kura Medical
==========================================
Robust error handling with user-friendly messages and logging.
"""
import functools
import logging
import traceback
from typing import Any, Callable, Optional, TypeVar, Union
from datetime import datetime

logger = logging.getLogger("kura.errors")

T = TypeVar('T')


class KuraError(Exception):
    """Base exception for all Kura-specific errors."""

    def __init__(self, message: str, user_message: Optional[str] = None, **context):
        super().__init__(message)
        self.user_message = user_message or message
        self.context = context
        self.timestamp = datetime.now()


class ModelError(KuraError):
    """Raised when AI models fail to load or execute."""
    pass


class LicenseError(KuraError):
    """Raised for license validation failures."""
    pass


class AudioError(KuraError):
    """Raised for audio recording/processing issues."""
    pass


class BillingError(KuraError):
    """Raised for billing/compliance issues."""
    pass


class ConfigurationError(KuraError):
    """Raised for configuration issues."""
    pass


def safe_execute(
    func: Callable[..., T],
    default: T = None,
    log_errors: bool = True,
    reraise: bool = False
) -> Callable[..., T]:
    """
    Decorator that wraps a function with safe execution and error logging.

    Args:
        func: Function to wrap
        default: Default value to return on error
        log_errors: Whether to log errors
        reraise: Whether to reraise exceptions after logging

    Returns:
        Wrapped function
    """
    # If already wrapped by safe_execute, unwrap to avoid stacking wrappers
    original = getattr(func, '_safe_execute_original', func)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as e:
            if log_errors:
                logger.error(
                    f"Error in {original.__name__}: {e}",
                    exc_info=True,
                    extra={
                        "function": original.__name__,
                        "func_args": str(args)[:100],
                        "func_kwargs": str(kwargs)[:100]
                    }
                )

            if reraise:
                raise

            return default

    wrapper._safe_execute_original = original
    return wrapper


def handle_startup_error(error: Exception, component: str) -> None:
    """
    Handle critical startup errors with user-friendly messaging.

    Args:
        error: The exception that occurred
        component: Name of the component that failed
    """
    logger.critical(f"Startup failed in {component}: {error}", exc_info=True)

    # Map technical errors to user-friendly messages
    if isinstance(error, MemoryError):
        user_msg = (
            f"Nicht genug Arbeitsspeicher verfügbar.\n\n"
            f"Bitte schließen Sie andere Programme und starten Sie Kura neu.\n"
            f"Mindestens 8GB RAM empfohlen."
        )
    elif isinstance(error, FileNotFoundError):
        user_msg = (
            f"Erforderliche Dateien fehlen: {error}\n\n"
            f"Bitte installieren Sie Kura neu oder kontaktieren Sie den Support."
        )
    elif "GPU" in str(error) or "Metal" in str(error) or "CUDA" in str(error):
        user_msg = (
            f"GPU-Fehler: {error}\n\n"
            f"Lösungsvorschläge:\n"
            f"• Computer neu starten\n"
            f"• GPU-intensive Programme schließen\n"
            f"• Grafiktreiber aktualisieren"
        )
    else:
        user_msg = (
            f"Kura konnte nicht gestartet werden.\n\n"
            f"Fehler: {str(error)[:200]}\n\n"
            f"Bitte kontaktieren Sie den Support mit dieser Meldung."
        )

    # Create error report file
    log_dir = logger.handlers[0].baseFilename if logger.handlers else "."
    error_file = f"{log_dir}/startup_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    try:
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"Kura Startup Error Report\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Component: {component}\n")
            f.write(f"Error Type: {type(error).__name__}\n")
            f.write(f"Error Message: {error}\n\n")
            f.write(f"Traceback:\n")
            f.write(traceback.format_exc())
            f.write(f"\n{'=' * 60}\n")
    except Exception as e:
        logger.error(f"Failed to write error report: {e}")

    return user_msg, error_file


def validate_input(
    value: Any,
    value_name: str,
    expected_type: type = None,
    min_value: Any = None,
    max_value: Any = None,
    allowed_values: list = None,
    not_empty: bool = False
) -> None:
    """
    Validate input parameters with detailed error messages.

    Args:
        value: Value to validate
        value_name: Name of the value for error messages
        expected_type: Expected type
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allowed_values: List of allowed values
        not_empty: Whether to check for empty strings/collections

    Raises:
        ValueError: If validation fails
    """
    if expected_type and not isinstance(value, expected_type):
        raise ValueError(
            f"{value_name} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )

    if not_empty:
        if not value:
            raise ValueError(f"{value_name} cannot be empty")

    if min_value is not None and value < min_value:
        raise ValueError(f"{value_name} must be >= {min_value}, got {value}")

    if max_value is not None and value > max_value:
        raise ValueError(f"{value_name} must be <= {max_value}, got {value}")

    if allowed_values is not None and value not in allowed_values:
        raise ValueError(
            f"{value_name} must be one of {allowed_values}, got {value}"
        )


class ErrorRecovery:
    """Manages error recovery strategies."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger("kura.recovery")

    def retry_on_failure(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Retry a function with exponential backoff.

        Args:
            func: Function to retry
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        import time

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(
                        f"Retry successful for {func.__name__} on attempt {attempt + 1}"
                    )
                return result

            except Exception as e:
                last_exception = e
                wait_time = self.backoff_factor ** attempt

                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for {func.__name__}: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )

                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

        self.logger.error(
            f"All {self.max_retries} attempts failed for {func.__name__}"
        )
        raise last_exception

