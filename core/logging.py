"""
Kura Medical Logging Configuration
===================================
Centralized logging setup with rotation, levels, and structured output.
"""
import logging
import logging.handlers
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class KuraLogFormatter(logging.Formatter):
    """Custom formatter with color coding for terminal output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[1;31m', # Bold Red
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        if sys.stderr.isatty():  # Only colorize if terminal
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                )
        return super().format(record)


def get_log_directory() -> Path:
    """Get platform-specific log directory."""
    if platform.system() == "Darwin":
        log_dir = Path.home() / "Library" / "Logs" / "Kura"
    elif platform.system() == "Windows":
        appdata = os.environ.get("LOCALAPPDATA", str(Path.home()))
        log_dir = Path(appdata) / "Kura" / "Logs"
    else:
        log_dir = Path.home() / ".local" / "share" / "kura" / "logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup centralized logging for Kura Medical.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to rotating file
        log_to_console: Whether to log to console
        max_bytes: Max log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured root logger
    """
    
    # Create root logger
    logger = logging.getLogger("kura")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()  # Remove any existing handlers
    
    # Console handler with color
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = KuraLogFormatter(
            '%(levelname)s [%(name)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Rotating file handler
    if log_to_file:
        log_dir = get_log_directory()
        log_file = log_dir / "kura.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed format for file logs
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging initialized - File: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"kura.{name}")


class PerformanceLogger:
    """Context manager for logging performance metrics."""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        self.operation = operation
        self.logger = logger or get_logger("performance")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} ({duration:.2f}s)")
        else:
            self.logger.error(
                f"Failed: {self.operation} ({duration:.2f}s) - {exc_type.__name__}: {exc_val}"
            )
        
        return False  # Don't suppress exceptions


def log_system_info():
    """Log system information for debugging."""
    logger = get_logger("system")
    
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"Architecture: {platform.machine()}")
    
    # Log memory info if available
    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info(f"RAM: {mem.total / (1024**3):.1f}GB total, {mem.available / (1024**3):.1f}GB available")
    except (ImportError, AttributeError):
        pass


# Initialize logging on module import
_default_logger = setup_logging(
    level=os.environ.get("KURA_LOG_LEVEL", "INFO"),
    log_to_file=True,
    log_to_console=not getattr(sys, 'frozen', False)  # Console only in dev mode
)

