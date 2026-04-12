"""Logging configuration with UTF-8 support, console output,
and rotating file handler with date-based filenames."""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGER_NAME = "lawnuri"

# Log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _get_log_filename() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"lawnuri_{today}.log"


def setup_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Set up and return a configured logger instance.

    Console handler at INFO level + rotating file handler at DEBUG level,
    both UTF-8 encoded. Files rotate at 10 MB, keeping 5 backups.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler - INFO level, UTF-8 for Windows compatibility
    console_handler = logging.StreamHandler(
        stream=sys.stdout
    )
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    # Windows console defaults to cp949; force UTF-8 to avoid encoding errors in Korean text
    if sys.platform == "win32" and hasattr(console_handler.stream, "reconfigure"):
        try:
            console_handler.stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logger.addHandler(console_handler)

    # Rotating file handler - DEBUG level, UTF-8 encoding
    log_file = LOG_DIR / _get_log_filename()
    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logger '%s' initialized. Log file: %s", name, log_file)
    return logger


# Module-level logger instance for convenient import
logger = setup_logger()
