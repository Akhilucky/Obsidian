"""
Centralized Logging Configuration
==================================
Single source of truth for all logging settings.
Import and call setup_logging() at application entry point.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    log_format: str = None,
    date_format: str = "%Y-%m-%d %H:%M:%S",
):
    """
    Configure logging for the entire application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        log_format: Custom format string
        date_format: Timestamp format
    """
    if log_format is None:
        log_format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )
    
    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "httpx", "httpcore", "yfinance", "peewee"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    
    logging.getLogger(__name__).info(f"Logging initialized at {level} level")
