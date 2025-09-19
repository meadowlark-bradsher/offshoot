"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_str: Optional[str] = None
) -> logging.Logger:
    """Set up logging to both console and file."""
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger("offshoot")
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(format_str)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger