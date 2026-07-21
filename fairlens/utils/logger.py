"""
Logging utilities for FairLens.

Author: FairLens
"""

from __future__ import annotations

import logging
from pathlib import Path

from .constants import LOG_DIR


def get_logger(
    name: str,
    level: str = "INFO",
) -> logging.Logger:
    """
    Return configured logger.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console Handler

    console = logging.StreamHandler()

    console.setFormatter(formatter)

    logger.addHandler(console)

    # File Handler

    logfile = Path(LOG_DIR) / f"{name}.log"

    file_handler = logging.FileHandler(logfile)

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger