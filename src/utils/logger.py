"""Logging configuration for the Network Attack Detection project.

Provides a single configured logger instance. Use this instead of print()
for all production logging.
"""

import logging
import sys
from pathlib import Path

from src.utils.config import get_project_root


def get_logger(name: str = "network_attack_detection") -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name. Use __name__ for module-level logger.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = get_project_root() / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
