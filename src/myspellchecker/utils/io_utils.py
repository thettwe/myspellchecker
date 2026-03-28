"""
I/O utilities for file handling and system checks.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..core.exceptions import InsufficientStorageError
from .logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "check_disk_space",
]


def check_disk_space(path: str | Path, required_mb: int = 500) -> None:
    """
    Check if the disk partition containing `path` has enough free space.

    Args:
        path: Path to check (directory or file)
        required_mb: Minimum required free space in MB (default: 500MB)

    Raises:
        InsufficientStorageError: If free space is less than required.
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = path_obj.absolute()

    # If path doesn't exist, find nearest existing parent
    check_path = path_obj
    while not check_path.exists():
        parent = check_path.parent
        if parent == check_path:  # Reached root or cannot go up
            break
        check_path = parent

    try:
        total, used, free = shutil.disk_usage(check_path)
        free_mb = free / (1024 * 1024)

        if free_mb < required_mb:
            raise InsufficientStorageError(
                f"Insufficient disk space at {check_path}. "
                f"Available: {free_mb:.2f} MB, Required: {required_mb} MB."
            )

        logger.debug(f"Disk space check passed: {free_mb:.2f} MB free at {check_path}")

    except FileNotFoundError:
        logger.warning(f"Could not check disk space for {path}")
    except OSError as e:
        logger.warning(f"Error checking disk space: {e}")
