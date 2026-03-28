"""
Centralized logging configuration for myspellchecker.

This module provides a unified logging abstraction that enables:
- Centralized configuration of log levels, formats, and handlers
- Consistent logger naming across the package
- Easy switching between development and production logging modes
- Optional JSON structured logging support

Usage:
    # In application entry points (CLI, main scripts):
    from myspellchecker.utils.logging_utils import configure_logging
    configure_logging(level="INFO", json_output=False)

    # In modules (backward compatible with standard pattern):
    from myspellchecker.utils.logging_utils import get_logger
    logger = get_logger(__name__)

    # Or continue using the standard pattern (still works):
    import logging
    logger = logging.getLogger(__name__)
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from typing import TextIO

__all__ = [
    "configure_logging",
    "get_logger",
]

# Package-wide logger name prefix
LOGGER_PREFIX = "myspellchecker"

# Default format strings
DEFAULT_FORMAT = "%(levelname)s: %(name)s: %(message)s"
DEBUG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
JSON_FORMAT = (
    '{"time": "%(asctime)s", "level": "%(levelname)s", '
    '"logger": "%(name)s", "message": "%(message)s"}'
)


def configure_logging(
    level: int | str = logging.INFO,
    format_string: str | None = None,
    stream: TextIO | None = None,
    json_output: bool = False,
    debug_mode: bool = False,
) -> None:
    """
    Configure logging for the entire myspellchecker package.

    This function should be called once at application startup (e.g., in CLI main()).
    It configures the root logger for the myspellchecker namespace.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL or int)
        format_string: Custom format string. If None, uses appropriate default.
        stream: Output stream. Defaults to sys.stderr.
        json_output: If True, use JSON format for structured logging.
        debug_mode: If True, use verbose debug format with timestamps and line numbers.

    Example:
        # Development mode
        configure_logging(level="DEBUG", debug_mode=True)

        # Production mode with JSON
        configure_logging(level="INFO", json_output=True)

        # Simple output for build commands
        configure_logging(level="INFO", format_string="%(message)s")
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Determine format string
    if format_string is None:
        if json_output:
            format_string = JSON_FORMAT
        elif debug_mode:
            format_string = DEBUG_FORMAT
        else:
            format_string = DEFAULT_FORMAT

    # Configure the package logger
    package_logger = logging.getLogger(LOGGER_PREFIX)
    package_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in package_logger.handlers[:]:
        package_logger.removeHandler(handler)

    # Create and configure handler
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_string))

    package_logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate logs
    package_logger.propagate = False


@lru_cache(maxsize=256)
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with consistent naming under the myspellchecker namespace.

    This is a convenience function that ensures all loggers are properly
    namespaced under 'myspellchecker'. It caches logger instances for efficiency.

    Args:
        name: Logger name. Typically __name__ from the calling module.
              If it already starts with 'myspellchecker', it's used as-is.
              Otherwise, 'myspellchecker.' prefix is added.

    Returns:
        A configured logging.Logger instance.

    Example:
        # In src/myspellchecker/algorithms/symspell.py
        logger = get_logger(__name__)
        # Returns logger named "myspellchecker.algorithms.symspell"

        # Works with short names too
        logger = get_logger("custom")
        # Returns logger named "myspellchecker.custom"
    """
    # If name already has the package prefix, use as-is
    if name.startswith(LOGGER_PREFIX):
        return logging.getLogger(name)

    # Strip 'src.' prefix if present (common in development)
    if name.startswith("src."):
        name = name[4:]

    # Add package prefix if not present
    if not name.startswith(LOGGER_PREFIX):
        return logging.getLogger(f"{LOGGER_PREFIX}.{name}")

    return logging.getLogger(name)
