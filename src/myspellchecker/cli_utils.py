"""CLI utility functions for mySpellChecker."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import TextIO

from myspellchecker.core.constants import DEFAULT_FILE_ENCODING


def confidence_type(value: str) -> float:
    """Validate confidence value is between 0.0 and 1.0."""
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value}") from None

    if not 0.0 <= fvalue <= 1.0:
        raise argparse.ArgumentTypeError(f"Confidence must be between 0.0 and 1.0, got {fvalue}")
    return fvalue


def _validate_file_path(file_path: str) -> str:
    """
    Validate and normalize a file path for security.

    This function resolves paths to absolute and logs warnings for
    potentially suspicious patterns (defense-in-depth).

    Args:
        file_path: Path to validate

    Returns:
        Resolved absolute path
    """
    # Resolve to absolute path
    resolved = os.path.abspath(os.path.realpath(file_path))

    # Warn about path traversal patterns (not blocking - CLI users have full control)
    if ".." in file_path:
        logging.getLogger(__name__).debug(
            f"Path contains parent reference: {file_path} -> {resolved}"
        )

    return resolved


def open_input_file(file_path: str | None, encoding: str = DEFAULT_FILE_ENCODING) -> TextIO:
    """
    Open input file for reading, with proper error handling.

    Args:
        file_path: Path to input file, None or "-" for stdin
        encoding: File encoding (default: utf-8)

    Returns:
        Opened file object or sys.stdin

    Raises:
        SystemExit: If file cannot be opened
    """
    if file_path is None or file_path == "-":
        return sys.stdin

    # Validate and resolve path
    resolved_path = _validate_file_path(file_path)

    try:
        return open(resolved_path, "r", encoding=encoding)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(2)
    except PermissionError:
        print(f"Error: Permission denied: {file_path}", file=sys.stderr)
        sys.exit(2)
    except (IsADirectoryError, OSError, UnicodeDecodeError) as e:
        print(f"Error: Cannot open file {file_path}: {e}", file=sys.stderr)
        sys.exit(2)


def open_output_file(file_path: str | None, encoding: str = DEFAULT_FILE_ENCODING) -> TextIO:
    """
    Open output file for writing, with proper error handling.

    Args:
        file_path: Path to output file, None or "-" for stdout
        encoding: File encoding (default: utf-8)

    Returns:
        Opened file object or sys.stdout

    Raises:
        SystemExit: If file cannot be opened
    """
    if file_path is None or file_path == "-":
        return sys.stdout

    # Validate and resolve path
    resolved_path = _validate_file_path(file_path)

    try:
        return open(resolved_path, "w", encoding=encoding)
    except PermissionError:
        print(f"Error: Permission denied: {file_path}", file=sys.stderr)
        sys.exit(2)
    except (IsADirectoryError, OSError) as e:
        print(f"Error: Cannot create file {file_path}: {e}", file=sys.stderr)
        sys.exit(2)
