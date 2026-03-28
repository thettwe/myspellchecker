"""
Zawgyi encoding detection and conversion support.

This module provides centralized utilities for working with Zawgyi-encoded
Myanmar text.

The module provides:
- Lazy-loaded ZawgyiDetector instance (thread-safe via lru_cache)
- Zawgyi to Unicode conversion utilities
- Availability checks for optional dependencies

Dependencies:
- myanmartools: Required for accurate Zawgyi detection
- python-myanmar (myanmar package): Optional, for Zawgyi conversion

Example:
    >>> from myspellchecker.text.zawgyi_support import (
    ...     get_zawgyi_detector,
    ...     is_zawgyi_converter_available,
    ...     convert_zawgyi_to_unicode,
    ... )
    >>> detector = get_zawgyi_detector()
    >>> if detector:
    ...     prob = detector.get_zawgyi_probability("ျမန္မာ")
    ...     print(f"Zawgyi probability: {prob:.2f}")
"""

from __future__ import annotations

import contextlib
import io
from functools import lru_cache
from typing import cast

from myspellchecker.utils.logging_utils import get_logger

__all__ = [
    "convert_zawgyi_to_unicode",
    "get_zawgyi_detector",
    "is_zawgyi_converter_available",
]

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_zawgyi_detector():
    """
    Get or create the ZawgyiDetector instance (lazy loading, thread-safe).

    Uses functools.lru_cache for thread-safe singleton pattern. The detector
    is created on first call and cached for subsequent calls.

    Returns:
        ZawgyiDetector instance, or None if myanmartools is not installed.

    Example:
        >>> detector = get_zawgyi_detector()
        >>> if detector:
        ...     prob = detector.get_zawgyi_probability(text)
    """
    try:
        from myanmartools import ZawgyiDetector

        return ZawgyiDetector()
    except ImportError:
        logger.warning(
            "myanmartools not available for Zawgyi detection. "
            "Install with: pip install myanmartools>=1.2.1"
        )
        return None


@lru_cache(maxsize=1)
def is_zawgyi_converter_available() -> bool:
    """
    Check if Zawgyi conversion is available (thread-safe).

    Uses functools.lru_cache for thread-safe caching of the availability check.

    Returns:
        True if python-myanmar converter is available, False otherwise.

    Example:
        >>> if is_zawgyi_converter_available():
        ...     converted = convert_zawgyi_to_unicode(text)
    """
    try:
        from myanmar import converter  # noqa: F401

        return True
    except ImportError:
        return False


def convert_zawgyi_to_unicode(text: str, threshold: float = 0.95) -> str:
    """
    Convert Zawgyi-encoded text to Unicode Myanmar.

    Uses myanmartools for detection and python-myanmar for conversion.
    Only converts text that is detected as Zawgyi above the threshold
    to avoid corrupting valid Unicode text.

    Args:
        text: Text to convert (may be Zawgyi or Unicode).
        threshold: Minimum Zawgyi probability to trigger conversion (default: 0.95).

    Returns:
        Converted Unicode text, or original text if not Zawgyi or conversion unavailable.

    Example:
        >>> convert_zawgyi_to_unicode("ျမန္မာ")  # Zawgyi
        'မြန်မာ'
        >>> convert_zawgyi_to_unicode("မြန်မာ")  # Unicode - unchanged
        'မြန်မာ'

    Note:
        Requires myanmartools and python-myanmar packages.
        Falls back to original text if packages not installed.
    """
    if not text:
        return text

    # Check if conversion is available
    if not is_zawgyi_converter_available():
        return text

    # Get detector
    detector = get_zawgyi_detector()
    if detector is None:
        return text

    try:
        # Check if text is Zawgyi
        prob = detector.get_zawgyi_probability(text)
        if prob < threshold:
            return text

        # Convert
        return _convert_zawgyi_internal(text)

    except (RuntimeError, UnicodeError, ValueError, KeyError) as e:
        logger.debug(f"Zawgyi conversion skipped: {e}")
        return text


def _convert_zawgyi_internal(text: str) -> str:
    """
    Internal function to convert Zawgyi text to Unicode.

    Args:
        text: Zawgyi text to convert.

    Returns:
        Converted Unicode text, or original text on error.
    """
    try:
        from myanmar import converter

        # Suppress debug print from python-myanmar package
        with contextlib.redirect_stdout(io.StringIO()):
            return cast(str, converter.convert(text, "zawgyi", "unicode"))
    except (RuntimeError, UnicodeError, ValueError, KeyError) as e:
        text_preview = text[:50] + "..." if len(text) > 50 else text
        logger.warning(f"Zawgyi conversion failed: '{text_preview}' - {e}")
        return text
