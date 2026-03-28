"""
Text segmentation interfaces and implementations.

This module provides abstract interfaces and concrete implementations for
Myanmar text segmentation (syllables and words).
"""

from __future__ import annotations

from myspellchecker.segmenters.base import Segmenter
from myspellchecker.segmenters.default import DefaultSegmenter
from myspellchecker.segmenters.regex import RegexSegmenter

__all__ = [
    "Segmenter",
    "DefaultSegmenter",
    "RegexSegmenter",
]
