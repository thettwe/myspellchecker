"""Shared type aliases for text processing modules."""

from __future__ import annotations

from collections.abc import Callable

__all__ = [
    "DictionaryCheck",
    "FrequencyCheck",
    "POSCheck",
]

DictionaryCheck = Callable[[str], bool]
FrequencyCheck = Callable[[str], int]
POSCheck = Callable[[str], str | None]
