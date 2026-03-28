"""Dictionary provider interfaces and implementations.

This module provides abstract interfaces and concrete implementations for
dictionary data storage and retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Eager import for lightweight abstract base class
from myspellchecker.providers.base import DictionaryProvider

# Static type checkers see concrete types directly; at runtime, __getattr__ handles lazy loading.
if TYPE_CHECKING:
    from myspellchecker.providers.csv_provider import CSVProvider as CSVProvider
    from myspellchecker.providers.json_provider import JSONProvider as JSONProvider
    from myspellchecker.providers.memory import MemoryProvider as MemoryProvider
    from myspellchecker.providers.sqlite import SQLiteProvider as SQLiteProvider

__all__ = [
    "CSVProvider",
    "DictionaryProvider",
    "JSONProvider",
    "MemoryProvider",
    "SQLiteProvider",
]


def __getattr__(name: str) -> Any:
    """Lazy import for concrete provider classes."""
    if name == "SQLiteProvider":
        from myspellchecker.providers.sqlite import SQLiteProvider

        return SQLiteProvider
    if name == "MemoryProvider":
        from myspellchecker.providers.memory import MemoryProvider

        return MemoryProvider
    if name == "JSONProvider":
        from myspellchecker.providers.json_provider import JSONProvider

        return JSONProvider
    if name == "CSVProvider":
        from myspellchecker.providers.csv_provider import CSVProvider

        return CSVProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
