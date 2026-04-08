"""
mySpellChecker: Myanmar (Burmese) Spell Checker Library

A high-performance Python library for spell checking in Myanmar (Burmese) language.
Implements a "Syllable-First" architecture for fast, reliable spell checking.

Basic Usage:
    >>> from myspellchecker import SpellChecker
    >>> from myspellchecker.providers import SQLiteProvider
    >>> checker = SpellChecker(provider=SQLiteProvider(database_path="mySpellChecker-default.db"))
    >>> result = checker.check("မြနမ်ာနိုငံ။")
    >>> print(result.has_errors)

For more information, see the documentation at: https://github.com/thettwe/my-spellchecker
"""

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

# Lightweight imports that don't require external dependencies
from myspellchecker.core import (
    ContextError,
    Error,
    Response,
    SyllableError,
    ValidationLevel,
    WordError,
)
from myspellchecker.core.i18n import (
    get_language,
    get_message,
    get_supported_languages,
    set_language,
)
from myspellchecker.core.response import ActionType, Suggestion, classify_action

# For type checking only - doesn't trigger runtime imports
if TYPE_CHECKING:
    from myspellchecker.core import SpellChecker
    from myspellchecker.core.builder import ConfigPresets, SpellCheckerBuilder
    from myspellchecker.core.check_options import CheckOptions
    from myspellchecker.core.config import SpellCheckerConfig
    from myspellchecker.core.response import GrammarError
    from myspellchecker.core.streaming import (
        ChunkResult,
        StreamingChecker,
        StreamingConfig,
        StreamingStats,
    )
    from myspellchecker.providers import SQLiteProvider

try:
    __version__ = version("myspellchecker")
except PackageNotFoundError:
    __version__ = "0.0.0"
__author__ = "Thet Twe Aung"
__license__ = "MIT"

# Set up null handler to avoid "No handler found" warnings for library users
# Uses standard logging since this is the package entry point (before logging_utils is loaded)
logging.getLogger("myspellchecker").addHandler(logging.NullHandler())


# Lazy import mechanism for heavy classes
_lazy_imports = {
    "SpellChecker": "myspellchecker.core",
    "SpellCheckerBuilder": "myspellchecker.core.builder",
    "ConfigPresets": "myspellchecker.core.builder",
    "SpellCheckerConfig": "myspellchecker.core.config",
    "GrammarError": "myspellchecker.core.response",
    "CheckOptions": "myspellchecker.core.check_options",
    "SQLiteProvider": "myspellchecker.providers",
    "StreamingChecker": "myspellchecker.core.streaming",
    "StreamingConfig": "myspellchecker.core.streaming",
    "StreamingStats": "myspellchecker.core.streaming",
    "ChunkResult": "myspellchecker.core.streaming",
    "MissingDatabaseError": "myspellchecker.core.exceptions",
}


def __getattr__(name: str):
    """Lazy import for heavy classes to avoid loading dependencies until needed."""
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def check_text(
    text: str,
    level: str = "syllable",
    database_path: str | None = None,
) -> Response:
    """
    Convenience function to check text using default settings.

    Requires a database to be available. If no database has been built,
    raises MissingDatabaseError with build instructions.

    Args:
        text: Myanmar text to check.
        level: Validation level ('syllable' or 'word').
        database_path: Optional path to a SQLite dictionary database.
            When provided, constructs a SQLiteProvider with this path.

    Returns:
        Response object with results.

    Raises:
        MissingDatabaseError: If no database is available.
        ValueError: If level is not 'syllable' or 'word'.
    """
    # Validate level parameter before creating SpellChecker instance
    valid_levels = {"syllable", "word"}
    if level not in valid_levels:
        raise ValueError(
            f"Invalid validation level: '{level}'. "
            f"Must be one of: {', '.join(sorted(valid_levels))}"
        )

    # Import SpellChecker lazily
    from myspellchecker.core import SpellChecker

    kwargs: dict = {}
    if database_path is not None:
        from myspellchecker.providers.sqlite import SQLiteProvider

        kwargs["provider"] = SQLiteProvider(database_path=database_path)

    with SpellChecker(**kwargs) as checker:
        return checker.check(text, level=ValidationLevel(level))


__all__ = [
    "__version__",
    "SpellChecker",
    "SpellCheckerBuilder",
    "ConfigPresets",
    "SpellCheckerConfig",
    "ActionType",
    "classify_action",
    "Response",
    "Error",
    "SyllableError",
    "WordError",
    "ContextError",
    "GrammarError",
    "Suggestion",
    "CheckOptions",
    "ValidationLevel",
    "SQLiteProvider",
    "StreamingChecker",
    "StreamingConfig",
    "StreamingStats",
    "ChunkResult",
    "MissingDatabaseError",
    "check_text",
    "set_language",
    "get_language",
    "get_message",
    "get_supported_languages",
]
