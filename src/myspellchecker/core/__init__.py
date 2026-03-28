"""
Core spell checking functionality.

This module contains the main SpellChecker class and response objects.

Imports are lazy to avoid loading heavy dependencies (pycrfsuite, etc.)
until they are actually needed.
"""

from typing import TYPE_CHECKING

# These imports are lightweight and don't trigger heavy dependencies
from myspellchecker.core.config import (
    JointConfig,
    NgramContextConfig,
    PhoneticConfig,
    ProviderConfig,
    SemanticConfig,
    SpellCheckerConfig,
    SymSpellConfig,
    ValidationConfig,
)
from myspellchecker.core.constants import ValidationLevel
from myspellchecker.core.exceptions import (
    CacheError,
    # Configuration errors
    ConfigurationError,
    ConnectionPoolError,
    # Data loading errors
    DataLoadingError,
    InferenceError,
    IngestionError,
    InsufficientStorageError,
    InvalidConfigError,
    MissingDatabaseError,
    # Resource errors
    MissingDependencyError,
    # Model errors
    ModelError,
    ModelLoadError,
    # Base exception
    MyanmarSpellcheckError,
    NormalizationError,
    PackagingError,
    # Pipeline errors
    PipelineError,
    # Processing errors
    ProcessingError,
    # Provider errors
    ProviderError,
    TokenizationError,
    ValidationError,
)
from myspellchecker.core.i18n import (
    get_language,
    get_message,
    get_supported_languages,
    set_language,
)
from myspellchecker.core.response import (
    ContextError,
    Error,
    GrammarError,
    Response,
    SyllableError,
    WordError,
)

# For type checking only - doesn't trigger runtime imports
if TYPE_CHECKING:
    from myspellchecker.core.builder import ConfigPresets, SpellCheckerBuilder
    from myspellchecker.core.spellchecker import SpellChecker
    from myspellchecker.core.streaming import (
        ChunkResult,
        StreamingChecker,
        StreamingConfig,
        StreamingStats,
    )
    from myspellchecker.core.syllable_rules import SyllableRuleValidator

__all__ = [
    # Response classes
    "Response",
    "Error",
    "SyllableError",
    "WordError",
    "ContextError",
    "GrammarError",
    # Core classes (lazy loaded)
    "SpellChecker",
    "SpellCheckerBuilder",
    "ConfigPresets",
    "SyllableRuleValidator",
    # Streaming classes (lazy loaded)
    "StreamingChecker",
    "StreamingConfig",
    "StreamingStats",
    "ChunkResult",
    # Configuration classes
    "SpellCheckerConfig",
    "SymSpellConfig",
    "NgramContextConfig",
    "PhoneticConfig",
    "SemanticConfig",
    "JointConfig",
    "ValidationConfig",
    "ProviderConfig",
    # Constants
    "ValidationLevel",
    # i18n (Localization)
    "set_language",
    "get_language",
    "get_message",
    "get_supported_languages",
    # Exceptions - Base
    "MyanmarSpellcheckError",
    # Exceptions - Configuration
    "ConfigurationError",
    "InvalidConfigError",
    # Exceptions - Data Loading
    "DataLoadingError",
    "MissingDatabaseError",
    # Exceptions - Processing
    "ProcessingError",
    "ValidationError",
    "TokenizationError",
    "NormalizationError",
    # Exceptions - Provider
    "ProviderError",
    "ConnectionPoolError",
    # Exceptions - Pipeline
    "PipelineError",
    "IngestionError",
    "PackagingError",
    # Exceptions - Model
    "ModelError",
    "ModelLoadError",
    "InferenceError",
    # Exceptions - Resource
    "MissingDependencyError",
    "InsufficientStorageError",
    "CacheError",
]


# Lazy import mechanism for heavy classes
_lazy_imports = {
    "SpellChecker": "myspellchecker.core.spellchecker",
    "SpellCheckerBuilder": "myspellchecker.core.builder",
    "ConfigPresets": "myspellchecker.core.builder",
    "SyllableRuleValidator": "myspellchecker.core.syllable_rules",
    "StreamingChecker": "myspellchecker.core.streaming",
    "StreamingConfig": "myspellchecker.core.streaming",
    "StreamingStats": "myspellchecker.core.streaming",
    "ChunkResult": "myspellchecker.core.streaming",
}


def __getattr__(name: str):
    """Lazy import for heavy classes to avoid loading dependencies until needed."""
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
