"""
Utilities module.

Contains caching helpers, centralized logging configuration, and lazy
accessors for phonetic hashing, edit distance, stemming, and tone
disambiguation.

Note: Text-processing helpers (normalize, NameHeuristic, etc.) and
keyboard-distance helpers live in their own packages. Import them
from the original source modules:
    from myspellchecker.text.normalize import normalize, is_myanmar_text, remove_punctuation
    from myspellchecker.text.ner import NameHeuristic, HONORIFICS
    from myspellchecker.algorithms.distance.keyboard import KEY_POSITIONS, is_keyboard_adjacent
    from myspellchecker.text.stemmer import Stemmer
    from myspellchecker.text.tone import ToneDisambiguator
"""

from myspellchecker.utils.cache import (
    Cache,
    CacheConfig,
    CacheManager,
    LRUCache,
)
from myspellchecker.utils.logging_utils import (
    configure_logging,
    get_logger,
)

__all__ = [
    # Logging utilities
    "configure_logging",
    "get_logger",
    # Cache utilities
    "Cache",
    "CacheConfig",
    "CacheManager",
    "LRUCache",
    # Lazy imports
    "PhoneticHasher",
    "levenshtein_distance",
    "damerau_levenshtein_distance",
    "weighted_damerau_levenshtein_distance",
    "myanmar_syllable_edit_distance",
    "Stemmer",
    "ToneDisambiguator",
    "create_disambiguator",
]


def __getattr__(name: str):
    """Lazy import for modules with circular dependency risks."""
    if name == "PhoneticHasher":
        from myspellchecker.text.phonetic import PhoneticHasher

        return PhoneticHasher
    if name in (
        "levenshtein_distance",
        "damerau_levenshtein_distance",
        "weighted_damerau_levenshtein_distance",
        "myanmar_syllable_edit_distance",
    ):
        from myspellchecker.algorithms.distance import edit_distance

        return getattr(edit_distance, name)
    if name == "Stemmer":
        from myspellchecker.text.stemmer import Stemmer

        return Stemmer
    if name in ("ToneDisambiguator", "create_disambiguator"):
        from myspellchecker.text import tone

        return getattr(tone, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
