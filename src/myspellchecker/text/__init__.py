"""Text processing utilities for Myanmar language.

This package provides text processing utilities organized in a 3-layer
normalization hierarchy for Myanmar (Burmese) text:

Normalization Hierarchy
=======================

**Layer 1: Cython Core** (``normalize_c.pyx``)
    Performance-critical functions (~20x faster than pure Python):
    - ``remove_zero_width_chars()``, ``reorder_myanmar_diacritics()``
    - ``get_myanmar_ratio()``, ``is_myanmar_string()``

**Layer 2: Python Wrapper** (``normalize.py``)
    Higher-level normalization functions:
    - ``normalize()``: Main normalization with configurable steps
    - ``normalize_for_lookup()``: Full normalization for dictionary lookups
    - Zawgyi detection/conversion, character variants, nasal endings

**Layer 3: Service Layer** (``normalization_service.py``)
    Purpose-specific normalization through unified interface:
    - ``NormalizationService.for_spell_checking()``: Fast, no Zawgyi
    - ``NormalizationService.for_dictionary_lookup()``: Full with Zawgyi
    - ``NormalizationService.for_comparison()``: Aggressive matching
    - ``NormalizationService.for_display()``: Minimal, preserve formatting

Quick Start
===========
>>> from myspellchecker.text import get_normalization_service
>>> service = get_normalization_service()
>>> normalized = service.for_spell_checking("မြန်မာ")

See Also
========
- ``normalize.py``: Full documentation of normalization hierarchy
- ``normalization_service.py``: Service layer documentation
- ``normalize_c.pyx``: Cython implementation details
"""

from __future__ import annotations

from typing import Any

# Eager imports for lightweight utilities
from .ner import NameHeuristic
from .normalization_service import (
    # Presets
    PRESET_COMPARISON,
    PRESET_DICTIONARY_LOOKUP,
    PRESET_DISPLAY,
    PRESET_INGESTION,
    PRESET_SPELL_CHECK,
    NormalizationOptions,
    NormalizationService,
    get_normalization_service,
    normalize_for_comparison,
    normalize_for_lookup,
    normalize_for_spell_checking,
)
from .normalize import normalize as normalize_myanmar_text

# Lazy imports for heavy classes (deferred until first use)

__all__ = [
    # Legacy NER (heuristic)
    "NameHeuristic",
    # New NER model classes
    "Entity",
    "EntityType",
    "HeuristicNER",
    "HybridNER",
    "NERConfig",
    "NERFactory",
    "NERModel",
    "TransformerNER",
    # Other utilities
    "PhoneticHasher",
    "Stemmer",
    "ToneDisambiguator",
    "normalize_myanmar_text",
    # Normalization Service
    "NormalizationOptions",
    "NormalizationService",
    "get_normalization_service",
    "normalize_for_comparison",
    "normalize_for_lookup",
    "normalize_for_spell_checking",
    "PRESET_COMPARISON",
    "PRESET_DICTIONARY_LOOKUP",
    "PRESET_DISPLAY",
    "PRESET_INGESTION",
    "PRESET_SPELL_CHECK",
]


def __getattr__(name: str) -> Any:
    """Lazy import for heavy classes."""
    if name == "PhoneticHasher":
        from .phonetic import PhoneticHasher

        return PhoneticHasher
    if name == "Stemmer":
        from .stemmer import Stemmer

        return Stemmer
    if name == "ToneDisambiguator":
        from .tone import ToneDisambiguator

        return ToneDisambiguator
    # NER model classes (lazy loaded due to optional transformer dependency)
    if name in (
        "Entity",
        "EntityType",
        "HeuristicNER",
        "HybridNER",
        "NERConfig",
        "NERFactory",
        "NERModel",
        "TransformerNER",
    ):
        from .ner_model import (
            Entity,
            EntityType,
            HeuristicNER,
            HybridNER,
            NERConfig,
            NERFactory,
            NERModel,
            TransformerNER,
        )

        return {
            "Entity": Entity,
            "EntityType": EntityType,
            "HeuristicNER": HeuristicNER,
            "HybridNER": HybridNER,
            "NERConfig": NERConfig,
            "NERFactory": NERFactory,
            "NERModel": NERModel,
            "TransformerNER": TransformerNER,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
