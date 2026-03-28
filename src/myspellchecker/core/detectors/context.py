"""Detector context — explicit dependency bundle for detector mixins.

Constructed once per ``check()`` call, this replaces scattered
``self.provider`` / ``self.segmenter`` / ``self.symspell`` accesses
across detector mixins with explicit, typed dependency injection.

Detectors can access this via ``self._detector_ctx`` during gradual
migration from direct ``self.*`` attribute access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from myspellchecker.core.config import SpellCheckerConfig
    from myspellchecker.core.detectors.tokenized_text import TokenizedText
    from myspellchecker.providers.base import DictionaryProvider


@dataclass(frozen=True, slots=True)
class DetectorContext:
    """Read-only context bundle for text-level detector methods.

    All fields are set at the start of a ``check()`` call and remain
    immutable throughout the detection phase.  This makes detector
    dependencies explicit and testable — you can construct a
    ``DetectorContext`` with mock objects for unit testing without
    needing a full ``SpellChecker`` instance.
    """

    provider: DictionaryProvider
    """Dictionary provider for word/frequency lookups."""

    segmenter: Any
    """Word/syllable segmenter (may be None if not configured)."""

    symspell: Any | None
    """SymSpell instance for edit-distance suggestions (may be None)."""

    semantic_checker: Any | None
    """Semantic (MLM) checker for context-aware scoring (may be None)."""

    config: SpellCheckerConfig
    """SpellChecker configuration."""

    tokenized: TokenizedText
    """Pre-computed space-delimited tokenization of the normalized text."""
