"""
Hidden Compound Typo Validation Strategy.

Detects compound-word typos that are hidden by segmenter over-splitting.

Canonical example: the input ``ခုန်ကျစရိတ်`` (intended: ``ကုန်ကျစရိတ်``,
"costs/expenses") is segmented as ``['ခုန်', 'ကျ', 'စရိတ်']`` — each piece is
a valid standalone word, so no upstream strategy flags it. The correct
compound ``ကုန်ကျစရိတ်`` has frequency 27677 in the production dictionary.

The strategy recovers these errors by:

1. Walking token windows where both w_i and w_{i+1} are curated vocabulary.
2. Generating confusable variants of w_i via
   :func:`generate_confusable_variants` (phonetic / tonal / medial / nasal /
   stop-coda / stacking / kinzi).
3. Checking whether ``variant + w_{i+1}`` is a high-frequency dictionary
   compound.
4. For freq=0 subsumed compounds (e.g. ``ကုန်ကျ`` is valid but freq=0 because
   the corpus only attests the larger ``ကုန်ကျစရိတ်``), extending the window
   by one token and checking the trigram variant.
5. Emitting a multi-token-span WordError via
   :data:`~myspellchecker.core.constants.ET_HIDDEN_COMPOUND_TYPO` without
   calling ``_mark_positions`` — downstream strategies keep running at the
   same position and a post-processing suppression rule handles deduplication.

Priority: **23** (structural phase, before StatisticalConfusable 24 and
BrokenCompound 25, surviving the fast-path cutoff at 25).

Sprint A scaffold: ``validate`` is a no-op returning ``[]``. Detection logic
lands in Sprint B. See
``~/Documents/myspellchecker/Workstreams/v1.5.0/hidden-compound-typo-plan.md``
for the full plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.core.response import Error
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import WordRepository
    from myspellchecker.text.phonetic import PhoneticHasher

logger = get_logger(__name__)

_PRIORITY = 23


class HiddenCompoundStrategy(ValidationStrategy):
    """
    Detect hidden compound typos exposed by segmenter over-splitting.

    See module docstring for the full algorithm. This Sprint A scaffold
    implements only the constructor, priority, and a no-op ``validate``.
    Detection logic (variant generation, bigram / trigram lookup, confidence
    scoring, span-error emission) is added in Sprint B.

    Args:
        provider: DictionaryProvider for word / frequency / vocabulary lookups.
        hasher: PhoneticHasher instance (held by the strategy so the
            LRU-cached variant method can key on word-only without making
            the hasher a cache key — PhoneticHasher is not hashable).
        config: Configuration snapshot read from
            ``ValidationConfig.hidden_compound_*`` fields.
    """

    def __init__(
        self,
        provider: WordRepository,
        hasher: PhoneticHasher | None,
        *,
        enabled: bool = False,
        max_token_syllables: int = 3,
        max_variants_per_token: int = 20,
        compound_min_frequency: int = 100,
        confidence_floor: float = 0.75,
        enable_trigram_lookahead: bool = True,
        variant_cache_size: int = 8192,
        require_typo_prone_chars: bool = True,
        curated_only: bool = True,
    ) -> None:
        self.provider = provider
        self._hasher = hasher
        self.enabled = enabled
        self.max_token_syllables = max_token_syllables
        self.max_variants_per_token = max_variants_per_token
        self.compound_min_frequency = compound_min_frequency
        self.confidence_floor = confidence_floor
        self.enable_trigram_lookahead = enable_trigram_lookahead
        self.variant_cache_size = variant_cache_size
        self.require_typo_prone_chars = require_typo_prone_chars
        self.curated_only = curated_only
        self.logger = logger

    def priority(self) -> int:
        """Return strategy execution priority (23).

        Placed before StatisticalConfusable (24) and BrokenCompound (25) so
        the strategy runs inside the structural phase (priority <= 25) and
        survives the fast-path cutoff at
        :attr:`ContextValidator._FAST_PATH_PRIORITY_CUTOFF`. Hidden-compound
        failures predominantly occur on structurally-clean sentences.
        """
        return _PRIORITY

    def validate(self, context: ValidationContext) -> list[Error]:
        """Sprint A scaffold: always returns an empty list.

        Sprint B fills in the detection logic (variant generation, bigram
        window, lookahead, confidence gating, span error emission).
        """
        if not self.enabled:
            return []
        # Sprint B: real implementation lands here.
        return []

    def __repr__(self) -> str:
        return (
            f"HiddenCompoundStrategy(priority={self.priority()}, "
            f"enabled={self.enabled}, "
            f"max_token_syllables={self.max_token_syllables}, "
            f"confidence_floor={self.confidence_floor})"
        )
