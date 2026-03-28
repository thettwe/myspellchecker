"""
Reduplication engine for validating productive Myanmar reduplications.

Myanmar frequently creates valid words through reduplication:
- AA: ကောင်းကောင်း ("well", from ကောင်း "good")
- AABB: သေသေချာချာ ("carefully", each syllable doubles)
- ABAB: ခဏခဏ ("frequently", whole word repeats)
- RHYME: Known rhyme pairs from grammar/patterns.py

This engine validates OOV words that are productive reduplications of known
dictionary words, suppressing false positive WordErrors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from myspellchecker.grammar.patterns import (
    RHYME_REDUPLICATION_PATTERNS,
    detect_reduplication_pattern,
)
from myspellchecker.text.types import DictionaryCheck, FrequencyCheck, POSCheck
from myspellchecker.utils.cache import LRUCache
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.core.config.algorithm_configs import ReduplicationConfig
    from myspellchecker.segmenters.base import Segmenter

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_ALLOWED_BASE_POS",
    "ReduplicationEngine",
    "ReduplicationResult",
]


@dataclass(frozen=True)
class ReduplicationResult:
    """Result of reduplication analysis.

    Attributes:
        word: The original word analyzed.
        pattern: Reduplication pattern type (AA, AABB, ABAB, RHYME).
        base_word: The base word from which reduplication is derived.
        is_valid: Whether this is a valid productive reduplication.
        pos_tag: POS tag of the base word (if available).
        confidence: Confidence score for the analysis.
    """

    word: str
    pattern: str
    base_word: str
    is_valid: bool
    pos_tag: str | None = None
    confidence: float = 0.0


# POS tags that can productively reduplicate in Myanmar
DEFAULT_ALLOWED_BASE_POS: frozenset[str] = frozenset({"V", "ADJ", "ADV", "N"})


class ReduplicationEngine:
    """Engine for validating productive Myanmar reduplications.

    Validates OOV words that are formed by reduplicating known dictionary words.
    Uses frequency floors and POS filters to prevent false positives.

    Args:
        segmenter: Segmenter for syllable segmentation.
        min_base_frequency: Minimum frequency for the base word (default: 5).
        cache_size: Maximum cache entries (default: 1024).
        allowed_base_pos: POS tags allowed to reduplicate.
        config: Optional ReduplicationConfig for algorithm tuning parameters.
            When provided, overrides min_base_frequency and cache_size with
            config values.
    """

    def __init__(
        self,
        segmenter: Segmenter,
        min_base_frequency: int = 5,
        cache_size: int = 1024,
        allowed_base_pos: frozenset[str] = DEFAULT_ALLOWED_BASE_POS,
        config: ReduplicationConfig | None = None,
    ):
        if config is not None:
            self.min_base_frequency = config.min_base_frequency
            cache_size = config.cache_size
        else:
            self.min_base_frequency = min_base_frequency

        self.segmenter = segmenter
        self.cache_size = cache_size
        self.allowed_base_pos = allowed_base_pos
        self._config = config
        self._cache: LRUCache[ReduplicationResult | None] = LRUCache(maxsize=cache_size)

    def analyze(
        self,
        word: str,
        dictionary_check: DictionaryCheck,
        frequency_check: FrequencyCheck,
        pos_check: POSCheck,
    ) -> ReduplicationResult | None:
        """Analyze a word for productive reduplication patterns.

        Args:
            word: The OOV word to analyze.
            dictionary_check: Callable returning True if word is in dictionary.
            frequency_check: Callable returning word frequency count.
            pos_check: Callable returning POS tag string or None.

        Returns:
            ReduplicationResult if valid reduplication detected, None otherwise.
        """
        _sentinel = object()
        cached = self._cache.get(word, _sentinel)
        if cached is not _sentinel:
            return cached

        result = self._analyze_impl(word, dictionary_check, frequency_check, pos_check)
        self._cache.set(word, result)
        return result

    def _analyze_impl(
        self,
        word: str,
        dictionary_check: DictionaryCheck,
        frequency_check: FrequencyCheck,
        pos_check: POSCheck,
    ) -> ReduplicationResult | None:
        """Internal implementation of reduplication analysis."""
        # Check known rhyme reduplication patterns first (fast path)
        if word in RHYME_REDUPLICATION_PATTERNS:
            rhyme_conf = self._config.pattern_confidence_rhyme if self._config is not None else 0.95
            return ReduplicationResult(
                word=word,
                pattern="RHYME",
                base_word=word,
                is_valid=True,
                confidence=rhyme_conf,
            )

        # Segment into syllables
        syllables = self.segmenter.segment_syllables(word)
        if len(syllables) < 2:
            return None

        # Detect reduplication pattern
        pattern = detect_reduplication_pattern(syllables)
        if pattern == "NONE":
            return None

        # Extract base form based on pattern
        base_word = self._extract_base(syllables, pattern)
        if not base_word:
            return None

        # Validate base word: must be in dictionary
        if not dictionary_check(base_word):
            return None

        # Validate frequency floor
        freq = frequency_check(base_word)
        if freq < self.min_base_frequency:
            return None

        # Validate POS: particles can't reduplicate
        pos = pos_check(base_word)
        if pos is not None and pos not in self.allowed_base_pos:
            return None

        # Compute confidence based on pattern type and frequency
        confidence = self._compute_confidence(pattern, freq)

        return ReduplicationResult(
            word=word,
            pattern=pattern,
            base_word=base_word,
            is_valid=True,
            pos_tag=pos,
            confidence=confidence,
        )

    def _extract_base(self, syllables: list[str], pattern: str) -> str | None:
        """Extract the base word from syllables given the reduplication pattern.

        Args:
            syllables: List of syllables.
            pattern: Detected pattern type (AB, AABB, ABAB).

        Returns:
            Base word string, or None if extraction fails.
        """
        n = len(syllables)

        if pattern == "AB" and n == 2:
            # AA: base is the first (= second) syllable
            return syllables[0]

        if pattern == "AABB" and n == 4:
            # AABB: each syllable doubles (A-A-B-B)
            # syl[0]==syl[1] and syl[2]==syl[3] (e.g., သေသေချာချာ)
            # Base is the disyllabic word formed by one from each pair: syl[0]+syl[2]
            return syllables[0] + syllables[2]

        if pattern == "ABAB" and n == 4:
            # ABAB: whole disyllabic unit repeats (AB-AB)
            # syl[0]==syl[2] and syl[1]==syl[3] (e.g., ခဏခဏ)
            # Base is the AB unit = syl[0] + syl[1]
            return "".join(syllables[:2])

        return None

    def _compute_confidence(self, pattern: str, frequency: int) -> float:
        """Compute confidence score for a reduplication result.

        Args:
            pattern: Reduplication pattern type.
            frequency: Frequency of the base word.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        cfg = self._config
        # Base confidence per pattern type
        if cfg is not None:
            pattern_confidence = {
                "AB": cfg.pattern_confidence_ab,
                "AABB": cfg.pattern_confidence_aabb,
                "ABAB": cfg.pattern_confidence_abab,
                "RHYME": cfg.pattern_confidence_rhyme,
            }
            default_conf = cfg.pattern_confidence_default
        else:
            pattern_confidence = {
                "AB": 0.90,
                "AABB": 0.85,
                "ABAB": 0.85,
                "RHYME": 0.95,
            }
            default_conf = 0.80
        base = pattern_confidence.get(pattern, default_conf)

        # Frequency boost (higher frequency = more confidence)
        high_thresh = cfg.high_freq_threshold if cfg is not None else 100
        med_thresh = cfg.medium_freq_threshold if cfg is not None else 50
        high_boost = cfg.high_freq_boost if cfg is not None else 0.05
        high_cap = cfg.high_freq_cap if cfg is not None else 0.98
        med_boost = cfg.medium_freq_boost if cfg is not None else 0.03
        med_cap = cfg.medium_freq_cap if cfg is not None else 0.95

        if frequency >= high_thresh:
            base = min(base + high_boost, high_cap)
        elif frequency >= med_thresh:
            base = min(base + med_boost, med_cap)

        return base
