"""
POS Inference Engine for Myanmar Text.

This module provides rule-based Part-of-Speech (POS) inference for words
that don't have POS tags in the database. It uses morphological analysis,
prefix/suffix patterns, and a registry of known ambiguous words.

The engine is designed to be used during database packaging to populate
POS tags for untagged words, improving overall POS coverage.

Features:
- Rule-based morphological inference (prefixes, suffixes)
- Numeral detection (Myanmar digits and numeral words)
- Proper noun pattern detection (country, city, university names)
- Multi-POS support for ambiguous words
- Confidence scoring for inference quality
- Batch processing for efficient database updates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from myspellchecker.core.constants import MYANMAR_NUMERALS
from myspellchecker.grammar.config import get_grammar_config
from myspellchecker.text.morphology import (
    MorphologyAnalyzer,
    get_cached_analyzer,
    is_numeral_word,
)
from myspellchecker.utils.singleton import ThreadSafeSingleton

__all__ = [
    "BatchInferenceStats",
    "InferenceSource",
    "POSInferenceEngine",
    "POSInferenceResult",
    "get_pos_inference_engine",
    "infer_pos",
    "infer_pos_batch",
]


class InferenceSource(Enum):
    """Source of POS inference."""

    DATABASE = "database"  # POS from database (seed data)
    AMBIGUOUS_REGISTRY = "ambiguous_registry"  # From AMBIGUOUS_WORDS
    NUMERAL_DETECTION = "numeral_detection"  # Numeral pattern match
    PREFIX_PATTERN = "prefix_pattern"  # Prefix-based inference
    PROPER_NOUN_SUFFIX = "proper_noun_suffix"  # Proper noun suffix match
    SUFFIX_PATTERN = "suffix_pattern"  # Suffix-based inference
    MORPHOLOGICAL = "morphological"  # General morphological analysis
    UNKNOWN = "unknown"  # Could not determine


@dataclass
class POSInferenceResult:
    """
    Result of POS inference for a single word.

    Attributes:
        word: The analyzed word.
        inferred_pos: Primary inferred POS tag (single tag for storage).
        all_pos_tags: All possible POS tags (for multi-POS words).
        confidence: Confidence score (0.0-1.0).
        source: How the POS was determined.
        is_ambiguous: True if word has multiple possible POS.
        requires_context: True if context needed for disambiguation.
        details: Additional inference details.
    """

    word: str
    inferred_pos: str | None = None
    all_pos_tags: frozenset[str] = field(default_factory=frozenset)
    confidence: float = 0.0
    source: InferenceSource = InferenceSource.UNKNOWN
    is_ambiguous: bool = False
    requires_context: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_multi_pos_string(self) -> str | None:
        """
        Convert all POS tags to pipe-separated string for database storage.

        Returns:
            Pipe-separated POS string (e.g., "N|V|ADJ") or None if no tags.

        Example:
            >>> result.all_pos_tags = frozenset({'N', 'V', 'ADJ'})
            >>> result.to_multi_pos_string()
            'ADJ|N|V'
        """
        if not self.all_pos_tags:
            return None
        # Sort for consistent ordering
        return "|".join(sorted(self.all_pos_tags))

    def __repr__(self) -> str:
        return (
            f"POSInferenceResult(word='{self.word}', "
            f"pos='{self.inferred_pos}', "
            f"all_tags={self.all_pos_tags}, "
            f"conf={self.confidence:.2f}, "
            f"source={self.source.value})"
        )


@dataclass
class BatchInferenceStats:
    """Statistics from batch POS inference."""

    total_words: int = 0
    inferred_count: int = 0
    ambiguous_count: int = 0
    numeral_count: int = 0
    prefix_count: int = 0
    suffix_count: int = 0
    unknown_count: int = 0
    avg_confidence: float = 0.0

    def __repr__(self) -> str:
        return (
            f"BatchInferenceStats(total={self.total_words}, "
            f"inferred={self.inferred_count}, "
            f"ambiguous={self.ambiguous_count}, "
            f"unknown={self.unknown_count})"
        )


class POSInferenceEngine:
    """
    Rule-based POS inference engine for Myanmar text.

    Uses multiple strategies to infer POS for untagged words:
    1. Numeral detection (highest priority)
    2. Ambiguous word registry lookup
    3. Prefix-based inference (e.g., အ prefix → Noun)
    4. Proper noun suffix detection
    5. General morphological suffix analysis

    Example:
        >>> engine = POSInferenceEngine()
        >>> result = engine.infer_pos("အလုပ်")
        >>> print(result.inferred_pos)
        'N'
        >>> print(result.source)
        InferenceSource.PREFIX_PATTERN

        >>> # Batch inference
        >>> words = ["အလုပ်", "ကြီး", "၁၂၃"]
        >>> results = engine.infer_pos_batch(words)
    """

    def __init__(self, analyzer: MorphologyAnalyzer | None = None, config_path: str | None = None):
        """
        Initialize the POS inference engine.

        Args:
            analyzer: Optional MorphologyAnalyzer instance.
                     If not provided, uses cached singleton.
            config_path: Optional path to grammar config.
        """
        self._analyzer = analyzer or get_cached_analyzer()
        self.config = get_grammar_config(config_path)

        # Initialize lookups from config
        self.ambiguous_words = self.config.ambiguous_words_map
        self.prefix_patterns: dict[str, dict[str, Any]] = {}
        self.proper_noun_suffixes: dict[str, tuple[str, float, str]] = {}

        self._load_from_config()

    def _load_from_config(self) -> None:
        """Load POS inference rules from config."""
        pos_config = self.config.pos_inference_config

        # Transform prefixes list to dict
        if "prefixes" in pos_config:
            for item in pos_config["prefixes"]:
                prefix = item["prefix"]
                self.prefix_patterns[prefix] = {
                    "pos": item["pos"],
                    "confidence": item["confidence"],
                    "description": item.get("description", ""),
                    "exceptions": frozenset(item.get("exceptions", [])),
                }

        # Transform proper noun suffixes list to dict
        if "proper_noun_suffixes" in pos_config:
            for item in pos_config["proper_noun_suffixes"]:
                suffix = item["suffix"]
                # Store as tuple (pos, confidence, description)
                self.proper_noun_suffixes[suffix] = (
                    item["pos"],
                    item["confidence"],
                    item.get("description", ""),
                )

    def is_ambiguous_word(self, word: str) -> bool:
        """Check if word is in ambiguous registry."""
        return word in self.ambiguous_words

    def infer_pos(self, word: str) -> POSInferenceResult:
        """
        Infer the POS tag for a single word.

        Args:
            word: The word to analyze.

        Returns:
            POSInferenceResult with inference details.

        Example:
            >>> engine = POSInferenceEngine()
            >>> result = engine.infer_pos("မြန်မာနိုင်ငံ")
            >>> print(result.inferred_pos)
            'N'
            >>> print(result.source)
            InferenceSource.PROPER_NOUN_SUFFIX
        """
        if not word:
            return POSInferenceResult(word=word)

        # 1. Check for numerals (highest priority, unambiguous)
        if is_numeral_word(word):
            return self._create_numeral_result(word)

        # 2. Check ambiguous word registry
        if self.is_ambiguous_word(word):
            return self._create_ambiguous_result(word)

        # 3. Check proper noun suffix patterns (before prefix patterns)
        # Proper nouns like "မြန်မာနိုင်ငံ" should not be caught by "မ" prefix
        proper_noun_result = self._check_proper_noun_suffixes(word)
        if proper_noun_result:
            return proper_noun_result

        # 4. Check prefix patterns
        prefix_result = self._check_prefix_patterns(word)
        if prefix_result:
            return prefix_result

        # 5. Use general morphological analysis
        return self._morphological_inference(word)

    def _create_numeral_result(self, word: str) -> POSInferenceResult:
        """Create result for numeral words."""
        # Determine if it's a digit or word numeral
        is_digit = all(char in MYANMAR_NUMERALS for char in word)
        confidence = 0.99 if is_digit else 0.95
        reason = "Myanmar numeral digits" if is_digit else "numeral word"

        return POSInferenceResult(
            word=word,
            inferred_pos="NUM",
            all_pos_tags=frozenset({"NUM"}),
            confidence=confidence,
            source=InferenceSource.NUMERAL_DETECTION,
            is_ambiguous=False,
            requires_context=False,
            details={"reason": reason},
        )

    def _create_ambiguous_result(self, word: str) -> POSInferenceResult:
        """Create result for ambiguous words."""
        tags: frozenset[str] = frozenset(self.ambiguous_words.get(word, frozenset()))

        # For storage, use the most common tag (N > V > ADJ)
        # This can be refined based on frequency data
        primary_tag = self._select_primary_tag(tags)

        return POSInferenceResult(
            word=word,
            inferred_pos=primary_tag,
            all_pos_tags=tags,
            confidence=0.70,
            source=InferenceSource.AMBIGUOUS_REGISTRY,
            is_ambiguous=True,
            requires_context=True,
            details={"possible_tags": sorted(tags)},
        )

    def _select_primary_tag(self, tags: frozenset[str]) -> str | None:
        """
        Select primary tag from multiple options.

        Priority order based on frequency in Myanmar:
        N (noun) > V (verb) > ADJ > ADV > others

        Args:
            tags: Set of possible POS tags.

        Returns:
            Most likely primary tag.
        """
        if not tags:
            return None

        priority = ["N", "V", "ADJ", "ADV", "CONJ", "PRON", "INT"]
        for tag in priority:
            if tag in tags:
                return tag

        # Return first tag if none in priority list
        return next(iter(sorted(tags)))

    def _check_prefix_patterns(self, word: str) -> POSInferenceResult | None:
        """Check prefix patterns for POS inference."""
        for prefix, pattern_info in self.prefix_patterns.items():
            if word.startswith(prefix) and len(word) > len(prefix):
                # Check exceptions
                exceptions = pattern_info.get("exceptions", frozenset())
                if word in exceptions:
                    continue

                pos_tag = pattern_info["pos"]
                confidence = pattern_info["confidence"]
                description = pattern_info["description"]

                return POSInferenceResult(
                    word=word,
                    inferred_pos=pos_tag,
                    all_pos_tags=frozenset({pos_tag}),
                    confidence=confidence,
                    source=InferenceSource.PREFIX_PATTERN,
                    is_ambiguous=False,
                    requires_context=False,
                    details={
                        "prefix": prefix,
                        "description": description,
                    },
                )

        return None

    def _check_proper_noun_suffixes(self, word: str) -> POSInferenceResult | None:
        """Check proper noun suffix patterns for POS inference."""
        for suffix, (pos_tag, confidence, description) in self.proper_noun_suffixes.items():
            if word.endswith(suffix) and len(word) > len(suffix):
                return POSInferenceResult(
                    word=word,
                    inferred_pos=pos_tag,
                    all_pos_tags=frozenset({pos_tag}),
                    confidence=confidence,
                    source=InferenceSource.PROPER_NOUN_SUFFIX,
                    is_ambiguous=False,
                    requires_context=False,
                    details={
                        "suffix": suffix,
                        "description": description,
                    },
                )

        return None

    def _morphological_inference(self, word: str) -> POSInferenceResult:
        """
        Perform general morphological inference using MorphologyAnalyzer.
        """
        tags, confidence, source = self._analyzer.guess_pos_multi(word)

        if not tags:
            return POSInferenceResult(
                word=word,
                source=InferenceSource.UNKNOWN,
            )

        # Determine primary tag
        primary_tag = self._select_primary_tag(tags)
        is_ambiguous = len(tags) > 1

        # Map source string to enum
        source_map = {
            "ambiguous_registry": InferenceSource.AMBIGUOUS_REGISTRY,
            "numeral_detection": InferenceSource.NUMERAL_DETECTION,
            "morphological_inference": InferenceSource.MORPHOLOGICAL,
            "unknown": InferenceSource.UNKNOWN,
        }
        inference_source = source_map.get(source, InferenceSource.MORPHOLOGICAL)

        return POSInferenceResult(
            word=word,
            inferred_pos=primary_tag,
            all_pos_tags=tags,
            confidence=confidence,
            source=inference_source,
            is_ambiguous=is_ambiguous,
            requires_context=is_ambiguous,
            details={"inferred_tags": sorted(tags)},
        )

    def infer_pos_batch(
        self,
        words: list[str],
        existing_pos: dict[str, str] | None = None,
    ) -> tuple[list[POSInferenceResult], BatchInferenceStats]:
        """
        Infer POS for a batch of words.

        Args:
            words: List of words to analyze.
            existing_pos: Optional dict of word -> existing POS tag.
                         Words with existing POS are skipped.

        Returns:
            Tuple of (list of results, statistics).

        Example:
            >>> engine = POSInferenceEngine()
            >>> words = ["အလုပ်", "ကြီး", "၁၂၃", "ကျောင်း"]
            >>> results, stats = engine.infer_pos_batch(words)
            >>> print(stats.inferred_count)
            4
        """
        existing_pos = existing_pos or {}
        results: list[POSInferenceResult] = []
        # Count words that actually need inference (not already in existing_pos)
        words_to_infer = [w for w in words if w not in existing_pos or not existing_pos[w]]
        stats = BatchInferenceStats(total_words=len(words_to_infer))

        total_confidence = 0.0

        for word in words:
            # Skip if already has POS from database
            if word in existing_pos and existing_pos[word]:
                continue

            result = self.infer_pos(word)
            results.append(result)

            # Update statistics
            if result.inferred_pos:
                stats.inferred_count += 1
                total_confidence += result.confidence

            if result.is_ambiguous:
                stats.ambiguous_count += 1

            # Count by source
            if result.source == InferenceSource.NUMERAL_DETECTION:
                stats.numeral_count += 1
            elif result.source == InferenceSource.PREFIX_PATTERN:
                stats.prefix_count += 1
            elif result.source in (
                InferenceSource.SUFFIX_PATTERN,
                InferenceSource.PROPER_NOUN_SUFFIX,
                InferenceSource.MORPHOLOGICAL,
            ):
                stats.suffix_count += 1
            elif result.source == InferenceSource.UNKNOWN:
                stats.unknown_count += 1

        # Calculate average confidence
        if stats.inferred_count > 0:
            stats.avg_confidence = total_confidence / stats.inferred_count

        return results, stats


# Module-level singleton for convenience (thread-safe)
_singleton: ThreadSafeSingleton[POSInferenceEngine] = ThreadSafeSingleton()


def get_pos_inference_engine() -> POSInferenceEngine:
    """
    Get the module-level POSInferenceEngine singleton (thread-safe).

    Uses ThreadSafeSingleton for thread-safe singleton initialization.

    Returns:
        POSInferenceEngine instance.
    """
    return _singleton.get(POSInferenceEngine)


def infer_pos(word: str) -> POSInferenceResult:
    """
    Infer POS for a single word using the default engine.

    Args:
        word: The word to analyze.

    Returns:
        POSInferenceResult with inference details.

    Example:
        >>> from myspellchecker.algorithms.pos_inference import infer_pos
        >>> result = infer_pos("အလုပ်")
        >>> print(result.inferred_pos)
        'N'
    """
    return get_pos_inference_engine().infer_pos(word)


def infer_pos_batch(
    words: list[str],
    existing_pos: dict[str, str] | None = None,
) -> tuple[list[POSInferenceResult], BatchInferenceStats]:
    """
    Infer POS for a batch of words using the default engine.

    Args:
        words: List of words to analyze.
        existing_pos: Optional dict of word -> existing POS tag.

    Returns:
        Tuple of (list of results, statistics).
    """
    return get_pos_inference_engine().infer_pos_batch(words, existing_pos)
