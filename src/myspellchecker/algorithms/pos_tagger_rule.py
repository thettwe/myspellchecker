"""
Rule-based POS tagger using morphological suffix analysis.

This module provides a fast, dependency-free POS tagger that analyzes word
suffixes to guess Part-of-Speech tags. It wraps the existing MorphologyAnalyzer
and adds caching for performance.

Features:
- No external dependencies (pure Python)
- Fast performance (~100K words/second)
- LRU cache for repeated lookups
- Fork-safe (can be used in multiprocessing)
- Optional pos_map for manual word-to-tag overrides

Example:
    >>> from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger
    >>>
    >>> # Default initialization (morphology-based)
    >>> tagger = RuleBasedPOSTagger()
    >>> tagger.tag_word("စားတယ်")
    'P_SENT'
"""

from __future__ import annotations

from functools import lru_cache

from myspellchecker.algorithms.pos_tagger_base import (
    POSPrediction,
    POSTaggerBase,
    TaggerType,
)
from myspellchecker.text.morphology import MorphologyAnalyzer


class RuleBasedPOSTagger(POSTaggerBase):
    """
    Rule-based POS tagger using morphological analysis.

    Uses suffix patterns and a predefined POS map to guess word tags.
    Falls back to morphological analysis for unknown words.

    Attributes:
        pos_map: Dictionary mapping words to sets of POS tags
        morphology_analyzer: MorphologyAnalyzer for suffix-based guessing
        cache_size: Size of LRU cache for performance (default: 10000)

    Example:
        >>> pos_map = {
        ...     "မြန်မာ": {"N"},
        ...     "နိုင်ငံ": {"N"},
        ...     "သည်": {"P_SENT"}
        ... }
        >>> tagger = RuleBasedPOSTagger(pos_map=pos_map)
        >>>
        >>> # Tag single word
        >>> tagger.tag_word("မြန်မာ")
        'N'
        >>>
        >>> # Tag sequence
        >>> tagger.tag_sequence(["မြန်မာ", "နိုင်ငံ", "သည်"])
        ['N', 'N', 'P_SENT']
        >>>
        >>> # With confidence scores
        >>> pred = tagger.tag_word_with_confidence("မြန်မာ")
        >>> print(f"{pred.tag} ({pred.confidence:.2f})")
        N (1.00)
    """

    def __init__(
        self,
        pos_map: dict[str, set[str]] | None = None,
        use_morphology_fallback: bool = True,
        cache_size: int = 10000,
        unknown_tag: str = "UNK",
    ):
        """
        Initialize rule-based POS tagger.

        Args:
            pos_map: Dictionary mapping words to sets of POS tags.
                    If None, uses morphological analysis only.
            use_morphology_fallback: Whether to use MorphologyAnalyzer
                                    for unknown words (default: True)
            cache_size: Size of LRU cache for tag lookups (default: 10000)
            unknown_tag: Tag to return for completely unknown words (default: "UNK")

        Example:
            >>> # Default (morphology only)
            >>> tagger = RuleBasedPOSTagger()
            >>>
            >>> # With POS map
            >>> pos_map = {"word": {"N"}}
            >>> tagger = RuleBasedPOSTagger(pos_map=pos_map)
            >>>
            >>> # Disable morphology fallback
            >>> tagger = RuleBasedPOSTagger(
            ...     pos_map=pos_map,
            ...     use_morphology_fallback=False
            ... )
        """
        self.pos_map = pos_map or {}
        self.use_morphology_fallback = use_morphology_fallback
        self.unknown_tag = unknown_tag

        # Initialize morphology analyzer
        self.morphology_analyzer = MorphologyAnalyzer() if use_morphology_fallback else None

        # Set up caching
        self._cache_size = cache_size
        self._setup_cache()

    def _setup_cache(self) -> None:
        """Set up LRU cache for tag lookups."""
        # Create cached version of _tag_word_impl
        self._tag_word_cached = lru_cache(maxsize=self._cache_size)(self._tag_word_impl)

    def __getstate__(self) -> dict:
        """Remove non-picklable LRU cache wrapper before pickling."""
        state = self.__dict__.copy()
        state.pop("_tag_word_cached", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state and recreate LRU cache after unpickling."""
        self.__dict__.update(state)
        self._setup_cache()

    def _tag_word_impl(self, word: str) -> str:
        """
        Core tagging implementation.

        This is the core tagging logic that is wrapped by the cache.

        Args:
            word: Word to tag

        Returns:
            POS tag string
        """
        # 1. Check POS map first (highest priority)
        if word in self.pos_map:
            tags = self.pos_map[word]
            # Return first tag (or use some priority logic)
            # Could be enhanced to use tag priority ordering
            return sorted(tags)[0]

        # 2. Fall back to morphological analysis
        if self.morphology_analyzer:
            best_tag = self.morphology_analyzer.guess_pos_best(word)
            if best_tag:
                return best_tag

        # 3. Final fallback to unknown tag
        return self.unknown_tag

    def tag_word(self, word: str) -> str:
        """
        Tag a single word with its POS tag.

        Uses cached lookups for better performance on repeated words.

        Args:
            word: The word to tag

        Returns:
            POS tag string (e.g., "N", "V", "P_SENT")

        Example:
            >>> tagger = RuleBasedPOSTagger()
            >>> tagger.tag_word("စားတယ်")
            'P_SENT'
        """
        if not word:
            return self.unknown_tag

        return self._tag_word_cached(word)

    def tag_sequence(self, words: list[str]) -> list[str]:
        """
        Tag a sequence of words with their POS tags.

        Note: This is a simple implementation that tags each word independently.
        It does not use contextual information. For context-aware tagging,
        use ViterbiTagger instead.

        Args:
            words: List of words to tag

        Returns:
            List of POS tags corresponding to input words

        Example:
            >>> tagger = RuleBasedPOSTagger()
            >>> tagger.tag_sequence(["မြန်မာ", "နိုင်ငံ", "သည်"])
            ['N', 'N', 'P_SENT']
        """
        return [self.tag_word(word) for word in words]

    def tag_word_with_confidence(self, word: str) -> POSPrediction:
        """
        Tag a word and return prediction with confidence score.

        Confidence is based on the source of the tag:
        - pos_map lookup: 1.0 (high confidence)
        - morphological guess: uses MorphologyAnalyzer confidence
        - unknown: 0.0 (no confidence)

        Args:
            word: The word to tag

        Returns:
            POSPrediction with tag and confidence score

        Example:
            >>> tagger = RuleBasedPOSTagger()
            >>> pred = tagger.tag_word_with_confidence("စားတယ်")
            >>> print(f"{pred.tag} (conf: {pred.confidence:.2f})")
            P_SENT (conf: 0.67)
        """
        if not word:
            return POSPrediction(
                word=word, tag=self.unknown_tag, confidence=0.0, metadata={"source": "unknown"}
            )

        # Check POS map first
        if word in self.pos_map:
            tags = self.pos_map[word]
            tag = sorted(tags)[0]
            return POSPrediction(
                word=word,
                tag=tag,
                confidence=1.0,
                metadata={"source": "pos_map", "all_tags": sorted(tags)},
            )

        # Try morphological analysis
        if self.morphology_analyzer:
            ranked_guesses = self.morphology_analyzer.guess_pos_ranked(word)
            if ranked_guesses:
                best_guess = ranked_guesses[0]
                return POSPrediction(
                    word=word,
                    tag=best_guess.tag,
                    confidence=best_guess.confidence,
                    metadata={
                        "source": "morphology",
                        "reason": best_guess.reason,
                        "alternatives": len(ranked_guesses),
                    },
                )

        # Unknown word
        return POSPrediction(
            word=word, tag=self.unknown_tag, confidence=0.0, metadata={"source": "fallback"}
        )

    def clear_cache(self) -> None:
        """
        Clear the LRU cache.

        Useful for freeing memory or when the POS map has been updated.

        Example:
            >>> tagger = RuleBasedPOSTagger()
            >>> tagger.tag_word("word")  # Cached
            >>> tagger.clear_cache()  # Clear cache
        """
        if hasattr(self, "_tag_word_cached"):
            self._tag_word_cached.cache_clear()

    def cache_info(self) -> tuple:
        """
        Get cache statistics.

        Returns:
            Named tuple with cache statistics (hits, misses, maxsize, currsize)

        Example:
            >>> tagger = RuleBasedPOSTagger()
            >>> tagger.tag_word("word")
            >>> info = tagger.cache_info()
            >>> print(f"Cache hits: {info.hits}, misses: {info.misses}")
        """
        if hasattr(self, "_tag_word_cached"):
            return self._tag_word_cached.cache_info()
        import functools

        return functools._CacheInfo(
            hits=0, misses=0, maxsize=self._cache_size, currsize=0
        )

    @property
    def tagger_type(self) -> TaggerType:
        """Return the tagger type identifier."""
        return TaggerType.RULE_BASED

    @property
    def supports_batch(self) -> bool:
        """Rule-based tagger does not benefit from batching."""
        return False

    @property
    def is_fork_safe(self) -> bool:
        """Rule-based tagger is fork-safe (pure Python, no CUDA)."""
        return True
