"""
SymSpell algorithm implementation for Myanmar spell checking.

SymSpell is a symmetric delete spelling correction algorithm that provides
fast, accurate suggestions for misspelled words. The key innovation is
generating delete-only variations at indexing time, enabling O(1) lookups.

Algorithm Overview:
1. **Indexing**: Generate all delete variations within max_edit_distance
2. **Lookup**: Generate deletes of query word, find matches in index
3. **Ranking**: Sort candidates by edit distance, then by frequency

For Myanmar spell checking, SymSpell is used at both syllable and word levels.

Performance Characteristics:
    Time Complexity:
        - build_index(): O(V * L^d) where V=vocabulary size, L=avg term length,
          d=max_edit_distance. Bounded by max_deletes_per_term.
        - lookup(): O(L^d + C) where L=query length, d=max_edit_distance,
          C=number of candidates. Typically O(1) for small d.
        - lookup_compound(): O(N * M * L^d) where N=text length, M=max_word_length

    Space Complexity:
        - Delete index: O(V * L^d) entries, bounded by max_deletes_per_term
        - Each entry stores (term, distance) tuples

    Cache Behavior:
        - No internal LRU cache in SymSpell itself
        - Edit distance calculations may benefit from external caching
        - Ranker may have its own caching strategy

Memory Considerations:
    - max_edit_distance > 2 causes exponential memory growth
    - max_deletes_per_term (default 5000) prevents runaway memory usage
    - prefix_length (default 10) reduces index size for long terms
    - Thread-safe via RLock for concurrent build_index/lookup

References:
    - Original SymSpell: https://github.com/wolfgarbe/SymSpell
    - Paper: "SymSpell: 1 million times faster through Symmetric Delete"
"""

from __future__ import annotations

import functools
import math
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myspellchecker.segmenters.base import Segmenter
    from myspellchecker.text.phonetic import PhoneticHasher

from myspellchecker.algorithms.distance.edit_distance import (
    damerau_levenshtein_distance,
    myanmar_syllable_edit_distance,
    weighted_damerau_levenshtein_distance,
)
from myspellchecker.algorithms.ranker import (
    DefaultRanker,
    SuggestionData,
    SuggestionRanker,
)
from myspellchecker.core.config import RankerConfig, SymSpellConfig
from myspellchecker.core.constants import ValidationLevel
from myspellchecker.providers import DictionaryProvider
from myspellchecker.text.normalize import (
    get_nasal_variants,
    has_same_nasal_ending,
    is_true_nasal_variant,
)
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "Suggestion",
    "SymSpell",
]

# ---------------------------------------------------------------------------
# Fast nasal-content guard (avoids full variant generation for ~60-70% of
# Myanmar words that contain no nasal endings).
# ---------------------------------------------------------------------------

_NASAL_CHARS = frozenset({"\u1036", "\u1014", "\u1019", "\u1004"})  # anusvara, na, ma, nga
_ASAT = "\u103a"


@functools.lru_cache(maxsize=8192)
def _has_nasal_ending(word: str) -> bool:
    """Fast check for nasal content (avoids full variant generation)."""
    if "\u1036" in word:  # anusvara — always relevant
        return True
    for i, ch in enumerate(word):
        if ch in _NASAL_CHARS and i + 1 < len(word) and word[i + 1] == _ASAT:
            return True
    return False


@dataclass
class Suggestion:
    """
    A spelling correction suggestion.

    Attributes:
        term: The suggested correction
        edit_distance: Damerau-Levenshtein distance from query
        frequency: Corpus frequency of the suggestion
        phonetic_score: Phonetic similarity score (0.0-1.0, higher is better)
        syllable_distance: Myanmar syllable-aware weighted distance (optional)
        weighted_distance: Myanmar-weighted edit distance using substitution costs (optional)
        is_nasal_variant: True if this is a nasal variant (န် ↔ ံ substitution only)
        has_same_nasal_ending: True if same nasal ending as the query term
        score: Combined score for ranking (lower is better), computed by SuggestionRanker

    Note:
        The score is always computed externally by a SuggestionRanker instance.
        This allows pluggable ranking strategies without duplicating scoring logic.
        SymSpell always creates a DefaultRanker if no custom ranker is provided.
    """

    term: str
    edit_distance: int
    frequency: int
    phonetic_score: float = 0.0
    syllable_distance: float | None = None  # Weighted syllable distance
    weighted_distance: float | None = None  # Myanmar-weighted edit distance
    is_nasal_variant: bool = False  # True if န် ↔ ံ substitution
    has_same_nasal_ending: bool = False  # True if same nasal ending as query
    score: float = 0.0  # Score from SuggestionRanker (lower is better)

    def __lt__(self, other: "Suggestion") -> bool:
        """Enable sorting suggestions by score."""
        return self.score < other.score

    def __eq__(self, other: object) -> bool:
        """Check equality based on term."""
        if not isinstance(other, Suggestion):
            return NotImplemented
        return self.term == other.term

    def __hash__(self) -> int:
        """Enable using Suggestion in sets."""
        return hash(self.term)


class SymSpell:
    """
    SymSpell spell correction algorithm.

    This implementation provides fast spelling correction using symmetric
    delete variations. It works with a DictionaryProvider to access
    dictionary data and frequencies.

    Attributes:
        provider: DictionaryProvider for dictionary access
        max_edit_distance: Maximum edit distance for suggestions (default: 2)
        prefix_length: Prefix length for optimization (default: 10 for Myanmar)
        count_threshold: Minimum frequency threshold (default: 1)
        max_word_length: Max word length for compound segmentation (default: 15)
        compound_lookup_count: Suggestions to consider per word in compound (default: 3)
        beam_width: Beam width for compound segmentation DP (default: 50)
        compound_max_suggestions: Max suggestions for compound queries (default: 5)

    Performance:
        - lookup() is O(1) average case for typical queries with small max_edit_distance
        - build_index() is O(V * L^d) but only runs once at startup
        - Thread-safe: concurrent lookups are safe after index is built

    Memory Usage:
        - Index size scales with vocabulary and max_edit_distance
        - Typical Myanmar corpus (100K terms, d=2): ~50-100MB index
        - Use max_deletes_per_term to cap memory per term

    Example:
        >>> from myspellchecker.providers import MemoryProvider
        >>> from myspellchecker.algorithms import SymSpell
        >>>
        >>> provider = MemoryProvider()
        >>> provider.add_syllable("မြန်", frequency=1000)
        >>> provider.add_syllable("မာ", frequency=800)
        >>>
        >>> symspell = SymSpell(provider, max_edit_distance=2)
        >>> symspell.build_index(['syllable'])  # Index syllables
        >>>
        >>> suggestions = symspell.lookup("မြ", level='syllable')
        >>> for s in suggestions[:3]:
        ...     print(f"{s.term} (distance: {s.edit_distance}, freq: {s.frequency})")
    """

    @staticmethod
    def compute_frequency_denominator(
        provider: DictionaryProvider,
        level: str = "syllable",
        percentile: float = 0.5,
    ) -> float:
        """
        Compute a data-driven frequency denominator from corpus statistics.

        The frequency denominator affects how corpus frequency influences
        suggestion ranking. Using a value derived from actual corpus statistics
        ensures consistent scoring across different corpora.

        Args:
            provider: DictionaryProvider with frequency data
            level: 'syllable' or 'word' (default: 'syllable')
            percentile: Percentile to use (0.5 = median, 0.9 = 90th percentile)
                       Default 0.5 (median) provides balanced normalization.

        Returns:
            Computed denominator, or 10000.0 if corpus is empty.

        Example:
            >>> denom = SymSpell.compute_frequency_denominator(provider)
            >>> symspell = SymSpell(provider, frequency_denominator=denom)
        """
        frequencies: list[int] = []

        try:
            if level == "syllable" and hasattr(provider, "get_all_syllables"):
                for _, freq in provider.get_all_syllables():
                    frequencies.append(freq)
            elif level == "word" and hasattr(provider, "get_all_words"):
                for _, freq in provider.get_all_words():
                    frequencies.append(freq)
        except (AttributeError, TypeError, StopIteration) as e:
            logger.debug(f"Failed to compute frequency denominator: {e}")
            # Fall through to default if provider methods fail

        if not frequencies:
            return 10000.0  # Default fallback

        # Sort and find percentile value
        frequencies.sort()
        idx = int(len(frequencies) * percentile)
        idx = min(idx, len(frequencies) - 1)
        percentile_freq = frequencies[idx]

        # Use 10x the percentile frequency as denominator
        # This ensures that frequency bonus asymptotes reasonably
        # (50% of corpus is below median, gets partial bonus)
        return max(float(percentile_freq) * 10.0, 100.0)

    def __init__(
        self,
        provider: DictionaryProvider,
        max_edit_distance: int = 2,
        prefix_length: int = 10,
        count_threshold: int = 1,
        phonetic_hasher: PhoneticHasher | None = None,
        max_word_length: int = 15,
        compound_lookup_count: int = 3,
        beam_width: int = 50,
        compound_max_suggestions: int = 5,
        damerau_cache_size: int = 4096,
        frequency_denominator: float = 10000.0,
        phonetic_bonus_weight: float = 0.4,
        use_syllable_distance: bool | None = None,
        syllable_bonus_weight: float | None = None,
        use_weighted_distance: bool = True,
        weighted_distance_bonus_weight: float | None = None,
        ranker: SuggestionRanker | None = None,
        max_deletes_per_term: int | None = None,
        syllable_segmenter: "Segmenter" | None = None,
        phonetic_bypass_threshold: float = 0.85,
        phonetic_extra_distance: int = 1,
        config: SymSpellConfig | None = None,
    ):
        """
        Initialize SymSpell with a dictionary provider.

        Args:
            provider: DictionaryProvider for accessing dictionary data
            max_edit_distance: Maximum edit distance for suggestions (1-3 recommended)
            prefix_length: Prefix length for index optimization (default: 10 for Myanmar).
                Myanmar syllables are typically 3-5 Unicode codepoints, so a word with
                2-3 syllables can easily exceed 7 characters. Using 10 ensures better
                suggestion quality for longer compound words.
            count_threshold: Minimum frequency threshold for suggestions
            phonetic_hasher: Optional PhoneticHasher for phonetic similarity
            max_word_length: Max word length for compound segmentation
            compound_lookup_count: Suggestions to consider per word in compound
            beam_width: Beam width for compound segmentation DP
            compound_max_suggestions: Max suggestions for compound queries
            damerau_cache_size: Cache size for edit distance calculations.
            frequency_denominator: Denominator for frequency bonus in scoring (default: 10000.0).
                Used to configure the default ranker if no custom ranker is provided.
                Use SymSpell.compute_frequency_denominator(provider) for a data-driven value.
            phonetic_bonus_weight: Weight for phonetic similarity bonus (default: 0.4).
                Used to configure the default ranker if no custom ranker is provided.
            use_syllable_distance: Enable Myanmar syllable-aware edit distance (default: True)
                                  Treats medial confusions (ျ vs ြ) as single edits
            syllable_bonus_weight: Weight bonus for syllable-aware scoring (default: 0.3).
                Used to configure the default ranker if no custom ranker is provided.
            use_weighted_distance: Enable Myanmar-weighted edit distance (default: True).
                Uses MYANMAR_SUBSTITUTION_COSTS to give lower cost to phonetically
                similar character substitutions (e.g., aspirated consonant pairs
                like က↔ခ, medial confusions ျ↔ြ, vowel length ိ↔ီ).
            weighted_distance_bonus_weight: Weight for weighted distance bonus (default: 0.35).
                Applied when weighted_distance < edit_distance, indicating
                phonetically related character substitutions.
                Used to configure the default ranker if no custom ranker is provided.
            ranker: Optional SuggestionRanker for custom ranking strategies.
                If provided, it will be used instead of the DefaultRanker.
                If None, a DefaultRanker is created with the scoring parameters above.
                See myspellchecker.algorithms.ranker for available rankers.
            max_deletes_per_term: Maximum number of delete variations to generate per term
                (default: 5000). This prevents O(n^d) memory growth for long terms or
                high max_edit_distance values. If the limit is exceeded, delete generation
                stops early and a warning is logged.
            syllable_segmenter: Optional Segmenter for syllable-level validation during
                compound segmentation. If provided, compound segmentation will
                also validate that each segmented word has valid syllable structure,
                not just dictionary presence.
            phonetic_bypass_threshold: Minimum graded phonetic similarity (0.0-1.0)
                required to bypass max_edit_distance cap (default: 0.85).
                Candidates with similarity >= this threshold can exceed
                max_edit_distance by up to phonetic_extra_distance.
            phonetic_extra_distance: Maximum additional edit distance allowed
                for high-similarity phonetic candidates (default: 1).
                Only applies when phonetic_similarity >= phonetic_bypass_threshold.

        Note:
            The index must be built using build_index() before performing lookups.

        Example:
            >>> from myspellchecker.algorithms.ranker import FrequencyFirstRanker
            >>> symspell = SymSpell(provider, ranker=FrequencyFirstRanker())
        """
        # Resolve config for previously hardcoded defaults
        cfg = config or SymSpellConfig()

        self.provider = provider
        # Validate max_edit_distance to prevent unbounded memory growth
        # and excessive iteration in delete generation
        if max_edit_distance < 0:
            raise ValueError(f"max_edit_distance must be non-negative, got {max_edit_distance}")
        if max_edit_distance > 5:
            raise ValueError(
                f"max_edit_distance={max_edit_distance} is too high "
                f"(max 5). This would cause O(n^d) memory growth. "
                f"Use 1-3 for best results."
            )
        if max_edit_distance > 3:
            import warnings

            warnings.warn(
                f"max_edit_distance={max_edit_distance} may cause high "
                f"memory usage and slow index building. "
                f"Consider using 1-2 for Myanmar text.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.max_edit_distance = max_edit_distance

        # Resolve params that can come from explicit args or config
        effective_max_deletes = (
            max_deletes_per_term if max_deletes_per_term is not None else cfg.max_deletes_per_term
        )
        effective_use_syllable_distance = (
            use_syllable_distance
            if use_syllable_distance is not None
            else cfg.use_syllable_distance
        )
        effective_syllable_bonus = (
            syllable_bonus_weight
            if syllable_bonus_weight is not None
            else cfg.syllable_bonus_weight
        )
        effective_weighted_bonus = (
            weighted_distance_bonus_weight
            if weighted_distance_bonus_weight is not None
            else cfg.weighted_distance_bonus_weight
        )

        # Validate max_deletes_per_term to prevent resource exhaustion
        if effective_max_deletes < 100:
            raise ValueError(
                f"max_deletes_per_term must be at least 100, got {effective_max_deletes}"
            )
        if effective_max_deletes > 100000:
            import warnings

            warnings.warn(
                f"max_deletes_per_term={effective_max_deletes} is very "
                f"high and may cause excessive memory usage. "
                f"Consider using 5000-10000.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._max_deletes_per_term = effective_max_deletes
        self.prefix_length = prefix_length
        self.count_threshold = count_threshold
        self.phonetic_hasher = phonetic_hasher
        self.max_word_length = max_word_length
        self.compound_lookup_count = compound_lookup_count
        self.beam_width = beam_width
        self.compound_max_suggestions = compound_max_suggestions
        self.damerau_cache_size = damerau_cache_size
        self.use_syllable_distance = effective_use_syllable_distance
        self.use_weighted_distance = use_weighted_distance
        self.syllable_segmenter = syllable_segmenter

        # Create ranker - use provided or create DefaultRanker
        if ranker is not None:
            self.ranker = ranker
        else:
            if frequency_denominator <= 0:
                raise ValueError(
                    f"frequency_denominator must be positive, got {frequency_denominator}"
                )
            ranker_config = RankerConfig(
                frequency_denominator=frequency_denominator,
                phonetic_bonus_weight=phonetic_bonus_weight,
                syllable_bonus_weight=effective_syllable_bonus,
                weighted_distance_bonus_weight=(effective_weighted_bonus),
            )
            self.ranker = DefaultRanker(ranker_config=ranker_config)

        # Phonetic bypass thresholds for graded similarity
        self._phonetic_bypass_threshold = phonetic_bypass_threshold
        self._phonetic_extra_distance = phonetic_extra_distance

        # Threshold for "known word" bypass optimization
        # Words with frequency >= this threshold skip phonetic lookup
        self._known_word_frequency_threshold = cfg.known_word_frequency_threshold

        # Myanmar variant generation config
        self._use_myanmar_variants = cfg.use_myanmar_variants
        self._myanmar_variant_max_candidates = cfg.myanmar_variant_max_candidates

        # Delete index: maps delete variations to original terms
        # Key: delete variation, Value: set of (original_term, edit_distance)
        self._deletes: dict[str, set[tuple[str, int]]] = {}

        # Cache for _get_deletes results: term → set of delete variations.
        # SymSpell lookup_compound calls lookup() repeatedly for overlapping
        # substrings, regenerating the same deletes.  This cache eliminates
        # ~70% of the delete-generation work.
        self._DELETES_CACHE_MAX = cfg.deletes_cache_max

        self._COMPOUND_SEG_CACHE_MAX = cfg.compound_seg_cache_max
        self._LOOKUP_CACHE_MAX = cfg.lookup_cache_max

        # Session caches are thread-local to prevent races in check_batch_async
        self._thread_local = threading.local()

        # Compound segmentation limits from config
        self._MAX_COMPOUND_SEG_LEN = cfg.max_compound_seg_len
        self._MAX_SYLLABLES_PER_WORD = cfg.max_syllables_per_word

        # Track which levels have been indexed
        self._indexed_levels: set[str] = set()

        # RLock for thread-safe index access
        # Uses RLock (reentrant) so the same thread can acquire multiple times.
        # Both reads and writes are protected to prevent race conditions between
        # concurrent build_index() and lookup() calls.
        self._index_lock = threading.RLock()

    @property
    def _deletes_cache(self) -> dict:
        """Thread-local deletes cache."""
        tl = self._thread_local
        if not hasattr(tl, "deletes_cache"):
            tl.deletes_cache: dict[str, set[str]] = {}
        return tl.deletes_cache

    @property
    def _lookup_cache(self) -> dict:
        """Thread-local lookup cache."""
        tl = self._thread_local
        if not hasattr(tl, "lookup_cache"):
            tl.lookup_cache: dict[tuple, list[Suggestion]] = {}
        return tl.lookup_cache

    @property
    def _compound_seg_cache(self) -> dict:
        """Thread-local compound segmentation cache."""
        tl = self._thread_local
        if not hasattr(tl, "compound_seg_cache"):
            tl.compound_seg_cache: dict[tuple[str, int], list[tuple[list[str], int, int]]] = {}
        return tl.compound_seg_cache

    def clear_session_cache(self) -> None:
        """Clear per-check session caches.

        Called at the start of each SpellChecker.check() to avoid stale
        results leaking across sentences while keeping the long-lived
        delete index and compound segment cache intact.
        Thread-safe: each thread clears only its own caches.
        """
        self._lookup_cache.clear()
        self._compound_seg_cache.clear()

    def build_index(self, levels: list[str]) -> None:
        """
        Build the delete index for specified levels.

        This must be called before performing lookups. It generates all
        delete variations for dictionary terms within max_edit_distance.

        This method is thread-safe - multiple threads can call it concurrently
        without causing data corruption or duplicate work.

        Args:
            levels: List of levels to index ('syllable', 'word', or both)
                   - 'syllable': Index syllables from provider
                   - 'word': Index words from provider

        Example:
            >>> symspell.build_index(['syllable'])  # Index only syllables
            >>> symspell.build_index(['syllable', 'word'])  # Index both
        """
        for level in levels:
            # Fast path: check without lock first (double-checked locking)
            if level in self._indexed_levels:
                continue  # Already indexed

            # Acquire lock for thread-safe index building
            with self._index_lock:
                # Re-check after acquiring lock (another thread may have indexed)
                if level in self._indexed_levels:
                    continue

                terms_iterator: Iterator[tuple[str, int]] = iter([])
                if level == ValidationLevel.SYLLABLE.value:
                    if hasattr(self.provider, "get_all_syllables"):
                        terms_iterator = self.provider.get_all_syllables()
                elif level == ValidationLevel.WORD.value:
                    if hasattr(self.provider, "get_all_words"):
                        terms_iterator = self.provider.get_all_words()

                skipped = 0
                indexed = 0
                for term, frequency in terms_iterator:
                    if frequency < self.count_threshold:
                        skipped += 1
                        continue
                    self._add_to_index(term)
                    indexed += 1

                if skipped > 0:
                    logger.info(
                        "Index '%s': indexed %d terms, skipped %d below count_threshold=%d",
                        level,
                        indexed,
                        skipped,
                        self.count_threshold,
                    )

                self._indexed_levels.add(level)

    def _add_to_index(self, term: str) -> None:
        """
        Add a term to the delete index.

        Uses prefix filtering optimization: for long terms, only generates
        delete variations for the prefix (up to prefix_length characters).
        This significantly reduces index size and speeds up lookups.

        Args:
            term: The term to add (must be valid dictionary word/syllable)
        """
        # Prefix filtering: For long terms, only index prefix
        # This reduces the number of delete variations significantly
        term_for_deletes = term
        if len(term) > self.prefix_length:
            term_for_deletes = term[: self.prefix_length]

        # Generate all deletes for the (possibly truncated) term
        deletes = self._get_deletes(term_for_deletes, self.max_edit_distance)

        for delete_var in deletes:
            # Calculate edit distance (deletion count from prefix)
            # Since delete_var is derived from term_for_deletes by deletions only:
            dist = len(term_for_deletes) - len(delete_var)

            if delete_var not in self._deletes:
                self._deletes[delete_var] = set()
            self._deletes[delete_var].add((term, dist))

    def lookup(
        self,
        term: str,
        level: str = ValidationLevel.SYLLABLE.value,
        max_suggestions: int = 5,
        include_known: bool = False,
        use_phonetic: bool = False,
    ) -> list[Suggestion]:
        """
        Find spelling suggestions for a term.

        Generates delete variations of the query term and finds matches
        in the indexed dictionary. Results are ranked by edit distance
        (primary) and frequency (secondary).

        Supports phonetic similarity matching to find suggestions based
        on pronunciation similarity.

        Args:
            term: Term to find suggestions for
            level: Dictionary level to search ('syllable' or 'word')
            max_suggestions: Maximum number of suggestions to return
            include_known: Include the term itself if it's in dictionary
            use_phonetic: Use phonetic similarity matching

        Returns:
            List of Suggestion objects, sorted by relevance (best first)

        Example:
            >>> suggestions = symspell.lookup("မြ", level='syllable')
            >>> if suggestions:
            ...     print(f"Did you mean: {suggestions[0].term}?")
            >>> # With phonetic matching
            >>> suggestions = symspell.lookup("မျန်", use_phonetic=True)
            >>> # Finds "မြန်" (phonetically similar)
        """
        if not term:
            return []

        # Session-level cache: same (term, params) may be looked up 2-3x
        # within a single check() call across validators and detectors.
        cache_key = (term, level, max_suggestions, include_known, use_phonetic)
        cached = self._lookup_cache.get(cache_key)
        if cached is not None:
            return cached

        # Check if term is already correct
        is_valid = (
            self.provider.is_valid_syllable(term)
            if level == ValidationLevel.SYLLABLE.value
            else self.provider.is_valid_word(term)
        )

        # Also check frequency threshold to filter out noise/typos in training data
        if is_valid:
            frequency = (
                self.provider.get_syllable_frequency(term)
                if level == ValidationLevel.SYLLABLE.value
                else self.provider.get_word_frequency(term)
            )
            if frequency < self.count_threshold:
                is_valid = False  # Low-frequency terms are likely noise

        if is_valid and not include_known:
            self._lookup_cache[cache_key] = []
            return []  # No suggestions needed for valid terms

        # Generate candidates from multiple sources
        loan_word_terms: set[str] = set()
        candidate_scores = self._collect_candidates(
            term, level, is_valid, include_known, use_phonetic, loan_word_terms
        )

        # Rank, filter, and truncate candidates
        result = self._rank_candidates(
            term,
            level,
            candidate_scores,
            include_known,
            max_suggestions,
            loan_word_terms=loan_word_terms,
        )

        # Store in session cache (bounded, evict oldest entry on overflow)
        if len(self._lookup_cache) >= self._LOOKUP_CACHE_MAX:
            oldest_key = next(iter(self._lookup_cache))
            del self._lookup_cache[oldest_key]
        self._lookup_cache[cache_key] = result

        return result

    def _rank_candidates(
        self,
        term: str,
        level: str,
        candidate_scores: dict[str, float],
        include_known: bool,
        max_suggestions: int,
        *,
        loan_word_terms: set[str] | None = None,
    ) -> list[Suggestion]:
        """
        Evaluate, score, sort, and truncate candidates into ranked suggestions.

        For each candidate this method:
        1. Filters out the input term (unless include_known), low-frequency terms,
           and candidates beyond max_edit_distance (with phonetic bypass).
        2. Computes edit distance, phonetic score, syllable distance, weighted
           distance, nasal variant flags, and the final ranker score.
        3. Sorts by score and truncates to max_suggestions.

        Args:
            term: Original query term.
            level: Dictionary level ('syllable' or 'word').
            candidate_scores: Dict mapping candidate terms to their max phonetic
                scores, as returned by ``_collect_candidates()``.
            include_known: Whether to include the term itself if it appears as
                a candidate.
            max_suggestions: Maximum number of suggestions to return.

        Returns:
            Sorted list of Suggestion objects, truncated to max_suggestions.
        """
        suggestions = []
        for candidate, p_score in candidate_scores.items():
            # Skip if same as input (unless include_known=True)
            if candidate == term and not include_known:
                continue

            # Get frequency
            frequency = (
                self.provider.get_syllable_frequency(candidate)
                if level == ValidationLevel.SYLLABLE.value
                else self.provider.get_word_frequency(candidate)
            )

            # Skip if below threshold
            if frequency < self.count_threshold:
                continue

            # Calculate edit distance
            distance = damerau_levenshtein_distance(term, candidate)

            # Check if TRUE nasal variant (only န် vs ံ substitution)
            # NOTE: We use is_true_nasal_variant() which only considers
            # န် ↔ ံ as true variants (same /n/ phoneme). Other nasals
            # (င်=/ŋ/, မ်=/m/) are different phonemes and NOT variants.
            is_nasal_variant = is_true_nasal_variant(term, candidate)

            # Check if same nasal ending (preserves nasal phoneme)
            # This helps rank suggestions that keep the same nasal consonant higher
            same_nasal = has_same_nasal_ending(term, candidate)

            # Calculate syllable-aware distance for Myanmar text
            syllable_dist: float | None = None
            if self.use_syllable_distance:
                _, syllable_dist = myanmar_syllable_edit_distance(term, candidate)

            # Calculate Myanmar-weighted edit distance (uses MYANMAR_SUBSTITUTION_COSTS)
            # This gives lower costs for phonetically similar substitutions
            weighted_dist: float | None = None
            if self.use_weighted_distance:
                weighted_dist = weighted_damerau_levenshtein_distance(term, candidate)

            # Only include if within max_edit_distance
            # OR if it's a high-confidence phonetic match within extended distance.
            # This allows phonetic variants to bypass the edit-distance cap by a small margin
            # when the graded similarity score is above the threshold.
            phonetic_bypass = (
                p_score >= self._phonetic_bypass_threshold
                and distance <= self.max_edit_distance + self._phonetic_extra_distance
            )
            # Loan word candidates bypass the edit distance gate entirely.
            # These are pre-validated variant→standard mappings from loan_words.yaml
            # that can have edit distances well beyond SymSpell's default max (2),
            # e.g. ဟတ်ဝဲ→ဟာ့ဒ်ဝဲ (hardware, edit_dist=3).
            loan_word_bypass = loan_word_terms is not None and candidate in loan_word_terms
            if distance <= self.max_edit_distance or phonetic_bypass or loan_word_bypass:
                # Compute score using ranker (always available)
                suggestion_data = SuggestionData(
                    term=candidate,
                    edit_distance=distance,
                    frequency=frequency,
                    phonetic_score=p_score,
                    syllable_distance=syllable_dist,
                    weighted_distance=weighted_dist,
                    is_nasal_variant=is_nasal_variant,
                    has_same_nasal_ending=same_nasal,
                )
                score = self.ranker.score(suggestion_data)

                suggestions.append(
                    Suggestion(
                        term=candidate,
                        edit_distance=distance,
                        frequency=frequency,
                        phonetic_score=p_score,
                        syllable_distance=syllable_dist,
                        weighted_distance=weighted_dist,
                        is_nasal_variant=is_nasal_variant,
                        has_same_nasal_ending=same_nasal,
                        score=score,
                    )
                )

        # Sort by score (edit distance primary, frequency secondary)
        suggestions.sort()

        return suggestions[:max_suggestions]

    def _get_deletes(self, term: str, max_distance: int) -> set[str]:
        """
        Generate all delete variations of a term within max_distance.

        This is the core of the SymSpell algorithm - generating all possible
        strings that can be created by deleting characters.

        Args:
            term: Input term
            max_distance: Maximum number of characters to delete

        Returns:
            Set of all delete variations (including original term)

        Time Complexity:
            O(L^d) where L=len(term), d=max_distance.
            For d=2 and L=10: ~55 deletes. For d=3 and L=10: ~220 deletes.
            Bounded by max_deletes_per_term to prevent exponential blowup.

        Space Complexity:
            O(L^d) for the result set, bounded by max_deletes_per_term.

        Note:
            Delete generation is limited by `max_deletes_per_term` to prevent
            O(n^d) memory exhaustion. If the limit is reached, the method
            returns early with a partial set and logs a warning.

        Memory Optimization:
            - Early exit when estimated deletes would exceed limit
            - Limit checks happen BEFORE allocating new level sets
            - Uses set intersection to avoid redundant membership checks
        """
        # Check cache (max_distance is constant per instance, so term alone is key)
        cached = self._deletes_cache.get(term)
        if cached is not None:
            return cached

        deletes = {term}

        if max_distance <= 0:
            self._deletes_cache[term] = deletes
            return deletes

        # Resource limit from instance configuration
        limit = self._max_deletes_per_term

        # Early exit: estimate if we'll exceed limit based on term length
        # For a term of length n with max_distance d, max deletes ≈ C(n,1) + C(n,2) + ... + C(n,d)
        # This is a rough upper bound; actual count is lower due to duplicates
        term_len = len(term)
        if term_len > 0:
            # Estimate: for distance 1, max n deletes; for distance 2, max n*(n-1)/2, etc.
            # If even the first level would exceed limit, warn early
            if term_len > limit:
                logger.warning(
                    "Term '%s' (length=%d) exceeds max_deletes_per_term=%d. "
                    "Limiting to first %d deletes.",
                    term[:20] + "..." if term_len > 20 else term,
                    term_len,
                    limit,
                    limit,
                )

        limit_exceeded = False

        # Use BFS to generate deletes level by level
        current_level = {term}

        for _distance in range(1, max_distance + 1):
            # Check limit BEFORE allocating next level
            # This prevents memory pressure from unused set allocation
            if len(deletes) >= limit:
                limit_exceeded = True
                logger.warning(
                    "Delete generation limit reached for term '%s' "
                    "(max_deletes_per_term=%d, generated=%d). "
                    "Some delete variations may be missing.",
                    term[:20] + "..." if len(term) > 20 else term,
                    limit,
                    len(deletes),
                )
                break

            # Calculate remaining capacity before allocating
            remaining_capacity = limit - len(deletes)

            # Estimate deletes for this level: each item generates len(item) deletes
            # Use min to avoid over-counting
            estimated_new = sum(len(item) for item in current_level)
            if estimated_new > remaining_capacity * 2:
                # Likely to exceed limit - process items one by one
                next_level: set[str] = set()
                for item in current_level:
                    if len(deletes) >= limit:
                        limit_exceeded = True
                        break
                    for i in range(len(item)):
                        delete = item[:i] + item[i + 1 :]
                        if delete and delete not in deletes:
                            deletes.add(delete)
                            next_level.add(delete)
                            if len(deletes) >= limit:
                                limit_exceeded = True
                                break
                    if limit_exceeded:
                        break
            else:
                # Fast path: generate all deletes for this level
                # Add limit checks to match slow path
                next_level = set()
                for item in current_level:
                    if len(deletes) >= limit:
                        limit_exceeded = True
                        break
                    for i in range(len(item)):
                        delete = item[:i] + item[i + 1 :]
                        if delete and delete not in deletes:
                            deletes.add(delete)
                            next_level.add(delete)
                            if len(deletes) >= limit:
                                limit_exceeded = True
                                break
                    if limit_exceeded:
                        break

            if limit_exceeded:
                logger.warning(
                    "Delete generation limit reached for term '%s' "
                    "(max_deletes_per_term=%d, generated=%d). "
                    "Some delete variations may be missing.",
                    term[:20] + "..." if len(term) > 20 else term,
                    limit,
                    len(deletes),
                )
                break

            current_level = next_level

        # Bound cache size (evict oldest entry on overflow)
        if len(self._deletes_cache) >= self._DELETES_CACHE_MAX:
            oldest_key = next(iter(self._deletes_cache))
            del self._deletes_cache[oldest_key]
        self._deletes_cache[term] = deletes
        return deletes

    def _find_similar_terms(self, delete_var: str, level: str) -> set[str]:
        """
        Find terms in dictionary that could match this delete variation.

        Uses the pre-built delete index for O(1) lookup.
        Also expands nasal variants (န်/မ်/င်/ံ) for better Myanmar matching.

        This method is thread-safe - it acquires _index_lock for reading _deletes
        to prevent race conditions with concurrent build_index() calls.

        Args:
            delete_var: Delete variation to match
            level: Dictionary level ('syllable' or 'word')

        Returns:
            Set of similar terms from dictionary
        """
        similar: set[str] = set()

        # 1. Lookup in Deletes Index (Primary SymSpell logic)
        # Acquire lock for thread-safe read access
        with self._index_lock:
            if delete_var in self._deletes:
                for original_term, _ in self._deletes[delete_var]:
                    similar.add(original_term)
                    # Add nasal variants for better matching —
                    # only expand when the term actually contains nasal endings
                    if _has_nasal_ending(original_term):
                        nasal_vars = get_nasal_variants(original_term)
                        for variant in nasal_vars:
                            if variant == original_term:
                                continue  # skip self-variant (already added)
                            if level == ValidationLevel.SYLLABLE.value:
                                if self.provider.is_valid_syllable(variant):
                                    similar.add(variant)
                            else:
                                if self.provider.is_valid_word(variant):
                                    similar.add(variant)

            # 2. Fallback for unindexed terms (e.g. if build_index wasn't called)
            # This handles simple deletion errors where the delete_var IS the word
            # e.g. input "appleX" -> delete "apple" -> "apple" is valid word
            indexed_levels_empty = not self._indexed_levels

        if indexed_levels_empty:
            if level == ValidationLevel.SYLLABLE.value:
                if self.provider.is_valid_syllable(delete_var):
                    similar.add(delete_var)
            else:
                if self.provider.is_valid_word(delete_var):
                    similar.add(delete_var)

        return similar

    def _collect_candidates(
        self,
        term: str,
        level: str,
        is_valid: bool,
        include_known: bool,
        use_phonetic: bool,
        loan_word_terms: set[str] | None = None,
    ) -> dict[str, float]:
        """
        Collect candidates from all sources.

        Combines candidates from:
        1. The term itself (if valid and include_known=True)
        2. Delete variations (core SymSpell algorithm)
        3. Nasal variants (Myanmar-specific)
        4. Myanmar-specific variant candidates (if enabled and term is OOV)
        5. Phonetic variants (if use_phonetic=True)
        6. Loan word transliteration variants (word level only)

        Args:
            term: Input term to find candidates for
            level: Dictionary level ('syllable' or 'word')
            is_valid: Whether the term itself is valid
            include_known: Include the term itself if valid
            use_phonetic: Use phonetic similarity matching
            loan_word_terms: Output set to track loan word candidates for
                edit distance filter bypass in ``_rank_candidates()``.

        Returns:
            Dict mapping candidate terms to their max phonetic scores
        """
        candidate_scores: dict[str, float] = {}

        # 1. Add term itself
        if is_valid and include_known:
            candidate_scores[term] = 0.0

        # 2. Generate delete candidates (core SymSpell)
        self._add_delete_candidates(term, level, candidate_scores)

        # 3. Add nasal variant candidates (Myanmar-specific)
        self._add_nasal_candidates(term, level, candidate_scores)

        # 4. Add Myanmar-specific variant candidates for OOV terms
        # Only when the term is not a valid word (OOV) and delete-based
        # candidates are insufficient. Covers medial swaps, aspiration swaps,
        # etc. that SymSpell's delete approach cannot find.
        if self._use_myanmar_variants and not is_valid:
            self._add_myanmar_variant_candidates(
                term, level, candidate_scores, self._myanmar_variant_max_candidates
            )

        # 5. Add phonetic candidates
        if use_phonetic:
            self._add_phonetic_candidates(term, level, candidate_scores)

        # 6. Add loan word transliteration variants (word level only)
        if level == ValidationLevel.WORD.value:
            self._add_loan_word_candidates(term, level, candidate_scores, loan_word_terms)

        return candidate_scores

    def _add_delete_candidates(
        self, term: str, level: str, candidate_scores: dict[str, float]
    ) -> None:
        """
        Add candidates from delete variations.

        Uses prefix filtering for consistency with indexing.

        Args:
            term: Input term
            level: Dictionary level
            candidate_scores: Dict to add candidates to (modified in place)
        """
        # Use prefix filtering for consistency with indexing
        term_for_deletes = term
        if len(term) > self.prefix_length:
            term_for_deletes = term[: self.prefix_length]

        deletes = self._get_deletes(term_for_deletes, self.max_edit_distance)

        # Add deletes candidates (phonetic_score=0.0)
        # Local memo avoids redundant _find_similar_terms calls for the same delete variant
        similar_cache: dict[str, set[str]] = {}
        for delete_var in deletes:
            if delete_var not in similar_cache:
                similar_cache[delete_var] = self._find_similar_terms(delete_var, level)
            for sim in similar_cache[delete_var]:
                candidate_scores.setdefault(sim, 0.0)

    def _add_nasal_candidates(
        self, term: str, level: str, candidate_scores: dict[str, float]
    ) -> None:
        """
        Add nasal variant candidates for Myanmar text.

        This helps find suggestions when only nasal ending differs (e.g., န်/မ်/င်/ံ).
        NOTE: We do NOT give these a phonetic bonus because different nasals
        (/n/, /ŋ/, /m/) are distinct phonemes in Myanmar.

        Args:
            term: Input term
            level: Dictionary level
            candidate_scores: Dict to add candidates to (modified in place)
        """
        # Only expand nasal variants when the term actually contains nasal endings
        if not _has_nasal_ending(term):
            return
        nasal_variants = get_nasal_variants(term)
        for variant in nasal_variants:
            if variant == term:
                continue  # skip self-variant
            is_valid_variant = (
                self.provider.is_valid_syllable(variant)
                if level == ValidationLevel.SYLLABLE.value
                else self.provider.is_valid_word(variant)
            )
            if is_valid_variant:
                # Add as candidate without phonetic bonus (score=0.0)
                candidate_scores[variant] = max(candidate_scores.get(variant, 0.0), 0.0)

    def _add_myanmar_variant_candidates(
        self,
        term: str,
        level: str,
        candidate_scores: dict[str, float],
        max_variants: int = 20,
    ) -> None:
        """
        Add Myanmar-specific variant candidates for OOV terms.

        Generates variants via medial swaps, aspiration swaps, nasal confusion,
        stop-coda mergers, tone mark and vowel length swaps.  Each valid variant
        is added with edit_distance=1 (conceptually a single Myanmar-specific
        operation).

        This complements the standard delete-based lookup and phonetic candidate
        generation by covering Myanmar confusions that SymSpell's character-deletion
        approach cannot find (e.g., medial ျ↔ြ swaps, anusvara ံ insertions).

        Args:
            term: Input term (OOV -- not valid in the dictionary).
            level: Dictionary level ('syllable' or 'word').
            candidate_scores: Dict to add candidates to (modified in place).
            max_variants: Maximum number of variant candidates to add.
        """
        # Lazy import to avoid circular dependency:
        # myanmar_confusables → phonetic → algorithms → symspell → myanmar_confusables
        from myspellchecker.core.myanmar_confusables import generate_myanmar_variants

        variants = generate_myanmar_variants(term)

        added = 0
        for variant in variants:
            if added >= max_variants:
                break

            # Skip variants already in the candidate set
            if variant in candidate_scores:
                continue

            # Check dictionary validity
            is_valid = (
                self.provider.is_valid_syllable(variant)
                if level == ValidationLevel.SYLLABLE.value
                else self.provider.is_valid_word(variant)
            )
            if not is_valid:
                continue

            # Check frequency threshold
            frequency = (
                self.provider.get_syllable_frequency(variant)
                if level == ValidationLevel.SYLLABLE.value
                else self.provider.get_word_frequency(variant)
            )
            if frequency < self.count_threshold:
                continue

            candidate_scores[variant] = 0.0
            added += 1

    def _add_phonetic_candidates(
        self, term: str, level: str, candidate_scores: dict[str, float]
    ) -> None:
        """
        Add phonetic candidates.

        Uses phonetic hasher to find phonetically similar terms.
        Skips lookup for high-frequency known words to save computation.

        Args:
            term: Input term
            level: Dictionary level
            candidate_scores: Dict to add candidates to (modified in place)
        """
        if not self.phonetic_hasher:
            return

        # Early exit for known high-frequency words
        # No need to search phonetic variants if the term itself is a common word
        term_frequency = (
            self.provider.get_syllable_frequency(term)
            if level == ValidationLevel.SYLLABLE.value
            else self.provider.get_word_frequency(term)
        )
        if term_frequency >= self._known_word_frequency_threshold:
            # Term is a known high-frequency word, skip phonetic expansion
            return

        phonetic_variants = self.phonetic_hasher.get_phonetic_variants(term)
        for variant in phonetic_variants:
            is_valid_variant = (
                self.provider.is_valid_syllable(variant)
                if level == ValidationLevel.SYLLABLE.value
                else self.provider.is_valid_word(variant)
            )
            if is_valid_variant:
                # Use graded phonetic similarity instead of flat 1.0 score
                similarity = self.phonetic_hasher.compute_phonetic_similarity(term, variant)
                candidate_scores[variant] = max(candidate_scores.get(variant, 0.0), similarity)

    def _add_loan_word_candidates(
        self,
        term: str,
        level: str,
        candidate_scores: dict[str, float],
        loan_word_terms: set[str] | None = None,
    ) -> None:
        """Add loan word transliteration variants as candidates.

        If the term is a known non-standard loan word variant, adds the
        standard form(s).  If the term is a standard loan word, adds its
        known variants.  This covers transliteration differences that are
        too large for SymSpell's delete-based approach.

        Candidates added here are tracked in *loan_word_terms* so that
        ``_rank_candidates()`` can bypass the ``max_edit_distance`` filter
        for them — loan word variant→standard mappings are pre-validated
        and may have edit distances well beyond SymSpell's default max.

        Args:
            term: Input term
            level: Dictionary level (only called for 'word')
            candidate_scores: Dict to add candidates to (modified in place)
            loan_word_terms: Optional set to track loan word candidates.
        """
        from myspellchecker.core.loan_word_variants import (
            get_loan_word_standard,
            get_loan_word_variants,
        )

        # variant -> standard (most common case: user typed non-standard form)
        standards = get_loan_word_standard(term)
        for std in standards:
            if self.provider.is_valid_word(std):
                candidate_scores.setdefault(std, 0.0)
                if loan_word_terms is not None:
                    loan_word_terms.add(std)

        # standard -> variants (less common: standard form typed, variants exist)
        variants = get_loan_word_variants(term)
        for variant in variants:
            if self.provider.is_valid_word(variant):
                candidate_scores.setdefault(variant, 0.0)
                if loan_word_terms is not None:
                    loan_word_terms.add(variant)

        # Exact-match correction table (Tier 1 unconditional corrections).
        # These are high-confidence corrections where the incorrect form is
        # never a valid Myanmar word. Inject with score 1.0 so they rank
        # above generic edit-distance candidates.
        try:
            from myspellchecker.grammar.config import get_grammar_config

            correction = get_grammar_config().get_loan_word_correction(term)
            if correction:
                correct = correction["correct"]
                candidate_scores[correct] = max(candidate_scores.get(correct, 0.0), 1.0)
                if loan_word_terms is not None:
                    loan_word_terms.add(correct)
        except Exception:
            pass  # Grammar config not available — skip silently

    def lookup_compound(
        self, text: str, max_suggestions: int = 5, max_edit_distance: int = 2
    ) -> list[tuple[list[str], int, int]]:
        """
        Find suggestions for compound terms (multi-word corrections).

        This method attempts to find corrections for phrases by
        considering word splits and joins. It's particularly useful for
        Myanmar text where word boundaries can be ambiguous.

        The algorithm:
        1. Try splitting compound into individual words
        2. Check each part for validity
        3. Suggest corrections for invalid parts
        4. Try joining adjacent words if they form valid compounds

        Args:
            text: Input text (may contain multiple words or compound)
            max_suggestions: Maximum number of suggestions
            max_edit_distance: Maximum edit distance per word

        Returns:
            List of tuples: (word_list, total_distance, total_frequency)
            where word_list is the suggested word segmentation

        Example:
            >>> symspell.lookup_compound("မြန်မာနိုင်ငံ")
            [(['မြန်မာ', 'နိုင်ငံ'], 0, 15000), ...]

        Note:
            Uses dynamic programming to find optimal segmentation.
        """
        if not text or not text.strip():
            return []

        # Remove leading/trailing whitespace
        text = text.strip()

        # Split by existing spaces first
        initial_words = text.split()

        suggestions: list[tuple[list[str], int, int]] = []

        # Process each segment (space-separated part)
        for segment in initial_words:
            # Try to find best segmentation for this segment
            segment_suggestions = self._segment_compound(segment, max_edit_distance)
            suggestions.extend(segment_suggestions)

        # Sort by total edit distance (ascending) and frequency (descending)
        suggestions.sort(key=lambda x: (x[1], -x[2]))

        return suggestions[:max_suggestions]

    # Class-level defaults (overridden per-instance from SymSpellConfig in __init__)
    _MAX_COMPOUND_SEG_LEN: int = 24
    _MAX_SYLLABLES_PER_WORD: int = 5

    def _segment_compound(
        self, text: str, max_edit_distance: int
    ) -> list[tuple[list[str], int, int]]:
        """
        Find best word segmentation for a compound term.

        Uses syllable-aligned dynamic programming: Myanmar words are always
        sequences of whole syllables, so only syllable-boundary-aligned
        substrings are considered.  This reduces O(n²) character-level
        substrings to O(s²) where s = number of syllables (typically 90%
        fewer iterations for compound words).

        Falls back to character-level DP when no syllable segmenter is
        available.

        Args:
            text: Compound text to segment
            max_edit_distance: Maximum edit distance per word

        Returns:
            List of (word_list, total_distance, total_frequency) tuples

        Time Complexity:
            O(S² * K) where S=syllable count, K=beam_width.
            Much faster than the old O(N * M * K) character-level approach.
        """
        n = len(text)
        if n == 0:
            return []

        # Fast path: return cached result (same compound often segmented 2x)
        cache_key = (text, max_edit_distance)
        cached = self._compound_seg_cache.get(cache_key)
        if cached is not None:
            return cached

        # Guard: skip DP for very long strings
        if n > self._MAX_COMPOUND_SEG_LEN:
            result = [([text], n, 0)]
            self._compound_seg_cache[cache_key] = result
            return result

        # Build syllable boundary positions for aligned DP.
        # Myanmar words are sequences of whole syllables, so we only
        # need to consider substrings that start/end at boundaries.
        if self.syllable_segmenter is not None:
            syllables = self.syllable_segmenter.segment_syllables(text)
            syl_positions = [0]
            pos = 0
            for syl in syllables:
                pos += len(syl)
                syl_positions.append(pos)
            # Sanity: positions must reach end of text
            if syl_positions[-1] != n:
                syl_positions.append(n)
            # Limit by syllable count (most words are 1-5 syllables)
            max_span = self._MAX_SYLLABLES_PER_WORD
        else:
            # Fallback: every character position is a boundary.
            # Use max_word_length to limit span (not syllable count).
            syl_positions = list(range(n + 1))
            max_span = self.max_word_length

        num_positions = len(syl_positions)

        # Pre-compute validity, frequency, and lookup for each unique
        # syllable-aligned substring (dedup across overlapping spans).
        substr_valid: dict[str, bool] = {}
        substr_freq: dict[str, int] = {}
        substr_syl_valid: dict[str, bool] = {}
        substr_lookup: dict[str, list[Suggestion]] = {}

        for i_idx in range(1, num_positions):
            i_pos = syl_positions[i_idx]
            start_idx = max(0, i_idx - max_span)
            for j_idx in range(start_idx, i_idx):
                j_pos = syl_positions[j_idx]
                if i_pos - j_pos > self.max_word_length:
                    continue
                word = text[j_pos:i_pos]
                if word in substr_valid:
                    continue
                is_valid = self.provider.is_valid_word(word)
                freq = self.provider.get_word_frequency(word)
                substr_valid[word] = is_valid
                substr_freq[word] = freq

                syl_ok = True
                if is_valid and self.syllable_segmenter is not None:
                    word_syls = self.syllable_segmenter.segment_syllables(word)
                    for ws in word_syls:
                        if not self.provider.is_valid_syllable(ws):
                            syl_ok = False
                            break
                substr_syl_valid[word] = syl_ok

                if not (is_valid and freq >= self.count_threshold and syl_ok):
                    substr_lookup[word] = self.lookup(
                        word,
                        level=ValidationLevel.WORD.value,
                        max_suggestions=self.compound_lookup_count,
                        include_known=False,
                    )

        # DP over syllable boundary positions.
        # dp_by_pos[char_position] = list of (words, total_distance, total_frequency)
        dp_by_pos: dict[int, list[tuple[list[str], int, int]]] = {syl_positions[0]: [([], 0, 0)]}

        for i_idx in range(1, num_positions):
            i_pos = syl_positions[i_idx]
            candidates: list[tuple[list[str], int, int]] = []

            start_idx = max(0, i_idx - max_span)
            for j_idx in range(start_idx, i_idx):
                j_pos = syl_positions[j_idx]
                if i_pos - j_pos > self.max_word_length:
                    continue
                if j_pos not in dp_by_pos or not dp_by_pos[j_pos]:
                    continue

                word = text[j_pos:i_pos]
                is_valid = substr_valid[word]
                frequency = substr_freq[word]
                syllables_valid = substr_syl_valid.get(word, True)

                if is_valid and frequency >= self.count_threshold and syllables_valid:
                    for prev_words, prev_dist, prev_freq in dp_by_pos[j_pos]:
                        candidates.append((prev_words + [word], prev_dist, prev_freq + frequency))
                else:
                    suggestions = substr_lookup.get(word, [])
                    if suggestions:
                        top = suggestions[0]
                        for prev_words, prev_dist, prev_freq in dp_by_pos[j_pos]:
                            new_words = prev_words + [top.term]
                            new_dist = prev_dist + top.edit_distance
                            new_freq = prev_freq + top.frequency
                            max_total_dist = max_edit_distance + (len(new_words) - 1) * math.ceil(
                                max_edit_distance / 2
                            )
                            if new_dist <= max_total_dist:
                                candidates.append((new_words, new_dist, new_freq))

            if len(candidates) > self.beam_width:
                candidates.sort(key=lambda x: (x[1], -x[2]))
                candidates = candidates[: self.beam_width]
            if candidates:
                dp_by_pos[i_pos] = candidates

        # Return best segmentations for full text
        final = dp_by_pos.get(n, [])
        if final:
            final.sort(key=lambda x: (x[1], -x[2]))
            result = final[: self.compound_max_suggestions]
        else:
            # No valid segmentation found - return original as single word
            result = [([text], 0, 0)]

        # Cache result (bounded)
        if len(self._compound_seg_cache) >= self._COMPOUND_SEG_CACHE_MAX:
            self._compound_seg_cache.clear()
        self._compound_seg_cache[cache_key] = result
        return result
