"""
N-gram Context Checker for Myanmar spell checking.

This module implements context-aware spell checking using N-gram probabilities
to detect and suggest corrections for contextually inappropriate words.

The N-gram Context Checker operates at Layer 3 of the spell checking pipeline,
analyzing word sequences to identify words that are individually valid but
contextually unlikely.

Algorithm Overview:
    1. **Detection**: Use bigram probabilities P(w2|w1) to detect unlikely sequences
    2. **Suggestion**: Find alternative words with higher conditional probability
    3. **Ranking**: Sort suggestions by combined probability and edit distance

Design Pattern:
    - **Provider Pattern**: Uses DictionaryProvider for N-gram data access
    - **Strategy Pattern**: Uses SmoothingStrategy enum for configurable smoothing
    - **Hybrid Integration**: Optionally integrates with SymSpell for candidate generation

Result Types:
    - ContextSuggestion: Dataclass containing term, probability, score (higher = better)
    - list[ContextSuggestion]: Sorted suggestions, best first (highest score)
    - is_contextual_error returns: bool indicating error detection

Parameter Validation:
    - edit_distance_weight, probability_weight: Validated to be in [0.0, 1.0] range
    - pos_score_weight, backoff_weight: Validated to be in [0.0, 1.0] range
    - Weight sum validation: Warns if scoring weights don't sum to 1.0
    - All validation occurs at initialization time

Smoothing Strategies:
    - **NONE**: No smoothing, returns raw probabilities (use for pre-smoothed data)
    - **STUPID_BACKOFF**: Default. Simple backoff with configurable weight.
      P(unseen_bigram) = backoff_weight * P(unigram)
    - **ADD_K**: Add-k (Laplace) smoothing. Adds constant k to all counts.

Example:
    Given the sequence "သူ ကျောင်း", if P(ကျောင်း|သူ) is very low, the checker
    might suggest "သူ သွား" or "သူ ရှိ" based on higher bigram probabilities.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from myspellchecker.algorithms._ngram_candidates import (
    apply_pos_context_scoring as _apply_pos_context_scoring,
)
from myspellchecker.algorithms._ngram_candidates import (
    calculate_combined_score as _calculate_combined_score,
)
from myspellchecker.algorithms._ngram_candidates import (
    calculate_pos_context_score as _calculate_pos_context_score,
)
from myspellchecker.algorithms._ngram_candidates import (
    combine_context_probabilities as _combine_context_probabilities,
)
from myspellchecker.algorithms._ngram_candidates import (
    generate_candidates as _generate_candidates,
)
from myspellchecker.algorithms._ngram_candidates import (
    get_phonetic_variants as _get_phonetic_variants,
)
from myspellchecker.algorithms._ngram_candidates import (
    get_pos_tags as _get_pos_tags,
)
from myspellchecker.algorithms._ngram_candidates import (
    max_pos_bigram_prob as _max_pos_bigram_prob,
)
from myspellchecker.algorithms._ngram_candidates import (
    score_candidate as _score_candidate,
)
from myspellchecker.algorithms._ngram_scoring import (
    compute_bidirectional_prob as _compute_bidirectional_prob,
)
from myspellchecker.algorithms._ngram_scoring import (
    has_ngram_context as _has_ngram_context,
)
from myspellchecker.algorithms._ngram_scoring import (
    has_ngram_context_directional as _has_ngram_context_directional,
)
from myspellchecker.algorithms._ngram_smoothing import (
    SmoothingStrategy,
)
from myspellchecker.algorithms._ngram_smoothing import (
    get_best_left_probability as _get_best_left_probability,
)
from myspellchecker.algorithms._ngram_smoothing import (
    get_best_right_probability as _get_best_right_probability,
)
from myspellchecker.algorithms._ngram_smoothing import (
    get_smoothed_bigram_probability as _get_smoothed_bigram_probability,
)
from myspellchecker.algorithms._ngram_smoothing import (
    get_smoothed_fivegram_probability as _get_smoothed_fivegram_probability,
)
from myspellchecker.algorithms._ngram_smoothing import (
    get_smoothed_fourgram_probability as _get_smoothed_fourgram_probability,
)
from myspellchecker.algorithms._ngram_smoothing import (
    get_smoothed_trigram_probability as _get_smoothed_trigram_probability,
)
from myspellchecker.core.config import NgramContextConfig
from myspellchecker.core.constants import (
    CONFUSABLE_EXEMPT_PAIRS,
    CONFUSABLE_EXEMPT_SUFFIX_PAIRS,
    ValidationLevel,
)
from myspellchecker.providers import DictionaryProvider
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.symspell import SymSpell


def _is_confusable_exempt(word: str, alt: str, provider: DictionaryProvider | None = None) -> bool:
    """Check if word<->alt pair is exempt from confusable detection.

    Checks the DB confusable_pairs table suppress flag (preferred path),
    falling back to hardcoded CONFUSABLE_EXEMPT_PAIRS when DB is unavailable.

    Also handles compound prefix matching: if stems (word[:N], alt[:N])
    form a suppressed pair and the remainders are identical, the compound
    pair is exempt (e.g. ဖေးမှာ<->ဘေးမှာ -> stems ဖေး<->ဘေး are exempt).
    """
    # DB-driven suppression (preferred path)
    if provider is not None and hasattr(provider, "is_confusable_suppressed"):
        if provider.is_confusable_suppressed(word, alt) is True:
            return True
        if provider.is_confusable_suppressed(alt, word) is True:
            return True

    # Fallback to hardcoded constants (for DBs without enrichment tables)
    if (word, alt) in CONFUSABLE_EXEMPT_PAIRS:
        return True
    for exempt_w, exempt_a in CONFUSABLE_EXEMPT_PAIRS:
        if (
            word.startswith(exempt_w)
            and alt.startswith(exempt_a)
            and word[len(exempt_w) :] == alt[len(exempt_a) :]
        ):
            return True
        if (
            word.startswith(exempt_a)
            and alt.startswith(exempt_w)
            and word[len(exempt_a) :] == alt[len(exempt_w) :]
        ):
            return True
    return False


# Re-export SmoothingStrategy for backward compatibility
__all__ = [
    "SmoothingStrategy",
    "ContextSuggestion",
    "NgramVerdict",
    "NgramContextChecker",
]


@dataclass
class ContextSuggestion:
    """
    A context-aware spelling correction suggestion.

    Attributes:
        term: The suggested replacement word
        probability: Bigram probability P(term|prev_word)
        edit_distance: Damerau-Levenshtein distance from original
        score: Combined score for ranking (higher is better)
        confidence: Confidence in this suggestion [0.0, 1.0]
    """

    term: str
    probability: float
    edit_distance: int
    score: float
    confidence: float

    def __lt__(self, other: "ContextSuggestion") -> bool:
        """Enable sorting suggestions by score (descending)."""
        return self.score > other.score  # Higher score is better

    def __eq__(self, other: object) -> bool:
        """Check equality based on term."""
        if not isinstance(other, ContextSuggestion):
            return NotImplemented
        return self.term == other.term

    def __hash__(self) -> int:
        """Enable using ContextSuggestion in sets."""
        return hash(self.term)


@dataclass
class NgramVerdict:
    """
    Result of a unified context check combining absolute threshold and comparison.

    Carries enough information for validation strategies to create Error objects
    without re-querying the n-gram checker.

    Attributes:
        is_error: Whether the word is flagged as a contextual error.
        best_alternative: The best replacement word found, if any.
        confidence: Confidence in the detection (0.0-1.0).
        error_type: Type of error detected -- ``"context_probability"`` for
            absolute threshold violations, ``"confusable_error"`` when a
            candidate is substantially more probable in context.
        ratio: Probability ratio between the best alternative and the word.
            Zero when no comparison was performed.
        probability: Combined bidirectional probability of the word in context.
        suggestions: Suggested replacement words, best first.
    """

    is_error: bool
    best_alternative: str | None = None
    confidence: float = 0.0
    error_type: str = "context_probability"
    ratio: float = 0.0
    probability: float = 0.0
    suggestions: list[str] = field(default_factory=list)


class NgramContextChecker:
    """
    N-gram based context-aware spell checker.

    This checker uses bigram probabilities to identify and correct contextually
    inappropriate words. It implements a Naive Bayesian Classifier approach
    to find replacements with higher conditional probability given the context.

    Attributes:
        provider: DictionaryProvider for bigram probability access
        threshold: Minimum probability threshold for flagging errors (default: 0.01)
        max_suggestions: Maximum number of suggestions to generate (default: 5)
        edit_distance_weight: Weight for edit distance in scoring (default: 0.6)
        probability_weight: Weight for probability in scoring (default: 0.4)

    Example:
        >>> from myspellchecker.providers import SQLiteProvider
        >>> from myspellchecker.algorithms import NgramContextChecker
        >>>
        >>> provider = SQLiteProvider("myspell.db")
        >>> checker = NgramContextChecker(provider)
        >>>
        >>> # Check if word is contextually appropriate
        >>> prob = provider.get_bigram_probability("သူ", "ကျောင်း")
        >>> if prob < checker.threshold:
        ...     suggestions = checker.suggest("သူ", "ကျောင်း")
        ...     print(f"Consider: {[s.term for s in suggestions]}")
    """

    # --- Class-level constants (extracted from inline magic numbers) ---

    # Confusable ratio: base ratio required for low-frequency words (freq < 5000)
    _CONFUSABLE_BASE_RATIO: float = 8.0

    # Confusable ratio: stricter ratio required for high-frequency words (freq >= 5000)
    _CONFUSABLE_HIGH_FREQ_RATIO: float = 50.0

    # Confusable ratio: frequency threshold above which the high-freq ratio applies
    _CONFUSABLE_HIGH_FREQ_THRESHOLD: int = 5000

    # Overlap-based confusable ratio interpolation:
    # overlap 0.0 -> _OVERLAP_RATIO_BASE, overlap 1.0 -> BASE + RANGE
    _OVERLAP_RATIO_BASE: float = 5.0
    _OVERLAP_RATIO_RANGE: float = 45.0

    # Aspirated confusable minimum ratio (floor regardless of base_ratio)
    _ASPIRATED_MIN_RATIO: float = 20.0

    def __init__(
        self,
        provider: DictionaryProvider,
        config: NgramContextConfig | None = None,
        symspell: "SymSpell" | None = None,
        pos_unigram_probs: dict[str, float] | None = None,
        pos_bigram_probs: dict[tuple[str, str], float] | None = None,
    ):
        """
        Initialize the N-gram context checker.

        Args:
            provider: DictionaryProvider for n-gram probabilities.
            config: NgramContextConfig with all thresholds and weights.
                Defaults to NgramContextConfig() if not provided.
            symspell: Optional SymSpell instance for hybrid candidate generation.
            pos_unigram_probs: POS unigram probabilities for POS-aware scoring.
            pos_bigram_probs: POS bigram probabilities for POS-aware scoring.

        Note:
            edit_distance_weight + probability_weight should equal 1.0 for
            normalized scoring, but this is not enforced.
        """
        cfg = config or NgramContextConfig()

        # Validate weight parameters (0.0-1.0 range)
        for name, value in [
            ("edit_distance_weight", cfg.edit_distance_weight),
            ("probability_weight", cfg.probability_weight),
            ("pos_score_weight", cfg.pos_score_weight),
            ("backoff_weight", cfg.backoff_weight),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")

        # Validate that scoring weights sum to 1.0
        scoring_weight_sum = cfg.edit_distance_weight + cfg.probability_weight
        if abs(scoring_weight_sum - 1.0) > 1e-6:
            warnings.warn(
                f"edit_distance_weight + probability_weight "
                f"= {scoring_weight_sum:.4f} "
                f"(expected 1.0). Scoring may not be normalized.",
                UserWarning,
                stacklevel=2,
            )

        self.logger = get_logger(__name__)
        self.provider = provider
        self.threshold = cfg.bigram_threshold
        self.trigram_threshold = cfg.trigram_threshold
        self.fourgram_threshold = cfg.fourgram_threshold
        self.fivegram_threshold = cfg.fivegram_threshold
        self.right_context_threshold = cfg.right_context_threshold
        self.max_suggestions = cfg.max_suggestions
        self._CASE1_FREQ_GUARD = cfg.case1_freq_guard
        self.edit_distance_weight = cfg.edit_distance_weight
        self.probability_weight = cfg.probability_weight
        self.symspell = symspell
        self.candidate_limit = cfg.candidate_limit
        self.min_prob_denom = cfg.min_prob_denominator
        self.heuristic_multiplier = cfg.heuristic_multiplier
        self.min_unigram_threshold = cfg.min_unigram_threshold
        self.pos_unigram_probs = pos_unigram_probs if pos_unigram_probs is not None else {}
        self.pos_bigram_probs = pos_bigram_probs if pos_bigram_probs is not None else {}
        self.pos_score_weight = cfg.pos_score_weight
        self.pos_distance_reduction_factor_multiplier = cfg.pos_distance_reduction_factor
        self.score_scaling_factor = cfg.score_scaling_factor
        self.use_smoothing = cfg.use_smoothing
        self.smoothing_strategy = SmoothingStrategy(cfg.smoothing_strategy)
        self.backoff_weight = cfg.backoff_weight
        self.add_k_smoothing = cfg.add_k_smoothing
        # Try to load total_word_count from DB metadata for accurate unigram probabilities
        dynamic_denominator = None
        if hasattr(provider, "get_metadata"):
            try:
                total_words_str = provider.get_metadata("total_word_count")
                if total_words_str and isinstance(total_words_str, str):
                    dynamic_denominator = float(total_words_str)
            except (ValueError, TypeError) as e:
                self.logger.debug("Could not parse total_word_count metadata: %s", e)
        self.unigram_denominator = max(dynamic_denominator or cfg.unigram_denominator, 1.0)
        self.unigram_prob_cap = cfg.unigram_prob_cap
        self.backoff_floor_multiplier = cfg.backoff_floor_multiplier

        # Detection thresholds from config
        self._MIN_MEANINGFUL_PROB = cfg.min_meaningful_prob
        self._collocation_pmi_threshold = cfg.collocation_pmi_threshold
        self._confidence_zero_prob = cfg.confidence_zero_prob
        self._confidence_with_prob = cfg.confidence_with_prob
        self._confusable_confidence_cap = cfg.confusable_confidence_cap
        self._confusable_confidence_base = cfg.confusable_confidence_base
        self._confusable_confidence_scale = cfg.confusable_confidence_scale
        self._confusable_confidence_ratio_divisor = cfg.confusable_confidence_ratio_divisor
        self._confusable_confidence_ratio_cap = cfg.confusable_confidence_ratio_cap

        # Sentence-level cache for bidirectional probability computations.
        # Keyed by (word, tuple(prev_words), tuple(next_words)).
        # Avoids redundant DB queries when the same word+context is
        # evaluated multiple times (e.g. check_word_in_context computes
        # word_prob once, then _has_ngram_context re-queries the same
        # bigrams/trigrams).
        # Sentence-level caches are thread-local for safe check_batch_async
        self._thread_local = threading.local()
        # Override class-level confusable detection constants from config
        self._CONFUSABLE_BASE_RATIO = cfg.confusable_base_ratio
        self._CONFUSABLE_HIGH_FREQ_RATIO = cfg.confusable_high_freq_ratio
        self._CONFUSABLE_HIGH_FREQ_THRESHOLD = cfg.confusable_high_freq_threshold
        self._OVERLAP_RATIO_BASE = cfg.overlap_ratio_base
        self._OVERLAP_RATIO_RANGE = cfg.overlap_ratio_range
        self._ASPIRATED_MIN_RATIO = cfg.aspirated_min_ratio
        self._BIDIR_CACHE_MAX_SIZE: int = cfg.bidir_cache_max_size

        if not cfg.use_smoothing:
            self.smoothing_strategy = SmoothingStrategy.NONE

    @property
    def _bidir_prob_cache(self) -> dict:
        """Thread-local bidirectional probability cache."""
        tl = self._thread_local
        if not hasattr(tl, "bidir_prob_cache"):
            tl.bidir_prob_cache: dict = {}
        return tl.bidir_prob_cache

    @property
    def _ngram_context_cache(self) -> dict:
        """Thread-local n-gram context cache."""
        tl = self._thread_local
        if not hasattr(tl, "ngram_context_cache"):
            tl.ngram_context_cache: dict = {}
        return tl.ngram_context_cache

    # ------------------------------------------------------------------
    # Smoothing helpers (delegate to _ngram_smoothing module)
    # ------------------------------------------------------------------

    def _smoothing_kwargs(self) -> dict:
        """Common keyword arguments for smoothing functions."""
        return {
            "use_smoothing": self.use_smoothing,
            "strategy": self.smoothing_strategy,
            "add_k": self.add_k_smoothing,
            "backoff_weight": self.backoff_weight,
            "unigram_denominator": self.unigram_denominator,
            "unigram_prob_cap": self.unigram_prob_cap,
        }

    def get_smoothed_bigram_probability(self, word1: str, word2: str) -> float:
        """
        Get bigram probability with configurable smoothing strategy.

        If the bigram is unseen (P=0), applies the configured smoothing strategy
        to estimate a probability.

        Smoothing Strategies:
        - NONE: Return raw probability
        - STUPID_BACKOFF: Backoff to alpha * P(unigram)
        - ADD_K: Add constant k to probability

        Args:
            word1: First word (context)
            word2: Second word (target)

        Returns:
            Smoothed probability (never returns 0 if smoothing is enabled)
        """
        return _get_smoothed_bigram_probability(
            self.provider, word1, word2, **self._smoothing_kwargs()
        )

    def get_smoothed_trigram_probability(self, word1: str, word2: str, word3: str) -> float:
        """
        Get trigram probability with configurable smoothing strategy.

        Backoff Chain (for STUPID_BACKOFF):
        1. Try exact trigram P(w3|w1,w2)
        2. If unseen, backoff: alpha * P(w3|w2)
        3. If bigram unseen, backoff: alpha^2 * P(w3)

        Args:
            word1: First word
            word2: Second word
            word3: Third word (target)

        Returns:
            Smoothed probability
        """
        return _get_smoothed_trigram_probability(
            self.provider,
            word1,
            word2,
            word3,
            **self._smoothing_kwargs(),
            trigram_threshold=self.trigram_threshold,
            backoff_floor_multiplier=self.backoff_floor_multiplier,
        )

    def get_smoothed_fourgram_probability(
        self, word1: str, word2: str, word3: str, word4: str
    ) -> float:
        """
        Get 4-gram probability with Stupid Backoff.

        Backoff Chain:
        1. Try exact 4-gram P(w4|w1,w2,w3)
        2. If unseen, backoff: alpha * P(w4|w2,w3)  [trigram]
        3. If trigram unseen, backoff: alpha^2 * P(w4|w3)  [bigram]
        4. If bigram unseen, backoff: alpha^3 * P(w4)  [unigram]
        """
        return _get_smoothed_fourgram_probability(
            self.provider,
            word1,
            word2,
            word3,
            word4,
            **self._smoothing_kwargs(),
            fourgram_threshold=self.fourgram_threshold,
            backoff_floor_multiplier=self.backoff_floor_multiplier,
        )

    def get_smoothed_fivegram_probability(
        self, word1: str, word2: str, word3: str, word4: str, word5: str
    ) -> float:
        """
        Get 5-gram probability with Stupid Backoff.

        Backoff Chain:
        1. Try exact 5-gram P(w5|w1,w2,w3,w4)
        2. If unseen, backoff: alpha * P(w5|w2,w3,w4)  [4-gram]
        3. If 4-gram unseen, backoff: alpha^2 * P(w5|w3,w4)  [trigram]
        4. If trigram unseen, backoff: alpha^3 * P(w5|w4)  [bigram]
        5. If bigram unseen, backoff: alpha^4 * P(w5)  [unigram]
        """
        return _get_smoothed_fivegram_probability(
            self.provider,
            word1,
            word2,
            word3,
            word4,
            word5,
            **self._smoothing_kwargs(),
            fivegram_threshold=self.fivegram_threshold,
            backoff_floor_multiplier=self.backoff_floor_multiplier,
        )

    def get_best_left_probability(self, prev_words: list[str], word: str) -> float:
        """Get best available left-context probability with 4-gram -> trigram -> bigram fallback.

        Uses the highest-order n-gram available:
        - 4-gram P(word | prev[-3], prev[-2], prev[-1]) when prev_words has 3+ entries
        - Trigram P(word | prev[-2], prev[-1]) when prev_words has 2+ entries
        - Bigram  P(word | prev[-1])           when only one prev word available

        Note: 4-gram and trigram hits use raw provider probabilities (+ add_k)
        rather than the full smoothing chain.  This is intentional -- a nonzero
        hit at 4-gram/trigram is already strong evidence, and the raw value is
        better for relative ranking across candidates.

        Keeps left and right probabilities separate so callers can apply
        independent directional weights (unlike _compute_bidirectional_prob).

        Args:
            prev_words: Left context words, ordered oldest-first.
            word: Candidate word being scored.

        Returns:
            Smoothed probability (higher = more likely given left context).
        """
        return _get_best_left_probability(
            self.provider,
            prev_words,
            word,
            add_k=self.add_k_smoothing,
            get_smoothed_bigram=self.get_smoothed_bigram_probability,
        )

    def get_best_right_probability(self, word: str, next_words: list[str]) -> float:
        """Get best available right-context probability with 4-gram -> trigram -> bigram fallback.

        Uses the highest-order n-gram available:
        - 4-gram P(next[2] | word, next[0], next[1]) when next_words has 3+ entries
        - Trigram P(next[1] | word, next[0]) when next_words has 2+ entries
        - Bigram  P(next[0] | word)           when only one next word available

        Keeps left and right probabilities separate so callers can apply
        independent directional weights (unlike _compute_bidirectional_prob).

        Args:
            word: Candidate word being scored.
            next_words: Right context words, ordered left-to-right.

        Returns:
            Smoothed probability (higher = more likely given right context).
        """
        return _get_best_right_probability(
            self.provider,
            word,
            next_words,
            add_k=self.add_k_smoothing,
            get_smoothed_bigram=self.get_smoothed_bigram_probability,
        )

    # ------------------------------------------------------------------
    # Detection: is_contextual_error
    # ------------------------------------------------------------------

    def is_contextual_error(
        self,
        prev_word: str,
        current_word: str,
        prev_prev_word: str | None = None,
        next_word: str | None = None,
        threshold: float | None = None,
        prev_prev_prev_word: str | None = None,
        prev_prev_prev_prev_word: str | None = None,
    ) -> bool:
        """
        Check if a word is a contextual error given the surrounding context.

        Tries the highest-order n-gram available first, falling back to lower:
        5-gram -> 4-gram -> trigram -> bigram (with bidirectional heuristics).

        Only enters a higher-order path when a raw (non-backoff) n-gram is
        found in the corpus. This prevents smoothed backoff values from
        locking out lower-order heuristics.

        Args:
            prev_word: Previous word providing left context
            current_word: Current word to check
            prev_prev_word: Word before previous word (for trigram check)
            next_word: Next word providing right context (optional)
            threshold: Custom threshold (uses self.threshold if None)
            prev_prev_prev_word: 3 words back (for 4-gram check)
            prev_prev_prev_prev_word: 4 words back (for 5-gram check)

        Returns:
            True if current_word is contextually inappropriate, False otherwise
        """
        # Use provided threshold or fall back to instance default (thread-safe)
        effective_threshold = threshold if threshold is not None else self.threshold

        # --- 5-gram path ---
        if prev_prev_prev_prev_word and prev_prev_prev_word and prev_prev_word:
            raw_fivegram_prob = self.provider.get_fivegram_probability(
                prev_prev_prev_prev_word,
                prev_prev_prev_word,
                prev_prev_word,
                prev_word,
                current_word,
            )
            if raw_fivegram_prob > 0:
                if self.use_smoothing:
                    prob = self.get_smoothed_fivegram_probability(
                        prev_prev_prev_prev_word,
                        prev_prev_prev_word,
                        prev_prev_word,
                        prev_word,
                        current_word,
                    )
                else:
                    prob = raw_fivegram_prob
                fivegram_thresh = threshold if threshold is not None else self.fivegram_threshold
                return prob < fivegram_thresh

        # --- 4-gram path ---
        if prev_prev_prev_word and prev_prev_word:
            raw_fourgram_prob = self.provider.get_fourgram_probability(
                prev_prev_prev_word, prev_prev_word, prev_word, current_word
            )
            if raw_fourgram_prob > 0:
                if self.use_smoothing:
                    prob = self.get_smoothed_fourgram_probability(
                        prev_prev_prev_word, prev_prev_word, prev_word, current_word
                    )
                else:
                    prob = raw_fourgram_prob
                fourgram_thresh = threshold if threshold is not None else self.fourgram_threshold
                return prob < fourgram_thresh

        # --- Trigram path ---
        trigram_prob = 0.0
        raw_trigram_found = False
        if prev_prev_word:
            raw_trigram_prob = self.provider.get_trigram_probability(
                prev_prev_word, prev_word, current_word
            )
            raw_trigram_found = raw_trigram_prob > 0

            if raw_trigram_found:
                if self.use_smoothing:
                    trigram_prob = self.get_smoothed_trigram_probability(
                        prev_prev_word, prev_word, current_word
                    )
                else:
                    trigram_prob = raw_trigram_prob

        if raw_trigram_found:
            trigram_effective_threshold = (
                threshold if threshold is not None else self.trigram_threshold
            )
            return trigram_prob < trigram_effective_threshold

        # Left context: P(current_word | prev_word)
        left_prob = self.provider.get_bigram_probability(prev_word, current_word)

        # Right context: P(next_word | current_word)
        # Track if we have actual right context data (non-zero probability)
        right_prob = 0.0
        has_right_context_data = False
        if next_word:
            right_prob = self.provider.get_bigram_probability(current_word, next_word)
            # Only consider right context as valid if we have actual probability data
            # An unseen bigram (0) is not evidence either way
            if right_prob > 0:
                has_right_context_data = True

        # For backward compatibility, use left_prob as the primary signal
        bigram_prob = left_prob

        # Case 1: Low probability but seen (0 < P < threshold)
        # With bidirectional context: if right context is KNOWN and strong, don't flag
        if 0 < bigram_prob < effective_threshold:
            # Only use right context to prevent flagging when:
            # 1. We have actual probability data for it (not unseen)
            # 2. That probability is above right_context_threshold (strong right context)
            # Use separate threshold for right context - typically higher than
            # left context threshold since rescue requires stronger evidence
            if has_right_context_data and right_prob > self.right_context_threshold:
                return False
            # When right context is absent (unseen, not weak), a nonzero left
            # bigram for a common word is sparse-data noise, not an error signal.
            # The pair HAS been observed in the corpus -- just infrequently.
            # NOTE: Do NOT extend this to weak right context -- TPs like ကာ/ကား
            # have weak right context but ARE genuine errors.
            # Guard: only when next_word exists. For sentence-final words
            # (next_word=None), absent right context is expected -- the left
            # bigram is the only signal and should not be dismissed.
            if next_word and not has_right_context_data:
                word_freq = self.provider.get_word_frequency(current_word)
                if word_freq >= self._CASE1_FREQ_GUARD:
                    return False
            return True

        # Case 2: Unseen (P=0).
        if bigram_prob == 0:
            # Collocation PMI rescue: if the word pair is a known strong
            # collocation (from mined data), trust it even without bigram evidence.
            # This handles valid word pairs that were unseen in n-gram training
            # but have strong co-occurrence in the corpus.
            if hasattr(self.provider, "get_collocation_pmi"):
                pmi = self.provider.get_collocation_pmi(prev_word, current_word)
                if isinstance(pmi, (int, float)) and pmi >= self._collocation_pmi_threshold:
                    return False  # Strong collocation -- not an error

            # Bidirectional weak context: left=unseen AND right=very weak
            # -> override unigram backoff.  When a word has NEVER been seen
            # after its predecessor AND its forward bigram is extremely rare,
            # the positional evidence is very strong regardless of word freq.
            # Guard: only when the prev word has non-trivial frequency.
            # Zero-freq prev words (rare/broken) make zero bigrams unreliable.
            if has_right_context_data and right_prob < self.right_context_threshold:
                prev_freq = self.provider.get_word_frequency(prev_word)
                if prev_freq > 0:
                    return True

            # Sub-strategy A: Backoff Smoothing (Unigram Check)
            # If the word itself is common, it's likely correct but in a new context.
            word_freq = self.provider.get_word_frequency(current_word)
            if word_freq >= self.min_unigram_threshold:
                return False

            # Sub-strategy B: Typo Heuristic (SymSpell)
            # Strategy: If P=0 AND word is rare, look for a neighbor (dist=1)
            # with P > threshold * multiplier.
            # This avoids flagging valid unique phrases, but catches "near miss" typos.
            if self.symspell:
                # Only check if we have SymSpell to find neighbors efficiently
                # Limit to edit_distance=1 for speed and precision
                neighbors = self.symspell.lookup(
                    current_word,
                    level=ValidationLevel.WORD.value,
                    max_suggestions=5,
                    include_known=True,  # generate neighbors even if word is valid
                    use_phonetic=True,
                )

                for neighbor in neighbors:
                    term = neighbor.term
                    if term == current_word:
                        continue

                    # Only consider close neighbors (edit distance <= 1) for typo detection.
                    # Distance-2 neighbors are too far to reliably assume a typo.
                    if neighbor.edit_distance > 1:
                        continue

                    # Check probability of neighbor
                    neighbor_prob = self.provider.get_bigram_probability(prev_word, term)

                    # If a neighbor is "Very Likely" (e.g., > 1%), assume current is a typo
                    # We use threshold * multiplier as a heuristic for "Likely"
                    if neighbor_prob > self.threshold * self.heuristic_multiplier:
                        return True

        return False

    # ------------------------------------------------------------------
    # Comparison: compare_contextual_probability
    # ------------------------------------------------------------------

    def compare_contextual_probability(
        self,
        word: str,
        alternatives: list[str],
        prev_words: list[str],
        next_words: list[str],
        min_ratio: float = 5.0,
        high_freq_threshold: int = 10000,
        high_freq_min_ratio: float = 10.0,
        min_meaningful_prob: float | None = None,
    ) -> tuple[str | None, float]:
        """
        Compare n-gram probability of word vs alternatives in context.

        Unlike ``is_contextual_error()`` which only asks "is this word unlikely?",
        this method asks "is another word MORE LIKELY in this context?" -- enabling
        detection of real-word confusions where both the wrong word and the correct
        word are valid dictionary entries with nonzero n-gram probabilities.

        The method computes a combined bidirectional n-gram probability for the
        current word and each alternative, using the same smoothing and backoff
        as the existing n-gram checker. It returns the best alternative if its
        combined probability exceeds the current word's by at least ``min_ratio``.

        Args:
            word: The current word to compare against alternatives.
            alternatives: List of candidate replacement words.
            prev_words: Previous words providing left context, ordered
                ``[..., prev_prev, prev]`` (closest last). May be empty.
            next_words: Next words providing right context, ordered
                ``[next, next_next, ...]`` (closest first). May be empty.
            min_ratio: Minimum probability ratio for an alternative to win
                (default: 5.0). An alternative must be at least this many
                times more probable than the current word.
            high_freq_threshold: Frequency above which ``high_freq_min_ratio``
                is used instead of ``min_ratio`` (default: 10000).
            high_freq_min_ratio: Stricter ratio for high-frequency words
                (default: 10.0). Prevents FPs on common words.
            min_meaningful_prob: Minimum combined probability for either the
                word or the best alternative to be considered meaningful
                (default: ``_MIN_MEANINGFUL_PROB``). When both are below this
                floor, the context is too sparse to draw conclusions and
                ``(None, 0.0)`` is returned.

        Returns:
            ``(best_alternative, ratio)`` if an alternative has >= min_ratio
            times higher combined probability, else ``(None, 0.0)``.
        """
        if not alternatives:
            return None, 0.0

        # Default to class-level constant when caller does not override
        if min_meaningful_prob is None:
            min_meaningful_prob = self._MIN_MEANINGFUL_PROB

        # Determine effective ratio based on word frequency
        effective_ratio = min_ratio
        word_freq = self.provider.get_word_frequency(word)
        if isinstance(word_freq, (int, float)) and word_freq >= high_freq_threshold:
            effective_ratio = max(min_ratio, high_freq_min_ratio)

        # Check if the current word has real n-gram context (bigram/trigram).
        word_has_ngram = self._has_ngram_context(word, prev_words, next_words)

        # Compute bidirectional probability for the current word
        word_prob = self._compute_bidirectional_prob(word, prev_words, next_words)

        best_alt: str | None = None
        best_ratio: float = 0.0

        for alt in alternatives:
            if alt == word:
                continue

            # Skip confusable-exempt pairs
            if _is_confusable_exempt(word, alt, self.provider):
                continue

            alt_prob = self._compute_bidirectional_prob(alt, prev_words, next_words)

            # Skip when both probabilities are too low to be meaningful
            if word_prob < min_meaningful_prob and alt_prob < min_meaningful_prob:
                continue

            # Guard: when neither the current word nor the alternative has
            # real n-gram context (bigrams/trigrams), both probabilities
            # come purely from unigram frequency fallback. This comparison
            # only reflects which word is more common in the corpus overall,
            # not which word fits the context better. Skip to avoid FPs on
            # valid but low-frequency words (e.g., ပြသ freq=0 vs ပြဿ freq=4808).
            if not word_has_ngram and not self._has_ngram_context(alt, prev_words, next_words):
                continue

            # Compute ratio: how much better is the alternative?
            if word_prob > 0:
                ratio = alt_prob / word_prob
            elif alt_prob > 0:
                # Current word has zero prob but alternative has nonzero.
                # Guard: when the current word lacks n-gram context entirely,
                # the "infinite ratio" from zero-denominator is misleading --
                # it means we have no data about the current word, not that
                # the alternative is infinitely better. Require the alternative
                # to have strong contextual evidence (both directions), not
                # just a single weak bigram (e.g., bigram(က,ပြဿ)=4e-6 is
                # noise-level, 7x weaker than the weakest real confusable).
                if not word_has_ngram:
                    alt_has_left = self._has_ngram_context_directional(
                        alt, prev_words, direction="left"
                    )
                    alt_has_right = self._has_ngram_context_directional(
                        alt, next_words, direction="right"
                    )
                    has_left_ctx = len(prev_words) > 0
                    has_right_ctx = len(next_words) > 0
                    # When both directions have context words, require
                    # bidirectional evidence to prevent single-direction
                    # noise from triggering inf-ratio corrections.
                    # At sentence edges (only one direction available),
                    # accept single-direction evidence.
                    if has_left_ctx and has_right_ctx:
                        if not (alt_has_left and alt_has_right):
                            continue
                    elif has_left_ctx and not alt_has_left:
                        continue
                    elif has_right_ctx and not alt_has_right:
                        continue
                ratio = float("inf")
            else:
                # Both zero -- no signal
                continue

            if ratio >= effective_ratio and ratio > best_ratio:
                best_ratio = ratio
                best_alt = alt

        if best_alt is not None:
            return best_alt, best_ratio

        return None, 0.0

    # ------------------------------------------------------------------
    # Ratio computation
    # ------------------------------------------------------------------

    def compute_required_ratio(
        self,
        word_freq: int,
        candidate_freq: int,
        confusable_type: str | None = None,
        word: str | None = None,
        candidate: str | None = None,
    ) -> float:
        """
        Compute the minimum n-gram probability ratio required to flag a word
        as confusable with a candidate.

        Uses DB context_overlap scores when available (data-driven path),
        falling back to hardcoded frequency-dependent scaling.

        Args:
            word_freq: Frequency of the word being checked.
            candidate_freq: Frequency of the candidate replacement.
            confusable_type: Optional type hint for the confusable pair.
            word: The actual word string (for DB context_overlap lookup).
            candidate: The candidate string (for DB context_overlap lookup).

        Returns:
            Minimum ratio required for the candidate to be considered better.
            Returns ``float('inf')`` when the inverse frequency guard fires.
        """
        # Inverse guard: if the current word is significantly more common
        # than the candidate, the candidate is unlikely to be the intended
        # word.  Return infinity to effectively skip the comparison.
        if candidate_freq > 0 and word_freq > 3 * candidate_freq:
            return float("inf")

        # Data-driven path: use DB context_overlap to scale the required ratio.
        # High overlap = pair is hard to distinguish -> need higher ratio.
        # Low overlap = easily separable by context -> lower ratio suffices.
        if word and candidate and hasattr(self.provider, "get_confusable_context_overlap"):
            overlap = self.provider.get_confusable_context_overlap(word, candidate)
            if isinstance(overlap, (int, float)):
                # Linear interpolation between easy and hard separability
                return self._OVERLAP_RATIO_BASE + overlap * self._OVERLAP_RATIO_RANGE

        # Fallback: hardcoded frequency-dependent scaling
        base_ratio = self._CONFUSABLE_BASE_RATIO

        # High-frequency words need much stronger evidence
        if word_freq >= self._CONFUSABLE_HIGH_FREQ_THRESHOLD:
            base_ratio = self._CONFUSABLE_HIGH_FREQ_RATIO

        # Adjust by confusable type
        if confusable_type == "medial":
            return base_ratio
        if confusable_type == "aspirated":
            return max(base_ratio, self._ASPIRATED_MIN_RATIO)

        return base_ratio

    # ------------------------------------------------------------------
    # Unified check: check_word_in_context
    # ------------------------------------------------------------------

    def check_word_in_context(
        self,
        word: str,
        prev_words: list[str],
        next_words: list[str],
        candidates: list[tuple[str, str | None]] | None = None,
        word_freq: int = 0,
    ) -> NgramVerdict:
        """
        Unified context check: absolute threshold + candidate comparison.

        This method combines the logic of ``is_contextual_error()`` (absolute
        threshold detection) and ``compare_contextual_probability()`` (relative
        comparison against candidates) into a single call that validation
        strategies can delegate to.

        Flow:
        1. Compute the absolute n-gram probability for *word* in context.
        2. If the absolute check flags an error AND no candidates are provided,
           return a verdict with ``error_type="context_probability"``.
        3. If candidates are provided, compare each against *word* using
           bidirectional n-gram probabilities and ``compute_required_ratio()``.
        4. If a candidate is substantially more probable, return a verdict
           with ``error_type="confusable_error"`` and the best alternative.
        5. If neither check fires, return ``is_error=False``.

        Args:
            word: The word to check.
            prev_words: Previous context words, ordered
                ``[..., prev_prev, prev]`` (closest last). May be empty.
            next_words: Following context words, ordered
                ``[next, next_next, ...]`` (closest first). May be empty.
            candidates: Optional list of ``(alternative_word, confusable_type)``
                to compare against.  ``confusable_type`` may be ``"medial"``,
                ``"aspirated"``, or ``None``.
            word_freq: Frequency of the word (0 if unknown).  Used for
                ratio scaling in ``compute_required_ratio()``.

        Returns:
            NgramVerdict with the detection result.
        """
        # --- Step 1: Absolute threshold check ---
        # Translate prev_words/next_words to the positional args that
        # is_contextual_error() expects.
        prev_word = prev_words[-1] if prev_words else ""
        prev_prev_word = prev_words[-2] if len(prev_words) >= 2 else None
        prev_prev_prev_word = prev_words[-3] if len(prev_words) >= 3 else None
        prev_prev_prev_prev_word = prev_words[-4] if len(prev_words) >= 4 else None
        next_word = next_words[0] if next_words else None

        absolute_is_error = False
        if prev_word:
            absolute_is_error = self.is_contextual_error(
                prev_word=prev_word,
                current_word=word,
                prev_prev_word=prev_prev_word,
                next_word=next_word,
                prev_prev_prev_word=prev_prev_prev_word,
                prev_prev_prev_prev_word=prev_prev_prev_prev_word,
            )

        # --- Step 2: No candidates -- return absolute result ---
        if not candidates:
            if absolute_is_error:
                prob = self._compute_bidirectional_prob(word, prev_words, next_words)
                return NgramVerdict(
                    is_error=True,
                    confidence=(
                        self._confidence_zero_prob if prob == 0.0 else self._confidence_with_prob
                    ),
                    error_type="context_probability",
                    probability=prob,
                )
            return NgramVerdict(is_error=False, probability=0.0)

        # --- Step 3: Candidate comparison ---
        word_prob = self._compute_bidirectional_prob(word, prev_words, next_words)
        word_has_ngram = self._has_ngram_context(word, prev_words, next_words)

        best_alt: str | None = None
        best_ratio: float = 0.0

        for alt, confusable_type in candidates:
            if alt == word:
                continue

            # Skip exempt word-variant pairs (e.g. ဖေး<->ဘေး loanword).
            # Also check stem pair when words share a common suffix
            # (e.g. ဖေးမှာ<->ဘေးမှာ -> stems ဖေး<->ဘေး are exempt).
            if _is_confusable_exempt(word, alt, self.provider):
                continue

            # Skip exempt suffix pairs (e.g. သည်<->သည့် syntactic distinction)
            _is_suffix_exempt = False
            for sfx_a, sfx_b in CONFUSABLE_EXEMPT_SUFFIX_PAIRS:
                if word.endswith(sfx_a) and alt.endswith(sfx_b):
                    _is_suffix_exempt = True
                    break
            if _is_suffix_exempt:
                continue

            alt_freq = self.provider.get_word_frequency(alt)
            if not isinstance(alt_freq, (int, float)):
                alt_freq = 0

            required_ratio = self.compute_required_ratio(
                word_freq=word_freq,
                candidate_freq=int(alt_freq),
                confusable_type=confusable_type,
                word=word,
                candidate=alt,
            )
            if required_ratio == float("inf"):
                continue

            alt_prob = self._compute_bidirectional_prob(alt, prev_words, next_words)

            # Skip when both probabilities are too low to be meaningful
            if word_prob < self._MIN_MEANINGFUL_PROB and alt_prob < self._MIN_MEANINGFUL_PROB:
                continue

            # Guard: when neither word has real n-gram context, both
            # probabilities come purely from unigram frequency -- not
            # contextual evidence.  Skip to avoid frequency-only FPs.
            if not word_has_ngram and not self._has_ngram_context(alt, prev_words, next_words):
                continue

            # Compute ratio
            if word_prob > 0:
                ratio = alt_prob / word_prob
            elif alt_prob > 0:
                # Zero-denominator: require bidirectional evidence when
                # the current word lacks n-gram context entirely.
                if not word_has_ngram:
                    has_left_ctx = len(prev_words) > 0
                    has_right_ctx = len(next_words) > 0
                    alt_has_left = self._has_ngram_context_directional(
                        alt, prev_words, direction="left"
                    )
                    alt_has_right = self._has_ngram_context_directional(
                        alt, next_words, direction="right"
                    )
                    if has_left_ctx and has_right_ctx:
                        if not (alt_has_left and alt_has_right):
                            continue
                    elif has_left_ctx and not alt_has_left:
                        continue
                    elif has_right_ctx and not alt_has_right:
                        continue
                ratio = float("inf")
            else:
                # Both zero -- no signal
                continue

            if ratio >= required_ratio and ratio > best_ratio:
                best_ratio = ratio
                best_alt = alt

        # --- Step 4: Return best comparison result ---
        if best_alt is not None:
            return NgramVerdict(
                is_error=True,
                best_alternative=best_alt,
                confidence=min(
                    self._confusable_confidence_cap,
                    self._confusable_confidence_base
                    + self._confusable_confidence_scale
                    * min(
                        best_ratio / self._confusable_confidence_ratio_divisor,
                        self._confusable_confidence_ratio_cap,
                    ),
                ),
                error_type="confusable_error",
                ratio=best_ratio,
                probability=word_prob,
                suggestions=[best_alt],
            )

        # --- Step 5: Fall back to absolute result ---
        if absolute_is_error:
            return NgramVerdict(
                is_error=True,
                confidence=(
                    self._confidence_zero_prob if word_prob == 0.0 else self._confidence_with_prob
                ),
                error_type="context_probability",
                probability=word_prob,
            )

        return NgramVerdict(is_error=False, probability=word_prob)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_context_cache(self) -> None:
        """Clear sentence-level bidirectional probability and n-gram context caches.

        Call between sentences or documents to prevent stale results.
        """
        self._bidir_prob_cache.clear()
        self._ngram_context_cache.clear()

    # ------------------------------------------------------------------
    # Bidirectional scoring (delegate to _ngram_scoring module)
    # ------------------------------------------------------------------

    def _compute_bidirectional_prob(
        self,
        word: str,
        prev_words: list[str],
        next_words: list[str],
    ) -> float:
        """
        Compute combined bidirectional n-gram probability for *word*.

        Uses the highest-order n-gram available in each direction and
        combines left and right probabilities. Falls back through the
        backoff chain: trigram -> bigram -> unigram (smoothed).

        The left probability is P(word | prev context) and the right
        probability is P(next context | word).  When both are available,
        they are averaged; when only one direction has data, that value
        is used alone.

        Args:
            word: Target word to compute probability for.
            prev_words: Left context ``[..., prev_prev, prev]``.
            next_words: Right context ``[next, next_next, ...]``.

        Returns:
            Combined bidirectional probability (float >= 0).
        """
        return _compute_bidirectional_prob(
            self.provider,
            word,
            prev_words,
            next_words,
            unigram_denominator=self.unigram_denominator,
            unigram_prob_cap=self.unigram_prob_cap,
            bidir_cache=self._bidir_prob_cache,
            bidir_cache_max_size=self._BIDIR_CACHE_MAX_SIZE,
        )

    def _has_ngram_context(
        self,
        word: str,
        prev_words: list[str],
        next_words: list[str],
    ) -> bool:
        """
        Check if *word* has any real bigram/trigram context.

        Returns True if at least one bigram or trigram involving *word* and
        the surrounding context words has a nonzero probability. Returns
        False when the only available signal is the unigram frequency of
        the word itself.

        This is used by ``compare_contextual_probability`` to distinguish
        genuine contextual evidence from mere corpus frequency differences.
        """
        return _has_ngram_context(
            self.provider,
            word,
            prev_words,
            next_words,
            ngram_cache=self._ngram_context_cache,
            ngram_cache_max_size=self._BIDIR_CACHE_MAX_SIZE,
        )

    def _has_ngram_context_directional(
        self,
        word: str,
        context_words: list[str],
        direction: str = "left",
    ) -> bool:
        """
        Check if *word* has n-gram context in a single direction.

        Args:
            word: Target word to check.
            context_words: Context words for the given direction.
                For ``"left"``: ``[..., prev_prev, prev]`` (closest last).
                For ``"right"``: ``[next, next_next, ...]`` (closest first).
            direction: ``"left"`` or ``"right"``.

        Returns:
            True if at least one bigram or trigram in this direction
            has a nonzero probability.
        """
        return _has_ngram_context_directional(self.provider, word, context_words, direction)

    # ------------------------------------------------------------------
    # Suggestion generation (delegate to _ngram_candidates module)
    # ------------------------------------------------------------------

    def suggest(
        self,
        prev_word: str,
        current_word: str,
        max_edit_distance: int = 2,
        next_word: str | None = None,
    ) -> list[ContextSuggestion]:
        """
        Generate context-aware suggestions for a word.

        This method finds alternative words that:
        1. Have higher bigram probability given prev_word
        2. Are similar to current_word (within max_edit_distance)
        3. Fit well with next_word (if provided)

        Strategy:
        1. Find candidate words with high P(candidate|prev_word)
        2. Filter candidates by edit distance from current_word
        3. Score by weighted combination of probability and similarity
        4. Return top-ranked suggestions

        Args:
            prev_word: Previous word providing context
            current_word: Current word to find suggestions for
            max_edit_distance: Maximum edit distance for candidates (default: 2)
            next_word: Next word providing right context (optional)

        Returns:
            List of ContextSuggestion objects, sorted by score (best first)

        Example:
            >>> suggestions = checker.suggest("သူ", "ကျောင်း")
            >>> for s in suggestions:
            ...     print(f"{s.term} (P={s.probability:.3f}, dist={s.edit_distance})")
            သွား (P=0.234, dist=2)
            သည် (P=0.189, dist=2)
            ရှိ (P=0.156, dist=2)

        Note:
            This is a simplified implementation. Production version should use
            more sophisticated candidate generation (e.g., SymSpell + N-grams).
        """
        suggestions: list[ContextSuggestion] = []

        # Get current bigram probability as baseline
        current_prob = self.provider.get_bigram_probability(prev_word, current_word)

        # Generate candidates using a hybrid approach
        candidates = _generate_candidates(
            self.provider,
            prev_word,
            current_word,
            max_edit_distance,
            candidate_limit=self.candidate_limit,
            symspell=self.symspell,
        )

        # Pre-calculate phonetic variants to boost their score
        phonetic_variants = _get_phonetic_variants(current_word, self.symspell)

        for candidate in candidates:
            result = _score_candidate(
                self.provider,
                candidate=candidate,
                prev_word=prev_word,
                current_word=current_word,
                current_prob=current_prob,
                next_word=next_word,
                max_edit_distance=max_edit_distance,
                phonetic_variants=phonetic_variants,
                pos_bigram_probs=self.pos_bigram_probs,
                pos_score_weight=self.pos_score_weight,
                pos_distance_reduction_factor_multiplier=self.pos_distance_reduction_factor_multiplier,
                probability_weight=self.probability_weight,
                edit_distance_weight=self.edit_distance_weight,
                score_scaling_factor=self.score_scaling_factor,
                min_prob_denom=self.min_prob_denom,
            )
            if result is not None:
                term, prob, distance, combined_score, confidence = result
                suggestions.append(
                    ContextSuggestion(
                        term=term,
                        probability=prob,
                        edit_distance=distance,
                        score=combined_score,
                        confidence=confidence,
                    )
                )

        suggestions.sort()

        return suggestions[: self.max_suggestions]

    def _get_phonetic_variants(self, word: str) -> set:
        """Get phonetic variants for a word using SymSpell's phonetic hasher."""
        return _get_phonetic_variants(word, self.symspell)

    def _score_candidate(
        self,
        candidate: str,
        prev_word: str,
        current_word: str,
        current_prob: float,
        next_word: str | None,
        max_edit_distance: int,
        phonetic_variants: set,
    ) -> ContextSuggestion | None:
        """
        Score a single candidate and return a ContextSuggestion if valid.

        Returns None if the candidate should be filtered out.
        """
        result = _score_candidate(
            self.provider,
            candidate=candidate,
            prev_word=prev_word,
            current_word=current_word,
            current_prob=current_prob,
            next_word=next_word,
            max_edit_distance=max_edit_distance,
            phonetic_variants=phonetic_variants,
            pos_bigram_probs=self.pos_bigram_probs,
            pos_score_weight=self.pos_score_weight,
            pos_distance_reduction_factor_multiplier=self.pos_distance_reduction_factor_multiplier,
            probability_weight=self.probability_weight,
            edit_distance_weight=self.edit_distance_weight,
            score_scaling_factor=self.score_scaling_factor,
            min_prob_denom=self.min_prob_denom,
        )
        if result is None:
            return None
        term, prob, distance, combined_score, confidence = result
        return ContextSuggestion(
            term=term,
            probability=prob,
            edit_distance=distance,
            score=combined_score,
            confidence=confidence,
        )

    def _combine_context_probabilities(
        self, prob_left: float, candidate: str, next_word: str | None
    ) -> float:
        """
        Combine left and right context probabilities.

        If right context is available and has probability > 0, average the two.
        Otherwise, use left probability only (conservative approach).
        """
        return _combine_context_probabilities(self.provider, prob_left, candidate, next_word)

    def _apply_pos_context_scoring(
        self,
        effective_distance: float,
        prev_word: str,
        candidate: str,
        next_word: str | None,
    ) -> float:
        """
        Apply POS context scoring to reduce effective distance.

        Uses POS bigram probabilities to boost candidates that have
        appropriate POS tag transitions.
        """
        return _apply_pos_context_scoring(
            self.provider,
            effective_distance,
            prev_word,
            candidate,
            next_word,
            pos_bigram_probs=self.pos_bigram_probs,
            pos_score_weight=self.pos_score_weight,
            pos_distance_reduction_factor_multiplier=self.pos_distance_reduction_factor_multiplier,
        )

    def _calculate_pos_context_score(
        self, prev_word: str, candidate: str, next_word: str | None
    ) -> float:
        """
        Calculate POS context score based on POS bigram probabilities.

        Returns a score between 0.0 and 1.0 representing how well the
        candidate's POS tag fits the context.
        """
        return _calculate_pos_context_score(
            self.provider,
            prev_word,
            candidate,
            next_word,
            pos_bigram_probs=self.pos_bigram_probs,
        )

    def calculate_pos_context_score(
        self, prev_word: str, candidate: str, next_word: str | None = None
    ) -> float:
        """
        Public interface for POS context scoring.

        Returns a score between 0.0 and 1.0 representing how well the
        candidate's POS tag fits the surrounding context based on POS
        bigram probabilities from the production database.
        """
        return self._calculate_pos_context_score(prev_word, candidate, next_word)

    def _get_pos_tags(self, word: str) -> set:
        """Get set of POS tags for a word from the provider."""
        return _get_pos_tags(self.provider, word)

    def _max_pos_bigram_prob(self, tags1: set, tags2: set) -> float:
        """
        Find maximum POS bigram probability between two sets of tags.

        Iterates through all combinations of tags and returns the maximum
        probability found in pos_bigram_probs.
        """
        return _max_pos_bigram_prob(tags1, tags2, self.pos_bigram_probs)

    def _calculate_combined_score(self, prob: float, effective_distance: float) -> float:
        """
        Calculate combined score using log-probability model.

        Score = probability_weight * log(prob) - edit_distance_weight * distance * scaling_factor

        Higher scores are better.
        """
        return _calculate_combined_score(
            prob,
            effective_distance,
            probability_weight=self.probability_weight,
            edit_distance_weight=self.edit_distance_weight,
            score_scaling_factor=self.score_scaling_factor,
        )

    def _generate_candidates(
        self, prev_word: str, current_word: str, max_edit_distance: int
    ) -> list[str]:
        """
        Generate candidate words for context-aware suggestions.

        This method uses a hybrid approach:
        1. Query high-probability continuations from bigram data
        2. Filter candidates by edit distance from current_word
        3. If SymSpell is available, add orthographically/phonetically similar words
        4. Return deduplicated candidate list

        Args:
            prev_word: Previous word for context
            current_word: Current word to find alternatives for
            max_edit_distance: Maximum edit distance

        Returns:
            List of candidate words
        """
        return _generate_candidates(
            self.provider,
            prev_word,
            current_word,
            max_edit_distance,
            candidate_limit=self.candidate_limit,
            symspell=self.symspell,
        )
