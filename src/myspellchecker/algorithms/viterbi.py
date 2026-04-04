"""
Viterbi Decoder for POS Tagging.

This module provides a trigram HMM-based POS tagger using the Viterbi algorithm
with deleted interpolation smoothing for robust probability estimation.

Performance Characteristics:
    Time Complexity:
        - tag_sequence(): O(N * T² * B) where N=sequence length, T=tagset size,
          B=beam_width. With beam pruning, effectively O(N * B² * T).
        - _get_smoothed_trigram_prob(): O(1) with caching, O(log operations) without.
        - _get_emission_score(): O(1) with caching.

    Space Complexity:
        - Viterbi table: O(N * B) where N=sequence length, B=beam_width
        - Backpointer table: O(N * B)
        - Probability caches: O(cache_size) bounded by EMISSION_CACHE_SIZE,
          TRANSITION_CACHE_SIZE constants

    Cache Behavior:
        - Emission cache (4096 entries): Caches word-tag emission scores.
          High hit rate for repeated words in document.
        - Bigram cache (2048 entries): Caches smoothed P(t2|t1) values.
          Bounded by tagset², typically <1000 unique pairs.
        - Trigram cache (2048 entries): Caches smoothed P(t3|t1,t2) values.
          Bounded by tagset³, but beam pruning limits active states.

Memory Considerations:
    The ViterbiTagger stores probability tables in memory for fast O(1) lookups.
    For typical Myanmar POS tagging:
    - Unigram table: ~1KB (small tagset)
    - Bigram table: ~100KB (tagset² entries)
    - Trigram table: ~10MB (tagset³ entries, sparse)

    For memory-constrained environments:
    1. Use beam_width to limit state space (reduces computation, not memory)
    2. Consider using the rule-based or transformer tagger instead
    3. Future: Lazy loading from provider with LRU eviction (planned)

    The emission and transition caches help reduce redundant computations
    but have bounded size to prevent memory growth.

Performance Tuning:
    - beam_width=10 (default): Good balance of speed vs accuracy
    - beam_width=5: Faster, slight accuracy loss on long sequences
    - beam_width=20: More accurate, ~2x slower
    - adaptive_beam=True: Auto-adjusts based on sequence length
"""

from __future__ import annotations

import math
import warnings
from heapq import nlargest
from typing import Any

from myspellchecker.core.config.tagger_configs import POSTaggerConfig
from myspellchecker.providers import DictionaryProvider
from myspellchecker.text.morphology import MorphologyAnalyzer
from myspellchecker.utils.cache import LRUCache
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Sentinel for distinguishing "not provided" from intentional default values
_UNSET: Any = object()

# Default cache sizes (used when no config is provided)
EMISSION_CACHE_SIZE = 4096
TRANSITION_CACHE_SIZE = 2048

# Try to import Cythonized Viterbi Tagger
try:
    from myspellchecker.algorithms import viterbi_c  # type: ignore[attr-defined]

    _HAS_CYTHON_VITERBI = True
except ImportError:
    _HAS_CYTHON_VITERBI = False


class ViterbiTagger:
    """
    Finds the optimal POS tag sequence using the Viterbi algorithm.

    Uses trigram HMM with:
    - Transition probabilities: P(tag | prev_tags) from bigram/trigram tables
    - Emission probabilities: P(tag | word) from seed data word-tag counts
    - Emission prior fallback: P(tag) from unigram probabilities

    Performance:
        - tag_sequence() runs in O(N * B² * T) with beam pruning
        - Cython implementation available for 2-5x speedup
        - LRU caches reduce redundant probability lookups

    Thread Safety:
        - Instance is NOT thread-safe (shared caches, Viterbi tables)
        - Create separate instances for concurrent tagging
        - Or use external synchronization

    Cache Management:
        - cache_stats(): Get hit/miss statistics for tuning
    """

    def __init__(
        self,
        provider: DictionaryProvider,
        pos_bigram_probs: dict[tuple[str, str], float],
        pos_trigram_probs: dict[tuple[str, str, str], float],
        pos_unigram_probs: dict[str, float] | None = None,
        unknown_word_tag: str = "UNK",  # Fallback tag for completely unknown words
        min_prob: float = _UNSET,
        beam_width: int = _UNSET,  # Keep top-k states per position for performance
        emission_weight: float = _UNSET,  # Weight for emission probabilities (tuned for accuracy)
        use_morphology_fallback: bool = True,  # Use MorphologyAnalyzer for unknown words
        lambda_unigram: float = _UNSET,  # Deleted interpolation: unigram weight
        lambda_bigram: float = _UNSET,  # Deleted interpolation: bigram weight
        lambda_trigram: float = _UNSET,  # Deleted interpolation: trigram weight
        adaptive_beam: bool = False,  # Enable adaptive beam width based on sequence length
        min_beam_width: int = _UNSET,  # Minimum beam width for adaptive mode
        max_beam_width: int = _UNSET,  # Maximum beam width for adaptive mode
        config: POSTaggerConfig | None = None,  # Optional config for centralized settings
    ):
        """
        Initialize ViterbiTagger with deleted interpolation smoothing.

        Args:
            provider: DictionaryProvider for word-POS lookups
            pos_bigram_probs: Bigram transition probabilities P(t2|t1)
            pos_trigram_probs: Trigram transition probabilities P(t3|t1,t2)
            pos_unigram_probs: Unigram probabilities P(t) for smoothing
            unknown_word_tag: Tag for unknown words (default: "UNK")
            min_prob: Minimum probability floor (default: 1e-10)
            beam_width: Beam search width for pruning (default: 10)
            emission_weight: Weight for emission scores (default: 1.2)
            use_morphology_fallback: Use MorphologyAnalyzer for unknown words
            lambda_unigram: Interpolation weight for unigram (default: 0.1)
            lambda_bigram: Interpolation weight for bigram (default: 0.3)
            lambda_trigram: Interpolation weight for trigram (default: 0.6)
            adaptive_beam: Enable adaptive beam width based on sequence length.
                          When True, beam width is adjusted dynamically:
                          - Short sequences (≤5 words): wider beam for accuracy
                          - Long sequences (>20 words): narrower beam for speed
            min_beam_width: Minimum beam width for adaptive mode (default: 5)
            max_beam_width: Maximum beam width for adaptive mode (default: 20)
            config: Optional POSTaggerConfig for centralized settings. When
                provided, its viterbi_* fields are used as defaults for
                parameters that are still at their original default values.

        Note:
            lambda_unigram + lambda_bigram + lambda_trigram should equal 1.0.
            The smoothed transition probability uses deleted interpolation:
            P(t3|t1,t2) = λ3*P(t3|t1,t2) + λ2*P(t3|t2) + λ1*P(t3)
        """
        # Resolve config-driven defaults: use config values when the caller
        # hasn't explicitly provided the parameter (sentinel check).
        if config is not None:
            if min_prob is _UNSET:
                min_prob = config.viterbi_min_prob
            if beam_width is _UNSET:
                beam_width = config.viterbi_beam_width
            if emission_weight is _UNSET:
                emission_weight = config.viterbi_emission_weight
            if lambda_unigram is _UNSET:
                lambda_unigram = config.viterbi_lambda_unigram
            if lambda_bigram is _UNSET:
                lambda_bigram = config.viterbi_lambda_bigram
            if lambda_trigram is _UNSET:
                lambda_trigram = config.viterbi_lambda_trigram
            if min_beam_width is _UNSET:
                min_beam_width = config.viterbi_min_beam_width
            if max_beam_width is _UNSET:
                max_beam_width = config.viterbi_max_beam_width

        # Apply hardcoded defaults for any parameters still unset
        if min_prob is _UNSET:
            min_prob = 1e-10
        if beam_width is _UNSET:
            beam_width = 10
        if emission_weight is _UNSET:
            emission_weight = 1.2
        if lambda_unigram is _UNSET:
            lambda_unigram = 0.1
        if lambda_bigram is _UNSET:
            lambda_bigram = 0.3
        if lambda_trigram is _UNSET:
            lambda_trigram = 0.6
        if min_beam_width is _UNSET:
            min_beam_width = 5
        if max_beam_width is _UNSET:
            max_beam_width = 20

        # Store config for adaptive beam threshold lookups
        self._config = config

        self.provider = provider
        self.pos_bigram_probs = pos_bigram_probs
        self.pos_trigram_probs = pos_trigram_probs
        self.pos_unigram_probs = pos_unigram_probs or {}
        self.unknown_word_tag = unknown_word_tag
        self.min_prob = min_prob
        self.log_min_prob = math.log(min_prob)
        self.beam_width = beam_width
        self.emission_weight = emission_weight
        self.use_morphology_fallback = use_morphology_fallback

        # Adaptive beam width configuration
        self.adaptive_beam = adaptive_beam
        self.min_beam_width = min_beam_width
        self.max_beam_width = max_beam_width

        # Normalize interpolation weights
        weight_sum = lambda_unigram + lambda_bigram + lambda_trigram
        if weight_sum > 0 and abs(weight_sum - 1.0) > 1e-6:
            # Warn before automatic normalization
            warnings.warn(
                f"lambda_unigram + lambda_bigram + lambda_trigram = {weight_sum:.4f} "
                f"(expected 1.0). Weights will be normalized automatically.",
                UserWarning,
                stacklevel=2,
            )
            lambda_unigram /= weight_sum
            lambda_bigram /= weight_sum
            lambda_trigram /= weight_sum

        # Ensure lambda values are positive to prevent math.log(0) domain error
        # Use min_prob as floor since it's already a safe small positive value
        # This also handles the edge case where all lambdas are 0
        lambda_unigram = max(lambda_unigram, min_prob)
        lambda_bigram = max(lambda_bigram, min_prob)
        lambda_trigram = max(lambda_trigram, min_prob)

        # Re-normalize after flooring to ensure weights sum close to 1.0
        # This prevents probability distortion in deleted interpolation
        floored_sum = lambda_unigram + lambda_bigram + lambda_trigram
        lambda_unigram = lambda_unigram / floored_sum
        lambda_bigram = lambda_bigram / floored_sum
        lambda_trigram = lambda_trigram / floored_sum

        # Re-apply floor after normalization to handle floating-point precision
        # When one lambda is tiny and others are large, division can push the
        # tiny value slightly below min_prob due to floating-point arithmetic
        lambda_unigram = max(lambda_unigram, min_prob)
        lambda_bigram = max(lambda_bigram, min_prob)
        lambda_trigram = max(lambda_trigram, min_prob)

        # Renormalize after flooring so weights still sum to 1.0
        final_sum = lambda_unigram + lambda_bigram + lambda_trigram
        if final_sum > 0:
            self.lambda_unigram = lambda_unigram / final_sum
            self.lambda_bigram = lambda_bigram / final_sum
            self.lambda_trigram = lambda_trigram / final_sum
        else:
            # All zero (impossible with default min_prob>0, but guard anyway)
            self.lambda_unigram = 1 / 3
            self.lambda_bigram = 1 / 3
            self.lambda_trigram = 1 / 3

        # Morphology analyzer for OOV word tagging
        self.morphology_analyzer = MorphologyAnalyzer() if use_morphology_fallback else None

        # Word-level emission probabilities P(tag|word)
        # Currently empty - emission scores fall back to unigram probabilities
        self.word_tag_probs: dict[str, dict[str, float]] = {}

        # Emission score cache: Cache computed scores for word-tag pairs
        # This avoids redundant dictionary lookups when the same word appears
        # multiple times in a sequence.
        # Uses LRUCache from utils/cache.py for proper LRU eviction and thread safety.
        emission_cache_size = (
            config.viterbi_emission_cache_size if config is not None else EMISSION_CACHE_SIZE
        )
        self._emission_cache: LRUCache[float] = LRUCache(maxsize=emission_cache_size)

        # Transition probability cache: Cache smoothed log probabilities
        # Avoids repeated log() and log-sum-exp computations for common transitions.
        # Uses LRUCache for proper LRU eviction instead of silent stop-caching.
        transition_cache_size = (
            config.viterbi_transition_cache_size if config is not None else TRANSITION_CACHE_SIZE
        )
        self._bigram_cache: LRUCache[float] = LRUCache(maxsize=transition_cache_size)
        self._trigram_cache: LRUCache[float] = LRUCache(maxsize=transition_cache_size)

        # Cython tagger with deleted interpolation smoothing
        if _HAS_CYTHON_VITERBI:
            self._cython_tagger = viterbi_c.CythonViterbiTagger(
                provider,
                pos_bigram_probs,
                pos_trigram_probs,
                unknown_word_tag,
                min_prob,
                pos_unigram_probs=self.pos_unigram_probs,
                word_tag_probs=self.word_tag_probs,
                emission_weight=emission_weight,
                beam_width=beam_width,
                lambda_unigram=lambda_unigram,
                lambda_bigram=lambda_bigram,
                lambda_trigram=lambda_trigram,
            )
        else:
            self._cython_tagger = None  # Use Python implementation

    def _get_emission_score(self, word: str, tag: str) -> float:
        """
        Get emission score for a word-tag pair.

        Uses word-level P(tag|word) if available, otherwise falls back to P(tag).
        Results are cached using LRUCache to avoid redundant dictionary lookups
        when the same word appears multiple times in a sequence.
        """
        # Check cache first (use string key for LRUCache)
        cache_key = f"{word}:{tag}"
        cached_score = self._emission_cache.get(cache_key)
        if cached_score is not None:
            return cached_score

        # Calculate emission score
        score = 0.0

        # Try word-level emission first
        if word in self.word_tag_probs:
            prob = self.word_tag_probs[word].get(tag, self.min_prob)
            score = self.emission_weight * math.log(prob)
        # Fall back to tag unigram probability
        elif self.pos_unigram_probs:
            prob = self.pos_unigram_probs.get(tag, self.min_prob)
            score = self.emission_weight * math.log(prob)
        else:
            # Fallback when no probability data available (empty pos_unigram_probs)
            # Use uniform distribution over all tags (semantically correct: no preference)
            # This prevents silent degradation to 0.0 emission scores for all tags
            score = self.emission_weight * self.log_min_prob

        # Cache the result (LRUCache handles eviction automatically)
        self._emission_cache.set(cache_key, score)

        return score

    def _log_sum_exp(self, log_values: list[float]) -> float:
        """
        Compute log(sum(exp(x))) with numerical stability.

        Uses the identity: log(sum(exp(x_i))) = max_x + log(sum(exp(x_i - max_x)))

        Time Complexity: O(n) where n=len(log_values)
        Space Complexity: O(1) auxiliary space

        Note:
            This avoids underflow/overflow that would occur with naive
            log(sum(exp(x))) computation for large/small values.
        """
        if not log_values:
            return float("-inf")
        max_val = max(log_values)
        if max_val == float("-inf"):
            return float("-inf")
        return max_val + math.log(sum(math.exp(x - max_val) for x in log_values))

    def _get_smoothed_bigram_prob(self, tag_prev: str, tag_curr: str) -> float:
        """
        Get smoothed bigram transition probability using deleted interpolation.

        P_smooth(t2|t1) = λ2 * P(t2|t1) + λ1 * P(t2)

        Returns log probability: log(λ2 * exp(log_bigram) + λ1 * exp(log_unigram))

        Results are cached using LRUCache to avoid repeated log computations.
        """
        # Check cache first (use string key for LRUCache)
        cache_key = f"{tag_prev}:{tag_curr}"
        cached_result = self._bigram_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Get bigram probability (use raw probability, convert to log)
        bigram_prob = self.pos_bigram_probs.get((tag_prev, tag_curr), self.min_prob)
        log_bigram = math.log(bigram_prob)

        # Get unigram probability for backoff
        unigram_prob = self.pos_unigram_probs.get(tag_curr, self.min_prob)
        log_unigram = math.log(unigram_prob)

        # Deleted interpolation in log space:
        # log(λ2 * P_bigram + λ1 * P_unigram)
        # = log(exp(log(λ2) + log_bigram) + exp(log(λ1) + log_unigram))
        log_lambda_bigram = math.log(self.lambda_bigram)
        log_lambda_unigram = math.log(self.lambda_unigram)

        result = self._log_sum_exp(
            [
                log_lambda_bigram + log_bigram,
                log_lambda_unigram + log_unigram,
            ]
        )

        # Cache the result (LRUCache handles eviction automatically)
        self._bigram_cache.set(cache_key, result)

        return result

    def _get_smoothed_trigram_prob(self, tag_prev_prev: str, tag_prev: str, tag_curr: str) -> float:
        """
        Get smoothed trigram transition probability using deleted interpolation.

        P_smooth(t3|t1,t2) = λ3 * P(t3|t1,t2) + λ2 * P(t3|t2) + λ1 * P(t3)

        Returns log probability: log(λ3*exp(log_tri) + λ2*exp(log_bi) + λ1*exp(log_uni))

        Results are cached using LRUCache to avoid repeated log computations.
        """
        # Check cache first (use string key for LRUCache)
        cache_key = f"{tag_prev_prev}:{tag_prev}:{tag_curr}"
        cached_result = self._trigram_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Get trigram probability
        trigram_prob = self.pos_trigram_probs.get(
            (tag_prev_prev, tag_prev, tag_curr), self.min_prob
        )
        log_trigram = math.log(trigram_prob)

        # Get bigram probability for backoff
        bigram_prob = self.pos_bigram_probs.get((tag_prev, tag_curr), self.min_prob)
        log_bigram = math.log(bigram_prob)

        # Get unigram probability for further backoff
        unigram_prob = self.pos_unigram_probs.get(tag_curr, self.min_prob)
        log_unigram = math.log(unigram_prob)

        # Deleted interpolation in log space using log-sum-exp
        log_lambda_trigram = math.log(self.lambda_trigram)
        log_lambda_bigram = math.log(self.lambda_bigram)
        log_lambda_unigram = math.log(self.lambda_unigram)

        result = self._log_sum_exp(
            [
                log_lambda_trigram + log_trigram,
                log_lambda_bigram + log_bigram,
                log_lambda_unigram + log_unigram,
            ]
        )

        # Apply probability floor to prevent extreme -inf propagation
        # This ensures very rare tag sequences still have minimal probability
        # rather than completely eliminating paths with -inf
        result = max(result, self.log_min_prob * 3)

        # Cache the result (LRUCache handles eviction automatically)
        self._trigram_cache.set(cache_key, result)

        return result

    def tag_sequence(self, words: list[str]) -> list[str]:
        """
        Tag a sequence of words with their most likely POS tags.

        Uses the Viterbi algorithm with beam pruning to find the optimal
        tag sequence under the trigram HMM model.

        Args:
            words: List of words to tag

        Returns:
            List of POS tags, one per word

        Time Complexity:
            O(N * B² * T) where N=len(words), B=beam_width, T=avg tags per word.
            With Cython: ~2-5x faster due to optimized inner loops.

        Space Complexity:
            O(N * B) for Viterbi and backpointer tables.

        Note:
            Uses Cython implementation if available (_HAS_CYTHON_VITERBI=True).
            Falls back to pure Python implementation otherwise.
        """
        if _HAS_CYTHON_VITERBI and self._cython_tagger:
            result: list[str] = self._cython_tagger.tag_sequence(words)
            return result
        else:
            # Fallback to pure Python implementation
            if not words:
                return []

            # Check if we have probabilities to work with
            if not self.pos_bigram_probs:
                return [self.unknown_word_tag] * len(words)

            # 1. Initialization for t = 0 (first word)
            viterbi: list[dict[tuple[str, str], float]] = []
            backpointer: list[dict[tuple[str, str], str]] = []

            first_word_tags = self._get_valid_tags(words[0])
            initial_step_viterbi = {}
            initial_step_backpointer = {}

            for tag_0 in first_word_tags:
                # Add emission score for first word tag
                emission_score = self._get_emission_score(words[0], tag_0)
                initial_step_viterbi[(self.unknown_word_tag, tag_0)] = emission_score
                initial_step_backpointer[(self.unknown_word_tag, tag_0)] = self.unknown_word_tag

            viterbi.append(initial_step_viterbi)
            backpointer.append(initial_step_backpointer)

            # 2. Initialization for t = 1 (second word)
            if len(words) > 1:
                second_word_tags = self._get_valid_tags(words[1])
                second_step_viterbi = {}
                second_step_backpointer = {}

                for tag_1 in second_word_tags:
                    max_prob = -float("inf")
                    best_prev_tag_0 = None

                    # Emission score for second word tag
                    emission_score = self._get_emission_score(words[1], tag_1)

                    for tag_0 in first_word_tags:
                        # Use smoothed bigram probability (deleted interpolation)
                        log_trans_prob = self._get_smoothed_bigram_prob(tag_0, tag_1)

                        # Use stored viterbi value from t=0 instead of
                        # recalculating emission. This is consistent with t>1
                        # and avoids potential double-counting issues.
                        prev_path_prob = initial_step_viterbi.get(
                            (self.unknown_word_tag, tag_0), -float("inf")
                        )
                        prob = prev_path_prob + log_trans_prob + emission_score

                        if prob > max_prob:
                            max_prob = prob
                            best_prev_tag_0 = tag_0

                    if best_prev_tag_0 is not None:
                        second_step_viterbi[(best_prev_tag_0, tag_1)] = max_prob
                        second_step_backpointer[(best_prev_tag_0, tag_1)] = self.unknown_word_tag

                viterbi.append(second_step_viterbi)
                backpointer.append(second_step_backpointer)

            # 3. Recursion for t > 1 (third word onwards)
            for t in range(2, len(words)):
                curr_word = words[t]
                curr_tags = self._get_valid_tags(curr_word)

                prev_states = viterbi[t - 1].keys()

                current_step_viterbi: dict[tuple[str, str], float] = {}
                current_step_backpointer: dict[tuple[str, str], str] = {}

                for curr_tag in curr_tags:
                    max_prob_for_curr_state = -float("inf")
                    best_prev_state_for_curr = None

                    # Emission score for current word-tag
                    emission_score = self._get_emission_score(curr_word, curr_tag)

                    for tag_prev_prev, tag_prev in prev_states:
                        # Use smoothed trigram probability (deleted interpolation)
                        log_trans_prob = self._get_smoothed_trigram_prob(
                            tag_prev_prev, tag_prev, curr_tag
                        )

                        prev_path_prob = viterbi[t - 1].get(
                            (tag_prev_prev, tag_prev), -float("inf")
                        )

                        # Add emission score to total probability
                        prob = prev_path_prob + log_trans_prob + emission_score

                        if prob > max_prob_for_curr_state:
                            max_prob_for_curr_state = prob
                            best_prev_state_for_curr = (tag_prev_prev, tag_prev)

                    if (
                        max_prob_for_curr_state != -float("inf")
                        and best_prev_state_for_curr is not None
                    ):
                        state_key = (best_prev_state_for_curr[1], curr_tag)
                        # Only update if this path has higher probability than existing
                        # This prevents losing the optimal path when multiple paths converge
                        if (
                            state_key not in current_step_viterbi
                            or max_prob_for_curr_state > current_step_viterbi[state_key]
                        ):
                            current_step_viterbi[state_key] = max_prob_for_curr_state
                            current_step_backpointer[state_key] = best_prev_state_for_curr[0]

                # Apply beam pruning to limit state space
                # Use adaptive beam width if enabled
                effective_beam_width = self._get_adaptive_beam_width(len(words))
                if len(current_step_viterbi) > effective_beam_width:
                    # Keep only top-k states by probability using heapq for efficiency
                    top_states = nlargest(
                        effective_beam_width, current_step_viterbi.items(), key=lambda x: x[1]
                    )
                    current_step_viterbi = dict(top_states)
                    # Prune backpointer to match
                    current_step_backpointer = {
                        k: current_step_backpointer[k] for k in current_step_viterbi.keys()
                    }

                viterbi.append(current_step_viterbi)
                backpointer.append(current_step_backpointer)

            # 4. Termination
            # Handle empty viterbi dictionary (can happen after beam pruning)
            if not viterbi[-1]:
                # Log fallback for debugging
                effective_beam = self._get_adaptive_beam_width(len(words))
                logger.debug(
                    "Beam pruning resulted in empty state space for %d words. "
                    "Returning all %s tags. Consider increasing beam_width "
                    "(effective: %d, configured: %d, adaptive: %s).",
                    len(words),
                    self.unknown_word_tag,
                    effective_beam,
                    self.beam_width,
                    self.adaptive_beam,
                )
                return [self.unknown_word_tag] * len(words)

            # Select best final state with tie-breaking
            # When multiple states have equal probabilities, break ties using:
            # 1. Primary: Viterbi path probability (higher is better)
            # 2. Secondary: Sum of unigram frequencies for both tags (prefer common tags)
            # 3. Tertiary: Alphabetical ordering for deterministic behavior
            def _tie_break_key(state: tuple[str, str]) -> tuple[float, float, str]:
                prob = viterbi[-1][state]
                # Sum unigram frequencies for tie-breaking (prefer common tag sequences)
                freq_sum = self.pos_unigram_probs.get(state[0], 0.0) + self.pos_unigram_probs.get(
                    state[1], 0.0
                )
                # Alphabetical ordering as final tie-breaker for determinism
                alpha_key = state[0] + state[1]
                return (prob, freq_sum, alpha_key)

            best_final_state = max(viterbi[-1], key=_tie_break_key)

            # 5. Backtracking
            # Handle simple cases explicitly for clarity
            if len(words) == 1:
                return [best_final_state[1]]

            if len(words) == 2:
                # For 2-word sequences, best_final_state = (tag_0, tag_1)
                # Return directly to avoid confusing backtracking logic
                return [best_final_state[0], best_final_state[1]]

            # General case: 3+ words
            best_path_tags: list[str] = [best_final_state[1]]
            curr_state = best_final_state

            for t in range(len(words) - 1, 0, -1):
                prev_tag_in_path = curr_state[0]
                best_path_tags.append(prev_tag_in_path)

                # Handle KeyError after beam pruning
                # After beam pruning, backpointer[t] may not contain curr_state
                if curr_state not in backpointer[t]:
                    # Fallback: use unknown_word_tag for remaining positions
                    remaining = t - 1
                    logger.debug(
                        f"Beam pruning fallback: state {curr_state} not in backpointer "
                        f"at t={t}, using '{self.unknown_word_tag}' for {remaining} positions"
                    )
                    best_path_tags.extend([self.unknown_word_tag] * remaining)
                    break

                prev_prev_tag = backpointer[t][curr_state]
                curr_state = (prev_prev_tag, prev_tag_in_path)

            return list(reversed(best_path_tags))

    def _get_valid_tags(self, word: str) -> set[str]:
        """
        Get valid POS tags for a word.

        Merges tags from multiple sources in order of priority:
        1. Provider (database) - most reliable
        2. Seed file emission probs - word-tag counts from training
        3. Morphological analysis - suffix-based guessing for OOV words
        4. Unknown tag - final fallback if nothing else matches

        Args:
            word: The word to find tags for

        Returns:
            Set of valid POS tags
        """
        tags: set[str] = set()

        # Add tags from provider (database)
        pos_str = self.provider.get_word_pos(word)
        if pos_str:
            tags.update(pos_str.split("|"))

        # Also add tags from seed file word_tag_probs
        if word in self.word_tag_probs:
            tags.update(self.word_tag_probs[word].keys())

        # Morphological fallback for OOV words (suffix-based POS guessing)
        if not tags and self.morphology_analyzer:
            morpho_tags = self.morphology_analyzer.guess_pos(word)
            if morpho_tags:
                tags.update(morpho_tags)

        # Final fallback to unknown_word_tag if no tags found
        if not tags:
            return {self.unknown_word_tag}

        return tags

    def compute_marginals(self, words: list[str]) -> list[dict[str, float]]:
        """Compute per-position POS tag marginal probabilities.

        For each word position, returns a dict mapping each possible tag to
        its approximate marginal probability P(tag_i | words). Uses the
        Viterbi lattice scores with log-sum-exp normalization.

        This is an approximation of true forward-backward marginals using
        the Viterbi scores (which overcount due to max vs sum). For a
        spell checker, this approximation is sufficient to distinguish
        high-confidence tags (>0.8) from ambiguous ones (<0.5).

        Args:
            words: List of words to tag.

        Returns:
            List of dicts, one per word position. Each dict maps tag strings
            to probability floats summing to ~1.0.
        """
        if not words or not self.pos_bigram_probs:
            return [{self.unknown_word_tag: 1.0}] * len(words)

        # Run the full Viterbi forward pass to get the lattice
        # Reuse tag_sequence logic but capture the viterbi table
        viterbi_table: list[dict[tuple[str, str], float]] = []

        # t=0: initialization
        first_word_tags = self._get_valid_tags(words[0])
        initial_step: dict[tuple[str, str], float] = {}
        for tag_0 in first_word_tags:
            emission_score = self._get_emission_score(words[0], tag_0)
            initial_step[(self.unknown_word_tag, tag_0)] = emission_score
        viterbi_table.append(initial_step)

        # t=1: second word — build one state per (tag_0, tag_1) pair
        if len(words) > 1:
            second_word_tags = self._get_valid_tags(words[1])
            second_step: dict[tuple[str, str], float] = {}
            for tag_1 in second_word_tags:
                emission_score = self._get_emission_score(words[1], tag_1)
                for tag_0 in first_word_tags:
                    log_trans = self._get_smoothed_bigram_prob(tag_0, tag_1)
                    prev_prob = initial_step.get((self.unknown_word_tag, tag_0), float("-inf"))
                    score = prev_prob + log_trans + emission_score
                    if score > float("-inf"):
                        second_step[(tag_0, tag_1)] = score
            viterbi_table.append(second_step)

        # t>1: recursion
        for t in range(2, len(words)):
            curr_tags = self._get_valid_tags(words[t])
            prev_states = viterbi_table[t - 1]
            current_step: dict[tuple[str, str], float] = {}

            for curr_tag in curr_tags:
                emission_score = self._get_emission_score(words[t], curr_tag)
                # Group by prev_tag for the (prev_tag, curr_tag) state
                prev_tag_scores: dict[str, list[float]] = {}
                for (pp_tag, p_tag), p_score in prev_states.items():
                    log_trans = self._get_smoothed_trigram_prob(pp_tag, p_tag, curr_tag)
                    score = p_score + log_trans + emission_score
                    prev_tag_scores.setdefault(p_tag, []).append(score)

                for p_tag, scores in prev_tag_scores.items():
                    current_step[(p_tag, curr_tag)] = self._log_sum_exp(scores)

            viterbi_table.append(current_step)

        # Extract marginals from the lattice
        marginals: list[dict[str, float]] = []
        for t in range(len(words)):
            tag_log_probs: dict[str, list[float]] = {}
            for (_, curr_tag), score in viterbi_table[t].items():
                tag_log_probs.setdefault(curr_tag, []).append(score)

            # Log-sum-exp per tag, then normalize
            tag_scores: dict[str, float] = {}
            for tag, scores in tag_log_probs.items():
                tag_scores[tag] = self._log_sum_exp(scores)

            # Normalize to probabilities
            total = self._log_sum_exp(list(tag_scores.values()))
            probs: dict[str, float] = {}
            for tag, log_score in tag_scores.items():
                prob = math.exp(log_score - total) if total > float("-inf") else 0.0
                probs[tag] = prob

            marginals.append(probs)

        return marginals

    def cache_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics for all caches.

        Returns:
            Dictionary with stats for emission, bigram, and trigram caches.
            Each cache stats include: size, maxsize, hits, misses, hit_rate.
        """
        return {
            "emission": self._emission_cache.stats(),
            "bigram": self._bigram_cache.stats(),
            "trigram": self._trigram_cache.stats(),
        }

    def _get_adaptive_beam_width(self, sequence_length: int) -> int:
        """
        Calculate adaptive beam width based on sequence length.

        Strategy:
        - Short sequences (≤5 words): Use max_beam_width for best accuracy
        - Medium sequences (6-20 words): Linearly interpolate
        - Long sequences (>20 words): Use min_beam_width for speed

        This trade-off is based on the observation that:
        - Short sequences can afford wider beams (few positions)
        - Long sequences benefit from narrower beams (prevents state explosion)

        Args:
            sequence_length: Number of words in the sequence

        Returns:
            Adjusted beam width
        """
        if not self.adaptive_beam:
            return self.beam_width

        # Resolve thresholds from config or use defaults
        short_threshold = (
            self._config.viterbi_short_sequence_threshold if self._config is not None else 5
        )
        long_threshold = (
            self._config.viterbi_long_sequence_threshold if self._config is not None else 20
        )

        # Short sequences: use maximum beam for best accuracy
        if sequence_length <= short_threshold:
            return self.max_beam_width

        # Long sequences: use minimum beam for speed
        if sequence_length > long_threshold:
            return self.min_beam_width

        # Medium sequences: linear interpolation
        # As length increases from short to long threshold, beam decreases from max to min
        span = float(long_threshold - short_threshold) or 1.0
        ratio = (sequence_length - short_threshold) / span
        beam = int(self.max_beam_width - ratio * (self.max_beam_width - self.min_beam_width))
        return max(self.min_beam_width, min(self.max_beam_width, beam))
