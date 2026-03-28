"""
Joint Segmentation and POS Tagging for Myanmar Text.

This module implements a unified Viterbi decoder that simultaneously
finds optimal word boundaries AND POS tags in a single pass.

Benefits over sequential pipeline:
- Global optimization: considers word boundaries and tags jointly
- Better handling of ambiguous segmentations via POS context
- Single pass through the text (potential speedup)

State space: (position, word_start, current_tag, prev_tag)
Scoring: word_prob + tag_transition_prob + tag_emission_prob

Performance Characteristics:
    Time Complexity:
        - segment_and_tag(): O(N * M * B * T²) where N=text length,
          M=max_word_length, B=beam_width, T=tagset size.
        - With beam pruning: effectively O(N * M * B²) per position.

    Space Complexity:
        - DP table: O(N * B) states, each storing (score, backpointer)
        - Caches: O(cache_size) bounded by constants below

    Cache Behavior:
        - word_score_cache (8192 entries): Caches bigram word scores.
          Key: "word:prev_word", high hit rate for common word pairs.
        - transition_cache (2048 entries): Caches POS trigram scores.
          Bounded by tagset³, but typically much smaller in practice.
        - emission_cache (4096 entries): Caches word-tag emission scores.
        - valid_tags_cache (4096 entries): Caches valid tags per word.
          Stores frozensets to avoid redundant set construction.

Memory Considerations:
    - Joint state space is larger than separate segmentation + tagging
    - beam_width=15 (default) balances accuracy vs memory/speed
    - For very long texts, consider sentence-level processing
    - clear_cache() releases all cache memory

Performance Tuning:
    - beam_width: Lower (10) for speed, higher (20) for accuracy
    - max_word_length: 20 chars covers most Myanmar words
    - emission_weight: 1.2 (default) slightly favors emission evidence
    - word_score_weight: 1.0 (default) standard LM contribution
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from heapq import nlargest
from typing import Any

from myspellchecker.providers import DictionaryProvider
from myspellchecker.text.morphology import MorphologyAnalyzer
from myspellchecker.utils.cache import LRUCache
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Cache sizes for JointSegmentTagger
# Sized for typical document processing
WORD_SCORE_CACHE_SIZE = 8192  # Word bigram scores
TRANSITION_CACHE_SIZE = 2048  # POS trigram transitions
EMISSION_CACHE_SIZE = 4096  # Word-tag emission scores
VALID_TAGS_CACHE_SIZE = 4096  # Valid tags per word


@dataclass
class JointState:
    """State in the joint segmentation-tagging lattice."""

    word_start: int  # Character index where current word starts
    current_tag: str  # POS tag for current word
    prev_tag: str  # POS tag for previous word
    score: float  # Log probability score
    backpointer: JointState | None  # Previous state for backtracking


class JointSegmentTagger:
    """
    Joint word segmentation and POS tagging using unified Viterbi decoding.

    Combines word boundary detection (via n-gram language model) with
    POS tagging (via HMM) into a single optimization problem.

    Mathematical formulation:
        argmax P(words, tags | text)
        = argmax Π P(word_i) × P(tag_i | tag_{i-1}, tag_{i-2}) × P(tag_i | word_i)

    In log space:
        = argmax Σ [log P(word_i) + log P(tag_i | tags) + log P(tag_i | word_i)]

    Performance:
        - segment_and_tag() runs in O(N * M * B * T²) worst case
        - With beam pruning (default B=15), practical complexity is much lower
        - Four LRU caches reduce redundant lookups for common patterns

    Thread Safety:
        - Instance is NOT thread-safe (shared caches, DP tables)
        - Create separate instances for concurrent processing
        - Or use external synchronization

    Cache Management:
        - clear_cache(): Free all cache memory
        - cache_stats(): Get hit/miss statistics for each cache

    Example:
        >>> tagger = JointSegmentTagger(provider, bigram_probs, trigram_probs)
        >>> words, tags = tagger.segment_and_tag("မြန်မာနိုင်ငံ")
        >>> print(list(zip(words, tags)))
        [('မြန်မာ', 'N'), ('နိုင်ငံ', 'N')]
    """

    # Boundary tags for sequence start/end
    START_TAG = "<S>"
    END_TAG = "</S>"
    UNKNOWN_TAG = "UNK"

    def __init__(
        self,
        provider: DictionaryProvider,
        pos_bigram_probs: dict[tuple[str, str], float],
        pos_trigram_probs: dict[tuple[str, str, str], float],
        pos_unigram_probs: dict[str, float] | None = None,
        word_tag_probs: dict[str, dict[str, float]] | None = None,
        min_prob: float = 1e-10,
        max_word_length: int = 20,
        beam_width: int = 15,  # Larger beam for joint state space
        emission_weight: float = 1.2,
        word_score_weight: float = 1.0,
        use_morphology_fallback: bool = True,
        cache_config: Any | None = None,
    ):
        """
        Initialize JointSegmentTagger.

        Args:
            provider: Dictionary provider for word lookups and frequencies.
            pos_bigram_probs: P(tag | prev_tag) transition probabilities.
            pos_trigram_probs: P(tag | prev_prev_tag, prev_tag) trigram transitions.
            pos_unigram_probs: P(tag) prior probabilities for fallback.
            word_tag_probs: P(tag | word) emission probabilities.
            min_prob: Minimum probability for smoothing.
            max_word_length: Maximum word length in characters.
            beam_width: Number of states to keep per position (beam pruning).
            emission_weight: Weight for emission probabilities.
            word_score_weight: Weight for word n-gram scores.
            use_morphology_fallback: Use morphology for OOV word tagging.
            cache_config: Optional AlgorithmCacheConfig for configuring cache sizes.
        """
        self.provider = provider
        self.pos_bigram_probs = pos_bigram_probs
        self.pos_trigram_probs = pos_trigram_probs
        self.pos_unigram_probs = pos_unigram_probs or {}
        self.word_tag_probs = word_tag_probs or {}
        self.min_prob = min_prob
        self.log_min_prob = math.log(min_prob)
        self.max_word_length = max_word_length
        self.beam_width = beam_width
        self.emission_weight = emission_weight
        self.word_score_weight = word_score_weight
        self.use_morphology_fallback = use_morphology_fallback

        # Morphology analyzer for OOV words
        self.morphology_analyzer = MorphologyAnalyzer() if use_morphology_fallback else None

        # Cache sizes: use config if provided, otherwise module-level defaults
        _ws = WORD_SCORE_CACHE_SIZE
        _tr = TRANSITION_CACHE_SIZE
        _em = EMISSION_CACHE_SIZE
        _vt = VALID_TAGS_CACHE_SIZE
        if cache_config is not None:
            _ws = getattr(cache_config, "joint_word_score_cache_size", _ws)
            _tr = getattr(cache_config, "joint_transition_cache_size", _tr)
            _em = getattr(cache_config, "joint_emission_cache_size", _em)
            _vt = getattr(cache_config, "joint_valid_tags_cache_size", _vt)

        # Cache for word scores using LRUCache for proper eviction and thread safety.
        self._word_score_cache: LRUCache[float] = LRUCache(maxsize=_ws)

        # Pre-compute all valid tags from various sources
        self._all_tags = self._compute_all_tags()

        # Pre-computed transition scores cache using LRUCache for proper eviction.
        self._transition_cache: LRUCache[float] = LRUCache(maxsize=_tr)

        # Pre-computed emission scores cache using LRUCache for proper eviction.
        self._emission_cache: LRUCache[float] = LRUCache(maxsize=_em)

        # Valid tags per word cache using LRUCache for proper eviction.
        self._valid_tags_cache: LRUCache[frozenset] = LRUCache(maxsize=_vt)

        logger.debug(
            f"JointSegmentTagger initialized: "
            f"beam_width={beam_width}, "
            f"max_word_length={max_word_length}, "
            f"tags={len(self._all_tags)}"
        )

    def _compute_all_tags(self) -> set[str]:
        """Compute the set of all valid POS tags from available data."""
        tags = {self.START_TAG, self.END_TAG, self.UNKNOWN_TAG}

        # From unigram probabilities
        tags.update(self.pos_unigram_probs.keys())

        # From bigram probabilities
        for tag1, tag2 in self.pos_bigram_probs.keys():
            tags.add(tag1)
            tags.add(tag2)

        # From trigram probabilities
        for tag1, tag2, tag3 in self.pos_trigram_probs.keys():
            tags.add(tag1)
            tags.add(tag2)
            tags.add(tag3)

        # From word-tag emission probabilities
        for word_tags in self.word_tag_probs.values():
            tags.update(word_tags.keys())

        # Remove boundary tags from candidate set
        tags.discard(self.START_TAG)
        tags.discard(self.END_TAG)

        return tags

    def _get_word_score(self, word: str, prev_word: str) -> float:
        """
        Get word n-gram score: log P(word | prev_word).

        Uses bigram if available, falls back to unigram.
        Results are cached using LRUCache for proper eviction.

        Time Complexity: O(1) with cache hit, O(1) with cache miss (dict lookup)
        Cache: word_score_cache (8192 entries max)
        """
        cache_key = f"{word}:{prev_word}"
        cached_score = self._word_score_cache.get(cache_key)
        if cached_score is not None:
            return cached_score

        # Try bigram first
        bigram_prob = self.provider.get_bigram_probability(prev_word, word)
        if bigram_prob > self.min_prob:
            score = self.word_score_weight * math.log(bigram_prob)
        else:
            # Fallback to unigram
            freq = self.provider.get_word_frequency(word)
            if freq > 0:
                # Normalize by total frequency (approximate)
                score = self.word_score_weight * math.log(freq / 1e6)  # Rough normalization
            else:
                # Unknown word penalty (length-based)
                score = self.word_score_weight * (self.log_min_prob - len(word) * 0.5)

        self._word_score_cache.set(cache_key, score)
        return score

    def _get_tag_transition_score(self, tag: str, prev_tag: str, prev_prev_tag: str) -> float:
        """
        Get POS tag transition score: log P(tag | prev_prev_tag, prev_tag).

        Uses trigram if available, falls back to bigram, then unigram.
        Results are cached using LRUCache for proper eviction.

        Time Complexity: O(1) with cache hit, O(1) with cache miss (dict lookup)
        Cache: transition_cache (2048 entries max)

        Note:
            Backoff chain: trigram -> bigram -> unigram ensures non-zero
            probability for any tag sequence, even unseen trigrams.
        """
        # Check cache first (use string key for LRUCache)
        cache_key = f"{tag}:{prev_tag}:{prev_prev_tag}"
        cached_score = self._transition_cache.get(cache_key)
        if cached_score is not None:
            return cached_score

        # Try trigram
        trigram_prob = self.pos_trigram_probs.get((prev_prev_tag, prev_tag, tag), 0.0)
        if trigram_prob > self.min_prob:
            score = math.log(trigram_prob)
        else:
            # Fallback to bigram
            bigram_prob = self.pos_bigram_probs.get((prev_tag, tag), 0.0)
            if bigram_prob > self.min_prob:
                score = math.log(bigram_prob)
            else:
                # Fallback to unigram
                unigram_prob = self.pos_unigram_probs.get(tag, self.min_prob)
                score = math.log(unigram_prob)

        self._transition_cache.set(cache_key, score)
        return score

    def _get_emission_score(self, word: str, tag: str) -> float:
        """
        Get emission score: log P(tag | word).

        Uses word-level emissions if available, otherwise tag unigram prior.
        Results are cached using LRUCache for proper eviction.

        Time Complexity: O(1) with cache hit, O(1) with cache miss (dict lookup)
        Cache: emission_cache (4096 entries max)
        """
        # Check cache first (use string key for LRUCache)
        cache_key = f"{word}:{tag}"
        cached_score = self._emission_cache.get(cache_key)
        if cached_score is not None:
            return cached_score

        # Try word-specific emission
        if word in self.word_tag_probs:
            prob = self.word_tag_probs[word].get(tag, self.min_prob)
            score = self.emission_weight * math.log(prob)
        elif self.pos_unigram_probs:
            # Fallback to tag prior
            prob = self.pos_unigram_probs.get(tag, self.min_prob)
            score = self.emission_weight * math.log(prob)
        else:
            score = 0.0

        self._emission_cache.set(cache_key, score)
        return score

    def _get_valid_tags_for_word(self, word: str) -> set[str]:
        """
        Get valid POS tags for a word from all available sources.

        Results are cached using LRUCache for proper eviction.

        Time Complexity: O(T) with cache miss where T=number of tags for word.
                        O(1) with cache hit.
        Cache: valid_tags_cache (4096 entries max), stores frozensets.

        Sources (in priority order):
            1. Provider database (get_word_pos)
            2. Word-tag emission probabilities (word_tag_probs)
            3. Morphological analysis (for OOV words)
            4. UNKNOWN_TAG fallback
        """
        # Check cache first (word is already a string key)
        cached_tags = self._valid_tags_cache.get(word)
        if cached_tags is not None:
            return set(cached_tags)  # Convert frozenset back to set

        tags: set[str] = set()

        # From provider database
        pos_str = self.provider.get_word_pos(word)
        if pos_str:
            tags.update(pos_str.split("|"))

        # From word-tag emission probs
        if word in self.word_tag_probs:
            tags.update(self.word_tag_probs[word].keys())

        # Morphological fallback for OOV words
        if not tags and self.morphology_analyzer:
            morpho_tags = self.morphology_analyzer.guess_pos(word)
            if morpho_tags:
                tags.update(morpho_tags)

        # Final fallback
        if not tags:
            result = {self.UNKNOWN_TAG}
        else:
            result = tags

        # Store as frozenset for immutability in cache
        self._valid_tags_cache.set(word, frozenset(result))
        return result

    def segment_and_tag(self, text: str) -> tuple[list[str], list[str]]:
        """
        Perform joint word segmentation and POS tagging.

        Uses a unified Viterbi algorithm that optimizes both word
        boundaries and POS tags simultaneously.

        Args:
            text: Input Myanmar text to segment and tag.

        Returns:
            Tuple of (words, tags) where:
            - words: List of segmented words
            - tags: List of POS tags (same length as words)

        Time Complexity:
            O(N * M * B * T²) where N=len(text), M=max_word_length,
            B=beam_width, T=avg tags per word candidate.
            With beam pruning, effectively O(N * M * B²) in practice.

        Space Complexity:
            O(N * B) for the DP table, where each entry stores a dict
            mapping state tuples to (score, backpointer) pairs.

        Note:
            For very long texts (>1000 chars), consider processing
            sentence by sentence to bound memory usage.
        """
        if not text or not text.strip():
            return [], []

        text = text.strip()
        n = len(text)

        # Initialize DP table with start state
        dp = self._init_dp_table(n)

        # Forward pass: fill DP table
        self._forward_pass(text, n, dp)

        # Find best final state and backtrack
        if not dp[n]:
            logger.warning(f"Joint segmentation failed for text: {text[:50]}...")
            return [text], [self.UNKNOWN_TAG]

        return self._backtrack(text, n, dp)

    def _init_dp_table(
        self, n: int
    ) -> list[dict[tuple[int, str, str], tuple[float, tuple[int, str, str] | None]]]:
        """Initialize DP table with start state."""
        dp: list[dict[tuple[int, str, str], tuple[float, tuple[int, str, str] | None]]] = [
            {} for _ in range(n + 1)
        ]
        dp[0][(-1, self.START_TAG, self.START_TAG)] = (0.0, None)
        return dp

    def _forward_pass(
        self,
        text: str,
        n: int,
        dp: list[dict[tuple[int, str, str], tuple[float, tuple[int, str, str] | None]]],
    ) -> None:
        """Run forward pass of Viterbi algorithm."""
        for end_pos in range(1, n + 1):
            start_min = max(0, end_pos - self.max_word_length)

            for start_pos in range(start_min, end_pos):
                if not dp[start_pos]:
                    continue

                word = text[start_pos:end_pos]
                self._process_word_at_position(text, word, start_pos, end_pos, dp)

            self._prune_beam(dp, end_pos)

    def _process_word_at_position(
        self,
        text: str,
        word: str,
        start_pos: int,
        end_pos: int,
        dp: list[dict[tuple[int, str, str], tuple[float, tuple[int, str, str] | None]]],
    ) -> None:
        """Process a word candidate at given position."""
        valid_tags = self._get_valid_tags_for_word(word)

        for (prev_word_start, prev_tag, prev_prev_tag), (prev_score, _) in dp[start_pos].items():
            prev_word = text[prev_word_start:start_pos] if prev_word_start >= 0 else self.START_TAG
            word_score = self._get_word_score(word, prev_word)

            for tag in valid_tags:
                total_score = self._compute_total_score(
                    prev_score, word_score, word, tag, prev_tag, prev_prev_tag
                )
                new_state = (start_pos, tag, prev_tag)
                backptr = (prev_word_start, prev_tag, prev_prev_tag)

                if new_state not in dp[end_pos] or total_score > dp[end_pos][new_state][0]:
                    dp[end_pos][new_state] = (total_score, backptr)

    def _compute_total_score(
        self,
        prev_score: float,
        word_score: float,
        word: str,
        tag: str,
        prev_tag: str,
        prev_prev_tag: str,
    ) -> float:
        """Compute total score for a state transition."""
        transition_score = self._get_tag_transition_score(tag, prev_tag, prev_prev_tag)
        emission_score = self._get_emission_score(word, tag)
        return prev_score + word_score + transition_score + emission_score

    def _prune_beam(
        self,
        dp: list[dict[tuple[int, str, str], tuple[float, tuple[int, str, str] | None]]],
        end_pos: int,
    ) -> None:
        """
        Apply beam pruning to keep only top-k states.

        Time Complexity: O(S * log(B)) where S=number of states, B=beam_width.
                        Uses heapq.nlargest for efficient top-k selection.
        Space Complexity: O(B) for the pruned state dict.

        Note:
            Beam pruning is critical for tractability. Without it, state space
            grows exponentially with text length.
        """
        if len(dp[end_pos]) > self.beam_width:
            top_states = nlargest(
                self.beam_width,
                dp[end_pos].items(),
                key=lambda x: x[1][0],
            )
            dp[end_pos] = dict(top_states)

    def _backtrack(
        self,
        text: str,
        n: int,
        dp: list[dict[tuple[int, str, str], tuple[float, tuple[int, str, str] | None]]],
    ) -> tuple[list[str], list[str]]:
        """Backtrack through DP table to recover words and tags."""
        best_state = max(dp[n].items(), key=lambda x: x[1][0])
        best_final_state, (_, best_backptr) = best_state

        words: list[str] = []
        tags: list[str] = []

        current_end = n
        current_state = best_final_state
        current_backptr = best_backptr

        while current_state[0] >= 0:
            word_start, tag, _ = current_state
            words.append(text[word_start:current_end])
            tags.append(tag)

            current_end = word_start
            if current_backptr is None:
                break

            prev_word_start, prev_tag_actual, prev_prev_tag_actual = current_backptr
            current_state = (prev_word_start, prev_tag_actual, prev_prev_tag_actual)
            current_backptr = (
                dp[current_end].get(current_state, (None, None))[1] if current_end > 0 else None
            )

        words.reverse()
        tags.reverse()
        return words, tags

    def clear_cache(self) -> None:
        """Clear all internal caches."""
        self._word_score_cache.clear()
        self._transition_cache.clear()
        self._emission_cache.clear()
        self._valid_tags_cache.clear()

    def cache_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics for all caches.

        Returns:
            Dictionary with stats for all caches.
            Each cache stats include: size, maxsize, hits, misses, hit_rate.
        """
        return {
            "word_score": self._word_score_cache.stats(),
            "transition": self._transition_cache.stats(),
            "emission": self._emission_cache.stats(),
            "valid_tags": self._valid_tags_cache.stats(),
        }
