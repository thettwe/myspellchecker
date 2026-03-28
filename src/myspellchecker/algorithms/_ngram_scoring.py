"""Bidirectional probability computation and n-gram context checks.

This private helper module extracts the core probability scoring
and context detection logic from ``NgramContextChecker`` into
standalone functions.  Each function receives the provider and any
caches/thresholds as explicit parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myspellchecker.providers import DictionaryProvider


def compute_bidirectional_prob(
    provider: DictionaryProvider,
    word: str,
    prev_words: list[str],
    next_words: list[str],
    *,
    unigram_denominator: float,
    unigram_prob_cap: float,
    bidir_cache: dict[tuple[str, tuple[str, ...], tuple[str, ...]], float],
    bidir_cache_max_size: int,
) -> float:
    """Compute combined bidirectional n-gram probability for *word*.

    Uses the highest-order n-gram available in each direction and
    combines left and right probabilities. Falls back through the
    backoff chain: trigram -> bigram -> unigram (smoothed).

    The left probability is P(word | prev context) and the right
    probability is P(next context | word).  When both are available,
    they are averaged; when only one direction has data, that value
    is used alone.

    Args:
        provider: DictionaryProvider for n-gram data access.
        word: Target word to compute probability for.
        prev_words: Left context ``[..., prev_prev, prev]``.
        next_words: Right context ``[next, next_next, ...]``.
        unigram_denominator: Total word count for unigram probability estimation.
        unigram_prob_cap: Maximum unigram probability.
        bidir_cache: Sentence-level cache (mutated in place).
        bidir_cache_max_size: Maximum cache entries before clearing.

    Returns:
        Combined bidirectional probability (float >= 0).
    """
    cache_key = (word, tuple(prev_words), tuple(next_words))
    cached = bidir_cache.get(cache_key)
    if cached is not None:
        return cached

    result = _compute_bidirectional_prob_uncached(
        provider,
        word,
        prev_words,
        next_words,
        unigram_denominator=unigram_denominator,
        unigram_prob_cap=unigram_prob_cap,
    )

    # Bound cache size to prevent unbounded growth across many sentences.
    if len(bidir_cache) >= bidir_cache_max_size:
        bidir_cache.clear()
    bidir_cache[cache_key] = result
    return result


def _compute_bidirectional_prob_uncached(
    provider: DictionaryProvider,
    word: str,
    prev_words: list[str],
    next_words: list[str],
    *,
    unigram_denominator: float,
    unigram_prob_cap: float,
) -> float:
    """Uncached implementation of bidirectional probability computation."""
    left_prob = 0.0
    right_prob = 0.0
    has_left = False
    has_right = False

    # --- Left context (backward): P(word | prev...) ---
    if len(prev_words) >= 2:
        prev_prev = prev_words[-2]
        prev_word = prev_words[-1]
        tri = provider.get_trigram_probability(prev_prev, prev_word, word)
        if tri > 0:
            left_prob = tri
            has_left = True
        else:
            bi = provider.get_bigram_probability(prev_word, word)
            if bi > 0:
                left_prob = bi
                has_left = True
    elif len(prev_words) == 1:
        bi = provider.get_bigram_probability(prev_words[0], word)
        if bi > 0:
            left_prob = bi
            has_left = True

    # --- Right context (forward): P(next... | word) ---
    if len(next_words) >= 2:
        next_word = next_words[0]
        next_next = next_words[1]
        tri = provider.get_trigram_probability(word, next_word, next_next)
        if tri > 0:
            right_prob = tri
            has_right = True
        else:
            bi = provider.get_bigram_probability(word, next_word)
            if bi > 0:
                right_prob = bi
                has_right = True
    elif len(next_words) == 1:
        bi = provider.get_bigram_probability(word, next_words[0])
        if bi > 0:
            right_prob = bi
            has_right = True

    # Combine
    if has_left and has_right:
        return (left_prob + right_prob) / 2.0
    if has_left:
        return left_prob
    if has_right:
        return right_prob

    # No n-gram data at all -- fall back to smoothed unigram
    word_freq = provider.get_word_frequency(word)
    if isinstance(word_freq, (int, float)) and word_freq > 0:
        return min(word_freq / unigram_denominator, unigram_prob_cap)

    return 0.0


def has_ngram_context(
    provider: DictionaryProvider,
    word: str,
    prev_words: list[str],
    next_words: list[str],
    *,
    ngram_cache: dict[tuple[str, tuple[str, ...], tuple[str, ...]], bool],
    ngram_cache_max_size: int,
) -> bool:
    """Check if *word* has any real bigram/trigram context.

    Returns True if at least one bigram or trigram involving *word* and
    the surrounding context words has a nonzero probability. Returns
    False when the only available signal is the unigram frequency of
    the word itself.

    This is used by ``compare_contextual_probability`` to distinguish
    genuine contextual evidence from mere corpus frequency differences.
    """
    cache_key = (word, tuple(prev_words), tuple(next_words))
    cached = ngram_cache.get(cache_key)
    if cached is not None:
        return cached

    result = _has_ngram_context_uncached(provider, word, prev_words, next_words)

    if len(ngram_cache) >= ngram_cache_max_size:
        ngram_cache.clear()
    ngram_cache[cache_key] = result
    return result


def _has_ngram_context_uncached(
    provider: DictionaryProvider,
    word: str,
    prev_words: list[str],
    next_words: list[str],
) -> bool:
    """Uncached implementation of n-gram context check."""
    # --- Left context ---
    if len(prev_words) >= 2:
        prev_prev = prev_words[-2]
        prev_word = prev_words[-1]
        if provider.get_trigram_probability(prev_prev, prev_word, word) > 0:
            return True
        if provider.get_bigram_probability(prev_word, word) > 0:
            return True
    elif len(prev_words) == 1:
        if provider.get_bigram_probability(prev_words[0], word) > 0:
            return True

    # --- Right context ---
    if len(next_words) >= 2:
        next_word = next_words[0]
        next_next = next_words[1]
        if provider.get_trigram_probability(word, next_word, next_next) > 0:
            return True
        if provider.get_bigram_probability(word, next_word) > 0:
            return True
    elif len(next_words) == 1:
        if provider.get_bigram_probability(word, next_words[0]) > 0:
            return True

    return False


def has_ngram_context_directional(
    provider: DictionaryProvider,
    word: str,
    context_words: list[str],
    direction: str = "left",
) -> bool:
    """Check if *word* has n-gram context in a single direction.

    Args:
        provider: DictionaryProvider for n-gram data access.
        word: Target word to check.
        context_words: Context words for the given direction.
            For ``"left"``: ``[..., prev_prev, prev]`` (closest last).
            For ``"right"``: ``[next, next_next, ...]`` (closest first).
        direction: ``"left"`` or ``"right"``.

    Returns:
        True if at least one bigram or trigram in this direction
        has a nonzero probability.
    """
    if direction == "left":
        if len(context_words) >= 2:
            prev_prev = context_words[-2]
            prev_word = context_words[-1]
            if provider.get_trigram_probability(prev_prev, prev_word, word) > 0:
                return True
            if provider.get_bigram_probability(prev_word, word) > 0:
                return True
        elif len(context_words) == 1:
            if provider.get_bigram_probability(context_words[0], word) > 0:
                return True
    else:  # right
        if len(context_words) >= 2:
            next_word = context_words[0]
            next_next = context_words[1]
            if provider.get_trigram_probability(word, next_word, next_next) > 0:
                return True
            if provider.get_bigram_probability(word, next_word) > 0:
                return True
        elif len(context_words) == 1:
            if provider.get_bigram_probability(word, context_words[0]) > 0:
                return True
    return False
