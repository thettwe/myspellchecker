"""Smoothing functions for N-gram probability estimation.

This private helper module extracts smoothing logic from
``NgramContextChecker`` into standalone functions. Each function
receives the provider and the configuration values it needs as
explicit parameters, avoiding hidden coupling to the checker instance.

The ``SmoothingStrategy`` enum is defined here and re-exported by
``ngram_context_checker`` for backward compatibility.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myspellchecker.providers import DictionaryProvider


class SmoothingStrategy(Enum):
    """
    Smoothing strategies for N-gram probability estimation.

    Different strategies trade off between simplicity, accuracy, and
    computational cost.
    """

    NONE = "none"
    """No smoothing - returns raw probabilities."""

    STUPID_BACKOFF = "stupid_backoff"
    """
    Stupid Backoff (default): Simple, fast, works well in practice.
    For unseen n-grams, backs off to (n-1)-gram with a discount factor.
    P_backoff(w_n | w_{1..n-1}) = alpha * P(w_n | w_{2..n-1})
    """

    ADD_K = "add_k"
    """
    Add-k smoothing: Adds a constant offset k to all probabilities.
    Simple but tends to oversmooth, especially for large vocabularies.
    """


def get_smoothed_bigram_probability(
    provider: DictionaryProvider,
    word1: str,
    word2: str,
    *,
    use_smoothing: bool,
    strategy: SmoothingStrategy,
    add_k: float,
    backoff_weight: float,
    unigram_denominator: float,
    unigram_prob_cap: float,
) -> float:
    """Get bigram probability with configurable smoothing strategy.

    If the bigram is unseen (P=0), applies the configured smoothing strategy
    to estimate a probability.

    Smoothing Strategies:
    - NONE: Return raw probability
    - STUPID_BACKOFF: Backoff to alpha * P(unigram)
    - ADD_K: Add constant k to probability

    Args:
        provider: DictionaryProvider for n-gram data access.
        word1: First word (context).
        word2: Second word (target).
        use_smoothing: Whether smoothing is enabled.
        strategy: Which smoothing strategy to apply.
        add_k: Constant k for add-k smoothing.
        backoff_weight: Discount factor for stupid backoff.
        unigram_denominator: Total word count for unigram probability estimation.
        unigram_prob_cap: Maximum unigram probability.

    Returns:
        Smoothed probability (never returns 0 if smoothing is enabled).
    """
    bigram_prob = provider.get_bigram_probability(word1, word2)

    if not use_smoothing or strategy == SmoothingStrategy.NONE:
        return bigram_prob

    if strategy == SmoothingStrategy.ADD_K:
        return bigram_prob + add_k

    # Default: STUPID_BACKOFF
    if bigram_prob > 0:
        return bigram_prob

    # Stupid Backoff: use unigram probability with discount
    word_freq = provider.get_word_frequency(word2)
    if word_freq > 0:
        unigram_prob = min(word_freq / unigram_denominator, unigram_prob_cap)
        return backoff_weight * unigram_prob + add_k

    return add_k


def get_smoothed_trigram_probability(
    provider: DictionaryProvider,
    word1: str,
    word2: str,
    word3: str,
    *,
    use_smoothing: bool,
    strategy: SmoothingStrategy,
    add_k: float,
    backoff_weight: float,
    unigram_denominator: float,
    unigram_prob_cap: float,
    trigram_threshold: float,
    backoff_floor_multiplier: float,
) -> float:
    """Get trigram probability with configurable smoothing strategy.

    Backoff Chain (for STUPID_BACKOFF):
    1. Try exact trigram P(w3|w1,w2)
    2. If unseen, backoff: alpha * P(w3|w2)
    3. If bigram unseen, backoff: alpha^2 * P(w3)

    Args:
        provider: DictionaryProvider for n-gram data access.
        word1: First word.
        word2: Second word.
        word3: Third word (target).
        use_smoothing: Whether smoothing is enabled.
        strategy: Which smoothing strategy to apply.
        add_k: Constant k for add-k smoothing.
        backoff_weight: Discount factor for stupid backoff.
        unigram_denominator: Total word count for unigram probability estimation.
        unigram_prob_cap: Maximum unigram probability.
        trigram_threshold: Threshold for trigram floor.
        backoff_floor_multiplier: Multiplier for backoff floor.

    Returns:
        Smoothed probability.
    """
    trigram_prob = provider.get_trigram_probability(word1, word2, word3)

    if not use_smoothing or strategy == SmoothingStrategy.NONE:
        return trigram_prob

    if strategy == SmoothingStrategy.ADD_K:
        return trigram_prob + add_k

    # Default: STUPID_BACKOFF
    if trigram_prob > 0:
        return trigram_prob

    # First backoff: use bigram with discount
    bigram_prob = provider.get_bigram_probability(word2, word3)
    if bigram_prob > 0:
        return backoff_weight * bigram_prob + add_k

    # Second backoff: use unigram with double discount
    word_freq = provider.get_word_frequency(word3)
    if word_freq > 0:
        unigram_prob = min(word_freq / unigram_denominator, unigram_prob_cap)
        backoff_prob = (backoff_weight**2) * unigram_prob + add_k
        # Floor at configured % of trigram threshold to prevent false positives
        return max(backoff_prob, trigram_threshold * backoff_floor_multiplier)

    return add_k


def get_smoothed_fourgram_probability(
    provider: DictionaryProvider,
    word1: str,
    word2: str,
    word3: str,
    word4: str,
    *,
    use_smoothing: bool,
    strategy: SmoothingStrategy,
    add_k: float,
    backoff_weight: float,
    unigram_denominator: float,
    unigram_prob_cap: float,
    fourgram_threshold: float,
    backoff_floor_multiplier: float,
) -> float:
    """Get 4-gram probability with Stupid Backoff.

    Backoff Chain:
    1. Try exact 4-gram P(w4|w1,w2,w3)
    2. If unseen, backoff: alpha * P(w4|w2,w3)  [trigram]
    3. If trigram unseen, backoff: alpha^2 * P(w4|w3)  [bigram]
    4. If bigram unseen, backoff: alpha^3 * P(w4)  [unigram]
    """
    fourgram_prob = provider.get_fourgram_probability(word1, word2, word3, word4)

    if not use_smoothing or strategy == SmoothingStrategy.NONE:
        return fourgram_prob

    if strategy == SmoothingStrategy.ADD_K:
        return fourgram_prob + add_k

    if fourgram_prob > 0:
        return fourgram_prob

    # First backoff: trigram
    trigram_prob = provider.get_trigram_probability(word2, word3, word4)
    if trigram_prob > 0:
        return backoff_weight * trigram_prob + add_k

    # Second backoff: bigram
    bigram_prob = provider.get_bigram_probability(word3, word4)
    if bigram_prob > 0:
        return (backoff_weight**2) * bigram_prob + add_k

    # Third backoff: unigram with floor
    word_freq = provider.get_word_frequency(word4)
    if word_freq > 0:
        unigram_prob = min(word_freq / unigram_denominator, unigram_prob_cap)
        backoff_prob = (backoff_weight**3) * unigram_prob + add_k
        return max(backoff_prob, fourgram_threshold * backoff_floor_multiplier)

    return add_k


def get_smoothed_fivegram_probability(
    provider: DictionaryProvider,
    word1: str,
    word2: str,
    word3: str,
    word4: str,
    word5: str,
    *,
    use_smoothing: bool,
    strategy: SmoothingStrategy,
    add_k: float,
    backoff_weight: float,
    unigram_denominator: float,
    unigram_prob_cap: float,
    fivegram_threshold: float,
    backoff_floor_multiplier: float,
) -> float:
    """Get 5-gram probability with Stupid Backoff.

    Backoff Chain:
    1. Try exact 5-gram P(w5|w1,w2,w3,w4)
    2. If unseen, backoff: alpha * P(w5|w2,w3,w4)  [4-gram]
    3. If 4-gram unseen, backoff: alpha^2 * P(w5|w3,w4)  [trigram]
    4. If trigram unseen, backoff: alpha^3 * P(w5|w4)  [bigram]
    5. If bigram unseen, backoff: alpha^4 * P(w5)  [unigram]
    """
    fivegram_prob = provider.get_fivegram_probability(word1, word2, word3, word4, word5)

    if not use_smoothing or strategy == SmoothingStrategy.NONE:
        return fivegram_prob

    if strategy == SmoothingStrategy.ADD_K:
        return fivegram_prob + add_k

    if fivegram_prob > 0:
        return fivegram_prob

    # First backoff: 4-gram
    fourgram_prob = provider.get_fourgram_probability(word2, word3, word4, word5)
    if fourgram_prob > 0:
        return backoff_weight * fourgram_prob + add_k

    # Second backoff: trigram
    trigram_prob = provider.get_trigram_probability(word3, word4, word5)
    if trigram_prob > 0:
        return (backoff_weight**2) * trigram_prob + add_k

    # Third backoff: bigram
    bigram_prob = provider.get_bigram_probability(word4, word5)
    if bigram_prob > 0:
        return (backoff_weight**3) * bigram_prob + add_k

    # Fourth backoff: unigram with floor
    word_freq = provider.get_word_frequency(word5)
    if word_freq > 0:
        unigram_prob = min(word_freq / unigram_denominator, unigram_prob_cap)
        backoff_prob = (backoff_weight**4) * unigram_prob + add_k
        return max(backoff_prob, fivegram_threshold * backoff_floor_multiplier)

    return add_k


def get_best_left_probability(
    provider: DictionaryProvider,
    prev_words: list[str],
    word: str,
    *,
    add_k: float,
    get_smoothed_bigram: Callable[[str, str], float],
) -> float:
    """Get best available left-context probability with 4-gram -> trigram -> bigram fallback.

    Uses the highest-order n-gram available:
    - 4-gram P(word | prev[-3], prev[-2], prev[-1]) when prev_words has 3+ entries
    - Trigram P(word | prev[-2], prev[-1]) when prev_words has 2+ entries
    - Bigram  P(word | prev[-1])           when only one prev word available

    Note: 4-gram and trigram hits use raw provider probabilities (+ add_k)
    rather than the full smoothing chain.  This is intentional -- a nonzero
    hit at 4-gram/trigram is already strong evidence, and the raw value is
    better for relative ranking across candidates.

    Args:
        provider: DictionaryProvider for n-gram data access.
        prev_words: Left context words, ordered oldest-first.
        word: Candidate word being scored.
        add_k: Constant k for add-k smoothing.
        get_smoothed_bigram: Callable for smoothed bigram probability.

    Returns:
        Smoothed probability (higher = more likely given left context).
    """
    if len(prev_words) >= 3:
        four = provider.get_fourgram_probability(
            prev_words[-3], prev_words[-2], prev_words[-1], word
        )
        if four > 0:
            return four + add_k
    if len(prev_words) >= 2:
        tri = provider.get_trigram_probability(prev_words[-2], prev_words[-1], word)
        if tri > 0:
            return tri + add_k
    if prev_words:
        return get_smoothed_bigram(prev_words[-1], word)
    return 0.0


def get_best_right_probability(
    provider: DictionaryProvider,
    word: str,
    next_words: list[str],
    *,
    add_k: float,
    get_smoothed_bigram: Callable[[str, str], float],
) -> float:
    """Get best available right-context probability with 4-gram -> trigram -> bigram fallback.

    Uses the highest-order n-gram available:
    - 4-gram P(next[2] | word, next[0], next[1]) when next_words has 3+ entries
    - Trigram P(next[1] | word, next[0]) when next_words has 2+ entries
    - Bigram  P(next[0] | word)           when only one next word available

    Args:
        provider: DictionaryProvider for n-gram data access.
        word: Candidate word being scored.
        next_words: Right context words, ordered left-to-right.
        add_k: Constant k for add-k smoothing.
        get_smoothed_bigram: Callable for smoothed bigram probability.

    Returns:
        Smoothed probability (higher = more likely given right context).
    """
    if len(next_words) >= 3:
        four = provider.get_fourgram_probability(word, next_words[0], next_words[1], next_words[2])
        if four > 0:
            return four + add_k
    if len(next_words) >= 2:
        tri = provider.get_trigram_probability(word, next_words[0], next_words[1])
        if tri > 0:
            return tri + add_k
    if next_words:
        return get_smoothed_bigram(word, next_words[0])
    return 0.0
