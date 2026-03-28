"""Candidate scoring functions for N-gram suggestion generation.

This private helper module extracts candidate generation, scoring, and
POS-aware ranking logic from ``NgramContextChecker`` into standalone
functions.  Each function receives the provider and any configuration
values as explicit parameters.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from myspellchecker.algorithms.distance.edit_distance import damerau_levenshtein_distance

if TYPE_CHECKING:
    from myspellchecker.algorithms.symspell import SymSpell
    from myspellchecker.providers import DictionaryProvider

from myspellchecker.core.constants import ValidationLevel


def generate_candidates(
    provider: "DictionaryProvider",
    prev_word: str,
    current_word: str,
    max_edit_distance: int,
    *,
    candidate_limit: int,
    symspell: "SymSpell | None",
) -> list[str]:
    """Generate candidate words for context-aware suggestions.

    Uses a hybrid approach:
    1. Query high-probability continuations from bigram data
    2. Filter candidates by edit distance from current_word
    3. If SymSpell is available, add orthographically/phonetically similar words
    4. Return deduplicated candidate list

    Args:
        provider: DictionaryProvider for n-gram data access.
        prev_word: Previous word for context.
        current_word: Current word to find alternatives for.
        max_edit_distance: Maximum edit distance.
        candidate_limit: Maximum number of candidates per source.
        symspell: Optional SymSpell instance for orthographic/phonetic candidates.

    Returns:
        List of candidate words.
    """
    candidates: list[str] = []

    # 1. Contextual Candidates (N-gram continuations)
    continuations = provider.get_top_continuations(
        prev_word,
        limit=candidate_limit,
    )

    for word, _ in continuations:
        if word not in candidates:
            candidates.append(word)

    # 2. Orthographic/Phonetic Candidates (SymSpell)
    if symspell:
        sym_suggestions = symspell.lookup(
            current_word,
            level=ValidationLevel.WORD.value,
            max_suggestions=candidate_limit,
            include_known=True,
            use_phonetic=True,
        )
        for s in sym_suggestions:
            if s.term not in candidates:
                candidates.append(s.term)

    return candidates


def get_phonetic_variants(
    word: str,
    symspell: "SymSpell | None",
) -> set[str]:
    """Get phonetic variants for a word using SymSpell's phonetic hasher."""
    if symspell and symspell.phonetic_hasher:
        return symspell.phonetic_hasher.get_phonetic_variants(word)
    return set()


def score_candidate(
    provider: "DictionaryProvider",
    candidate: str,
    prev_word: str,
    current_word: str,
    current_prob: float,
    next_word: str | None,
    max_edit_distance: int,
    phonetic_variants: set[str],
    *,
    pos_bigram_probs: dict[tuple[str, str], float],
    pos_score_weight: float,
    pos_distance_reduction_factor_multiplier: float,
    probability_weight: float,
    edit_distance_weight: float,
    score_scaling_factor: float,
    min_prob_denom: float,
) -> tuple[str, float, int, float, float] | None:
    """Score a single candidate and return scoring data if valid.

    Returns None if the candidate should be filtered out.

    Returns:
        Tuple of (term, probability, edit_distance, score, confidence)
        or None.
    """
    # Skip if same as current word
    if candidate == current_word:
        return None

    # Get bigram probability (Left Context)
    prob_left = provider.get_bigram_probability(prev_word, candidate)

    # Only suggest if probability is higher than current
    if prob_left <= current_prob:
        return None

    # Combine left and right context probabilities
    prob = combine_context_probabilities(provider, prob_left, candidate, next_word)

    # Calculate edit distance and check phonetic equivalence
    distance = damerau_levenshtein_distance(current_word, candidate)
    is_phonetic = candidate in phonetic_variants

    # Skip if distance too large (allow phonetic matches to exceed)
    if distance > max_edit_distance and not is_phonetic:
        return None

    # Calculate effective distance (reduced for phonetic matches)
    effective_distance = 0.0 if is_phonetic else float(distance)

    # Apply POS context scoring to reduce effective distance
    effective_distance = apply_pos_context_scoring(
        provider,
        effective_distance,
        prev_word,
        candidate,
        next_word,
        pos_bigram_probs=pos_bigram_probs,
        pos_score_weight=pos_score_weight,
        pos_distance_reduction_factor_multiplier=pos_distance_reduction_factor_multiplier,
    )

    # Calculate final score
    combined_score = calculate_combined_score(
        prob,
        effective_distance,
        probability_weight=probability_weight,
        edit_distance_weight=edit_distance_weight,
        score_scaling_factor=score_scaling_factor,
    )

    # Calculate confidence based on probability improvement
    confidence = min(1.0, prob / max(current_prob, min_prob_denom))

    return (candidate, prob, distance, combined_score, confidence)


def combine_context_probabilities(
    provider: "DictionaryProvider",
    prob_left: float,
    candidate: str,
    next_word: str | None,
) -> float:
    """Combine left and right context probabilities.

    If right context is available and has probability > 0, average the two.
    Otherwise, use left probability only (conservative approach).
    """
    if not next_word:
        return prob_left

    prob_right = provider.get_bigram_probability(candidate, next_word)
    if prob_right > 0:
        return (prob_left + prob_right) / 2
    return prob_left


def apply_pos_context_scoring(
    provider: "DictionaryProvider",
    effective_distance: float,
    prev_word: str,
    candidate: str,
    next_word: str | None,
    *,
    pos_bigram_probs: dict[tuple[str, str], float],
    pos_score_weight: float,
    pos_distance_reduction_factor_multiplier: float,
) -> float:
    """Apply POS context scoring to reduce effective distance.

    Uses POS bigram probabilities to boost candidates that have
    appropriate POS tag transitions.
    """
    if not pos_bigram_probs or pos_score_weight <= 0:
        return effective_distance

    pos_context_score = calculate_pos_context_score(
        provider, prev_word, candidate, next_word, pos_bigram_probs=pos_bigram_probs
    )

    pos_distance_reduction_factor = (
        pos_context_score * pos_score_weight * pos_distance_reduction_factor_multiplier
    )
    return max(0, effective_distance - pos_distance_reduction_factor)


def calculate_pos_context_score(
    provider: "DictionaryProvider",
    prev_word: str,
    candidate: str,
    next_word: str | None,
    *,
    pos_bigram_probs: dict[tuple[str, str], float],
) -> float:
    """Calculate POS context score based on POS bigram probabilities.

    Returns a score between 0.0 and 1.0 representing how well the
    candidate's POS tag fits the context.
    """
    prev_pos_tags = get_pos_tags(provider, prev_word)
    candidate_pos_tags = get_pos_tags(provider, candidate)
    next_pos_tags = get_pos_tags(provider, next_word) if next_word else set()

    # Calculate left POS context: P(candidate_pos | prev_word_pos)
    max_pos_prob_left = max_pos_bigram_prob(prev_pos_tags, candidate_pos_tags, pos_bigram_probs)

    # Calculate right POS context: P(next_word_pos | candidate_pos)
    max_pos_prob_right = max_pos_bigram_prob(candidate_pos_tags, next_pos_tags, pos_bigram_probs)

    # Combine left and right POS probabilities
    if max_pos_prob_left > 0 and max_pos_prob_right > 0:
        return (max_pos_prob_left + max_pos_prob_right) / 2.0
    return max(max_pos_prob_left, max_pos_prob_right)


def get_pos_tags(provider: "DictionaryProvider", word: str) -> set[str]:
    """Get set of POS tags for a word from the provider."""
    pos_str = provider.get_word_pos(word)
    return set(pos_str.split("|")) if pos_str else set()


def max_pos_bigram_prob(
    tags1: set[str],
    tags2: set[str],
    pos_bigram_probs: dict[tuple[str, str], float],
) -> float:
    """Find maximum POS bigram probability between two sets of tags.

    Iterates through all combinations of tags and returns the maximum
    probability found in pos_bigram_probs.
    """
    if not tags1 or not tags2:
        return 0.0

    max_prob = 0.0
    for tag1 in tags1:
        for tag2 in tags2:
            prob = pos_bigram_probs.get((tag1, tag2), 0.0)
            if prob > max_prob:
                max_prob = prob
    return max_prob


def calculate_combined_score(
    prob: float,
    effective_distance: float,
    *,
    probability_weight: float,
    edit_distance_weight: float,
    score_scaling_factor: float,
) -> float:
    """Calculate combined score using log-probability model.

    Score = probability_weight * log(prob) - edit_distance_weight * distance * scaling_factor

    Higher scores are better.
    """
    safe_prob = max(prob, 1e-10)
    log_prob = math.log(safe_prob)

    return probability_weight * log_prob - (
        edit_distance_weight * effective_distance * score_scaling_factor
    )
