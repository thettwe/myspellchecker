"""Shared deduplication logic for suggestion pipelines.

Consolidates the duplicate-removal algorithm used by both
``UnifiedRanker._deduplicate()`` and
``CompositeSuggestionStrategy._deduplicate_suggestions()``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myspellchecker.algorithms.ranker import SuggestionData


def deduplicate_suggestions(
    suggestions: list[SuggestionData],
    weight_fn: Callable[[str], float] | None = None,
) -> list[SuggestionData]:
    """Remove duplicate terms, keeping the best version of each.

    When multiple sources suggest the same term, keeps the one with:
    1. Higher weighted confidence (``weight_fn(source) * confidence``)
    2. If tied, lower ``strategy_score`` (when present)
    3. If still tied, lower ``edit_distance``
    4. If still tied, higher ``frequency``

    Args:
        suggestions: List of SuggestionData that may contain duplicates.
        weight_fn: Optional callable ``(source) -> float`` for source-aware
            weighting. When provided, the primary comparison is
            ``weight_fn(source) * confidence``. When None (source-agnostic),
            compares raw ``confidence`` values.

    Returns:
        Deduplicated list preserving insertion order of first-seen terms.
    """
    best_by_term: dict[str, SuggestionData] = {}

    for suggestion in suggestions:
        term = suggestion.term
        if term not in best_by_term:
            best_by_term[term] = suggestion
            continue

        existing = best_by_term[term]

        # Primary comparison: weighted confidence
        if weight_fn is not None:
            existing_score = weight_fn(existing.source) * existing.confidence
            new_score = weight_fn(suggestion.source) * suggestion.confidence
        else:
            existing_score = existing.confidence
            new_score = suggestion.confidence

        if new_score > existing_score:
            best_by_term[term] = suggestion
        elif new_score == existing_score:
            if _is_better_tiebreak(suggestion, existing):
                best_by_term[term] = suggestion

    return list(best_by_term.values())


def _is_better_tiebreak(new: SuggestionData, existing: SuggestionData) -> bool:
    """Determine if ``new`` is better than ``existing`` on tie-break criteria.

    Tie-break order:
    1. Lower strategy_score (prefer having one over not)
    2. Lower edit_distance
    3. Higher frequency
    """
    new_strat = new.strategy_score
    existing_strat = existing.strategy_score

    if new_strat is not None and existing_strat is not None:
        if new_strat < existing_strat:
            return True
        if new_strat > existing_strat:
            return False
        # Equal strategy_score — fall through to edit_distance
    elif new_strat is not None:
        # Prefer suggestion with strategy_score over one without
        return True
    elif existing_strat is not None:
        # Existing has strategy_score, new doesn't — existing wins
        return False
    # Both None — fall through to edit_distance

    if new.edit_distance < existing.edit_distance:
        return True
    if new.edit_distance > existing.edit_distance:
        return False

    # Equal edit_distance — prefer higher frequency
    return new.frequency > existing.frequency
