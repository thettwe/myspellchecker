"""Conflict resolution rules for the validation pipeline.

Defines which strategies are allowed to override errors claimed by
earlier (lower-priority) strategies.  This replaces the ad-hoc inline
override in StatisticalConfusableStrategy and extends the pattern to
ConfusableSemantic and Semantic strategies.

The override matrix is intentionally small (~5 entries).  If it grows
past ~6 rules, migrate to the full candidate/voting system (v2.0.0).
"""

from __future__ import annotations

# Map: strategy class name -> set of error_type strings it may override.
# When a strategy encounters a position already claimed by an earlier
# strategy, it checks this table.  If the existing error type is in the
# override set, the strategy proceeds (re-diagnoses the position).
# Otherwise it skips the position (respects the earlier claim).
#
# Strategies not listed here (or with an empty set) never override --
# they either skip claimed positions or append suggestions.
STRATEGY_OVERRIDE_RULES: dict[str, set[str]] = {
    # Existing behavior: statistical confusable overrides POS sequence
    "StatisticalConfusableStrategy": {"pos_sequence_error"},
    # NEW: ConfusableSemantic (MLM) can override POS and n-gram claims
    "ConfusableSemanticStrategy": {"pos_sequence_error", "context_probability"},
    # NEW: Semantic (MLM proactive/contrast/animacy) can override POS and n-gram claims
    "SemanticValidationStrategy": {"pos_sequence_error", "context_probability"},
}


def should_skip_position(
    strategy_name: str,
    position: int,
    existing_errors: dict[int, str],
) -> bool:
    """Decide whether a strategy should skip a claimed position.

    Args:
        strategy_name: The ``__class__.__name__`` of the calling strategy.
        position: The absolute character position being considered.
        existing_errors: Map of position -> error_type for already-claimed
            positions (typically ``context.existing_errors``).

    Returns:
        ``True`` if the strategy should **skip** this position (the existing
        claim stands), ``False`` if the strategy may proceed (either the
        position is unclaimed, or the strategy is allowed to override).
    """
    if position not in existing_errors:
        return False

    existing_type = existing_errors[position]
    overridable = STRATEGY_OVERRIDE_RULES.get(strategy_name, set())
    return existing_type not in overridable
