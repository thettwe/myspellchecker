"""Candidate arbiter for multi-strategy conflict resolution.

When multiple strategies produce ErrorCandidates for the same position,
the arbiter selects the winner using tier-based priority with confidence
tiebreaking.

Tier hierarchy (higher tier wins):
    Tier 4 — Neural/MLM:   ConfusableSemanticStrategy, SemanticValidationStrategy
    Tier 3 — Contextual:   NgramContextValidationStrategy, HomophoneValidationStrategy,
                           POSSequenceValidationStrategy, QuestionStructureValidationStrategy
    Tier 2 — Structural:   StatisticalConfusableStrategy, SyntacticValidationStrategy,
                           BrokenCompoundStrategy
    Tier 1 — Deterministic: ToneValidationStrategy, OrthographyValidationStrategy

Within the same tier, the candidate with higher confidence wins.
On confidence tie, the candidate from the earlier (lower-priority) strategy
wins (it had more evidence available when it ran).

v1.3.0 scope: only arbitrates POS(30) vs ConfusableSemantic(48) and
POS(30) vs Semantic(70) conflict pairs.  All other single-candidate
positions pass through unchanged.
"""

from __future__ import annotations

from myspellchecker.core.validation_strategies.base import ErrorCandidate
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Strategy name -> tier number.  Higher tier = higher authority.
STRATEGY_TIER: dict[str, int] = {
    # Tier 1: Deterministic (hard veto — unambiguous Myanmar orthography)
    "ToneValidationStrategy": 1,
    "OrthographyValidationStrategy": 1,
    # Tier 2: Structural (can be wrong, but cheap and early)
    "SyntacticValidationStrategy": 2,
    "StatisticalConfusableStrategy": 2,
    "BrokenCompoundStrategy": 2,
    # Tier 3: Contextual (use context signals, medium cost)
    "POSSequenceValidationStrategy": 3,
    "QuestionStructureValidationStrategy": 3,
    "HomophoneValidationStrategy": 3,
    "NgramContextValidationStrategy": 3,
    # Tier 4: Neural (MLM-powered, highest accuracy, highest cost)
    "ConfusableSemanticStrategy": 4,
    "SemanticValidationStrategy": 4,
}

# Default tier for unknown strategies (treated as structural).
_DEFAULT_TIER = 2


def _get_tier(strategy_name: str) -> int:
    """Return the tier for a strategy, defaulting to structural."""
    return STRATEGY_TIER.get(strategy_name, _DEFAULT_TIER)


def select_winner(candidates: list[ErrorCandidate]) -> ErrorCandidate:
    """Select the best candidate from a list of competing candidates.

    Args:
        candidates: Non-empty list of ErrorCandidates at the same position.

    Returns:
        The winning ErrorCandidate.

    Raises:
        ValueError: If candidates list is empty.
    """
    if not candidates:
        raise ValueError("select_winner requires at least one candidate")

    if len(candidates) == 1:
        return candidates[0]

    # Highest tier first, then highest confidence.  On exact tie,
    # prefer the candidate that appeared first in the list (i.e. the
    # earlier-running strategy, which had more evidence when it ran).
    # The negated index (-i) ensures max() picks the first element on tie.
    winner = max(
        enumerate(candidates),
        key=lambda ic: (_get_tier(ic[1].strategy_name), ic[1].confidence, -ic[0]),
    )[1]

    if logger.isEnabledFor(10):  # DEBUG
        losers = [c for c in candidates if c is not winner]
        for loser in losers:
            logger.debug(
                "arbiter: %s (tier=%d, conf=%.2f) beats %s (tier=%d, conf=%.2f) at evidence=%s",
                winner.strategy_name,
                _get_tier(winner.strategy_name),
                winner.confidence,
                loser.strategy_name,
                _get_tier(loser.strategy_name),
                loser.confidence,
                winner.evidence,
            )

    return winner


def arbitrate_candidates(
    error_candidates: dict[int, list[ErrorCandidate]],
) -> dict[int, ErrorCandidate]:
    """Run arbiter on all positions with multiple candidates.

    Args:
        error_candidates: Map of position -> list of ErrorCandidates.

    Returns:
        Map of position -> winning ErrorCandidate (only for positions
        that had more than one candidate).
    """
    winners: dict[int, ErrorCandidate] = {}
    for position, candidates in error_candidates.items():
        if len(candidates) > 1:
            winners[position] = select_winner(candidates)
    return winners
