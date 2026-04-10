"""Candidate arbiter for multi-strategy conflict resolution.

When multiple strategies produce ErrorCandidates for the same position,
the arbiter selects the winner using tier-based priority with confidence
tiebreaking, and optionally fuses confidences using calibrated Noisy-OR.

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

v1.3.0: full candidate fusion pipeline with calibrated Noisy-OR across
independence clusters, gated by ``use_candidate_fusion`` config flag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.core.validation_strategies.base import ErrorCandidate
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.core.calibration import StrategyCalibrator

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
    # Tier 4: Neural (MLM/MLP-powered, highest accuracy, highest cost)
    "ConfusableCompoundClassifierStrategy": 4,
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


# ---------------------------------------------------------------------------
# Independence clusters (Component 4)
# ---------------------------------------------------------------------------

# Strategies within the same cluster are treated as correlated.
# Within a cluster we take max(reliability * calibrated_confidence).
# Across clusters we use Noisy-OR merge (Component 2).
INDEPENDENCE_CLUSTERS: dict[str, list[str]] = {
    "deterministic": ["ToneValidationStrategy", "OrthographyValidationStrategy"],
    "structural_grammar": ["SyntacticValidationStrategy", "POSSequenceValidationStrategy"],
    "statistical": [
        "StatisticalConfusableStrategy",
        "NgramContextValidationStrategy",
        "HomophoneValidationStrategy",
    ],
    "neural": [
        "ConfusableCompoundClassifierStrategy",
        "ConfusableSemanticStrategy",
        "SemanticValidationStrategy",
    ],
    "compound": ["BrokenCompoundStrategy"],
    "hidden_compound": ["HiddenCompoundStrategy"],
    "question": ["QuestionStructureValidationStrategy"],
}

# Reverse lookup: strategy name -> cluster name (built once at import).
_STRATEGY_TO_CLUSTER: dict[str, str] = {}
for _cluster_name, _cluster_strategies in INDEPENDENCE_CLUSTERS.items():
    for _strat in _cluster_strategies:
        _STRATEGY_TO_CLUSTER[_strat] = _cluster_name

_UNCLUSTERED_PREFIX = "unclustered_"


def _get_cluster(strategy_name: str) -> str:
    """Return cluster name for a strategy, creating a singleton cluster for unknowns."""
    return _STRATEGY_TO_CLUSTER.get(strategy_name, f"{_UNCLUSTERED_PREFIX}{strategy_name}")


# ---------------------------------------------------------------------------
# Noisy-OR confidence merge (Component 2)
# ---------------------------------------------------------------------------


def noisy_or_merge(cluster_scores: list[float]) -> float:
    """Merge independent cluster scores using Noisy-OR.

    ``P(error) = 1 - prod(1 - score_i)``

    Args:
        cluster_scores: Per-cluster max scores (already weighted by
            reliability and calibrated).

    Returns:
        Merged confidence in ``[0.0, 1.0]``.
    """
    prob_no_error = 1.0
    for score in cluster_scores:
        prob_no_error *= 1.0 - score
    return 1.0 - prob_no_error


# ---------------------------------------------------------------------------
# Candidate fusion (Component 5 entry point)
# ---------------------------------------------------------------------------


def fuse_candidates(
    candidates: list[ErrorCandidate],
    calibrator: "StrategyCalibrator",
) -> tuple[float, ErrorCandidate]:
    """Fuse multiple candidates at a position using calibrated Noisy-OR.

    Pipeline:
        1. Calibrate each candidate's confidence.
        2. Group by independence cluster.
        3. Within each cluster: ``max(reliability * calibrated_confidence)``.
        4. Across clusters: Noisy-OR merge.
        5. Return ``(merged_confidence, winner)`` where *winner* is
           selected via tier-based priority (for error details / suggestion).

    Args:
        candidates: Non-empty list of ErrorCandidates at the same position.
        calibrator: ``StrategyCalibrator`` instance.

    Returns:
        ``(merged_confidence, winner_candidate)``.
    """
    winner = select_winner(candidates)

    if len(candidates) == 1:
        c = candidates[0]
        cal_conf = calibrator.calibrate(c.strategy_name, c.confidence)
        reliability = calibrator.get_reliability(c.strategy_name)
        return min(reliability * cal_conf, 1.0), winner

    # Group by cluster, keep per-cluster max weighted score
    cluster_max: dict[str, float] = {}
    for c in candidates:
        cluster = _get_cluster(c.strategy_name)
        cal_conf = calibrator.calibrate(c.strategy_name, c.confidence)
        reliability = calibrator.get_reliability(c.strategy_name)
        weighted = min(reliability * cal_conf, 1.0)

        prev = cluster_max.get(cluster, -1.0)
        if weighted > prev:
            cluster_max[cluster] = weighted

    merged = noisy_or_merge(list(cluster_max.values()))
    return merged, winner


def fuse_all_candidates(
    error_candidates: dict[int, list[ErrorCandidate]],
    calibrator: "StrategyCalibrator",
    threshold: float = 0.5,
) -> dict[int, tuple[float, ErrorCandidate]]:
    """Run fusion on all positions and filter by confidence threshold.

    Args:
        error_candidates: Map of position -> list of ErrorCandidates.
        calibrator: ``StrategyCalibrator`` instance.
        threshold: Minimum fused confidence to keep a position.

    Returns:
        Map of position -> ``(fused_confidence, winner)`` for positions
        that meet the threshold.
    """
    results: dict[int, tuple[float, ErrorCandidate]] = {}
    for position, candidates in error_candidates.items():
        if not candidates:
            continue
        fused_conf, winner = fuse_candidates(candidates, calibrator)
        if fused_conf >= threshold:
            results[position] = (fused_conf, winner)
    return results
