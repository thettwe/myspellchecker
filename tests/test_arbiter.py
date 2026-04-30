"""Unit tests for the candidate arbiter (multi-strategy conflict resolution)."""

import pytest

from myspellchecker.core.validation_strategies.arbiter import (
    STRATEGY_TIER,
    _get_tier,
    arbitrate_candidates,
    select_winner,
)
from myspellchecker.core.validation_strategies.base import ErrorCandidate

# -- Helpers --


def _candidate(
    strategy: str,
    error_type: str = "test_error",
    confidence: float = 0.8,
    suggestion: str | None = "fix",
) -> ErrorCandidate:
    return ErrorCandidate(
        strategy_name=strategy,
        error_type=error_type,
        confidence=confidence,
        suggestion=suggestion,
    )


class TestStrategyTier:
    """Tests for tier assignments."""

    def test_tier1_deterministic(self):
        assert _get_tier("ToneValidationStrategy") == 1
        assert _get_tier("OrthographyValidationStrategy") == 1

    def test_tier2_structural(self):
        assert _get_tier("SyntacticValidationStrategy") == 2
        assert _get_tier("StatisticalConfusableStrategy") == 2
        assert _get_tier("BrokenCompoundStrategy") == 2

    def test_tier3_contextual(self):
        assert _get_tier("POSSequenceValidationStrategy") == 3
        assert _get_tier("QuestionStructureValidationStrategy") == 3
        assert _get_tier("HomophoneValidationStrategy") == 3
        assert _get_tier("NgramContextValidationStrategy") == 3

    def test_tier4_neural(self):
        assert _get_tier("ConfusableSemanticStrategy") == 4
        assert _get_tier("SemanticValidationStrategy") == 4

    def test_unknown_strategy_defaults_to_tier2(self):
        assert _get_tier("UnknownStrategy") == 2
        assert _get_tier("CustomStrategy") == 2

    def test_all_known_strategies_have_tiers(self):
        assert len(STRATEGY_TIER) == 17


class TestSelectWinner:
    """Tests for select_winner()."""

    def test_single_candidate_passthrough(self):
        c = _candidate("POSSequenceValidationStrategy")
        assert select_winner([c]) is c

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one candidate"):
            select_winner([])

    # -- Tier beats tier --

    def test_tier4_beats_tier3(self):
        pos = _candidate("POSSequenceValidationStrategy", confidence=0.9)
        sem = _candidate("ConfusableSemanticStrategy", confidence=0.7)
        winner = select_winner([pos, sem])
        assert winner.strategy_name == "ConfusableSemanticStrategy"

    def test_tier4_beats_tier2(self):
        stat = _candidate("StatisticalConfusableStrategy", confidence=0.95)
        sem = _candidate("SemanticValidationStrategy", confidence=0.6)
        winner = select_winner([stat, sem])
        assert winner.strategy_name == "SemanticValidationStrategy"

    def test_tier3_beats_tier2(self):
        syn = _candidate("SyntacticValidationStrategy", confidence=0.9)
        pos = _candidate("POSSequenceValidationStrategy", confidence=0.5)
        winner = select_winner([syn, pos])
        assert winner.strategy_name == "POSSequenceValidationStrategy"

    def test_tier3_beats_tier1(self):
        tone = _candidate("ToneValidationStrategy", confidence=0.99)
        pos = _candidate("POSSequenceValidationStrategy", confidence=0.5)
        winner = select_winner([tone, pos])
        assert winner.strategy_name == "POSSequenceValidationStrategy"

    # -- Same tier: confidence wins --

    def test_same_tier_higher_confidence_wins(self):
        a = _candidate("ConfusableSemanticStrategy", confidence=0.9)
        b = _candidate("SemanticValidationStrategy", confidence=0.7)
        winner = select_winner([a, b])
        assert winner.strategy_name == "ConfusableSemanticStrategy"

    def test_same_tier_reverse_order(self):
        a = _candidate("SemanticValidationStrategy", confidence=0.95)
        b = _candidate("ConfusableSemanticStrategy", confidence=0.6)
        winner = select_winner([b, a])
        assert winner.strategy_name == "SemanticValidationStrategy"

    # -- POS vs ConfusableSemantic --

    def test_pos_vs_confusable_semantic(self):
        """Primary conflict pair from scope doc."""
        pos = _candidate("POSSequenceValidationStrategy", "pos_sequence_error", 0.8, "pos_fix")
        conf = _candidate("ConfusableSemanticStrategy", "confusable_error", 0.75, "confusable_fix")
        winner = select_winner([pos, conf])
        assert winner.strategy_name == "ConfusableSemanticStrategy"
        assert winner.suggestion == "confusable_fix"

    def test_pos_vs_semantic(self):
        """Secondary conflict pair from scope doc."""
        pos = _candidate("POSSequenceValidationStrategy", "pos_sequence_error", 0.85, "pos_fix")
        sem = _candidate("SemanticValidationStrategy", "semantic_error", 0.6, "semantic_fix")
        winner = select_winner([pos, sem])
        assert winner.strategy_name == "SemanticValidationStrategy"
        assert winner.suggestion == "semantic_fix"

    # -- Three-way conflict --

    def test_three_candidates_highest_tier_wins(self):
        stat = _candidate("StatisticalConfusableStrategy", confidence=0.95)
        pos = _candidate("POSSequenceValidationStrategy", confidence=0.8)
        sem = _candidate("SemanticValidationStrategy", confidence=0.7)
        winner = select_winner([stat, pos, sem])
        assert winner.strategy_name == "SemanticValidationStrategy"


class TestArbitrateCandidates:
    """Tests for arbitrate_candidates()."""

    def test_single_candidate_positions_excluded(self):
        candidates = {
            0: [_candidate("POSSequenceValidationStrategy")],
            10: [_candidate("ConfusableSemanticStrategy")],
        }
        winners = arbitrate_candidates(candidates)
        assert winners == {}

    def test_multi_candidate_position_resolved(self):
        candidates = {
            0: [
                _candidate("POSSequenceValidationStrategy", confidence=0.8),
                _candidate("ConfusableSemanticStrategy", confidence=0.7),
            ],
        }
        winners = arbitrate_candidates(candidates)
        assert 0 in winners
        assert winners[0].strategy_name == "ConfusableSemanticStrategy"

    def test_mixed_single_and_multi(self):
        candidates = {
            0: [_candidate("ToneValidationStrategy")],  # single
            10: [
                _candidate("POSSequenceValidationStrategy", confidence=0.8),
                _candidate("SemanticValidationStrategy", confidence=0.6),
            ],
            20: [_candidate("SyntacticValidationStrategy")],  # single
        }
        winners = arbitrate_candidates(candidates)
        assert 0 not in winners
        assert 20 not in winners
        assert 10 in winners
        assert winners[10].strategy_name == "SemanticValidationStrategy"

    def test_empty_input(self):
        assert arbitrate_candidates({}) == {}
