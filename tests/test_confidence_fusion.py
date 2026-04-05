"""Unit tests for confidence fusion pipeline (Components 2, 4, 5)."""

import pytest

from myspellchecker.core.calibration import CalibrationData, StrategyCalibrator
from myspellchecker.core.validation_strategies.arbiter import (
    INDEPENDENCE_CLUSTERS,
    _get_cluster,
    fuse_all_candidates,
    fuse_candidates,
    noisy_or_merge,
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


# ---------------------------------------------------------------------------
# Noisy-OR (Component 2)
# ---------------------------------------------------------------------------


class TestNoisyOrMerge:
    """Tests for the Noisy-OR merge function."""

    def test_single_score(self):
        assert noisy_or_merge([0.7]) == pytest.approx(0.7)

    def test_two_independent_scores(self):
        # P(error) = 1 - (1-0.5)*(1-0.5) = 1 - 0.25 = 0.75
        assert noisy_or_merge([0.5, 0.5]) == pytest.approx(0.75)

    def test_zero_score_no_effect(self):
        assert noisy_or_merge([0.8, 0.0]) == pytest.approx(0.8)

    def test_one_score_certainty(self):
        assert noisy_or_merge([1.0, 0.5]) == pytest.approx(1.0)

    def test_all_zeros(self):
        assert noisy_or_merge([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_three_moderate_scores(self):
        # 1 - (1-0.6)*(1-0.6)*(1-0.6) = 1 - 0.064 = 0.936
        assert noisy_or_merge([0.6, 0.6, 0.6]) == pytest.approx(0.936)

    def test_empty_list(self):
        assert noisy_or_merge([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Independence Clusters (Component 4)
# ---------------------------------------------------------------------------


class TestIndependenceClusters:
    """Tests for cluster assignments."""

    def test_all_strategies_clustered(self):
        """Every strategy in the tier system should have a cluster."""
        from myspellchecker.core.validation_strategies.arbiter import STRATEGY_TIER

        for strategy_name in STRATEGY_TIER:
            cluster = _get_cluster(strategy_name)
            assert not cluster.startswith("unclustered_"), (
                f"{strategy_name} is not assigned to any cluster"
            )

    def test_cluster_structure(self):
        assert "deterministic" in INDEPENDENCE_CLUSTERS
        assert "neural" in INDEPENDENCE_CLUSTERS
        assert "statistical" in INDEPENDENCE_CLUSTERS

    def test_neural_cluster_strategies(self):
        assert "ConfusableCompoundClassifierStrategy" in INDEPENDENCE_CLUSTERS["neural"]
        assert "ConfusableSemanticStrategy" in INDEPENDENCE_CLUSTERS["neural"]
        assert "SemanticValidationStrategy" in INDEPENDENCE_CLUSTERS["neural"]

    def test_registry_tables_in_sync(self):
        """STRATEGY_TIER, STRATEGY_RELIABILITY, and INDEPENDENCE_CLUSTERS must cover
        exactly the same set of strategies."""
        from myspellchecker.core.calibration import STRATEGY_RELIABILITY
        from myspellchecker.core.validation_strategies.arbiter import STRATEGY_TIER

        tier_keys = set(STRATEGY_TIER.keys())
        reliability_keys = set(STRATEGY_RELIABILITY.keys())
        cluster_keys = set()
        for strategies in INDEPENDENCE_CLUSTERS.values():
            cluster_keys.update(strategies)

        assert tier_keys == reliability_keys, (
            f"TIER vs RELIABILITY mismatch: "
            f"only in TIER={tier_keys - reliability_keys}, "
            f"only in RELIABILITY={reliability_keys - tier_keys}"
        )
        assert tier_keys == cluster_keys, (
            f"TIER vs CLUSTERS mismatch: "
            f"only in TIER={tier_keys - cluster_keys}, "
            f"only in CLUSTERS={cluster_keys - tier_keys}"
        )

    def test_unknown_strategy_gets_singleton_cluster(self):
        cluster = _get_cluster("CustomStrategy")
        assert cluster == "unclustered_CustomStrategy"

    def test_correlated_strategies_same_cluster(self):
        assert _get_cluster("ConfusableSemanticStrategy") == _get_cluster(
            "SemanticValidationStrategy"
        )
        assert _get_cluster("SyntacticValidationStrategy") == _get_cluster(
            "POSSequenceValidationStrategy"
        )

    def test_independent_strategies_different_clusters(self):
        assert _get_cluster("ToneValidationStrategy") != _get_cluster(
            "ConfusableSemanticStrategy"
        )
        assert _get_cluster("BrokenCompoundStrategy") != _get_cluster(
            "NgramContextValidationStrategy"
        )


# ---------------------------------------------------------------------------
# Candidate Fusion (Component 5)
# ---------------------------------------------------------------------------


class TestFuseCandidates:
    """Tests for fuse_candidates() with calibrated Noisy-OR."""

    def test_single_candidate(self):
        cal = StrategyCalibrator()
        c = _candidate("OrthographyValidationStrategy", confidence=0.9)
        fused_conf, winner = fuse_candidates([c], cal)
        assert winner is c
        # fused = reliability(0.90) * calibrated(0.9) = 0.81
        assert fused_conf == pytest.approx(0.90 * 0.9)

    def test_same_cluster_takes_max(self):
        """Two neural strategies (same cluster) should NOT be Noisy-OR merged."""
        cal = StrategyCalibrator()
        c1 = _candidate("ConfusableSemanticStrategy", confidence=0.8)
        c2 = _candidate("SemanticValidationStrategy", confidence=0.6)
        fused_conf, winner = fuse_candidates([c1, c2], cal)
        # Same cluster: max(0.70*0.8, 0.65*0.6) = max(0.56, 0.39) = 0.56
        # Only 1 cluster score, so Noisy-OR of [0.56] = 0.56
        assert fused_conf == pytest.approx(0.56)
        # Winner by tier: both tier 4, so higher confidence wins
        assert winner.strategy_name == "ConfusableSemanticStrategy"

    def test_different_clusters_noisy_or(self):
        """POS (structural_grammar) + Confusable (neural) should be Noisy-OR merged."""
        cal = StrategyCalibrator()
        pos = _candidate("POSSequenceValidationStrategy", confidence=0.85)
        csem = _candidate("ConfusableSemanticStrategy", confidence=0.8)
        fused_conf, winner = fuse_candidates([pos, csem], cal)
        # Cluster 1 (structural_grammar): 0.60 * 0.85 = 0.51
        # Cluster 2 (neural):             0.70 * 0.80 = 0.56
        # Noisy-OR: 1 - (1-0.51)*(1-0.56) = 1 - 0.49*0.44 = 1 - 0.2156 = 0.7844
        assert fused_conf == pytest.approx(1 - (1 - 0.51) * (1 - 0.56))
        # Winner: tier 4 (neural) beats tier 3
        assert winner.strategy_name == "ConfusableSemanticStrategy"

    def test_three_clusters(self):
        """Tone + POS + Semantic across 3 independent clusters."""
        cal = StrategyCalibrator()
        tone = _candidate("ToneValidationStrategy", confidence=0.9)
        pos = _candidate("POSSequenceValidationStrategy", confidence=0.85)
        sem = _candidate("SemanticValidationStrategy", confidence=0.7)
        fused_conf, winner = fuse_candidates([tone, pos, sem], cal)
        # deterministic: 0.85 * 0.9 = 0.765
        # structural_grammar: 0.60 * 0.85 = 0.51
        # neural: 0.65 * 0.7 = 0.455
        expected = 1 - (1 - 0.765) * (1 - 0.51) * (1 - 0.455)
        assert fused_conf == pytest.approx(expected)
        # Winner: tier 4 beats all
        assert winner.strategy_name == "SemanticValidationStrategy"

    def test_winner_uses_tier_priority(self):
        """Fusion confidence from Noisy-OR, but winner by tier hierarchy."""
        cal = StrategyCalibrator()
        ortho = _candidate("OrthographyValidationStrategy", confidence=0.99)
        ngram = _candidate("NgramContextValidationStrategy", confidence=0.6)
        fused_conf, winner = fuse_candidates([ortho, ngram], cal)
        # Winner by tier: ngram is tier 3, ortho is tier 1
        assert winner.strategy_name == "NgramContextValidationStrategy"
        # But fused confidence is from both clusters
        assert fused_conf > 0.6


class TestFuseAllCandidates:
    """Tests for fuse_all_candidates() with threshold filtering."""

    def test_above_threshold_kept(self):
        cal = StrategyCalibrator()
        candidates = {
            0: [_candidate("OrthographyValidationStrategy", confidence=0.9)],
        }
        results = fuse_all_candidates(candidates, cal, threshold=0.5)
        assert 0 in results

    def test_below_threshold_filtered(self):
        cal = StrategyCalibrator()
        candidates = {
            0: [_candidate("QuestionStructureValidationStrategy", confidence=0.3)],
        }
        results = fuse_all_candidates(candidates, cal, threshold=0.5)
        # 0.55 * 0.3 = 0.165 < 0.5
        assert 0 not in results

    def test_empty_candidates_skipped(self):
        cal = StrategyCalibrator()
        results = fuse_all_candidates({0: []}, cal, threshold=0.0)
        assert results == {}

    def test_multi_position(self):
        cal = StrategyCalibrator()
        candidates = {
            0: [
                _candidate("OrthographyValidationStrategy", confidence=0.9),
                _candidate("POSSequenceValidationStrategy", confidence=0.85),
            ],
            10: [_candidate("ToneValidationStrategy", confidence=0.9)],
        }
        results = fuse_all_candidates(candidates, cal, threshold=0.3)
        assert 0 in results
        assert 10 in results

    def test_custom_calibration_data(self):
        """Calibration data can shift scores across the threshold."""
        data = {
            "ToneValidationStrategy": CalibrationData(
                x_thresholds=[0.0, 1.0],
                y_thresholds=[0.0, 0.1],  # Aggressively compress
            ),
        }
        cal = StrategyCalibrator(calibration_data=data)
        candidates = {
            0: [_candidate("ToneValidationStrategy", confidence=0.9)],
        }
        # Calibrated: 0.1*0.9 = 0.09. Reliability: 0.85. Weighted: 0.85*0.09 = 0.0765
        results = fuse_all_candidates(candidates, cal, threshold=0.5)
        assert 0 not in results


# ---------------------------------------------------------------------------
# word_indices on ErrorCandidate (Component 3)
# ---------------------------------------------------------------------------


class TestErrorCandidateWordIndices:
    """Tests for the word_indices field on ErrorCandidate."""

    def test_default_empty_tuple(self):
        c = _candidate("ToneValidationStrategy")
        assert c.word_indices == ()

    def test_single_word(self):
        c = ErrorCandidate(
            strategy_name="ToneValidationStrategy",
            error_type="tone_error",
            confidence=0.9,
            word_indices=(3,),
        )
        assert c.word_indices == (3,)

    def test_multi_word_span(self):
        c = ErrorCandidate(
            strategy_name="BrokenCompoundStrategy",
            error_type="broken_compound",
            confidence=0.85,
            word_indices=(2, 3),
        )
        assert c.word_indices == (2, 3)
        assert len(c.word_indices) == 2


# ---------------------------------------------------------------------------
# fusion_mode on ValidationContext (Component 6)
# ---------------------------------------------------------------------------


class TestValidationContextFusionMode:
    """Tests for the fusion_mode flag on ValidationContext."""

    def test_default_false(self):
        from myspellchecker.core.validation_strategies.base import ValidationContext

        ctx = ValidationContext(
            sentence="test",
            words=["test"],
            word_positions=[0],
        )
        assert ctx.fusion_mode is False

    def test_set_true(self):
        from myspellchecker.core.validation_strategies.base import ValidationContext

        ctx = ValidationContext(
            sentence="test",
            words=["test"],
            word_positions=[0],
            fusion_mode=True,
        )
        assert ctx.fusion_mode is True


# ---------------------------------------------------------------------------
# should_skip_position with fusion_mode (Component 6)
# ---------------------------------------------------------------------------


class TestShouldSkipPositionFusionMode:
    """Tests for mutex bypass in fusion mode."""

    def test_fusion_mode_never_skips(self):
        from myspellchecker.core.validation_strategies.conflict_rules import should_skip_position

        existing = {0: "pos_sequence_error"}
        # Without fusion_mode: unknown strategy should skip
        assert should_skip_position("UnknownStrategy", 0, existing) is True
        # With fusion_mode: never skips
        assert should_skip_position("UnknownStrategy", 0, existing, fusion_mode=True) is False

    def test_fusion_mode_unclaimed_still_false(self):
        from myspellchecker.core.validation_strategies.conflict_rules import should_skip_position

        assert should_skip_position("Any", 0, {}, fusion_mode=True) is False

    def test_non_fusion_mode_preserves_behavior(self):
        from myspellchecker.core.validation_strategies.conflict_rules import should_skip_position

        existing = {0: "tone_error"}
        # Non-overridable: should skip
        assert should_skip_position("POSSequenceValidationStrategy", 0, existing) is True
        assert (
            should_skip_position(
                "POSSequenceValidationStrategy", 0, existing, fusion_mode=False
            )
            is True
        )
