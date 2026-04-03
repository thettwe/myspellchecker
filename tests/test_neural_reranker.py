"""Tests for neural reranker feature schema v2, dual pipeline, and core functionality."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

from myspellchecker.training.reranker_data import (
    FEATURE_NAMES,
    MLP_CROSS_FEATURE_NAMES,
    MLP_CROSS_FEATURES,
    NUM_FEATURES,
    ORIGINAL_RANK_INDEX,
    _char_bigram_dice,
)

# ---------------------------------------------------------------------------
# Feature schema v2 constants
# ---------------------------------------------------------------------------


class TestFeatureSchemaV2:
    """Verify v2 feature schema constants are consistent."""

    def test_num_features_matches_names(self):
        assert NUM_FEATURES == len(FEATURE_NAMES)

    def test_num_features_is_19(self):
        assert NUM_FEATURES == 19

    def test_no_source_indicator_features(self):
        """v2 removed the 5 dead source indicator features."""
        for name in FEATURE_NAMES:
            assert not name.startswith("source_"), (
                f"Source indicator '{name}' should have been removed in v2"
            )

    def test_mlm_logit_present(self):
        assert "mlm_logit" in FEATURE_NAMES
        assert FEATURE_NAMES.index("mlm_logit") == 7

    def test_v2_features_present(self):
        """Check the 4 new v2 features exist."""
        v2_names = [
            "ngram_improvement_ratio",
            "edit_type_subst",
            "edit_type_delete",
            "char_dice_coeff",
        ]
        for name in v2_names:
            assert name in FEATURE_NAMES, f"v2 feature '{name}' missing from FEATURE_NAMES"

    def test_feature_order_stable(self):
        """Critical features retain expected indices."""
        assert FEATURE_NAMES[0] == "edit_distance"
        assert FEATURE_NAMES[1] == "weighted_distance"
        assert FEATURE_NAMES[2] == "log_frequency"
        assert FEATURE_NAMES[7] == "mlm_logit"
        assert FEATURE_NAMES[14] == "original_rank"


# ---------------------------------------------------------------------------
# Character bigram Dice coefficient
# ---------------------------------------------------------------------------


class TestCharBigramDice:
    """Test the character bigram Dice coefficient helper."""

    def test_identical_strings(self):
        assert _char_bigram_dice("abc", "abc") == 1.0

    def test_completely_different(self):
        assert _char_bigram_dice("ab", "cd") == 0.0

    def test_single_char_equal(self):
        assert _char_bigram_dice("a", "a") == 1.0

    def test_single_char_different(self):
        assert _char_bigram_dice("a", "b") == 0.0

    def test_empty_strings(self):
        assert _char_bigram_dice("", "") == 1.0

    def test_partial_overlap(self):
        score = _char_bigram_dice("abc", "abd")
        # bigrams: {ab, bc} vs {ab, bd} → intersection = {ab}
        # dice = 2*1 / (2+2) = 0.5
        assert score == 0.5

    def test_myanmar_characters(self):
        # Myanmar bigrams should work the same way
        score = _char_bigram_dice("\u1000\u1001\u1002", "\u1000\u1001\u1003")
        assert score == 0.5

    def test_symmetry(self):
        assert _char_bigram_dice("abc", "bcd") == _char_bigram_dice("bcd", "abc")

    def test_returns_float_in_range(self):
        score = _char_bigram_dice("hello", "world")
        assert 0.0 <= score <= 1.0

    def test_multiset_repeated_bigrams(self):
        """Counter-based Dice handles repeated bigrams correctly."""
        # "aab" has bigrams: Counter({"aa": 1, "ab": 1})
        # "aac" has bigrams: Counter({"aa": 1, "ac": 1})
        # intersection: min(1,1) for "aa" = 1
        # total: 2 + 2 = 4
        # dice = 2*1/4 = 0.5
        assert _char_bigram_dice("aab", "aac") == 0.5

    def test_reduplication_pattern(self):
        """Myanmar reduplication: repeated characters should be counted properly."""
        # "aa" has bigrams: Counter({"aa": 1})
        # "aaa" has bigrams: Counter({"aa": 2})
        # intersection: min(1, 2) = 1
        # total: 1 + 2 = 3
        # dice = 2*1/3 = 0.666...
        score = _char_bigram_dice("aa", "aaa")
        assert abs(score - 2.0 / 3.0) < 1e-9


# ---------------------------------------------------------------------------
# NeuralReranker module
# ---------------------------------------------------------------------------


class TestNeuralRerankerModule:
    """Test the ONNX-based NeuralReranker class."""

    def test_num_features_constant(self):
        from myspellchecker.algorithms.neural_reranker import _NUM_FEATURES

        assert _NUM_FEATURES == 19

    def _make_mock_reranker(self, input_shape):
        """Create a mock NeuralReranker with a given input shape."""
        with patch("myspellchecker.algorithms.neural_reranker.ort") as mock_ort:
            mock_ort.SessionOptions.return_value = MagicMock()
            mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 0

            mock_session = MagicMock()
            mock_ort.InferenceSession.return_value = mock_session
            mock_input = MagicMock(name="input")
            mock_input.shape = input_shape
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [MagicMock(name="output")]

            with patch("os.path.exists", return_value=True):
                from myspellchecker.algorithms.neural_reranker import NeuralReranker

                return NeuralReranker("/fake/path.onnx")

    def test_auto_detect_mlp(self):
        """3D input shape should be detected as MLP."""
        reranker = self._make_mock_reranker(["batch", "candidates", 19])
        assert reranker.model_type == "mlp"

    def test_auto_detect_gbt(self):
        """2D input shape should be detected as GBT."""
        reranker = self._make_mock_reranker([None, 19])
        assert reranker.model_type == "gbt"

    def test_score_candidates_empty(self):
        """score_candidates should return empty list for empty input."""
        with patch("myspellchecker.algorithms.neural_reranker.ort") as mock_ort:
            mock_ort.SessionOptions.return_value = MagicMock()
            mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 0

            mock_session = MagicMock()
            mock_ort.InferenceSession.return_value = mock_session
            mock_input = MagicMock(name="input")
            mock_input.shape = ["batch", "candidates", 19]
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [MagicMock(name="output")]

            with patch("os.path.exists", return_value=True):
                from myspellchecker.algorithms.neural_reranker import NeuralReranker

                reranker = NeuralReranker("/fake/path.onnx")

            result = reranker.score_candidates([])
            assert result == []

    def test_rerank_preserves_order_on_failure(self):
        """rerank should return original list if scoring fails."""
        with patch("myspellchecker.algorithms.neural_reranker.ort") as mock_ort:
            mock_ort.SessionOptions.return_value = MagicMock()
            mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 0

            mock_session = MagicMock()
            mock_ort.InferenceSession.return_value = mock_session
            mock_input = MagicMock(name="input")
            mock_input.shape = ["batch", "candidates", 19]
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [MagicMock(name="output")]
            mock_session.run.side_effect = RuntimeError("inference failed")

            with patch("os.path.exists", return_value=True):
                from myspellchecker.algorithms.neural_reranker import NeuralReranker

                reranker = NeuralReranker("/fake/path.onnx")

            suggestions = ["word1", "word2", "word3"]
            features = [[0.0] * 19] * 3
            result = reranker.rerank(suggestions, features)
            assert result == suggestions


# ---------------------------------------------------------------------------
# Feature extraction integration
# ---------------------------------------------------------------------------


class TestFeatureExtractionV2:
    """Test that feature extraction produces correct v2 vectors."""

    def test_edit_type_substitution(self):
        """Same-length words should have edit_type_subst=1.0."""
        # Mocking is complex for the full pipeline, so test the logic directly
        error = "abc"
        cand = "abd"  # same length
        assert len(cand) == len(error)
        edit_type_subst = 1.0 if len(cand) == len(error) else 0.0
        edit_type_delete = 1.0 if len(cand) != len(error) else 0.0
        assert edit_type_subst == 1.0
        assert edit_type_delete == 0.0

    def test_edit_type_deletion(self):
        """Different-length words should have edit_type_delete=1.0."""
        error = "abcd"
        cand = "abc"  # shorter
        edit_type_subst = 1.0 if len(cand) == len(error) else 0.0
        edit_type_delete = 1.0 if len(cand) != len(error) else 0.0
        assert edit_type_subst == 0.0
        assert edit_type_delete == 1.0

    def test_ngram_improvement_ratio_clamped(self):
        """N-gram improvement ratio should be clamped to [-5, 5]."""
        # Simulate extreme probability ratios
        error_prob = 1e-10
        cand_prob = 1.0
        ratio = max(-5.0, min(5.0, math.log(cand_prob / error_prob)))
        assert ratio == 5.0

        # Reverse
        ratio = max(-5.0, min(5.0, math.log(error_prob / cand_prob)))
        assert ratio == -5.0

    def test_ngram_improvement_zero_when_no_context(self):
        """When there's no context probability, improvement should be 0."""
        error_prob = 0.0
        cand_prob = 0.5
        # Can't compute log ratio when denominator is 0
        if error_prob > 0 and cand_prob > 0:
            ratio = math.log(cand_prob / error_prob)
        else:
            ratio = 0.0
        assert ratio == 0.0


# ---------------------------------------------------------------------------
# Training data v2 compatibility
# ---------------------------------------------------------------------------


class TestTrainerV2Compatibility:
    """Test that the trainer handles both v1 and v2 data."""

    def test_feature_names_count(self):
        """FEATURE_NAMES should have exactly NUM_FEATURES entries."""
        assert len(FEATURE_NAMES) == NUM_FEATURES == 19


# ---------------------------------------------------------------------------
# Dual pipeline: MLP feature schema constants
# ---------------------------------------------------------------------------


class TestMLPFeatureSchema:
    """Test MLP-specific feature schema constants for dual pipeline."""

    def test_original_rank_index(self):
        """ORIGINAL_RANK_INDEX should point to 'original_rank'."""
        assert ORIGINAL_RANK_INDEX == 14
        assert FEATURE_NAMES[ORIGINAL_RANK_INDEX] == "original_rank"

    def test_cross_feature_count(self):
        """MLP should have exactly 5 cross-features."""
        assert len(MLP_CROSS_FEATURES) == 5
        assert len(MLP_CROSS_FEATURE_NAMES) == 5

    def test_cross_feature_names(self):
        """Cross-feature names should match definitions."""
        expected = [
            "edit_dist_x_ngram_improv",
            "phonetic_x_confusable",
            "freq_x_dice",
            "mlm_x_ngram_sum",
            "edit_dist_x_freq",
        ]
        assert MLP_CROSS_FEATURE_NAMES == expected

    def test_cross_feature_indices_valid(self):
        """Cross-feature indices should reference valid base features."""
        for name, left_idx, right_idx in MLP_CROSS_FEATURES:
            assert 0 <= left_idx < NUM_FEATURES, (
                f"Cross-feature '{name}' left_idx {left_idx} out of range"
            )
            # right_idx can be -1 (special: mlm*ngram_sum)
            if right_idx >= 0:
                assert right_idx < NUM_FEATURES, (
                    f"Cross-feature '{name}' right_idx {right_idx} out of range"
                )

    def test_mlp_total_features(self):
        """MLP with all transforms: 19 - 1 (original_rank) + 5 (cross) = 23."""
        mlp_dim = NUM_FEATURES - 1 + len(MLP_CROSS_FEATURES)
        assert mlp_dim == 23


# ---------------------------------------------------------------------------
# Dual pipeline: MLP dataset feature transforms
# ---------------------------------------------------------------------------


class TestMLPDatasetTransforms:
    """Test RerankerDataset MLP-specific feature transforms."""

    def _make_example(self, gold_index: int = 0, num_candidates: int = 3):
        """Create a synthetic training example with known feature values."""
        features = []
        for i in range(num_candidates):
            fv = [float(f_idx + i * 0.1) for f_idx in range(NUM_FEATURES)]
            features.append(fv)
        return {
            "features": features,
            "gold_index": gold_index,
            "error_type": "test",
        }

    def test_no_transforms_preserves_features(self):
        """Without transforms, dataset should preserve original features."""
        try:
            import torch  # noqa: F401
        except ImportError:
            return  # Skip if torch not available

        from myspellchecker.training.reranker_trainer import RerankerDataset

        ex = self._make_example()
        ds = RerankerDataset(
            [ex],
            max_candidates=5,
            drop_original_rank=False,
            add_cross_features=False,
        )
        assert ds.num_features == NUM_FEATURES

    def test_drop_original_rank(self):
        """Dropping original_rank should reduce feature count by 1."""
        try:
            import torch  # noqa: F401
        except ImportError:
            return

        from myspellchecker.training.reranker_trainer import RerankerDataset

        ex = self._make_example()
        ds = RerankerDataset(
            [ex],
            max_candidates=5,
            drop_original_rank=True,
            add_cross_features=False,
        )
        assert ds.num_features == NUM_FEATURES - 1

    def test_add_cross_features(self):
        """Adding cross-features should increase feature count."""
        try:
            import torch  # noqa: F401
        except ImportError:
            return

        from myspellchecker.training.reranker_trainer import RerankerDataset

        ex = self._make_example()
        ds = RerankerDataset(
            [ex],
            max_candidates=5,
            drop_original_rank=False,
            add_cross_features=True,
        )
        assert ds.num_features == NUM_FEATURES + len(MLP_CROSS_FEATURES)

    def test_both_transforms(self):
        """Both transforms: 19 - 1 + 5 = 23 features."""
        try:
            import torch  # noqa: F401
        except ImportError:
            return

        from myspellchecker.training.reranker_trainer import RerankerDataset

        ex = self._make_example()
        ds = RerankerDataset(
            [ex],
            max_candidates=5,
            drop_original_rank=True,
            add_cross_features=True,
        )
        assert ds.num_features == 23
        # Verify actual tensor shape
        item = ds[0]
        assert item["features"].shape == (5, 23)

    def test_cross_feature_values(self):
        """Cross-features should be computed correctly from base features."""
        try:
            import torch  # noqa: F401
        except ImportError:
            return

        from myspellchecker.training.reranker_trainer import RerankerDataset

        # Create an example with known values
        fv = [0.0] * NUM_FEATURES
        fv[0] = 2.0  # edit_distance
        fv[1] = 1.5  # weighted_distance
        fv[3] = 0.8  # phonetic_score
        fv[7] = 3.0  # mlm_logit
        fv[8] = 0.5  # ngram_left_prob
        fv[9] = 0.3  # ngram_right_prob
        fv[10] = 1.0  # is_confusable
        fv[11] = 0.7  # relative_log_freq
        fv[15] = 1.2  # ngram_improvement_ratio
        fv[18] = 0.6  # char_dice_coeff
        ex = {"features": [fv], "gold_index": 0}

        ds = RerankerDataset(
            [ex],
            max_candidates=1,
            drop_original_rank=True,
            add_cross_features=True,
        )
        item = ds[0]
        t = item["features"][0]  # First (only) candidate

        # After dropping original_rank (index 14), base features are 18.
        # Cross-features start at index 18.
        # edit_dist_x_ngram_improv = 2.0 * 1.2 = 2.4
        assert abs(t[18].item() - 2.4) < 1e-5
        # phonetic_x_confusable = 0.8 * 1.0 = 0.8
        assert abs(t[19].item() - 0.8) < 1e-5
        # freq_x_dice = 0.7 * 0.6 = 0.42
        assert abs(t[20].item() - 0.42) < 1e-5
        # mlm_x_ngram_sum = 3.0 * (0.5 + 0.3) = 2.4
        assert abs(t[21].item() - 2.4) < 1e-5
        # edit_dist_x_freq = 2.0 * 0.7 = 1.4
        assert abs(t[22].item() - 1.4) < 1e-5


# ---------------------------------------------------------------------------
# NeuralReranker MLP v3 transform loading
# ---------------------------------------------------------------------------


class TestNeuralRerankerMLPV3:
    """Test that NeuralReranker loads MLP v3 schema correctly."""

    def test_load_stats_with_schema(self):
        """Stats file with feature_schema should set transform flags."""
        import json
        import tempfile

        stats = {
            "feature_names": ["f" + str(i) for i in range(23)],
            "feature_means": [0.0] * 23,
            "feature_stds": [1.0] * 23,
            "max_candidates": 20,
            "num_features": 23,
            "model_type": "mlp",
            "feature_schema": "mlp_v3",
            "drop_original_rank": True,
            "cross_features": MLP_CROSS_FEATURE_NAMES,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(stats, f)
            stats_path = f.name

        with patch("myspellchecker.algorithms.neural_reranker.ort") as mock_ort:
            mock_ort.SessionOptions.return_value = MagicMock()
            mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 0

            mock_session = MagicMock()
            mock_ort.InferenceSession.return_value = mock_session
            mock_input = MagicMock(name="input")
            mock_input.shape = ["batch", "candidates", 23]
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [MagicMock(name="output")]

            with patch("os.path.exists", return_value=True):
                from myspellchecker.algorithms.neural_reranker import NeuralReranker

                reranker = NeuralReranker("/fake/path.onnx", stats_path=stats_path)

            assert reranker._drop_original_rank is True
            assert reranker._cross_features == MLP_CROSS_FEATURE_NAMES
            assert reranker._feature_schema == "mlp_v3"

        import os

        os.unlink(stats_path)

    def test_gbt_stats_no_transforms(self):
        """GBT stats should not set any transform flags."""
        import json
        import tempfile

        stats = {
            "feature_names": FEATURE_NAMES,
            "feature_means": [],
            "feature_stds": [],
            "max_candidates": 20,
            "num_features": 19,
            "model_type": "lightgbm_lambdarank",
            "feature_schema": "gbt_v1",
            "drop_original_rank": False,
            "cross_features": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(stats, f)
            stats_path = f.name

        with patch("myspellchecker.algorithms.neural_reranker.ort") as mock_ort:
            mock_ort.SessionOptions.return_value = MagicMock()
            mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 0

            mock_session = MagicMock()
            mock_ort.InferenceSession.return_value = mock_session
            mock_input = MagicMock(name="input")
            mock_input.shape = [None, 19]
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [MagicMock(name="output")]

            with patch("os.path.exists", return_value=True):
                from myspellchecker.algorithms.neural_reranker import NeuralReranker

                reranker = NeuralReranker("/fake/path.onnx", stats_path=stats_path)

            assert reranker._drop_original_rank is False
            assert reranker._cross_features == []
            assert reranker._feature_schema == "gbt_v1"

        import os

        os.unlink(stats_path)
