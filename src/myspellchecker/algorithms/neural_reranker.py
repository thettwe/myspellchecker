"""ONNX-based suggestion reranker supporting both MLP and tree-based models.

This module provides a unified reranker that scores spell checker suggestion
candidates using extracted features.  It auto-detects the model type (MLP or
LightGBM/XGBoost) from the ONNX model's input shape and handles inference
accordingly.

Supported model types:
  - **MLP**: Input (batch, candidates, F) → Output (batch, candidates).
    Requires z-score normalization via stats file.
  - **GBT (LightGBM/XGBoost)**: Input (N, F) → Output (N, 1).
    No normalization needed (tree models are scale-invariant).

The reranker is integrated as the LAST step in the suggestion pipeline,
running after both n-gram and semantic reranking.

Feature vector layout (v2, 19 features — keep in sync with
``training/reranker_data.py::FEATURE_NAMES``):

    0. edit_distance              - raw Damerau-Levenshtein distance
    1. weighted_distance          - Myanmar-weighted edit distance
    2. log_frequency              - log1p(word_frequency)
    3. phonetic_score             - phonetic similarity [0, 1]
    4. syllable_count_diff        - absolute syllable count difference
    5. plausibility_ratio         - weighted_dist / raw_dist
    6. span_length_ratio          - len(candidate) / len(error)
    7. mlm_logit                  - MLM logit from semantic checker
    8. ngram_left_prob            - left context n-gram probability
    9. ngram_right_prob           - right context n-gram probability
   10. is_confusable              - 1.0 if Myanmar confusable variant
   11. relative_log_freq          - log_freq / max(log_freq) within candidates
   12. char_length_diff           - len(candidate) - len(error), signed
   13. is_substring               - 1.0 if substring relationship exists
   14. original_rank              - 1/(1+rank) prior ranking signal
   15. ngram_improvement_ratio    - log(P_cand_ctx / P_error_ctx)
   16. edit_type_subst            - 1.0 if primary edit is substitution
   17. edit_type_delete           - 1.0 if primary edit is deletion/insertion
   18. char_dice_coeff            - character bigram Dice coefficient
"""

from __future__ import annotations

import json
import os

import numpy as np

from myspellchecker.utils.logging_utils import get_logger

# Try importing ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

logger = get_logger(__name__)

_NUM_FEATURES = 19


class NeuralReranker:
    """ONNX-based suggestion reranker (supports MLP and GBT models)."""

    def __init__(
        self,
        model_path: str,
        stats_path: str | None = None,
    ):
        """Load ONNX model and optional normalization stats.

        Auto-detects model type from ONNX input shape:
        - 3D input (batch, candidates, features) → MLP mode
        - 2D input (N, features) → GBT mode (LightGBM/XGBoost)

        For MLP models with ``feature_schema == "mlp_v3"``, the reranker
        automatically applies feature transforms at inference time:
        dropping ``original_rank`` and computing cross-features.

        Args:
            model_path: Path to the ONNX model file.
            stats_path: Optional path to a JSON file containing
                normalization stats.  Required for MLP models,
                ignored for GBT models (scale-invariant).

        Raises:
            ImportError: If ``onnxruntime`` is not installed.
            FileNotFoundError: If *model_path* does not exist.
        """
        if ort is None:
            raise ImportError(
                "NeuralReranker requires onnxruntime. Install via: pip install onnxruntime"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Neural reranker model not found at {model_path}")

        # Load ONNX model
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, sess_options)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # Auto-detect model type from input shape
        input_shape = self._session.get_inputs()[0].shape
        if len(input_shape) == 3:
            self._model_type = "mlp"
        else:
            self._model_type = "gbt"

        # Load normalization stats (only used for MLP)
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

        # MLP v3 feature transform flags (loaded from stats file)
        self._drop_original_rank: bool = False
        self._cross_features: list[str] = []
        self._feature_schema: str = ""

        if stats_path and os.path.exists(stats_path):
            self._load_stats(stats_path)

        logger.info(
            "Loaded NeuralReranker from %s (type=%s, schema=%s, stats=%s)",
            model_path,
            self._model_type,
            self._feature_schema or "default",
            "yes" if self._feature_means is not None else "no",
        )

    @property
    def model_type(self) -> str:
        """Return the detected model type ('mlp' or 'gbt')."""
        return self._model_type

    def _load_stats(self, stats_path: str) -> None:
        """Load feature normalization statistics and schema from JSON.

        For MLP models, loads feature_means and feature_stds for z-score
        normalization, plus ``feature_schema`` to determine if MLP-specific
        transforms (drop original_rank, add cross-features) are needed.

        For GBT models, stats are loaded for metadata only (no normalization).
        """
        with open(stats_path, encoding="utf-8") as f:
            data = json.load(f)

        # Load schema info (applies to both model types)
        self._feature_schema = data.get("feature_schema", "")
        self._drop_original_rank = data.get("drop_original_rank", False)
        self._cross_features = data.get("cross_features", [])

        means = data.get("feature_means")
        stds = data.get("feature_stds")

        # GBT models don't need normalization — skip even if stats file exists
        if self._model_type == "gbt":
            return

        if not means or not stds:
            logger.warning(
                "Stats file %s missing feature_means or feature_stds; skipping normalization.",
                stats_path,
            )
            return

        self._feature_means = np.array(means, dtype=np.float32)
        self._feature_stds = np.array(stds, dtype=np.float32)

        # Prevent division by zero
        self._feature_stds = np.where(self._feature_stds < 1e-8, 1.0, self._feature_stds)

    def score_candidates(
        self,
        features: list[list[float]],
    ) -> list[float]:
        """Score each candidate using the loaded model.

        Automatically dispatches to MLP or GBT inference based on the
        detected model type.  For MLP models with ``feature_schema == "mlp_v3"``,
        applies feature transforms (drop original_rank, add cross-features)
        before scoring.

        Args:
            features: Feature matrix of shape ``(num_candidates, 19)``.
                Always expects the base 19-feature layout; MLP transforms
                are applied internally.

        Returns:
            List of scores (higher is better), one per candidate.
            Returns an empty list if features are empty or inference fails.
        """
        if not features:
            return []

        try:
            feat_array = np.array(features, dtype=np.float32)

            if self._model_type == "mlp":
                # Apply MLP v3 feature transforms if schema requires it
                if self._drop_original_rank or self._cross_features:
                    feat_array = self._apply_mlp_transforms(feat_array)
                return self._score_mlp(feat_array)
            else:
                return self._score_gbt(feat_array)
        except Exception as e:
            logger.debug("Neural reranker inference failed: %s", e)
            return []

    def _apply_mlp_transforms(self, feat_array: np.ndarray) -> np.ndarray:
        """Apply MLP-specific feature transforms to a 2D feature array.

        Transforms are applied in order:
        1. Compute cross-features from the base layout (before any drops).
        2. Drop ``original_rank`` column.
        3. Append cross-feature columns.

        Args:
            feat_array: Shape ``(num_candidates, 19)`` — base feature layout.

        Returns:
            Transformed array with shape ``(num_candidates, N)`` where
            N = 19 - dropped + cross_features.
        """
        from myspellchecker.training.reranker_data import (
            MLP_CROSS_FEATURES,
            ORIGINAL_RANK_INDEX,
        )

        base = feat_array  # (N, 19)

        # Compute cross-features from the base layout
        cross_cols: list[np.ndarray] = []
        if self._cross_features:
            for _name, left_idx, right_idx in MLP_CROSS_FEATURES:
                if right_idx == -1:
                    # mlm_logit * (ngram_left + ngram_right)
                    cross_cols.append(base[:, left_idx] * (base[:, 8] + base[:, 9]))
                else:
                    cross_cols.append(base[:, left_idx] * base[:, right_idx])

        # Drop original_rank
        if self._drop_original_rank:
            base = np.delete(base, ORIGINAL_RANK_INDEX, axis=1)

        # Append cross-features
        if cross_cols:
            cross_array = np.column_stack(cross_cols)
            base = np.concatenate([base, cross_array], axis=1)

        return base

    def _score_mlp(self, feat_array: np.ndarray) -> list[float]:
        """Score candidates using MLP model (3D input)."""
        # Normalize if stats are available
        if self._feature_means is not None and self._feature_stds is not None:
            feat_array = (feat_array - self._feature_means) / self._feature_stds

        # MLP expects (batch, candidates, features) — add batch dim
        if feat_array.ndim == 2:
            feat_array = feat_array[np.newaxis, :, :]  # (1, N, 19)

        outputs = self._session.run(
            [self._output_name],
            {self._input_name: feat_array},
        )
        raw_scores = outputs[0]

        # Output is (batch, candidates) or (batch, candidates, 1)
        if raw_scores.ndim == 3:
            raw_scores = raw_scores[:, :, 0]
        if raw_scores.ndim == 2:
            raw_scores = raw_scores[0]

        return raw_scores.tolist()

    def _score_gbt(self, feat_array: np.ndarray) -> list[float]:
        """Score candidates using GBT model (2D input)."""
        # GBT expects (N, features) — already 2D, no batch dim needed
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: feat_array},
        )
        raw_scores = outputs[0]

        # Output is (N, 1) — squeeze to (N,)
        if raw_scores.ndim == 2:
            raw_scores = raw_scores[:, 0]

        return raw_scores.tolist()

    def rerank(
        self,
        suggestions: list[str],
        features: list[list[float]],
    ) -> list[str]:
        """Rerank suggestions by model score.

        Args:
            suggestions: Candidate suggestion strings.
            features: Feature matrix aligned with *suggestions*.

        Returns:
            Reordered suggestion list (best first).
            Returns the original list unchanged if scoring fails.
        """
        scores = self.score_candidates(features)
        if not scores or len(scores) != len(suggestions):
            return suggestions

        paired = sorted(
            zip(scores, suggestions, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )
        return [s for _, s in paired]
