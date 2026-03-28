"""ONNX-based neural suggestion reranker using a feature MLP.

This module provides a lightweight MLP reranker that scores spell checker
suggestion candidates using 20 extracted features.  The model is trained
offline (see ``training/reranker_data.py`` for data generation) and
deployed as a quantized ONNX model for sub-millisecond inference.

The reranker is integrated as the LAST step in the suggestion pipeline,
running after both n-gram and semantic reranking.

Feature vector layout (20 features, keep in sync with
``training/reranker_data.py::FEATURE_NAMES``):

    0. edit_distance        - raw Damerau-Levenshtein distance
    1. weighted_distance    - Myanmar-weighted edit distance
    2. log_frequency        - log1p(word_frequency)
    3. phonetic_score       - phonetic similarity [0, 1]
    4. syllable_count_diff  - absolute syllable count difference
    5. plausibility_ratio   - weighted_dist / raw_dist
    6. span_length_ratio    - len(candidate) / len(error)
    7. mlm_logit            - MLM logit score (0 if unavailable)
    8. ngram_left_prob      - left context n-gram probability
    9. ngram_right_prob     - right context n-gram probability
   10. is_confusable        - 1.0 if Myanmar confusable variant
   11. source_symspell      - 1.0 if from SymSpell
   12. source_morpheme      - 1.0 if from morpheme strategy
   13. source_context       - 1.0 if from context strategy
   14. source_compound      - 1.0 if from compound reconstruction
   15. source_other         - 1.0 if from other source
   16. relative_log_freq    - log_freq / max(log_freq) within candidate list
   17. char_length_diff     - len(candidate) - len(error), signed
   18. is_substring         - 1.0 if candidate contains error or vice versa
   19. original_rank        - 1/(1+rank) prior ranking signal
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

# Expected number of features (must match training)
_NUM_FEATURES = 20


class NeuralReranker:
    """ONNX-based neural suggestion reranker using a feature MLP."""

    def __init__(
        self,
        model_path: str,
        stats_path: str | None = None,
    ):
        """Load ONNX model and optional normalization stats.

        Args:
            model_path: Path to the ONNX model file.
            stats_path: Optional path to a JSON file containing
                ``feature_means`` and ``feature_stds`` arrays used to
                z-score normalize input features before inference.
                If None, raw features are passed to the model.

        Raises:
            ImportError: If ``onnxruntime`` is not installed.
            FileNotFoundError: If *model_path* does not exist.
            ValueError: If the stats file has an unexpected structure.
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

        # Load normalization stats
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

        if stats_path and os.path.exists(stats_path):
            self._load_stats(stats_path)

        logger.info(
            "Loaded NeuralReranker from %s (stats=%s)",
            model_path,
            "yes" if self._feature_means is not None else "no",
        )

    def _load_stats(self, stats_path: str) -> None:
        """Load feature normalization statistics from JSON.

        Expected JSON structure::

            {
                "feature_means": [float, ...],  // length == 20
                "feature_stds": [float, ...]    // length == 20
            }
        """
        with open(stats_path, encoding="utf-8") as f:
            data = json.load(f)

        means = data.get("feature_means")
        stds = data.get("feature_stds")

        if means is None or stds is None:
            logger.warning(
                "Stats file %s missing feature_means or feature_stds; skipping normalization.",
                stats_path,
            )
            return

        self._feature_means = np.array(means, dtype=np.float32)
        self._feature_stds = np.array(stds, dtype=np.float32)

        # Prevent division by zero
        self._feature_stds = np.where(self._feature_stds < 1e-8, 1.0, self._feature_stds)

        if len(self._feature_means) != _NUM_FEATURES:
            logger.warning(
                "Expected %d features in stats, got %d; disabling normalization.",
                _NUM_FEATURES,
                len(self._feature_means),
            )
            self._feature_means = None
            self._feature_stds = None

    def score_candidates(
        self,
        features: list[list[float]],
    ) -> list[float]:
        """Score each candidate using the MLP.

        Args:
            features: Feature matrix of shape ``(num_candidates, 20)``.

        Returns:
            List of scores (higher is better), one per candidate.
            Returns an empty list if features are empty or inference
            fails.
        """
        if not features:
            return []

        try:
            feat_array = np.array(features, dtype=np.float32)

            # Normalize if stats are available
            if self._feature_means is not None and self._feature_stds is not None:
                feat_array = (feat_array - self._feature_means) / self._feature_stds

            # Model expects (batch, candidates, features) — add batch dim
            if feat_array.ndim == 2:
                feat_array = feat_array[np.newaxis, :, :]  # (1, N, 20)

            outputs = self._session.run(
                [self._output_name],
                {self._input_name: feat_array},
            )
            raw_scores = outputs[0]

            # Output is (batch, candidates) or (batch, candidates, 1)
            if raw_scores.ndim == 3:
                raw_scores = raw_scores[:, :, 0]
            # Remove batch dimension
            if raw_scores.ndim == 2:
                raw_scores = raw_scores[0]

            return raw_scores.tolist()
        except Exception as e:
            logger.debug("Neural reranker inference failed: %s", e)
            return []

    def rerank(
        self,
        suggestions: list[str],
        features: list[list[float]],
    ) -> list[str]:
        """Rerank suggestions by neural score.

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
