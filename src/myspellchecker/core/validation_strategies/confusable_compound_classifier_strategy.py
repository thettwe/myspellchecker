"""MLP-based confusable/compound detection strategy.

Uses a trained ONNX binary classifier to detect confusable word pairs
and broken compounds. Operates independently from the MLM-based
ConfusableSemanticStrategy — no dependency on error budget or MLM
logit diffs.

The classifier uses 22 features including frequency, n-gram, PMI,
POS tags, and morphological patterns. It was trained on benchmark
examples and mandatory compound data.

Priority: 47 (after Homophone 45, before ConfusableSemantic 48)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from myspellchecker.core.constants import ET_BROKEN_COMPOUND
from myspellchecker.core.response import Error, WordError
from myspellchecker.core.validation_strategies.base import (
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import WordRepository

logger = get_logger(__name__)

TITLE_SUFFIXES = frozenset(
    {
        "ကြီး",
        "ငယ်",
        "သား",
        "ဝန်",
        "တော်",
        "တန်း",
        "မှူး",
    }
)

CONFUSABLE_ERROR_TYPE = "confusable_error"


class ConfusableCompoundClassifierStrategy(ValidationStrategy):
    """MLP classifier for confusable pairs and broken compounds.

    Loads a pre-trained ONNX model and runs binary classification on
    each adjacent word pair. Does NOT depend on MLM or error budget.

    Priority: 47
    """

    _PRIORITY = 47

    def __init__(
        self,
        provider: WordRepository,
        model_path: str,
        stats_path: str | None = None,
        threshold: float = 0.5,
        confidence: float = 0.80,
    ):
        self.provider = provider
        self._threshold = threshold
        self._confidence = confidence
        self._session = None
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

        self._load_model(model_path, stats_path)

    def _load_model(self, model_path: str, stats_path: str | None) -> None:
        """Load ONNX model and normalization stats."""
        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                model_path,
                sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            logger.info("Loaded classifier from %s", model_path)
        except (ImportError, Exception) as e:
            logger.warning("Failed to load classifier: %s", e)
            self._session = None
            return

        # Load stats
        if stats_path is None:
            stats_path = str(Path(model_path).with_suffix(".stats.json"))
        stats_file = Path(stats_path)
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            self._feature_means = np.array(
                stats["feature_means"],
                dtype=np.float32,
            )
            self._feature_stds = np.array(
                stats["feature_stds"],
                dtype=np.float32,
            )
            self._feature_stds = np.where(
                self._feature_stds < 1e-8,
                1.0,
                self._feature_stds,
            )

    def validate(self, context: ValidationContext) -> list[Error]:
        """Check each adjacent word pair with the MLP classifier."""
        if self._session is None or len(context.words) < 2:
            return []

        errors: list[Error] = []

        for i in range(len(context.words) - 1):
            pos_i = context.word_positions[i]
            pos_next = context.word_positions[i + 1]

            # Skip already flagged
            if pos_i in context.existing_errors or pos_next in context.existing_errors:
                continue

            # Skip names
            if context.is_name_mask[i] or context.is_name_mask[i + 1]:
                continue

            w1 = context.words[i]
            w2 = context.words[i + 1]

            # Skip very short words
            if len(w1) < 2 or len(w2) < 2:
                continue

            # Extract features
            prev_word = context.words[i - 1] if i > 0 else ""
            next_word = context.words[i + 2] if i + 2 < len(context.words) else ""
            features = self._extract_features(
                w1,
                w2,
                prev_word,
                next_word,
            )

            # Run inference
            score = self._predict(features)
            if score is None:
                continue

            if score >= self._threshold:
                compound = w1 + w2
                # Determine error type
                if self._is_compound_pair(w1, w2, compound):
                    error = self._build_compound_error(
                        context,
                        i,
                        w1,
                        w2,
                        compound,
                        score,
                    )
                else:
                    error = self._build_confusable_error(
                        context,
                        i,
                        w1,
                        w2,
                        score,
                    )
                if error is not None:
                    errors.append(error)
                    context.existing_errors[pos_i] = error.error_type
                    context.existing_confidences[pos_i] = self._confidence * score

        return errors

    def _extract_features(
        self,
        w1: str,
        w2: str,
        prev_word: str,
        next_word: str,
    ) -> list[float]:
        """Extract 22 features for a word pair."""
        f1 = self.provider.get_word_frequency(w1)
        f2 = self.provider.get_word_frequency(w2)
        compound = w1 + w2
        fc = self.provider.get_word_frequency(compound)

        # Bigram
        bp = 0.0
        if hasattr(self.provider, "get_bigram_probability"):
            bp = self.provider.get_bigram_probability(w1, w2)

        # PMI / NPMI
        pmi = 0.0
        npmi = 0.0
        if hasattr(self.provider, "get_collocation_pmi"):
            pmi = self.provider.get_collocation_pmi(w1, w2) or 0.0
        if hasattr(self.provider, "get_collocation_npmi"):
            npmi = self.provider.get_collocation_npmi(w1, w2) or 0.0

        # POS
        pos1 = self._get_pos(w1)
        pos2 = self._get_pos(w2)

        # Context bigrams
        bp_left = 0.0
        bp_right = 0.0
        if hasattr(self.provider, "get_bigram_probability"):
            if prev_word:
                bp_left = self.provider.get_bigram_probability(
                    prev_word,
                    w1,
                )
            if next_word:
                bp_right = self.provider.get_bigram_probability(
                    w2,
                    next_word,
                )

        # Syllable counts
        syl1 = self._get_syllable_count(w1)
        syl2 = self._get_syllable_count(w2)

        return [
            math.log1p(f1),  # 0
            math.log1p(f2),  # 1
            math.log1p(fc),  # 2
            math.log1p(fc / max(min(f1, f2), 1)),  # 3
            1.0 if fc > 0 else 0.0,  # 4
            bp,  # 5
            pmi,  # 6
            npmi,  # 7
            1.0 if pos1 == "V" else 0.0,  # 8
            1.0 if pos1 == "N" else 0.0,  # 9
            1.0 if pos2 in ("PART", "PPM") else 0.0,  # 10
            1.0 if pos2 == "N" else 0.0,  # 11
            1.0 if w2 in TITLE_SUFFIXES else 0.0,  # 12
            1.0 if w1 == "\u1021" else 0.0,  # 13: အ
            1.0 if w1 == w2 else 0.0,  # 14
            float(syl1),  # 15
            float(syl2),  # 16
            float(len(compound)),  # 17
            bp_left,  # 18
            bp_right,  # 19
            math.log1p(f1) - math.log1p(f2),  # 20
            1.0 if fc > max(f1, f2) else 0.0,  # 21
        ]

    def _predict(self, features: list[float]) -> float | None:
        """Run ONNX inference, return sigmoid probability."""
        if self._session is None:
            return None

        feat_array = np.array([features], dtype=np.float32)

        # Normalize
        if self._feature_means is not None and self._feature_stds is not None:
            feat_array = (feat_array - self._feature_means) / self._feature_stds

        logit = self._session.run(
            [self._output_name],
            {self._input_name: feat_array},
        )[0]

        # Sigmoid
        logit_val = float(logit.flat[0])
        return 1.0 / (1.0 + math.exp(-logit_val))

    def _get_pos(self, word: str) -> str:
        """Get primary POS tag via provider interface."""
        if hasattr(self.provider, "get_word_pos"):
            result = self.provider.get_word_pos(word)
            if result:
                return result.split("|")[0]
        return "UNK"

    def _get_syllable_count(self, word: str) -> int:
        """Get syllable count via provider interface."""
        if hasattr(self.provider, "get_syllable_count"):
            result = self.provider.get_syllable_count(word)
            if result and result > 0:
                return result
        return max(1, len(word) // 3)

    def _is_compound_pair(
        self,
        w1: str,
        w2: str,
        compound: str,
    ) -> bool:
        """Determine if this is a compound (vs confusable) detection."""
        # If compound form exists in dictionary → compound
        if hasattr(self.provider, "is_valid_word"):
            return self.provider.is_valid_word(compound)
        return False

    def _build_compound_error(
        self,
        context: ValidationContext,
        i: int,
        w1: str,
        w2: str,
        compound: str,
        score: float,
    ) -> WordError | None:
        """Build a broken compound error."""
        pos_i = context.word_positions[i]
        local_start = context.sentence.find(w1)
        if local_start >= 0:
            local_end = context.sentence.find(w2, local_start + len(w1))
            if local_end >= 0:
                local_end += len(w2)
            else:
                local_end = local_start + len(w1)
            span_text = context.sentence[local_start:local_end]
        else:
            span_text = w1 + w2

        return WordError(
            text=span_text,
            position=pos_i,
            error_type=ET_BROKEN_COMPOUND,
            suggestions=[compound],
            confidence=self._confidence * score,
        )

    def _build_confusable_error(
        self,
        context: ValidationContext,
        i: int,
        w1: str,
        w2: str,
        score: float,
    ) -> WordError | None:
        """Build a confusable error (placeholder — needs variant gen)."""
        # For now, the MLP detects that this pair is confusable
        # but generating the correct suggestion requires the variant
        # pipeline. Return None to avoid flagging without a suggestion.
        return None

    def priority(self) -> int:
        """Return strategy execution priority (47)."""
        return self._PRIORITY

    def __repr__(self) -> str:
        loaded = self._session is not None
        return (
            f"ConfusableCompoundClassifierStrategy("
            f"priority={self._PRIORITY}, loaded={loaded}, "
            f"threshold={self._threshold})"
        )
