"""Probe-based syllable-span detection strategy (priority 85).

Drop-in replacement for the v3 GECToRValidationStrategy. Uses a frozen
GKLMIP-BERT encoder + a single Linear head to emit per-syllable detection
scores. High-prob syllables get projected onto words via direct overlap or
whitespace-adjacency (a high-prob whitespace syllable attaches to the
preceding Myanmar word — the broken_compound signal).

Suggestions are emitted empty; the downstream suggestion pipeline generates
SymSpell candidates. Composes with ProbeBoostedCompoundStrategy at priority 24.

See ``30_Audits/Probe Hybrid Ships at +0.0067 2026-05-03.md`` for design and
benchmark results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.core.response import Error
from myspellchecker.core.validation_strategies.base import (
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from myspellchecker.algorithms.probe.syllable_span_probe import (
        ProbeInferenceEngine,
    )

logger = get_logger(__name__)

ET_PROBE = "gector_correction"


class ProbeValidationStrategy(ValidationStrategy):
    """Frozen-encoder syllable-span detection (priority 85)."""

    bypass_fast_path = True

    def __init__(
        self,
        engine: "ProbeInferenceEngine",
        threshold: float = 0.75,
        max_existing_errors: int = 100,
    ):
        self._engine = engine
        self._threshold = threshold
        self._max_existing_errors = max_existing_errors

    def priority(self) -> int:
        return 85

    def validate(self, context: ValidationContext) -> list[Error]:
        if not context.words:
            return []
        if len(context.existing_errors) > self._max_existing_errors:
            return []
        if not context.sentence:
            return []

        try:
            probs, syl_spans = self._engine.score_sentence(context.sentence)
        except Exception:
            logger.error("Probe inference failed", exc_info=True)
            return []
        if not syl_spans:
            return []

        word_spans = [
            (
                context.word_positions[i],
                context.word_positions[i] + len(context.words[i]),
            )
            for i in range(min(len(context.words), len(context.word_positions)))
        ]

        # Per-word max prob: include syllables that overlap the word OR are a
        # whitespace syllable immediately following the word (compound boundary).
        word_max_prob = [0.0] * len(word_spans)
        for s_idx, span in enumerate(syl_spans):
            if probs[s_idx] < self._threshold:
                continue
            is_whitespace = span.text.strip() == ""
            for w_idx, (ws, we) in enumerate(word_spans):
                overlaps = span.start < we and span.end > ws
                adjacent = is_whitespace and span.start == we
                if overlaps or adjacent:
                    if probs[s_idx] > word_max_prob[w_idx]:
                        word_max_prob[w_idx] = float(probs[s_idx])

        errors: list[Error] = []
        for w_idx, max_prob in enumerate(word_max_prob):
            if max_prob < self._threshold:
                continue
            if w_idx >= len(context.words):
                continue
            if context.is_name_mask and w_idx < len(context.is_name_mask):
                if context.is_name_mask[w_idx]:
                    continue
            word = context.words[w_idx]
            position = context.word_positions[w_idx]
            errors.append(
                Error(
                    text=word,
                    position=position,
                    error_type=ET_PROBE,
                    suggestions=[],
                    confidence=max_prob,
                    source_strategy="GECToRValidationStrategy",
                )
            )
            context.existing_errors[position] = ET_PROBE
            context.existing_suggestions[position] = []
            context.existing_confidences[position] = max_prob
        return errors
