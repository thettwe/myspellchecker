"""Probe-boosted broken-compound detection strategy (priority 24).

Uses the same trained probe as ProbeValidationStrategy, but operates as a
pre-filter for the existing BrokenCompoundStrategy (priority 25). For each
adjacent word pair separated by whitespace, this strategy:

1. Reads the probe's per-syllable score on the whitespace syllable between
   the words (the broken_compound signal).
2. If the score >= threshold AND the merged compound exists in the dictionary
   at sufficient frequency, emits a broken_compound error with the merged
   compound as the top-1 suggestion.

This bypasses BrokenCompoundStrategy's rare_threshold and compound_ratio
heuristic gates (which reject many true positives) and replaces them with
neural evidence + dictionary membership.

Composes with ProbeValidationStrategy at priority 85.
See ``30_Audits/Probe Hybrid Ships at +0.0067 2026-05-03.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.core.response import Error, Suggestion
from myspellchecker.core.validation_strategies.base import (
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from myspellchecker.algorithms.probe.syllable_span_probe import (
        ProbeInferenceEngine,
    )
    from myspellchecker.providers.base import DictionaryProvider

logger = get_logger(__name__)

ET_BROKEN_COMPOUND = "broken_compound"


class ProbeBoostedCompoundStrategy(ValidationStrategy):
    """Probe + dict gate for broken-compound detection (priority 24)."""

    bypass_fast_path = True

    def __init__(
        self,
        engine: "ProbeInferenceEngine",
        provider: "DictionaryProvider",
        threshold: float = 0.7,
        compound_min_freq: int = 50,
        max_existing_errors: int = 100,
    ):
        self._engine = engine
        self._provider = provider
        self._threshold = threshold
        self._compound_min_freq = compound_min_freq
        self._max_existing_errors = max_existing_errors

    def priority(self) -> int:
        return 24

    def validate(self, context: ValidationContext) -> list[Error]:
        if len(context.words) < 2:
            return []
        if len(context.existing_errors) > self._max_existing_errors:
            return []
        if not context.sentence:
            return []

        try:
            probs, syl_spans = self._engine.score_sentence(context.sentence)
        except Exception:
            logger.error("Probe inference failed in compound strategy", exc_info=True)
            return []
        if not syl_spans:
            return []

        # Index probability of whitespace syllables by their start char position.
        ws_prob_at_pos: dict[int, float] = {}
        for s_idx, span in enumerate(syl_spans):
            if span.text.strip() == "":
                ws_prob_at_pos[span.start] = float(probs[s_idx])

        errors: list[Error] = []
        for i in range(len(context.words) - 1):
            if i >= len(context.word_positions):
                break
            if i + 1 >= len(context.word_positions):
                break
            pos_i = context.word_positions[i]
            pos_next = context.word_positions[i + 1]
            if pos_i in context.existing_errors or pos_next in context.existing_errors:
                continue
            w1 = context.words[i]
            w2 = context.words[i + 1]
            if context.is_name_mask:
                if i < len(context.is_name_mask) and context.is_name_mask[i]:
                    continue
                if i + 1 < len(context.is_name_mask) and context.is_name_mask[i + 1]:
                    continue
            w1_end = pos_i + len(w1)
            # Skip if no whitespace gap between the two words
            if pos_next == w1_end:
                continue
            # Skip Pali stacking compounds — segmenter splits these unreliably
            if "္" in w1 or "္" in w2:
                continue
            # Skip reduplication
            if w1 == w2:
                continue

            ws_prob = ws_prob_at_pos.get(w1_end, 0.0)
            if ws_prob < self._threshold:
                continue

            compound = w1 + w2
            try:
                if not self._provider.is_valid_word(compound):
                    continue
                compound_freq = self._provider.get_word_frequency(compound)
                if compound_freq < self._compound_min_freq:
                    continue
            except Exception:
                continue

            errors.append(
                Error(
                    text=f"{w1} {w2}",
                    position=pos_i,
                    error_type=ET_BROKEN_COMPOUND,
                    suggestions=[Suggestion(text=compound)],
                    confidence=min(0.95, ws_prob),
                    source_strategy="ProbeBoostedCompoundStrategy",
                )
            )
            context.existing_errors[pos_i] = ET_BROKEN_COMPOUND
            context.existing_errors[pos_next] = ET_BROKEN_COMPOUND
            context.existing_suggestions[pos_i] = [compound]
            context.existing_confidences[pos_i] = ws_prob
        return errors
