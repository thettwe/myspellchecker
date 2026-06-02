"""Probe-boosted no-whitespace over-segmentation rescue strategy (priority 26).

Targets the residual broken-compound bucket where the segmenter splits a
typo into adjacent dictionary-valid tokens (no whitespace between them).
Example: ``သံဂါ`` → segmented as ``သံ`` + ``ဂါ`` (both valid standalone),
so WordValidator never sees the typo. SymSpell would find the gold
``သံဃာ`` if given the merged form, but no upstream strategy passes it.

Algorithm: for each adjacent (no-whitespace) Myanmar pair (w1, w2), if the
trained probe scores syllables on the merge boundary above threshold AND a
SymSpell ed=1 lookup on (w1+w2) returns a high-frequency dict word that
isn't the merged form itself, emit a broken_compound error with that
candidate as the top-1 suggestion.

Composes with ProbeBoostedCompoundStrategy at priority 24 (which handles
the *whitespace-adjacent* compound case) and ProbeValidationStrategy at
priority 85 (general detection). All three share one ProbeInferenceEngine.

See ``30_Audits/Probe Phase 2A Ships at +0.0111 2026-05-04.md``.
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
    from myspellchecker.algorithms.symspell import SymSpell
    from myspellchecker.providers.base import DictionaryProvider

logger = get_logger(__name__)

ET_BROKEN_COMPOUND = "broken_compound"
VIRAMA = "္"  # U+1039 — Pali stacking marker; segmenter splits these unreliably

# Particles that should never participate in a "broken-compound" merge.
# Sourced from over-segmentation research 2026-04-29.
NEVER_MERGE_PARTICLES = frozenset(
    {
        "က",
        "ကို",
        "မှာ",
        "နဲ့",
        "တယ်",
        "ပါတယ်",
        "လား",
        "လဲ",
        "ပဲ",
        "တော့",
        "ပြီး",
        "သည်",
        "ဟာ",
        "၌",
        "မှ",
        "ရဲ့",
        "၏",
        "နှင့်",
        "များ",
        "ပါ",
        "နှ",
        "သို့",
    }
)


class ProbeSegmenterRescueStrategy(ValidationStrategy):
    """No-whitespace over-segmentation rescue (priority 26)."""

    bypass_fast_path = True

    def __init__(
        self,
        engine: "ProbeInferenceEngine",
        provider: "DictionaryProvider",
        symspell: "SymSpell",
        threshold: float = 0.75,
        min_freq: int = 2000,
        max_existing_errors: int = 100,
    ):
        self._engine = engine
        self._provider = provider
        self._symspell = symspell
        self._threshold = threshold
        self._min_freq = min_freq
        self._max_existing_errors = max_existing_errors

    def priority(self) -> int:
        return 26  # after BrokenCompoundStrategy (25), so it doesn't preempt

    def _is_excluded(self, w1: str, w2: str) -> bool:
        """Linguistic exclusion gates from over-segmentation research."""
        if VIRAMA in w1 or VIRAMA in w2:
            return True
        if w1 == w2:
            return True
        if w1 in NEVER_MERGE_PARTICLES or w2 in NEVER_MERGE_PARTICLES:
            return True
        if len(w1) <= 1 or len(w2) <= 1:
            return True
        return False

    def _lookup_merged(self, merged: str) -> str | None:
        """Return top dict-valid candidate at ed=1 with sufficient frequency.

        Restricted to ed=1 because ed=2 introduces too many false matches in
        the compound-merge use case (a different consonant cluster would
        silently substitute valid but unrelated words).
        """
        try:
            res = self._symspell.lookup(
                merged, level="word", max_suggestions=5, include_known=False
            )
        except Exception:
            return None
        for r in res:
            cand = r.term
            if cand == merged or r.edit_distance > 1:
                continue
            try:
                if not self._provider.is_valid_word(cand):
                    continue
                freq = self._provider.get_word_frequency(cand)
            except Exception:
                continue
            if freq is not None and freq >= self._min_freq:
                return cand
        return None

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
            logger.error("Probe inference failed in rescue strategy", exc_info=True)
            return []
        if not syl_spans:
            return []

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
            w1_end = pos_i + len(w1)
            # Only the no-whitespace path; whitespace cases are handled by
            # ProbeBoostedCompoundStrategy at priority 24.
            if pos_next != w1_end:
                continue
            if self._is_excluded(w1, w2):
                continue
            if context.is_name_mask:
                if i < len(context.is_name_mask) and context.is_name_mask[i]:
                    continue
                if i + 1 < len(context.is_name_mask) and context.is_name_mask[i + 1]:
                    continue

            merge_start = pos_i
            merge_end = pos_next + len(w2)
            max_prob = 0.0
            for s_idx, span in enumerate(syl_spans):
                if span.start < merge_end and span.end > merge_start:
                    if probs[s_idx] > max_prob:
                        max_prob = float(probs[s_idx])
            if max_prob < self._threshold:
                continue

            merged = w1 + w2
            cand = self._lookup_merged(merged)
            if cand is None:
                continue
            try:
                # Don't flag if the merged form itself is already a valid word
                # (no fix needed; user may have intentionally typed it).
                if self._provider.is_valid_word(merged):
                    continue
            except Exception:
                continue

            errors.append(
                Error(
                    text=merged,
                    position=pos_i,
                    error_type=ET_BROKEN_COMPOUND,
                    suggestions=[Suggestion(text=cand)],
                    confidence=min(0.9, max_prob),
                    source_strategy="ProbeSegmenterRescueStrategy",
                )
            )
            context.existing_errors[pos_i] = ET_BROKEN_COMPOUND
            context.existing_errors[pos_next] = ET_BROKEN_COMPOUND
            context.existing_suggestions[pos_i] = [cand]
            context.existing_confidences[pos_i] = max_prob
        return errors
