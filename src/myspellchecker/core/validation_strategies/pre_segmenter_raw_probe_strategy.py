"""Pre-segmenter raw-token SymSpell probe strategy.

Recovers compound typos that the segmenter fragments into individually-valid
subtokens. Canonical example: ``စွမ်းဆောင်ရည`` (missing the final ``်``) is
split by :class:`DefaultSegmenter` into ``[စွမ်းဆောင်, ရ, ည]``; each piece is
dictionary-valid, so no downstream strategy flags the span.

The fix is structural rather than algorithmic: probe
:meth:`SymSpell.lookup` directly on the raw, unsegmented token
(``SymSpell.lookup(raw_token, level='word')``) and emit a correction when
the top candidate is an edit-distance-≤2 high-frequency dictionary word.

Priority **23** — inside the structural phase, surviving the fast-path
cutoff at ``ContextValidator._FAST_PATH_PRIORITY_CUTOFF``. Gated by
:attr:`ValidationConfig.use_pre_segmenter_raw_probe`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from myspellchecker.algorithms.distance.edit_distance import (
    weighted_damerau_levenshtein_distance,
)
from myspellchecker.core.constants import ET_WORD
from myspellchecker.core.response import Error, Suggestion, WordError
from myspellchecker.core.validation_strategies.base import (
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.text.phonetic_data import is_colloquial_variant
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.symspell import SymSpell
    from myspellchecker.providers.interfaces import WordRepository

logger = get_logger(__name__)

_PRIORITY = 23

# Whitespace / punctuation delimited Myanmar spans. Matches the widest
# character set the library validates (core block plus Extended-A/B so that
# Shan/Mon tokens are still surfaced to downstream strategies, even though
# they will usually fall through without a SymSpell hit).
_RAW_TOKEN_REGEX = re.compile(r"[\u1000-\u109F\uA9E0-\uA9FF\uAA60-\uAA7F]+")


class PreSegmenterRawProbeStrategy(ValidationStrategy):
    """Probe SymSpell on raw whitespace-delimited tokens before segmentation.

    The strategy iterates the Myanmar-character spans of ``context.sentence``
    directly (bypassing :class:`Segmenter`). Each span is gated to avoid
    emissions on already-valid dictionary words, proper names, colloquial
    forms, and tokens that exceed the SymSpell index's length cap.

    Args:
        symspell: Configured :class:`SymSpell` instance (already indexed).
        provider: Dictionary provider for ``is_valid_word`` /
            ``get_word_frequency`` lookups.
        enabled: Master on/off switch. Wired from
            :attr:`ValidationConfig.use_pre_segmenter_raw_probe`.
        max_edit_distance: Upper bound on accepted SymSpell edit distance.
        min_frequency: Minimum dict frequency a candidate must clear.
        max_token_length: Skip raw tokens longer than this (matches
            SymSpell's ``max_word_length``).
        max_length_diff: Maximum allowed ``|len(candidate) - len(token)|``.
        confidence: Confidence stamped on the emitted :class:`WordError`.
        max_suggestions: SymSpell ``max_suggestions`` parameter per probe.
    """

    def __init__(
        self,
        symspell: SymSpell,
        provider: WordRepository,
        *,
        enabled: bool = False,
        max_edit_distance: int = 2,
        min_frequency: int = 100,
        max_token_length: int = 15,
        max_length_diff: int = 2,
        confidence: float = 0.85,
        max_suggestions: int = 5,
    ) -> None:
        self.symspell = symspell
        self.provider = provider
        self.enabled = enabled
        self.max_edit_distance = max_edit_distance
        self.min_frequency = min_frequency
        self.max_token_length = max_token_length
        self.max_length_diff = max_length_diff
        self.confidence = confidence
        self.max_suggestions = max_suggestions

    def priority(self) -> int:
        """Return strategy execution priority (23, structural phase)."""
        return _PRIORITY

    def validate(self, context: ValidationContext) -> list[Error]:
        """Emit :class:`WordError` for raw tokens that resolve via SymSpell."""
        if not self.enabled or self.symspell is None:
            return []

        sentence = context.sentence
        if not sentence:
            return []

        sentence_base = self._resolve_sentence_base(context)

        errors: list[Error] = []
        for match in _RAW_TOKEN_REGEX.finditer(sentence):
            raw_token = match.group(0)
            local_start = match.start()
            local_end = match.end()
            abs_start = sentence_base + local_start

            if not self._should_probe(raw_token, context, local_start, local_end, sentence_base):
                continue
            if abs_start in context.existing_errors:
                continue

            candidate = self._best_candidate(raw_token)
            if candidate is None:
                continue

            suggestion_text, cand_freq, cand_ed = candidate
            error = WordError(
                text=raw_token,
                position=abs_start,
                error_type=ET_WORD,
                suggestions=[Suggestion(text=suggestion_text, source="pre_segmenter_raw_probe")],
                confidence=self.confidence,
            )
            errors.append(error)
            context.existing_errors[abs_start] = ET_WORD
            context.existing_confidences[abs_start] = self.confidence
            context.existing_suggestions[abs_start] = [suggestion_text]
            logger.debug(
                "pre_segmenter_raw_probe: %s -> %s ed=%.2f freq=%d conf=%.2f",
                raw_token,
                suggestion_text,
                cand_ed,
                cand_freq,
                self.confidence,
            )

        return errors

    def _should_probe(
        self,
        raw_token: str,
        context: ValidationContext,
        local_start: int,
        local_end: int,
        sentence_base: int,
    ) -> bool:
        """Return True if ``raw_token`` is a viable probe target."""
        if not raw_token:
            return False
        if len(raw_token) > self.max_token_length:
            return False
        # Already a dict word → nothing to correct.
        if self.provider.is_valid_word(raw_token):
            return False
        # Respect the colloquial whitelist; lenient mode treats these as info,
        # not errors, and the probe should not upgrade them silently.
        if is_colloquial_variant(raw_token):
            return False
        # Skip tokens that overlap any name-masked word. We check overlap on
        # the segmented word positions the caller already computed.
        if self._overlaps_name(context, local_start, local_end, sentence_base):
            return False
        return True

    def _best_candidate(self, raw_token: str) -> tuple[str, int, float] | None:
        """Probe SymSpell for ``raw_token`` and return the best acceptable hit.

        Returns a ``(suggestion, frequency, edit_distance)`` tuple or
        ``None`` when no candidate clears the guards.
        """
        try:
            suggestions = self.symspell.lookup(
                raw_token,
                level="word",
                max_suggestions=self.max_suggestions,
            )
        except (RuntimeError, ValueError, KeyError):
            logger.debug("SymSpell lookup failed for %r", raw_token, exc_info=True)
            return None

        if not suggestions:
            return None

        for suggestion in suggestions:
            candidate_text = str(suggestion)
            if candidate_text == raw_token:
                continue
            if abs(len(candidate_text) - len(raw_token)) > self.max_length_diff:
                continue
            ed = weighted_damerau_levenshtein_distance(raw_token, candidate_text)
            if ed > self.max_edit_distance:
                continue
            freq = self.provider.get_word_frequency(candidate_text) or 0
            if freq < self.min_frequency:
                continue
            return candidate_text, int(freq), float(ed)

        return None

    @staticmethod
    def _resolve_sentence_base(context: ValidationContext) -> int:
        """Return the absolute offset of ``context.sentence`` in the full text.

        Mirrors the calculation in :class:`HiddenCompoundStrategy`: the first
        word's absolute position minus its local position inside the sentence.
        When ``context.words`` is empty we cannot derive the base and default
        to zero — callers without segmented words are exercised only by
        unit tests, where local and absolute positions coincide.

        Defensive ``max(0, ...)`` clamp: if ``context.sentence.find(words[0])``
        returns -1 (normalization mismatch between the raw sentence and the
        segmenter output) ``first_local`` is clamped to 0, which would make
        the returned base equal ``word_positions[0]`` and corrupt every
        downstream local-to-absolute translation. The outer clamp forces a
        conservative base of 0 in that case; ``abs_start`` then equals
        ``local_start``, which is at worst a small offset error on the first
        sentence rather than a document-wide position corruption. Logs a
        warning so the root-cause mismatch is surfaced.
        """
        if not context.words or not context.word_positions:
            return 0
        first_local = context.sentence.find(context.words[0]) if context.sentence else 0
        if first_local < 0:
            logger.warning(
                "pre_segmenter_raw_probe: first word %r not found in sentence; "
                "clamping sentence_base to 0",
                context.words[0],
            )
            first_local = 0
        return max(0, context.word_positions[0] - first_local)

    @staticmethod
    def _overlaps_name(
        context: ValidationContext,
        local_start: int,
        local_end: int,
        sentence_base: int,
    ) -> bool:
        """Return True if any name-masked word overlaps ``[local_start, local_end)``.

        ``sentence_base`` is threaded in from the caller so we do not pay a
        second ``_resolve_sentence_base`` walk (and do not risk diverging
        from the caller's reference frame).
        """
        if not context.is_name_mask:
            return False
        for idx, word in enumerate(context.words):
            if idx >= len(context.is_name_mask) or not context.is_name_mask[idx]:
                continue
            word_local_start = context.word_positions[idx] - sentence_base
            word_local_end = word_local_start + len(word)
            if word_local_start < local_end and word_local_end > local_start:
                return True
        return False

    def __repr__(self) -> str:
        return (
            f"PreSegmenterRawProbeStrategy(priority={self.priority()}, "
            f"enabled={self.enabled}, "
            f"max_edit_distance={self.max_edit_distance}, "
            f"min_frequency={self.min_frequency})"
        )
