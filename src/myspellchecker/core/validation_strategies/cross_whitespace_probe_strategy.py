"""Cross-whitespace compound probe strategy.

Detects compound words that users split with whitespace (e.g.,
``လူ သွား လမ်း`` → ``လူသွားလမ်း``). Each whitespace-delimited chunk passes
dictionary validation individually, so no downstream strategy sees the error.

The probe concatenates adjacent Myanmar spans from the raw sentence and checks
whether the concatenation is a known high-frequency dictionary word. When it is,
the original space-separated text is flagged as a spelling error with the
concatenated form as the suggestion.

Priority **21** — early structural phase, before
:class:`PreSegmenterRawProbeStrategy` (23) and all segmentation-dependent
strategies. Gated by :attr:`ValidationConfig.use_cross_whitespace_probe`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from myspellchecker.core.constants import ET_BROKEN_COMPOUND
from myspellchecker.core.constants.myanmar_constants import SKIPPED_CONTEXT_WORDS
from myspellchecker.core.response import Error, Suggestion, WordError
from myspellchecker.core.validation_strategies.base import (
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import WordRepository

logger = get_logger(__name__)

_PRIORITY = 21

_MYANMAR_SPAN_REGEX = re.compile(r"[က-႟ꧠ-꧿ꩠ-ꩿ]+")


class CrossWhitespaceProbeStrategy(ValidationStrategy):
    """Probe adjacent whitespace-delimited Myanmar spans for compound words.

    Args:
        provider: Dictionary provider for word validation and frequency.
        enabled: Master on/off switch.
        min_concat_freq: Minimum frequency the concatenated form must have.
        max_part_length: Maximum codepoint length per individual part.
        max_concat_length: Maximum codepoint length of the concatenated form.
        confidence: Confidence score for emitted errors.
    """

    def __init__(
        self,
        provider: WordRepository,
        *,
        enabled: bool = False,
        min_concat_freq: int = 50,
        max_part_length: int = 30,
        max_concat_length: int = 40,
        confidence: float = 0.90,
    ) -> None:
        self.provider = provider
        self.enabled = enabled
        self.min_concat_freq = min_concat_freq
        self.max_part_length = max_part_length
        self.max_concat_length = max_concat_length
        self.confidence = confidence

    def priority(self) -> int:
        return _PRIORITY

    def validate(self, context: ValidationContext) -> list[Error]:
        if not self.enabled:
            return []

        sentence = context.sentence
        if not sentence:
            return []

        sentence_base = self._resolve_sentence_base(context)

        spans = list(_MYANMAR_SPAN_REGEX.finditer(sentence))
        if len(spans) < 2:
            return []

        errors: list[Error] = []
        consumed: set[int] = set()

        for width in (3, 2):
            for i in range(len(spans) - width + 1):
                if any(j in consumed for j in range(i, i + width)):
                    continue

                window = spans[i : i + width]
                if not self._window_gaps_whitespace_only(sentence, window):
                    continue

                parts = [s.group(0) for s in window]
                if any(len(p) > self.max_part_length for p in parts):
                    continue

                if any(p in SKIPPED_CONTEXT_WORDS for p in parts):
                    continue

                concat = "".join(parts)
                if len(concat) > self.max_concat_length:
                    continue

                abs_start = sentence_base + window[0].start()
                if abs_start in context.existing_errors:
                    continue

                if not self._is_compound_split(parts, concat):
                    continue

                if self._overlaps_name(context, window[0].start(), window[-1].end(), sentence_base):
                    continue

                original_text = sentence[window[0].start() : window[-1].end()]
                error = WordError(
                    text=original_text,
                    position=abs_start,
                    error_type=ET_BROKEN_COMPOUND,
                    suggestions=[Suggestion(text=concat, source="cross_whitespace_probe")],
                    confidence=self.confidence,
                )
                errors.append(error)
                for j in range(i, i + width):
                    consumed.add(j)
                logger.debug(
                    "cross_whitespace_probe: '%s' -> '%s' (width=%d, freq=%d)",
                    original_text,
                    concat,
                    width,
                    self.provider.get_word_frequency(concat) or 0,
                )

        return errors

    @staticmethod
    def _window_gaps_whitespace_only(sentence: str, window: list[re.Match[str]]) -> bool:
        for k in range(len(window) - 1):
            gap = sentence[window[k].end() : window[k + 1].start()]
            if gap.strip():
                return False
        return True

    def _is_compound_split(self, parts: list[str], concat: str) -> bool:
        if not self.provider.is_valid_word(concat):
            return False
        concat_freq = self.provider.get_word_frequency(concat) or 0
        if concat_freq < self.min_concat_freq:
            return False
        for part in parts:
            if not self.provider.is_valid_word(part):
                return False
        return True

    @staticmethod
    def _resolve_sentence_base(context: ValidationContext) -> int:
        if not context.words or not context.word_positions:
            return 0
        first_local = context.sentence.find(context.words[0]) if context.sentence else 0
        if first_local < 0:
            first_local = 0
        return max(0, context.word_positions[0] - first_local)

    @staticmethod
    def _overlaps_name(
        context: ValidationContext,
        local_start: int,
        local_end: int,
        sentence_base: int,
    ) -> bool:
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
            f"CrossWhitespaceProbeStrategy(priority={self.priority()}, "
            f"enabled={self.enabled}, "
            f"min_concat_freq={self.min_concat_freq})"
        )
