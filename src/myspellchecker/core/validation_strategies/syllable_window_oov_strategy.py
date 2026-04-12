"""Syllable-window OOV detection strategy.

Detects multi-syllable typos that the word segmenter decomposes into
individually valid syllables. Example: an OOV compound ``ခုန်ကျစရိတ်``
(intended ``ကုန်ကျစရိတ်``, "costs") is segmented as ``['ခုန်', 'ကျ', 'စရိတ်']``
— each token valid, so no upstream strategy fires.

The strategy enumerates contiguous syllable windows (default sizes 2-4)
across adjacent words, joins them, and consults SymSpell for a high-
frequency near-match. When the joined string is OOV and a candidate exists
above the frequency / confidence floors, the window is emitted as a
:data:`~myspellchecker.core.constants.ET_SYLLABLE_WINDOW_OOV` error.

Priority 22 places it in the structural phase (≤ 25) so it runs on
"clean" sentences. The strategy does not populate ``existing_errors`` so
HiddenCompound and other priority-23+ strategies can still fire at
overlapping positions with their own error types.

FPR mitigations:

- ``skip_names``: reject windows spanning ``context.is_name_mask`` words.
- ``require_valid_source_words``: every contributing word must itself be valid.
- ``require_typo_prone``: joined window must contain a typo-prone character.
- Confidence floor and frequency threshold gate weak matches.
- One emission per start position; the highest-confidence window wins.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from myspellchecker.core.constants import ET_SYLLABLE_WINDOW_OOV
from myspellchecker.core.response import Error, Suggestion, WordError
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.text.normalize import normalize
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.symspell import SymSpell
    from myspellchecker.providers.interfaces import WordRepository

logger = get_logger(__name__)

_PRIORITY = 22

# Characters that frequently appear in typos. A candidate window must contain
# at least one of these to be considered. Mirrors HiddenCompoundStrategy.
_TYPO_PRONE_CHARS: frozenset[str] = frozenset(
    [
        # Aspirated / unaspirated consonant pairs
        "ခ",
        "က",
        "ဂ",
        "ဃ",
        "ပ",
        "ဖ",
        "ဗ",
        "ဘ",
        "ဒ",
        "ဓ",
        "တ",
        "ထ",
        "ဇ",
        "ဈ",
        "ဆ",
        "စ",
        # Medials
        "ျ",
        "ြ",
        "ွ",
        "ှ",
        # Rhotic / liquid confusion
        "ရ",
        "ယ",
        "လ",
        # Nasal confusion
        "န",
        "မ",
        "ံ",
        # Retroflex / dental confusion
        "ည",
        "ဏ",
        "ဋ",
        "ဍ",
        # Vowel length confusion
        "ိ",
        "ီ",
        "ု",
        "ူ",
        "ေ",
        "ဲ",
        "ော",
        "ို",
    ]
)


class SyllableWindowOOVStrategy(ValidationStrategy):
    """Detect multi-syllable OOV typos hidden by segmenter over-splitting.

    See the module docstring for the algorithm and FPR mitigations.
    """

    def __init__(
        self,
        provider: WordRepository,
        symspell: SymSpell | None,
        *,
        enabled: bool = True,
        window_sizes: tuple[int, ...] = (2, 3, 4),
        min_frequency: int = 50,
        confidence_floor: float = 0.70,
        max_edit_distance: int = 2,
        require_typo_prone: bool = True,
        skip_names: bool = True,
        require_valid_source_words: bool = True,
    ) -> None:
        self.provider = provider
        self._symspell = symspell
        self.enabled = enabled
        # Sort sizes ascending so the contiguity early-break works correctly.
        self.window_sizes = tuple(sorted(set(window_sizes)))
        self.min_frequency = min_frequency
        self.confidence_floor = confidence_floor
        self.max_edit_distance = max_edit_distance
        self.require_typo_prone = require_typo_prone
        self.skip_names = skip_names
        self.require_valid_source_words = require_valid_source_words
        self.logger = logger

        from myspellchecker.tokenizers.syllable import SyllableTokenizer

        self._syllable_tokenizer = SyllableTokenizer()

    def priority(self) -> int:
        return _PRIORITY

    def validate(self, context: ValidationContext) -> list[Error]:
        if not self.enabled:
            return []
        if self._symspell is None:
            return []
        if not context.words:
            return []

        try:
            flat = self._flatten_syllables(context)
        except (RuntimeError, ValueError, IndexError, AttributeError) as e:
            self.logger.error(f"flatten failed: {e}", exc_info=True)
            return []
        if len(flat) < 2:
            return []

        # Per start position, keep the highest-confidence window.
        best_by_pos: dict[int, tuple[float, WordError]] = {}

        try:
            self._walk_windows(context, flat, best_by_pos)
        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError, TypeError) as e:
            self.logger.error(f"walk failed: {e}", exc_info=True)

        return [err for _, err in sorted(best_by_pos.values(), key=lambda x: x[1].position)]

    def _flatten_syllables(
        self, context: ValidationContext
    ) -> list[tuple[str, int, int]]:
        """Return ``[(syllable, abs_char_pos, source_word_idx), ...]`` for the context.

        Skips words where the syllable tokenizer's output does not concatenate
        back to the original word — without that invariant the absolute char
        offsets cannot be trusted.
        """
        flat: list[tuple[str, int, int]] = []
        for wi, word in enumerate(context.words):
            if not word:
                continue
            word_pos = context.word_positions[wi]
            syllables = self._syllable_tokenizer.tokenize(word)
            if not syllables:
                continue
            if sum(len(s) for s in syllables) != len(word):
                self.logger.debug(
                    f"syllable length mismatch for {word!r}: {syllables}"
                )
                continue
            running = word_pos
            for syl in syllables:
                flat.append((syl, running, wi))
                running += len(syl)
        return flat

    def _walk_windows(
        self,
        context: ValidationContext,
        flat: list[tuple[str, int, int]],
        best_by_pos: dict[int, tuple[float, WordError]],
    ) -> None:
        """Enumerate contiguous windows of configured sizes and emit errors."""
        n = len(flat)
        is_name_mask = context.is_name_mask

        for i in range(n):
            start_syl, start_pos, start_wi = flat[i]

            if start_pos in context.existing_errors:
                continue

            local_best: tuple[float, WordError] | None = None

            for size in self.window_sizes:
                end = i + size
                if end > n:
                    break

                # Contiguity: consecutive syllables must touch (no whitespace
                # or punctuation gap). With sorted sizes, the first non-
                # contiguous size means all larger sizes also fail.
                contiguous = True
                for k in range(i, end - 1):
                    if flat[k + 1][1] != flat[k][1] + len(flat[k][0]):
                        contiguous = False
                        break
                if not contiguous:
                    break

                source_word_indices = {flat[k][2] for k in range(i, end)}

                if self.skip_names and any(
                    wi < len(is_name_mask) and is_name_mask[wi]
                    for wi in source_word_indices
                ):
                    continue

                if self.require_valid_source_words and not all(
                    self.provider.is_valid_word(context.words[wi])
                    for wi in source_word_indices
                ):
                    continue

                # Single-word substrings are already validated upstream;
                # the strategy targets cross-word over-splits.
                if len(source_word_indices) < 2:
                    continue

                joined = "".join(flat[k][0] for k in range(i, end))

                if self.require_typo_prone and not any(
                    c in _TYPO_PRONE_CHARS for c in joined
                ):
                    continue

                joined_norm = normalize(joined)

                if self.provider.is_valid_word(joined_norm):
                    continue

                suggestions = self._symspell.lookup(
                    joined_norm,
                    level="word",
                    max_suggestions=5,
                    include_known=False,
                )
                if not suggestions:
                    continue

                # Accept the first suggestion that passes all gates. SymSpell
                # already orders by edit distance + frequency. The
                # length-preserving check rejects deletion-style suggestions
                # that strip a valid trailing particle.
                best_suggestion = None
                joined_len = len(joined_norm)
                for sugg in suggestions:
                    if sugg.edit_distance > self.max_edit_distance:
                        continue
                    if sugg.frequency < self.min_frequency:
                        continue
                    if len(normalize(sugg.term)) < joined_len:
                        continue
                    best_suggestion = sugg
                    break
                if best_suggestion is None:
                    continue

                if normalize(best_suggestion.term) == joined_norm:
                    continue

                confidence = self._compute_confidence(
                    window_size=end - i,
                    suggestion_freq=best_suggestion.frequency,
                    edit_distance=best_suggestion.edit_distance,
                )
                if confidence < self.confidence_floor:
                    continue

                error = self._build_error(
                    context=context,
                    joined=joined,
                    start_pos=start_pos,
                    suggestion_term=best_suggestion.term,
                    confidence=confidence,
                    window_size=end - i,
                )
                if error is None:
                    continue

                if local_best is None or confidence > local_best[0]:
                    local_best = (confidence, error)

            if local_best is not None:
                existing = best_by_pos.get(start_pos)
                if existing is None or local_best[0] > existing[0]:
                    best_by_pos[start_pos] = local_best

    def _compute_confidence(
        self,
        *,
        window_size: int,
        suggestion_freq: int,
        edit_distance: int,
    ) -> float:
        """Confidence in [0.0, 1.0]: log-frequency × ed-penalty × size-penalty."""
        base = min(1.0, math.log10(max(suggestion_freq, 1)) / 5.0)
        if edit_distance <= 1:
            ed_penalty = 1.0
        elif edit_distance <= 2:
            ed_penalty = 0.85
        else:
            ed_penalty = 0.50
        size_penalty = {2: 1.0, 3: 0.95, 4: 0.90}.get(window_size, 0.85)
        return min(1.0, base * ed_penalty * size_penalty)

    def _build_error(
        self,
        *,
        context: ValidationContext,
        joined: str,
        start_pos: int,
        suggestion_term: str,
        confidence: float,
        window_size: int,
    ) -> WordError | None:
        return WordError(
            text=joined,
            position=start_pos,
            error_type=ET_SYLLABLE_WINDOW_OOV,
            suggestions=[Suggestion(text=suggestion_term)],
            confidence=confidence,
            syllable_count=window_size,
        )

    def __repr__(self) -> str:
        return (
            f"SyllableWindowOOVStrategy(priority={self.priority()}, "
            f"enabled={self.enabled}, "
            f"window_sizes={self.window_sizes}, "
            f"min_frequency={self.min_frequency}, "
            f"confidence_floor={self.confidence_floor})"
        )
