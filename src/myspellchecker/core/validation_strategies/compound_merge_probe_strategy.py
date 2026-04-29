"""Token-level compound merge probe strategy.

Recovers compound typos that the segmenter fragments into individually-valid
subtokens by sliding a window of 2–N adjacent segmented tokens, concatenating
their raw text, and probing :class:`SymSpell` for a high-frequency dictionary
match at edit distance ≤ 2.

Canonical example: ``စွမ်းဆောင်ရည`` (missing asat) segments as
``['စွမ်းဆောင်', 'ရ', 'ည']``; each piece is dictionary-valid so no
downstream strategy flags the span. This strategy concatenates the three
tokens → ``'စွမ်းဆောင်ရည'``, probes SymSpell → ``'စွမ်းဆောင်ရည်'``
(ED 1, high freq), and emits a correction at the merged span.

Unlike :class:`PreSegmenterRawProbeStrategy` (which operates on
whitespace-delimited raw tokens), this strategy uses segmented token
boundaries as its window anchors — essential for Myanmar text where words
are not separated by spaces.

Priority **46** — late detection phase, after most per-word strategies
have claimed positions. Only emits at unclaimed spans. Gated by
:attr:`ValidationConfig.use_compound_merge_probe`.
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

_PRIORITY = 46
_NON_MYANMAR_RE = re.compile(r"[^က-႟၊-၏ꩠ-ꩿ]")
_MYANMAR_PUNCT = frozenset("။၊")
_BARE_CONSONANT_RE = re.compile(r"^[က-အ]$")
_ASAT = "်"

# Particles that should NEVER be part of a merge window. A window containing
# any of these tokens is skipped — particles are standalone grammatical units,
# not fragments of a broken compound.
# NOTE: verbal complements (ကျ, ပြ, ချ) are intentionally EXCLUDED — they
# CAN be fragments of compound words.
_NEVER_MERGE_PARTICLES: frozenset[str] = frozenset(
    {
        # Subject / topic markers
        "က",
        "သည်",
        "ဟာ",
        # Object marker
        "ကို",
        # Locative
        "မှာ",
        "တွင်",
        "၌",
        # Ablative
        "မှ",
        # Genitive
        "ရဲ့",
        "၏",
        # Comitative
        "နဲ့",
        "နှင့်",
        # Plural
        "များ",
        # Question particles
        "လား",
        "လဲ",
        # Sentence-final particles
        "တယ်",
        "ပါတယ်",
        "ပြီ",
        # Politeness
        "ပါ",
        # Modal / emphasis
        "ပဲ",
        "ပေါ့",
        "တော့",
        "ပြီး",
        # Classifiers after numerals
        "ခု",
        "ယောက်",
        "လုံး",
        "ကောင်",
        "စု",
    }
)


class CompoundMergeProbeStrategy(ValidationStrategy):
    """Probe SymSpell on concatenations of adjacent segmented tokens.

    Args:
        symspell: Configured :class:`SymSpell` instance (already indexed).
        provider: Dictionary provider for ``is_valid_word`` /
            ``get_word_frequency`` lookups.
        enabled: Master on/off switch.
        max_window_tokens: Maximum number of adjacent tokens to merge (2–6).
        max_span_length: Skip merged spans longer than this.
        max_edit_distance: Upper bound on accepted SymSpell edit distance.
        min_candidate_freq: Minimum dict frequency a candidate must clear.
        fragment_freq_floor: Tokens with frequency below this are considered
            "fragment-like". At least one token in the window must be below
            this threshold (or OOV) for the probe to fire.
        max_length_diff: Maximum ``|len(candidate) - len(span)|``.
        confidence: Confidence stamped on emitted errors.
        max_suggestions: SymSpell ``max_suggestions`` parameter per probe.
    """

    def __init__(
        self,
        symspell: SymSpell,
        provider: WordRepository,
        *,
        enabled: bool = False,
        max_window_tokens: int = 3,
        max_span_length: int = 20,
        max_edit_distance: float = 0.4,
        min_candidate_freq: int = 5_000,
        fragment_freq_floor: int = 50_000,
        max_length_diff: int = 1,
        confidence: float = 0.70,
        max_suggestions: int = 5,
    ) -> None:
        self.symspell = symspell
        self.provider = provider
        self.enabled = enabled
        self.max_window_tokens = max_window_tokens
        self.max_span_length = max_span_length
        self.max_edit_distance = max_edit_distance
        self.min_candidate_freq = min_candidate_freq
        self.fragment_freq_floor = fragment_freq_floor
        self.max_length_diff = max_length_diff
        self.confidence = confidence
        self.max_suggestions = max_suggestions

    def priority(self) -> int:
        return _PRIORITY

    def validate(self, context: ValidationContext) -> list[Error]:
        if not self.enabled or self.symspell is None:
            return []

        words = context.words
        positions = context.word_positions
        if len(words) < 2:
            return []

        errors: list[Error] = []
        claimed_positions: set[int] = set()

        for window_size in range(2, min(self.max_window_tokens, len(words)) + 1):
            for i in range(len(words) - window_size + 1):
                span_tokens = words[i : i + window_size]
                span_start = positions[i]

                if any(positions[j] in context.existing_errors for j in range(i, i + window_size)):
                    continue
                if span_start in claimed_positions:
                    continue

                if self._has_punctuation(span_tokens):
                    continue

                if any(t in _NEVER_MERGE_PARTICLES for t in span_tokens):
                    continue

                span_text = "".join(span_tokens)

                if len(span_text) > self.max_span_length:
                    continue

                if self.provider.is_valid_word(span_text):
                    continue

                if not self._has_fragment_evidence(span_tokens):
                    continue

                if self._any_name_masked(context, i, i + window_size):
                    continue

                if is_colloquial_variant(span_text):
                    continue

                asat_result = self._try_asat_insertion(span_text, span_tokens)
                if asat_result is not None:
                    suggestion_text, cand_freq = asat_result
                    error = WordError(
                        text=span_text,
                        position=span_start,
                        error_type=ET_WORD,
                        suggestions=[
                            Suggestion(text=suggestion_text, source="compound_merge_asat")
                        ],
                        confidence=self.confidence,
                    )
                    errors.append(error)
                    claimed_positions.add(span_start)
                    logger.debug(
                        "compound_merge_asat: %s (%d tokens) -> %s freq=%d",
                        span_text,
                        window_size,
                        suggestion_text,
                        cand_freq,
                    )
                    continue

                candidate = self._best_candidate(span_text, span_tokens)
                if candidate is None:
                    continue

                suggestion_text, cand_freq, cand_ed = candidate

                error = WordError(
                    text=span_text,
                    position=span_start,
                    error_type=ET_WORD,
                    suggestions=[Suggestion(text=suggestion_text, source="compound_merge_probe")],
                    confidence=self.confidence,
                )
                errors.append(error)
                claimed_positions.add(span_start)

                logger.debug(
                    "compound_merge_probe: %s (%d tokens) -> %s ed=%.2f freq=%d",
                    span_text,
                    window_size,
                    suggestion_text,
                    cand_ed,
                    cand_freq,
                )

        return errors

    def _has_fragment_evidence(self, tokens: list[str]) -> bool:
        """All tokens must be valid AND high-frequency to skip (evidence = false).

        When every token is a common, standalone word above the freq floor,
        there's no evidence of fragmentation — the segmenter's split is
        intentional.
        """
        for token in tokens:
            if not self.provider.is_valid_word(token):
                return True
            freq = self.provider.get_word_frequency(token)
            freq_val = int(freq) if isinstance(freq, (int, float)) else 0
            if freq_val < self.fragment_freq_floor:
                return True
        return False

    def _try_asat_insertion(self, span_text: str, span_tokens: list[str]) -> tuple[str, int] | None:
        """Fast path: if the last token is a bare consonant, try appending asat."""
        last = span_tokens[-1]
        if not _BARE_CONSONANT_RE.match(last):
            return None
        candidate = span_text + _ASAT
        if not self.provider.is_valid_word(candidate):
            return None
        freq = self.provider.get_word_frequency(candidate) or 0
        freq_val = int(freq) if isinstance(freq, (int, float)) else 0
        if freq_val < self.min_candidate_freq:
            return None
        return candidate, freq_val

    def _best_candidate(
        self, span_text: str, span_tokens: list[str] | None = None
    ) -> tuple[str, int, float] | None:
        """Probe SymSpell and return the best acceptable hit."""
        try:
            suggestions = self.symspell.lookup(
                span_text,
                level="word",
                max_suggestions=self.max_suggestions,
            )
        except (RuntimeError, ValueError, KeyError):
            logger.debug("SymSpell lookup failed for %r", span_text, exc_info=True)
            return None

        if not suggestions:
            return None

        token_set = set(span_tokens) if span_tokens else set()
        sub_spans = self._build_sub_spans(span_tokens) if span_tokens else set()

        for suggestion in suggestions:
            candidate_text = getattr(suggestion, "term", None) or str(suggestion)
            if candidate_text == span_text:
                continue
            if candidate_text in token_set:
                continue
            if candidate_text in sub_spans:
                continue
            if abs(len(candidate_text) - len(span_text)) > self.max_length_diff:
                continue
            ed = weighted_damerau_levenshtein_distance(span_text, candidate_text)
            if ed > self.max_edit_distance:
                continue
            freq = self.provider.get_word_frequency(candidate_text) or 0
            freq_val = int(freq) if isinstance(freq, (int, float)) else 0
            if freq_val < self.min_candidate_freq:
                continue
            return candidate_text, freq_val, float(ed)

        return None

    @staticmethod
    def _has_punctuation(tokens: list[str]) -> bool:
        """Return True if any token contains punctuation or non-Myanmar chars."""
        for token in tokens:
            if any(c in _MYANMAR_PUNCT for c in token):
                return True
            if _NON_MYANMAR_RE.search(token):
                return True
        return False

    @staticmethod
    def _build_sub_spans(tokens: list[str]) -> set[str]:
        """Build all contiguous proper sub-sequences of the token list."""
        n = len(tokens)
        result: set[str] = set()
        for length in range(1, n):
            for start in range(n - length + 1):
                result.add("".join(tokens[start : start + length]))
        return result

    @staticmethod
    def _any_name_masked(context: ValidationContext, start_idx: int, end_idx: int) -> bool:
        """Return True if any token in [start_idx, end_idx) is name-masked."""
        if not context.is_name_mask:
            return False
        for idx in range(start_idx, end_idx):
            if idx < len(context.is_name_mask) and context.is_name_mask[idx]:
                return True
        return False

    def __repr__(self) -> str:
        return (
            f"CompoundMergeProbeStrategy(priority={self.priority()}, "
            f"enabled={self.enabled}, "
            f"max_window={self.max_window_tokens}, "
            f"max_ed={self.max_edit_distance}, "
            f"min_freq={self.min_candidate_freq})"
        )
