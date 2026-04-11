"""
Syllable-Window OOV Validation Strategy (Sprint I-1).

Detects multi-syllable typos that the word segmenter decomposes into
individually-valid syllables. The canonical failure mode: an OOV compound
like ``ခုန်ကျစရိတ်`` (intended ``ကုန်ကျစရိတ်``, "costs") is segmented as
``['ခုန်', 'ကျ', 'စရိတ်']`` — each token valid, so no upstream strategy
fires — and the error hides in plain sight.

The strategy enumerates 2-4 syllable contiguous windows across adjacent
words (adjacency measured at character-offset level), joins them, and
consults SymSpell for a high-frequency near-match. When the joined string
is OOV and a near-match exists above the frequency/confidence floors, the
window is emitted as a :data:`~myspellchecker.core.constants.ET_SYLLABLE_WINDOW_OOV`
error.

Priority: **22** (structural phase, before HiddenCompound 23,
StatisticalConfusable 24, and BrokenCompound 25). Unlike BrokenCompound,
this strategy does **not** populate ``context.existing_errors`` — that
preserves HiddenCompound's ability to fire at overlapping positions with
its own error type, so the two mechanisms remain complementary.

The design was hardened by a Define→Develop debate gate (codex+gemini).
Key mitigations against FPR risk on proper nouns and loanwords:

- ``skip_names``: reject windows spanning any ``context.is_name_mask`` word.
- ``require_valid_source_words``: every source word must be individually
  valid — prevents firing on upstream segmentation artifacts.
- Strict ``confidence_floor`` (0.70) and frequency threshold (50).
- ``require_typo_prone`` filter on candidate joined strings.
- Within-strategy dedup: one emission per start syllable position, longer
  windows preferred only when confidence is strictly higher.

See ``~/Documents/myspellchecker/Workstreams/v1.5.0/sprint-i-1-syllable-window-detector.md``
for the sprint plan and oracle-verified recall targets.
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

# Typo-prone characters. Borrowed from HiddenCompoundStrategy so both
# strategies share a consistent gating heuristic. A window must contain at
# least one of these to be considered as a potential typo.
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
    """
    Detect multi-syllable OOV typos hidden by segmenter over-splitting.

    See the module docstring for the full algorithm. The strategy runs at
    priority 22 inside the structural phase (priority <= 25 survives the
    fast-path cutoff in :class:`ContextValidator`) so it fires on clean
    sentences where 185/218 of the benchmark's clean-sentence FNs live.

    Args:
        provider: Dictionary provider for word validity / frequency lookups.
        symspell: Pre-built SymSpell instance (word-level index required).
        enabled: Master on/off switch (default True after oracle validation).
        window_sizes: Syllable window sizes to enumerate. Default (2, 3, 4).
        min_frequency: Minimum frequency for a SymSpell suggestion to count.
        confidence_floor: Minimum confidence to emit an error.
        max_edit_distance: Maximum SymSpell edit distance.
        require_typo_prone: If True, joined window must contain a typo-prone
            character before considering SymSpell lookup.
        skip_names: If True, skip windows spanning ``context.is_name_mask`` words.
        require_valid_source_words: If True, every source word contributing
            syllables to the window must itself be individually valid.
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
        # Sort window sizes ascending so the non-contiguity early-break works
        # correctly even if callers pass a non-monotone tuple.
        self.window_sizes = tuple(sorted(set(window_sizes)))
        self.min_frequency = min_frequency
        self.confidence_floor = confidence_floor
        self.max_edit_distance = max_edit_distance
        self.require_typo_prone = require_typo_prone
        self.skip_names = skip_names
        self.require_valid_source_words = require_valid_source_words
        self.logger = logger

        # Lazy-load SyllableTokenizer for per-word tokenization. Matches the
        # HiddenCompoundStrategy pattern so tests with mocked providers still
        # construct a real tokenizer.
        from myspellchecker.tokenizers.syllable import SyllableTokenizer

        self._syllable_tokenizer = SyllableTokenizer()

    def priority(self) -> int:
        """Return strategy execution priority (22).

        Placed before HiddenCompound (23), StatisticalConfusable (24), and
        BrokenCompound (25). Inside the structural phase (<=25), which
        survives the fast-path cutoff in
        :attr:`ContextValidator._FAST_PATH_PRIORITY_CUTOFF`. This is
        essential because the target FNs predominantly live in zero-error
        ("clean") sentences.
        """
        return _PRIORITY

    # ── Core walk ──────────────────────────────────────────────────────

    def validate(self, context: ValidationContext) -> list[Error]:
        """Enumerate syllable windows and emit OOV errors with near-match fixes.

        Returns a list of :class:`WordError` objects, one per flagged window.
        The strategy does NOT populate ``context.existing_errors`` so later
        strategies (notably HiddenCompound at priority 23) continue to fire
        at overlapping positions with their own error types.
        """
        if not self.enabled:
            return []
        if self._symspell is None:
            return []
        if not context.words:
            return []

        try:
            flat = self._flatten_syllables(context)
        except (RuntimeError, ValueError, IndexError, AttributeError) as e:
            self.logger.error(
                f"SyllableWindowOOVStrategy: flatten failed: {e}", exc_info=True
            )
            return []
        if len(flat) < 2:
            return []

        # Track best emission per start syllable position (char offset).
        # Key: char_pos of first syllable in window.
        # Value: (confidence, error) — higher confidence wins.
        best_by_pos: dict[int, tuple[float, WordError]] = {}

        try:
            self._walk_windows(context, flat, best_by_pos)
        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError, TypeError) as e:
            self.logger.error(
                f"SyllableWindowOOVStrategy: walk failed: {e}", exc_info=True
            )

        # Emit sorted by character position for deterministic output.
        return [err for _, err in sorted(best_by_pos.values(), key=lambda x: x[1].position)]

    # ── Syllable flattening ────────────────────────────────────────────

    def _flatten_syllables(
        self, context: ValidationContext
    ) -> list[tuple[str, int, int]]:
        """Flatten ``context.words`` into a syllable list with char positions.

        Returns a list of ``(syllable_text, abs_char_pos, source_word_idx)``
        tuples. ``abs_char_pos`` is the absolute character position in
        ``context.full_text`` (matches ``context.word_positions``).

        Syllable text starts at word position and accumulates by syllable
        length. This assumes ``SyllableTokenizer.tokenize(word)`` produces
        a concatenation equal to ``word`` itself.
        """
        flat: list[tuple[str, int, int]] = []
        for wi, word in enumerate(context.words):
            if not word:
                continue
            word_pos = context.word_positions[wi]
            syllables = self._syllable_tokenizer.tokenize(word)
            if not syllables:
                continue
            # Guard against tokenizer/word length mismatch — if the sum of
            # syllable lengths differs from len(word), we cannot trust the
            # char offsets and skip this word.
            if sum(len(s) for s in syllables) != len(word):
                self.logger.debug(
                    f"syllable length mismatch for word {word!r}: "
                    f"syllables={syllables}"
                )
                continue
            running = word_pos
            for syl in syllables:
                flat.append((syl, running, wi))
                running += len(syl)
        return flat

    # ── Window enumeration ─────────────────────────────────────────────

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

            # Skip if already flagged upstream at this char position.
            if start_pos in context.existing_errors:
                continue

            # Per-start: track best (conf, error) across window sizes.
            local_best: tuple[float, WordError] | None = None

            for size in self.window_sizes:
                end = i + size
                if end > n:
                    break

                # Contiguity check: each consecutive pair must have
                # syl[k+1].pos == syl[k].pos + len(syl[k]). This rejects
                # whitespace/punctuation gaps and windows that cross
                # discontiguous spans.
                contiguous = True
                for k in range(i, end - 1):
                    if flat[k + 1][1] != flat[k][1] + len(flat[k][0]):
                        contiguous = False
                        break
                if not contiguous:
                    # Larger window sizes at this start cannot be contiguous
                    # either, so break the size loop.
                    break

                source_word_indices = {flat[k][2] for k in range(i, end)}

                # Skip windows that span name-masked words.
                if self.skip_names and any(
                    wi < len(is_name_mask) and is_name_mask[wi]
                    for wi in source_word_indices
                ):
                    continue

                # Require every source word to be individually valid.
                if self.require_valid_source_words and not all(
                    self.provider.is_valid_word(context.words[wi])
                    for wi in source_word_indices
                ):
                    continue

                # Core invariant: this strategy exists to surface compound
                # typos that the segmenter OVER-SPLIT into multiple valid
                # tokens. Windows contained entirely within a single source
                # word are substrings of something already validated
                # upstream — firing on them risks flagging arbitrary
                # prefixes/suffixes of long valid compounds. Require the
                # window to span >=2 distinct source words.
                if len(source_word_indices) < 2:
                    continue

                joined = "".join(flat[k][0] for k in range(i, end))

                if self.require_typo_prone and not any(
                    c in _TYPO_PRONE_CHARS for c in joined
                ):
                    continue

                joined_norm = normalize(joined)

                # Already a valid word? Not an OOV candidate.
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

                # Accept the first suggestion that passes ALL gates:
                #   1. ed within the strategy's cap (tighter than SymSpell's)
                #   2. freq >= min_frequency (high-confidence dictionary entry)
                #   3. suggestion length >= joined length (REJECTS deletions
                #      where SymSpell is just stripping a valid trailing
                #      particle like က/ကို/မှာ — empirically the dominant
                #      false-positive mode on clean Burmese news text)
                #
                # SymSpell results are sorted by edit distance + frequency,
                # so the first one passing the gates is the strongest.
                best_suggestion = None
                joined_len = len(joined_norm)
                for sugg in suggestions:
                    if sugg.edit_distance > self.max_edit_distance:
                        continue
                    if sugg.frequency < self.min_frequency:
                        continue
                    if len(normalize(sugg.term)) < joined_len:
                        # Deletion-style suggestion (shorter than joined).
                        continue
                    best_suggestion = sugg
                    break
                if best_suggestion is None:
                    continue

                # Guard against self-suggestions (normalization noise).
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

    # ── Confidence scoring ─────────────────────────────────────────────

    def _compute_confidence(
        self,
        *,
        window_size: int,
        suggestion_freq: int,
        edit_distance: int,
    ) -> float:
        """Confidence score in [0.0, 1.0].

        - Base = log10(freq)/5 capped at 1.0 (matches HiddenCompound).
        - Edit-distance penalty: ed=1 is 1.0, ed=2 is 0.85, else 0.50.
        - Window-size penalty: 2-syl 1.0, 3-syl 0.95, 4-syl 0.90. Longer
          windows have more surface area for spurious matches.
        """
        base = min(1.0, math.log10(max(suggestion_freq, 1)) / 5.0)
        if edit_distance <= 1:
            ed_penalty = 1.0
        elif edit_distance <= 2:
            ed_penalty = 0.85
        else:
            ed_penalty = 0.50
        size_penalty = {2: 1.0, 3: 0.95, 4: 0.90}.get(window_size, 0.85)
        return min(1.0, base * ed_penalty * size_penalty)

    # ── Error construction ─────────────────────────────────────────────

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
        """Build a WordError for the flagged window.

        ``joined`` is the exact concatenation of the syllables as produced
        by :class:`SyllableTokenizer`, and by construction matches the
        substring of the source text from ``start_pos`` for ``len(joined)``
        characters. This is used directly as the error span text without
        re-slicing the sentence — re-slicing is fragile when
        ``context.sentence`` is a substring of a larger ``full_text`` and
        ``context.words[0]`` may appear multiple times.
        """
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
