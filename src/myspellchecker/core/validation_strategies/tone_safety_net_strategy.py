"""Tone-safety-net candidate generator for D2 missing / extra tone marks.

Recovers real-word confusions where the typo and the gold differ only by a
single trailing tone character from {း (U+1038), ့ (U+1037), ံ (U+1036),
် (U+103A)}. Canonical examples from the v1.5 benchmark D2 bucket:

    BM-139-E1:  သွင်  → သွင်း   (missing visarga)
    BM-528-E1:  ခဲ    → ခဲ့      (missing dot-below)
    BM-537-E2:  မှား  → မှာ     (extraneous visarga)

These cases are structurally invisible to SymSpell when both forms are valid
dictionary words (``target_in_dict=True, gold_in_dict=True``) — the
``is_valid_word(typo) == True`` short-circuit kills candidate generation.
The ``mined_confusable_pairs`` strategy already handles the ~85% of D2
covered by its corpus-mined table; this strategy targets the ~17 net-new
FN that mining missed because the freq_ratio gate in the miner was too
strict for rare-form pairs.

Priority **22** — structural phase, ahead of the fast-path cutoff and
ahead of :class:`StatisticalConfusableStrategy` (priority 24). Default-off
behind :attr:`ValidationConfig.use_tone_safety_net` until the
``tzn-benchmark-01`` gate measures composite delta + FPR impact.

See [[Tone-Zawgyi Slice 2026-04-19]] for the audit and
[[Myanmar Real-Word Confusion Taxonomy 2026-04-18]] §1 D2 for the class
definition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.core.constants import ET_WORD
from myspellchecker.core.response import Error, Suggestion, WordError
from myspellchecker.core.validation_strategies.base import (
    ValidationContext,
    ValidationStrategy,
)
from myspellchecker.text.phonetic_data import is_colloquial_variant
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import WordRepository

logger = get_logger(__name__)

_PRIORITY = 22

# Trailing tone characters the strategy probes for insert / delete.
_VISARGA = "\u1038"  # း
_DOT_BELOW = "\u1037"  # ့
_ANUSVARA = "\u1036"  # ံ
_ASAT = "\u103a"  # ်
_TONE_CHARS: tuple[str, ...] = (_VISARGA, _DOT_BELOW, _ANUSVARA)


class ToneSafetyNetStrategy(ValidationStrategy):
    """Probe trailing tone insert / delete candidates against the dictionary.

    For each raw Myanmar token the strategy builds up to four candidates:

    1. ``token + း``
    2. ``token + ့``
    3. ``token + ံ``
    4. ``token[:-1]`` if ``token[-1]`` is one of ``{း, ့, ံ, ်}``

    A candidate is emitted when:

    * the candidate is a valid dictionary word,
    * ``freq(candidate) >= min_frequency``,
    * ``freq(candidate) / max(freq(token), 1) >= freq_ratio``,
    * the current token's frequency does not exceed
      ``skip_above_freq`` (guards very common words from being flagged),
    * the token is not on the colloquial whitelist,
    * the token does not overlap a name-masked span,
    * no existing error has already been emitted at the token position.

    Args:
        provider: Dictionary provider for ``is_valid_word`` /
            ``get_word_frequency`` lookups.
        enabled: Master on/off switch. Wired from
            :attr:`ValidationConfig.use_tone_safety_net`.
        min_frequency: Minimum dict frequency a candidate must clear.
        freq_ratio: Minimum ``freq(candidate) / freq(token)`` ratio; higher
            is safer but may miss rare-form repairs.
        skip_above_freq: Do not probe tokens whose own frequency already
            exceeds this value. Guards against second-guessing high-freq
            dictionary words.
        confidence: Confidence stamped on the emitted :class:`WordError`.
    """

    def __init__(
        self,
        provider: WordRepository,
        *,
        enabled: bool = False,
        min_frequency: int = 1000,
        freq_ratio: float = 10.0,
        skip_above_freq: int = 50000,
        confidence: float = 0.80,
    ) -> None:
        self.provider = provider
        self.enabled = enabled
        self.min_frequency = min_frequency
        self.freq_ratio = freq_ratio
        self.skip_above_freq = skip_above_freq
        self.confidence = confidence

    def priority(self) -> int:
        """Return strategy execution priority (22, structural phase)."""
        return _PRIORITY

    def validate(self, context: ValidationContext) -> list[Error]:
        """Emit :class:`WordError` for tokens with a better tone-variant candidate."""
        if not self.enabled:
            return []

        words = context.words
        if not words:
            return []

        errors: list[Error] = []

        for idx, word in enumerate(words):
            if idx >= len(context.word_positions):
                continue
            position = context.word_positions[idx]

            if position in context.existing_errors:
                continue
            if not self._should_probe(word, context, idx):
                continue

            candidate = self._best_candidate(word)
            if candidate is None:
                continue

            suggestion_text, cand_freq, token_freq = candidate
            error = WordError(
                text=word,
                position=position,
                error_type=ET_WORD,
                suggestions=[Suggestion(text=suggestion_text, source="tone_safety_net")],
                confidence=self.confidence,
            )
            errors.append(error)
            context.existing_errors[position] = ET_WORD
            context.existing_confidences[position] = self.confidence
            context.existing_suggestions[position] = [suggestion_text]
            logger.debug(
                "tone_safety_net: %s (freq=%d) -> %s (freq=%d) ratio=%.1f conf=%.2f",
                word,
                token_freq,
                suggestion_text,
                cand_freq,
                cand_freq / max(token_freq, 1),
                self.confidence,
            )

        return errors

    def _should_probe(
        self,
        word: str,
        context: ValidationContext,
        idx: int,
    ) -> bool:
        """Return True if the segmented ``word`` is a viable probe target."""
        if not word:
            return False
        # Must contain Myanmar characters; Latin / punctuation tokens pass
        # through unchanged.
        if not any("\u1000" <= ch <= "\u109f" for ch in word):
            return False
        # The strategy only operates on real-word confusions; skip OOV
        # tokens because SymSpell / raw-probe already handle those.
        if not self.provider.is_valid_word(word):
            return False
        # Very common words are unlikely to be typos; skip to bound FPR.
        token_freq = self.provider.get_word_frequency(word) or 0
        if token_freq > self.skip_above_freq:
            return False
        # Colloquial whitelist: these are flagged informationally, not as
        # errors — the tone-safety-net must not upgrade them silently.
        if is_colloquial_variant(word):
            return False
        # Skip tokens flagged as proper names by the mask.
        if context.is_name_mask and idx < len(context.is_name_mask) and context.is_name_mask[idx]:
            return False
        return True

    def _candidate_variants(self, token: str) -> list[str]:
        """Return the list of tone-variant candidates to probe for ``token``.

        Generates up to four candidates per input:
        - Append each of {း, ့, ံ} to the token
        - Strip the trailing char if it is one of {း, ့, ံ, ်}
        """
        if not token:
            return []
        variants: list[str] = []
        for tone in _TONE_CHARS:
            if not token.endswith(tone):
                variants.append(token + tone)
        if token[-1] in (_VISARGA, _DOT_BELOW, _ANUSVARA, _ASAT):
            stripped = token[:-1]
            if stripped:
                variants.append(stripped)
        return variants

    def _best_candidate(self, raw_token: str) -> tuple[str, int, int] | None:
        """Return ``(candidate, cand_freq, token_freq)`` for the best variant.

        Returns ``None`` when no variant clears the frequency / ratio guards.
        """
        token_freq = self.provider.get_word_frequency(raw_token) or 0
        best: tuple[str, int] | None = None
        best_freq = 0
        for variant in self._candidate_variants(raw_token):
            if not self.provider.is_valid_word(variant):
                continue
            cand_freq = self.provider.get_word_frequency(variant) or 0
            if cand_freq < self.min_frequency:
                continue
            if cand_freq < self.freq_ratio * max(token_freq, 1):
                continue
            if cand_freq > best_freq:
                best = (variant, cand_freq)
                best_freq = cand_freq
        if best is None:
            return None
        return best[0], best[1], token_freq

    def __repr__(self) -> str:
        return (
            f"ToneSafetyNetStrategy(priority={self.priority()}, "
            f"enabled={self.enabled}, "
            f"min_frequency={self.min_frequency}, "
            f"freq_ratio={self.freq_ratio}, "
            f"skip_above_freq={self.skip_above_freq})"
        )
