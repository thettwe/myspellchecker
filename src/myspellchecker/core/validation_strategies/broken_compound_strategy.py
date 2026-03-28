"""
Broken Compound Validation Strategy.

Detects compound words that were incorrectly split by a space.
For example: "မနက် ဖြန်" should be "မနက်ဖြန်" (tomorrow).

This is the inverse of MergedWordChecker, which detects wrongly-merged words.

The strategy checks adjacent word pairs: if their concatenation forms a valid
dictionary word that is significantly more common than the rarer component,
it flags the pair as a broken compound.

Priority: 25 (after SyntacticValidation 20, before POS 30)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.core.config.algorithm_configs import BrokenCompoundStrategyConfig
from myspellchecker.core.constants import ET_BROKEN_COMPOUND
from myspellchecker.core.response import Error, WordError
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import WordRepository

logger = get_logger(__name__)


class BrokenCompoundStrategy(ValidationStrategy):
    """
    Detect compound words that were incorrectly split by a space.

    When adjacent words W_i and W_{i+1} concatenate to a valid dictionary word
    that is much more common than either individual word, this likely indicates
    a broken compound. The strategy flags the rarer component and suggests
    the merged compound form.

    Priority: 25
    """

    def __init__(
        self,
        provider: "WordRepository",
        rare_threshold: int | None = None,
        compound_min_frequency: int | None = None,
        compound_ratio: float | None = None,
        confidence: float | None = None,
        config: BrokenCompoundStrategyConfig | None = None,
    ):
        """
        Initialize broken compound strategy.

        Args:
            provider: Word repository with is_valid_word and get_word_frequency.
            rare_threshold: Maximum frequency for a word to be considered "rare".
                Words with frequency below this are candidates for compound merging.
            compound_min_frequency: Minimum frequency for the compound to be flagged.
                Prevents merging into rare compounds.
            compound_ratio: Minimum ratio of compound_freq / rare_word_freq.
                Higher values = more conservative (fewer flags).
            confidence: Confidence score for broken compound errors.
            config: BrokenCompoundStrategyConfig with all thresholds.
                When provided, its values are used as defaults that explicit
                kwargs can override.
        """
        self._config = config or BrokenCompoundStrategyConfig()
        self.provider = provider
        self.rare_threshold = (
            rare_threshold if rare_threshold is not None else self._config.rare_threshold
        )
        self.compound_min_frequency = (
            compound_min_frequency
            if compound_min_frequency is not None
            else self._config.compound_min_frequency
        )
        self.compound_ratio = (
            compound_ratio if compound_ratio is not None else self._config.compound_ratio
        )
        self.confidence = confidence if confidence is not None else self._config.confidence
        self._both_high_freq = self._config.both_high_freq
        self._min_compound_len = self._config.min_compound_len
        self.logger = logger

    def validate(self, context: ValidationContext) -> list[Error]:
        """Validate word pairs for broken compound errors."""
        if len(context.words) < 2:
            return []

        if not hasattr(self.provider, "is_valid_word") or not hasattr(
            self.provider, "get_word_frequency"
        ):
            return []

        errors: list[Error] = []

        try:
            for i in range(len(context.words) - 1):
                # Skip if either position is already flagged
                pos_i = context.word_positions[i]
                pos_next = context.word_positions[i + 1]
                if pos_i in context.existing_errors or pos_next in context.existing_errors:
                    continue

                # Skip names
                if context.is_name_mask[i] or context.is_name_mask[i + 1]:
                    continue

                w1 = context.words[i]
                w2 = context.words[i + 1]

                # Skip Pali/Sanskrit stacking fragments (virama U+1039)
                # The segmenter splits stacking words like ဗုဒ္ဓ into fragments
                # that falsely appear as broken compounds
                if "\u1039" in w1 or "\u1039" in w2:
                    continue

                # Both must be valid individual words (we're detecting split, not typo)
                if not self.provider.is_valid_word(w1) or not self.provider.is_valid_word(w2):
                    continue

                # At least one must be rare
                freq1 = self.provider.get_word_frequency(w1)
                freq2 = self.provider.get_word_frequency(w2)
                rare_freq = min(freq1, freq2)
                if rare_freq >= self.rare_threshold:
                    continue

                # Both-high-freq guard: when both tokens are well-established
                # multi-syllable compounds (freq >= both_high_freq, len >= min_compound_len),
                # their adjacency is intentional — don't suggest merging.
                if (
                    freq1 >= self._both_high_freq
                    and freq2 >= self._both_high_freq
                    and len(w1) >= self._min_compound_len
                    and len(w2) >= self._min_compound_len
                ):
                    continue

                # Guard: zero-frequency but valid words are curated dictionary
                # entries (e.g., adjective stems like လှပ). These are valid
                # standalone forms, not segmenter artifacts. Don't flag them
                # as broken compounds — their adjacency with particles like
                # သော is natural grammar, not a space error.
                if rare_freq == 0:
                    continue

                compound = w1 + w2
                if not self.provider.is_valid_word(compound):
                    continue

                compound_freq = self.provider.get_word_frequency(compound)
                if compound_freq < self.compound_min_frequency:
                    continue

                # Compound must be significantly more common than the rare word
                if rare_freq > 0 and compound_freq / rare_freq < self.compound_ratio:
                    continue

                # Flag the full span covering BOTH words so that
                # generate_corrected_text replaces "w1 w2" with the compound,
                # not just one of the two words.
                # Use absolute positions for error.position (needed by
                # generate_corrected_text on full text), but derive span_text
                # from the local sentence to avoid absolute/local mismatch.
                span_start = pos_i
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

                errors.append(
                    WordError(
                        text=span_text,
                        position=span_start,
                        error_type=ET_BROKEN_COMPOUND,
                        suggestions=[compound],
                        confidence=self.confidence,
                    )
                )
                # Mark both positions to prevent duplicate detection from
                # downstream strategies.
                context.existing_errors[pos_i] = ET_BROKEN_COMPOUND
                context.existing_suggestions[pos_i] = [compound]
                context.existing_confidences[pos_i] = self.confidence
                context.existing_errors[pos_next] = ET_BROKEN_COMPOUND

        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError, TypeError) as e:
            self.logger.error(f"Error in broken compound validation: {e}", exc_info=True)

        return errors

    def priority(self) -> int:
        """Return strategy execution priority (25)."""
        return 25

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BrokenCompoundStrategy(priority={self.priority()}, "
            f"rare_threshold={self.rare_threshold}, ratio={self.compound_ratio}x)"
        )
