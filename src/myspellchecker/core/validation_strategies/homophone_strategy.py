"""
Homophone Validation Strategy.

This strategy detects real-word errors where homophones (words with similar
pronunciation) are used incorrectly in context.

Delegates n-gram comparison to ``NgramContextChecker.check_word_in_context()``
so that ratio computation, frequency guards, and bidirectional probability
logic live in one place.

Priority: 45 (runs before n-gram validation to ensure homophone errors are detected first)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.core.constants import ET_HOMOPHONE_ERROR
from myspellchecker.core.response import ContextError, Error
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.ngram_context_checker import NgramContextChecker
    from myspellchecker.core.homophones import HomophoneChecker

# Productive prefixes that form compounds with the following word.
# When prefix+candidate is valid but prefix+word is not, favor candidate.
_COMPOUND_PREFIXES = frozenset(
    {
        "\u1019",  # မ — negation prefix (most productive)
        "\u1021",  # အ — nominalizer prefix (အစား, အသွား, etc.)
    }
)

# Common nominal suffixes that form compounds with the preceding word.
# When word+suffix is invalid but candidate+suffix is valid, favor candidate.
_COMPOUND_SUFFIXES = frozenset(
    {
        "\u101b\u1031\u1038",  # ရေး — affair/matter (e.g., စီးပွားရေး)
        "\u101b\u102c",  # ရာ — place/thing (e.g., နေရာ)
        "\u101e\u102c\u1038",  # သား — son/person (e.g., ကျောင်းသား)
        "\u101e\u1030",  # သူ — person who (e.g., လုပ်သူ)
    }
)


class HomophoneValidationStrategy(ValidationStrategy):
    """
    Homophone confusion detection strategy.

    For each word, looks up known homophones via ``HomophoneChecker`` and
    delegates the n-gram comparison to
    ``NgramContextChecker.check_word_in_context()`` with candidate type
    ``"homophone"``.

    Priority: 45 (runs before n-gram validation)
    """

    def __init__(
        self,
        homophone_checker: "HomophoneChecker | None",
        provider: object,
        context_checker: "NgramContextChecker | None" = None,
        confidence: float = 0.8,
        # Legacy kwargs accepted for backward-compat but ignored — the
        # unified NgramContextChecker.compute_required_ratio() owns these.
        **_kwargs: object,
    ):
        """
        Initialize homophone validation strategy.

        Args:
            homophone_checker: HomophoneChecker instance for homophone lookup.
                              If None, this strategy is disabled.
            provider: DictionaryProvider (used for word frequency lookups).
            context_checker: NgramContextChecker that performs the unified
                            n-gram comparison via ``check_word_in_context()``.
            confidence: Confidence score for homophone errors (default: 0.8).
        """
        self.homophone_checker = homophone_checker
        self.provider = provider
        self.context_checker = context_checker
        self.confidence = confidence
        self.logger = get_logger(__name__)

    def validate(self, context: ValidationContext) -> list[Error]:
        """
        Validate words for homophone confusions.

        Args:
            context: Validation context with sentence information.

        Returns:
            List of ContextError objects for homophone confusions.
        """
        if not self.homophone_checker or not self.context_checker or len(context.words) < 2:
            return []

        errors: list[Error] = []

        try:
            for i in range(len(context.words)):
                word = context.words[i]
                position = context.word_positions[i]

                # Skip names
                if context.is_name_mask[i]:
                    continue

                # Get homophones for the current word
                homophones = self.homophone_checker.get_homophones(word)
                if not homophones:
                    continue

                # Build context word lists, using corrected forms when available.
                # Higher-priority strategies (e.g., orthography at priority 15)
                # may have already flagged and corrected adjacent words.  Using
                # the corrected form gives the n-gram checker valid bigram evidence
                # instead of zero-probability lookups against misspelled words.
                prev_words: list[str] = []
                if i > 1:
                    prev_words.append(self._effective_word(i - 2, context))
                if i > 0:
                    prev_words.append(self._effective_word(i - 1, context))

                next_words: list[str] = []
                if i + 1 < len(context.words):
                    next_words.append(self._effective_word(i + 1, context))

                # Get word frequency
                word_freq = 0
                if hasattr(self.provider, "get_word_frequency"):
                    freq = self.provider.get_word_frequency(word)
                    if isinstance(freq, (int, float)):
                        word_freq = int(freq)

                # Build candidates as (homophone, "homophone") pairs
                candidates = [(h, "homophone") for h in sorted(homophones)]

                # Delegate to unified check
                verdict = self.context_checker.check_word_in_context(
                    word=word,
                    prev_words=prev_words,
                    next_words=next_words,
                    candidates=candidates,
                    word_freq=word_freq,
                )

                # Morphological compound guard: when the previous word is
                # a productive prefix (မ negation, အ nominalizer) and
                # prefix+candidate is a valid compound but prefix+word is
                # not, that is strong evidence the candidate is correct —
                # even if bidirectional n-gram averaging favors the word.
                if (
                    not verdict.is_error
                    and prev_words
                    and prev_words[-1] in _COMPOUND_PREFIXES
                    and hasattr(self.provider, "is_valid_word")
                    and hasattr(self.provider, "get_word_frequency")
                    and not self.provider.is_valid_word(prev_words[-1] + word)
                ):
                    for alt, _ctype in candidates:
                        if self.provider.is_valid_word(prev_words[-1] + alt):
                            alt_freq = self.provider.get_word_frequency(prev_words[-1] + alt)
                            if isinstance(alt_freq, (int, float)) and alt_freq > 0:
                                verdict = verdict.__class__(
                                    is_error=True,
                                    confidence=0.85,
                                    error_type="homophone_error",
                                    best_alternative=alt,
                                    probability=0.0,
                                )
                                break

                # Right-context compound guard: when the next word is a
                # common verbal suffix and word+suffix is invalid but
                # candidate+suffix is valid, favor the candidate.
                if (
                    not verdict.is_error
                    and next_words
                    and next_words[0] in _COMPOUND_SUFFIXES
                    and hasattr(self.provider, "is_valid_word")
                    and hasattr(self.provider, "get_word_frequency")
                    and not self.provider.is_valid_word(word + next_words[0])
                ):
                    for alt, _ctype in candidates:
                        if self.provider.is_valid_word(alt + next_words[0]):
                            alt_freq = self.provider.get_word_frequency(alt + next_words[0])
                            if isinstance(alt_freq, (int, float)) and alt_freq > 0:
                                verdict = verdict.__class__(
                                    is_error=True,
                                    confidence=0.82,
                                    error_type="homophone_error",
                                    best_alternative=alt,
                                    probability=0.0,
                                )
                                break

                if verdict.is_error and verdict.best_alternative:
                    if position in context.existing_errors:
                        # Position already flagged by higher-priority strategy —
                        # append our suggestion without creating a duplicate error
                        existing = context.existing_suggestions.get(position, [])
                        if verdict.best_alternative not in existing:
                            existing.append(verdict.best_alternative)
                            context.existing_suggestions[position] = existing
                    else:
                        errors.append(
                            ContextError(
                                text=word,
                                position=position,
                                error_type=ET_HOMOPHONE_ERROR,
                                suggestions=[verdict.best_alternative],
                                confidence=self.confidence,
                                probability=verdict.probability,
                                prev_word=prev_words[-1] if prev_words else "",
                            )
                        )

                        # Mark this position as having an error
                        context.existing_errors[position] = ET_HOMOPHONE_ERROR
                        context.existing_suggestions[position] = [verdict.best_alternative]
                        context.existing_confidences[position] = self.confidence

        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError, TypeError) as e:
            self.logger.error(f"Error in homophone validation: {e}", exc_info=True)

        return errors

    def _effective_word(self, idx: int, context: ValidationContext) -> str:
        """Return corrected form of a context word if available, else original.

        Checks two sources for corrections:
        1. ``existing_suggestions`` from higher-priority context strategies.
        2. If the word is invalid (OOV) and no context correction exists,
           check if the provider can identify a valid correction (e.g., a
           word flagged by the syllable validator in a previous layer whose
           correction is not propagated to the context validator).
        """
        word = context.words[idx]
        pos = context.word_positions[idx]
        corrections = context.existing_suggestions.get(pos)
        if corrections:
            return corrections[0]
        # If the word is not valid, its n-gram evidence is likely zero.
        # Try the confusable variant that is most common — this approximates
        # what the syllable/word validator would have suggested.
        if hasattr(self.provider, "is_valid_word") and not self.provider.is_valid_word(word):
            if hasattr(self.provider, "get_confusable_pairs"):
                best_variant = None
                best_freq = 0
                for (
                    variant,
                    _ctype,
                    _overlap,
                    _fratio,
                    _suppress,
                ) in self.provider.get_confusable_pairs(word):
                    freq = self.provider.get_word_frequency(variant)
                    freq_val = int(freq) if isinstance(freq, (int, float)) else 0
                    if freq_val > best_freq:
                        best_freq = freq_val
                        best_variant = variant
                if best_variant:
                    return best_variant
        return word

    def priority(self) -> int:
        """
        Return strategy execution priority.

        Returns:
            45 (runs before n-gram validation)
        """
        return 45

    def __repr__(self) -> str:
        """String representation."""
        enabled = "enabled" if self.homophone_checker else "disabled"
        return (
            f"HomophoneValidationStrategy(priority={self.priority()}, {enabled}, "
            f"confidence={self.confidence})"
        )
