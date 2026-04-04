"""
N-gram Context Validation Strategy.

This strategy validates word sequences using bigram and trigram probability models
to detect contextually unlikely word combinations.

Delegates the core detection to ``NgramContextChecker.check_word_in_context()``
(absolute threshold check, no candidate comparison).

Priority: 50 (runs after rule-based validation, before specialized detectors)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.core.config.algorithm_configs import NgramStrategyConfig
from myspellchecker.core.constants import ET_CONTEXT_PROBABILITY, SKIPPED_CONTEXT_WORDS
from myspellchecker.core.response import ContextError, Error
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.ngram_context_checker import NgramContextChecker
    from myspellchecker.providers.interfaces import NgramRepository


class NgramContextValidationStrategy(ValidationStrategy):
    """
    N-gram probability validation strategy.

    Uses ``NgramContextChecker.check_word_in_context()`` (absolute threshold
    mode, no candidates) to detect contextually unlikely word sequences and
    generates suggestions via ``NgramContextChecker.suggest()``.

    Priority: 50 (runs after rule-based methods and homophone detection)
    """

    def __init__(
        self,
        context_checker: "NgramContextChecker | None",
        provider: "NgramRepository",
        confidence_high: float | None = None,
        confidence_low: float | None = None,
        max_suggestions: int | None = None,
        edit_distance: int | None = None,
        config: NgramStrategyConfig | None = None,
        # Legacy kwargs accepted for backward-compat but ignored.
        **_kwargs: object,
    ):
        """
        Initialize n-gram context validation strategy.

        Args:
            context_checker: NgramContextChecker instance for context analysis.
                           If None, this strategy is disabled.
            provider: NgramRepository for n-gram probability lookups.
            confidence_high: Confidence for high-probability errors (default: 0.9).
            confidence_low: Confidence for low-probability errors (default: 0.6).
            max_suggestions: Maximum number of suggestions to generate (default: 5).
            edit_distance: Maximum edit distance for suggestions (default: 2).
            config: NgramStrategyConfig with all thresholds. Explicit
                   kwargs override config values.
        """
        self._config = config or NgramStrategyConfig()
        self.context_checker = context_checker
        self.provider = provider
        self.confidence_high = (
            confidence_high if confidence_high is not None else self._config.confidence_high
        )
        self.confidence_low = (
            confidence_low if confidence_low is not None else self._config.confidence_low
        )
        self.max_suggestions = (
            max_suggestions if max_suggestions is not None else self._config.max_suggestions
        )
        self.edit_distance = (
            edit_distance if edit_distance is not None else self._config.edit_distance
        )
        self.logger = get_logger(__name__)

    def validate(self, context: ValidationContext) -> list[Error]:
        """
        Validate word sequences using n-gram probability models.

        Args:
            context: Validation context with sentence information.

        Returns:
            List of ContextError objects for contextually unlikely sequences.
        """
        if not self.context_checker or len(context.words) < 2:
            return []

        # Literary register guard: skip n-gram for classical/literary sentences
        # whose rare word combinations produce FPs on standard-trained models
        sentence = context.sentence
        if any(marker in sentence for marker in self._config.literary_sentence_markers):
            return []

        errors: list[Error] = []

        try:
            # Filter out names and skipped words, preserving positions
            filtered_words: list[tuple[str, int, bool]] = []
            for i, word in enumerate(context.words):
                filtered_words.append((word, context.word_positions[i], context.is_name_mask[i]))

            # Analyze context for each word pair
            for i in range(len(filtered_words) - 1):
                current_word, current_pos, _ = filtered_words[i]
                next_word, next_pos, next_is_name = filtered_words[i + 1]

                # Skip names and common words that don't need context validation
                if next_is_name or next_word in SKIPPED_CONTEXT_WORDS:
                    continue

                # When the preceding word has an error, use its top
                # correction for n-gram lookups instead of skipping entirely.
                effective_current = current_word
                if current_pos in context.existing_errors:
                    corrected = context.existing_suggestions.get(current_pos, [])
                    if corrected:
                        effective_current = corrected[0]

                # Get surrounding context for better analysis, applying
                # error-correction to each context word so that higher-order
                # n-gram lookups use corrected forms (not raw misspellings).
                #
                # Variable naming is relative to the loop index i:
                #   word_at_im1 = filtered_words[i-1] (2 back from checked word)
                #   word_at_ip2 = filtered_words[i+2] (1 after checked word)
                word_at_im1 = (
                    self._get_effective_word(
                        filtered_words[i - 1][0], filtered_words[i - 1][1], context
                    )
                    if i > 0
                    else None
                )
                word_at_ip2 = filtered_words[i + 2][0] if i + 2 < len(filtered_words) else None

                if next_pos in context.existing_errors:
                    # Position already flagged by higher-priority strategy —
                    # generate suggestions to append without creating duplicate error
                    suggestions = self._generate_suggestions(
                        current_word=effective_current,
                        next_word=next_word,
                        next_next_word=word_at_ip2,
                    )
                    if suggestions:
                        existing = context.existing_suggestions.get(next_pos, [])
                        for s in suggestions:
                            if s not in existing:
                                existing.append(s)
                        context.existing_suggestions[next_pos] = existing
                    continue

                # Build context word lists for check_word_in_context.
                # The checked word is next_word (i+1).  prev_words are ordered
                # oldest-first, so [i-3, i-2, i-1, i] with closest last.
                prev_words: list[str] = []
                if i > 1:
                    word_at_im2 = self._get_effective_word(
                        filtered_words[i - 2][0], filtered_words[i - 2][1], context
                    )
                    prev_words.append(word_at_im2)
                if i > 2:
                    word_at_im3 = self._get_effective_word(
                        filtered_words[i - 3][0], filtered_words[i - 3][1], context
                    )
                    prev_words.insert(0, word_at_im3)
                if word_at_im1:
                    prev_words.append(word_at_im1)
                prev_words.append(effective_current)

                next_words: list[str] = [next_word]
                if word_at_ip2:
                    next_words.append(word_at_ip2)

                # check_word_in_context API:
                #   prev_words = [..., i-2, i-1, i] (closest last)
                #   next_words = [i+1, i+2, ...] (closest first)
                # When checking next_word (i+1), effective_current (i) is the closest prev.

                # Use check_word_in_context (absolute threshold, no candidates)
                verdict = self.context_checker.check_word_in_context(
                    word=next_word,
                    prev_words=prev_words,
                    next_words=next_words[1:],  # exclude next_word itself
                    candidates=None,
                    word_freq=0,
                )

                if not verdict.is_error:
                    continue

                # Generate context-aware suggestions
                suggestions = self._generate_suggestions(
                    current_word=effective_current,
                    next_word=next_word,
                    next_next_word=word_at_ip2,
                )

                # High-frequency word guard: suppress FP when a common word
                # has no better alternatives.
                _get_freq = getattr(self.provider, "get_word_frequency", None)
                if _get_freq is not None:
                    freq = _get_freq(next_word)
                    if (
                        isinstance(freq, (int, float))
                        and freq >= self._config.high_freq_ngram_guard
                    ):
                        if not suggestions:
                            continue
                        best_sugg_freq = max(
                            (_get_freq(s) for s in suggestions),
                            default=0,
                        )
                        if isinstance(best_sugg_freq, (int, float)) and best_sugg_freq < freq:
                            continue

                # Determine confidence based on suggestion quality
                confidence = self.confidence_high if suggestions else self.confidence_low

                error = ContextError(
                    text=next_word,
                    position=next_pos,
                    error_type=ET_CONTEXT_PROBABILITY,
                    suggestions=suggestions,
                    confidence=confidence,
                    probability=verdict.probability,
                    prev_word=effective_current,
                )

                errors.append(error)
                context.existing_errors[next_pos] = error.error_type
                context.existing_suggestions[next_pos] = list(error.suggestions)
                context.existing_confidences[next_pos] = error.confidence

        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError) as e:
            self.logger.error(f"Error in n-gram context validation: {e}", exc_info=True)
        return errors

    def _generate_suggestions(
        self,
        current_word: str,
        next_word: str,
        next_next_word: str | None,
    ) -> list[str]:
        """
        Generate context-aware suggestions for the unlikely word.

        Args:
            current_word: Previous word in sequence.
            next_word: The word flagged as unlikely.
            next_next_word: Following word (optional, for better ranking).

        Returns:
            List of suggested corrections based on n-gram continuations.
        """
        # Guard clause - strategy is disabled if context_checker is None
        if self.context_checker is None:
            return []

        try:
            # Get suggestions from context checker
            context_suggestions = self.context_checker.suggest(
                prev_word=current_word,
                current_word=next_word,
                max_edit_distance=self.edit_distance,
                next_word=next_next_word,
            )

            # Extract suggestion terms and limit to max_suggestions
            suggestions = [s.term for s in context_suggestions[: self.max_suggestions]]

            # Filter out suggestions that are the same as the original
            suggestions = [s for s in suggestions if s != next_word]

            return suggestions

        except (RuntimeError, ValueError, KeyError, IndexError) as e:
            self.logger.debug(f"Failed to generate n-gram suggestions: {e}")
            return []

    @staticmethod
    def _get_effective_word(word: str, position: int, context: ValidationContext) -> str:
        """Return the corrected form of *word* if it has an existing error, else the original."""
        if position in context.existing_errors:
            corrected = context.existing_suggestions.get(position, [])
            if corrected:
                return corrected[0]
        return word

    def priority(self) -> int:
        """
        Return strategy execution priority.

        Returns:
            50 (runs after rule-based validation, before specialized detectors)
        """
        return 50

    def __repr__(self) -> str:
        """String representation."""
        enabled = "enabled" if self.context_checker else "disabled"
        return (
            f"NgramContextValidationStrategy(priority={self.priority()}, {enabled}, "
            f"confidence_low={self.confidence_low}, "
            f"confidence_high={self.confidence_high})"
        )
