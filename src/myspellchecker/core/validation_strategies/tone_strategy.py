"""
Tone Validation Strategy.

This strategy handles context-based Myanmar tone mark disambiguation.
Tone marks in Myanmar script can be ambiguous and require context to
determine the correct usage.

Priority: 10 (runs early)
"""

from __future__ import annotations

from myspellchecker.core.config.algorithm_configs import ToneStrategyConfig
from myspellchecker.core.constants import ET_TONE_AMBIGUITY
from myspellchecker.core.response import ContextError, Error
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.text.tone import ToneDisambiguator
from myspellchecker.utils.logging_utils import get_logger


class ToneValidationStrategy(ValidationStrategy):
    """
    Tone mark disambiguation strategy.

    This strategy uses the ToneDisambiguator to detect and correct
    tone mark errors based on context. Tone marks are critical in
    Myanmar language as they change meaning and pronunciation.

    Common tone mark errors:
    - Missing tone marks: ပြီ → ပြီး
    - Wrong tone marks: ခဲ့ → ခဲ (context-dependent)

    Priority: 10 (runs first)
    - Tone errors should be fixed before other validations
    - High confidence corrections based on n-gram context

    Example:
        >>> strategy = ToneValidationStrategy(tone_disambiguator)
        >>> context = ValidationContext(
        ...     sentence="သူ ပြီ ပြီး သွား ခဲ့ တယ်",
        ...     words=["သူ", "ပြီ", "ပြီး", "သွား", "ခဲ့", "တယ်"],
        ...     word_positions=[0, 6, 12, 21, 30, 39]
        ... )
        >>> errors = strategy.validate(context)
    """

    def __init__(
        self,
        tone_disambiguator: ToneDisambiguator | None,
        confidence_threshold: float | None = None,
        provider: object | None = None,
        config: ToneStrategyConfig | None = None,
    ):
        """
        Initialize tone validation strategy.

        Args:
            tone_disambiguator: ToneDisambiguator instance for tone checking.
                               If None, this strategy is disabled.
            confidence_threshold: Minimum confidence to report an error (default: 0.5).
            provider: Optional DictionaryProvider for word frequency lookup.
                When both original and correction are high-frequency (>100K),
                tone_ambiguity is suppressed as both forms are valid.
            config: ToneStrategyConfig with all thresholds. Explicit
                   kwargs override config values.
        """
        self._config = config or ToneStrategyConfig()
        self.tone_disambiguator = tone_disambiguator
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self._config.confidence_threshold
        )
        self._provider = provider
        self.logger = get_logger(__name__)

    def validate(self, context: ValidationContext) -> list[Error]:
        """
        Validate words for tone mark errors.

        Args:
            context: Validation context with sentence information.

        Returns:
            List of ContextError objects for tone mark errors.
        """
        if not self.tone_disambiguator or not context.words:
            return []

        errors: list[Error] = []

        try:
            # Run tone disambiguation on word sequence
            tone_corrections = self.tone_disambiguator.check_sentence(context.words)

            for idx, original, correction, confidence in tone_corrections:
                # Only report if confidence meets threshold
                if confidence < self.confidence_threshold:
                    continue

                # Validate index bounds
                if idx < 0 or idx >= len(context.words):
                    self.logger.warning(
                        f"Tone correction index {idx} out of bounds for {len(context.words)} words"
                    )
                    continue

                # Skip if this word is a proper name
                if context.is_name_mask[idx]:
                    continue

                # Skip if this word already has an error from previous strategy
                if context.word_positions[idx] in context.existing_errors:
                    continue

                # Suppress when both forms are high-frequency — both are
                # grammatically valid in different contexts (e.g., သူ့ possessive
                # vs သူ subject). The disambiguator can't reliably distinguish
                # these without deeper syntactic analysis.
                if self._provider and hasattr(self._provider, "get_word_frequency"):
                    orig_freq = self._provider.get_word_frequency(original)
                    corr_freq = self._provider.get_word_frequency(correction)
                    if (
                        orig_freq >= self._config.high_freq_threshold
                        and corr_freq >= self._config.high_freq_threshold
                    ):
                        continue

                # Create tone ambiguity error
                errors.append(
                    ContextError(
                        text=original,
                        position=context.word_positions[idx],
                        error_type=ET_TONE_AMBIGUITY,
                        suggestions=[correction],
                        confidence=confidence,
                        probability=0.0,  # Tone disambiguation doesn't use n-gram probability
                        prev_word=context.words[idx - 1] if idx > 0 else "",
                    )
                )

                # Mark this position as having an error
                pos = context.word_positions[idx]
                context.existing_errors[pos] = ET_TONE_AMBIGUITY
                context.existing_suggestions[pos] = [correction]
                context.existing_confidences[pos] = confidence

        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError) as e:
            self.logger.error(f"Error in tone validation: {e}", exc_info=True)

        return errors

    def priority(self) -> int:
        """
        Return strategy execution priority.

        Returns:
            10 (runs first - tone errors should be fixed before other validations)
        """
        return 10

    def __repr__(self) -> str:
        """String representation."""
        enabled = "enabled" if self.tone_disambiguator else "disabled"
        return f"ToneValidationStrategy(priority={self.priority()}, {enabled})"
