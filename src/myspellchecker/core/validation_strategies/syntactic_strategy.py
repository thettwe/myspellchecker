"""
Syntactic Validation Strategy.

This strategy applies rule-based grammar checking to detect
syntactic errors like particle typos, incorrect verb forms,
and grammatical patterns.

Priority: 20 (runs after tone disambiguation)

When to Use:
    SyntacticRuleChecker is optional but recommended for:
    - Full spell checking with grammar validation
    - Production environments requiring grammatical accuracy
    - Detecting context-sensitive errors (e.g., particle misuse after verbs)

    Skip for:
    - Simple syllable/word validation only
    - Latency-critical applications (adds ~5-10ms overhead)
    - Testing without full grammar rules loaded

Grammar Checkers Integrated:
    The SyntacticRuleChecker orchestrates 6 specialized checkers:
    - AspectChecker: Validates aspect markers (ခဲ့, ပြီ, နေ, etc.)
    - ClassifierChecker: Validates numeral-classifier patterns
    - CompoundChecker: Validates compound words and reduplications
    - MergedWordChecker: Detects segmenter merge errors (e.g., ကစား → က + စား)
    - NegationChecker: Validates negation patterns (မ...ဘူး)
    - RegisterChecker: Detects formal/colloquial register mixing

Note:
    All checker-specific error types (aspect_error, classifier_error, etc.) are
    mapped to ``syntax_error`` in ``ContextError`` by this strategy. Downstream
    consumers should check the ``suggestions`` field for checker-specific context
    rather than relying on ``error_type`` differentiation.

See Also:
    - grammar/engine.py: SyntacticRuleChecker implementation
    - grammar/patterns.py: Centralized pattern definitions
"""

from __future__ import annotations

from myspellchecker.core.constants import ET_SYNTAX_ERROR
from myspellchecker.core.response import ContextError, Error
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.grammar.engine import SyntacticRuleChecker
from myspellchecker.grammar.patterns import SENTENCE_PARTICLES
from myspellchecker.text.normalize import normalize
from myspellchecker.utils.logging_utils import get_logger

_DUPLICATED_ENDING_BASES: tuple[str, ...] = (
    normalize("ပါသည်"),
    normalize("ပါတယ်"),
    normalize("သည်"),
    normalize("တယ်"),
    normalize("မည်"),
    normalize("မယ်"),
    normalize("ပြီ"),
    normalize("ဘူး"),
    normalize("လား"),
)
_EDGE_PUNCT = "၊။,.!?;:\"'()[]{}"


class SyntacticValidationStrategy(ValidationStrategy):
    """
    Rule-based syntactic validation strategy.

    This strategy uses the SyntacticRuleChecker to detect grammatical
    errors based on predefined linguistic rules. These are high-confidence
    patterns that don't require n-gram statistics.

    Common syntactic errors detected:
    - Particle typos: မှာ → မှ (context-specific)
    - Verb form errors: သည် → တယ် (formality)
    - Grammatical patterns: subject-verb agreement

    Priority: 20 (runs after tone disambiguation)
    - Rule-based checks are fast and deterministic
    - Should run before statistical methods
    - Confidence: 0.9 (very high for rule-based)

    Example:
        >>> strategy = SyntacticValidationStrategy(syntactic_checker)
        >>> context = ValidationContext(
        ...     sentence="သူ မှာ ကျောင်း သွား ခဲ့ တယ်",
        ...     words=["သူ", "မှာ", "ကျောင်း", "သွား", "ခဲ့", "တယ်"],
        ...     word_positions=[0, 6, 12, 27, 36, 45]
        ... )
        >>> errors = strategy.validate(context)
    """

    def __init__(
        self,
        syntactic_rule_checker: SyntacticRuleChecker | None,
        confidence: float = 0.9,
    ):
        """
        Initialize syntactic validation strategy.

        Args:
            syntactic_rule_checker: SyntacticRuleChecker instance for grammar checking.
                                   If None, this strategy is disabled.
            confidence: Confidence score for rule-based corrections (default: 0.9).
        """
        self.syntactic_rule_checker = syntactic_rule_checker
        self.confidence = confidence
        self.logger = get_logger(__name__)

    def validate(self, context: ValidationContext) -> list[Error]:
        """
        Validate words for syntactic/grammatical errors.

        Args:
            context: Validation context with sentence information.

        Returns:
            List of ContextError objects for syntactic errors.
        """
        if not self.syntactic_rule_checker or not context.words:
            return []

        errors: list[Error] = []

        try:
            # Fast-path for merged duplicated sentence endings (e.g., သည်သည်).
            self._add_duplicated_sentence_ending_errors(context, errors)

            # Run syntactic rule checking on word sequence
            rule_corrections = self.syntactic_rule_checker.check_sequence(context.words)

            for idx, error_word, suggestion, rule_confidence in rule_corrections:
                # Validate index bounds
                if idx < 0 or idx >= len(context.words):
                    self.logger.warning(
                        f"Syntactic correction index {idx} out of bounds for "
                        f"{len(context.words)} words"
                    )
                    continue

                # Skip if this word is a proper name (though particles are rarely names)
                if context.is_name_mask[idx]:
                    continue

                # Skip if this word already has an error from previous strategy
                if context.word_positions[idx] in context.existing_errors:
                    continue

                # Guard: SFP at position 0 is a segmenter artifact, not a syntax error
                if idx == 0 and error_word in SENTENCE_PARTICLES:
                    continue

                # Guard: SFP after ပါ is a split polite form (ပါတယ် → ပါ + တယ်).
                # The segmenter may split polite colloquial endings into two tokens;
                # this is not a register/syntax error.
                if (
                    idx > 0
                    and error_word in SENTENCE_PARTICLES
                    and context.words[idx - 1] == normalize("ပါ")
                ):
                    continue

                # In fusion mode, low-to-medium confidence syntax rules are
                # FP-prone because the mutex no longer blocks them. Discount
                # their confidence so the output gate filters them out.
                # Reliable rules (medial confusion 0.90+, particle typo 0.85+)
                # are above the 0.85 cutoff and pass through undiscounted.
                effective_confidence = rule_confidence
                if context.fusion_mode and rule_confidence <= 0.85:
                    effective_confidence = rule_confidence * 0.6

                # Create syntax error
                errors.append(
                    ContextError(
                        text=error_word,
                        position=context.word_positions[idx],
                        error_type=ET_SYNTAX_ERROR,
                        suggestions=[suggestion],
                        confidence=effective_confidence,
                        probability=0.0,  # Rule-based doesn't use n-gram probability
                        prev_word=context.words[idx - 1] if idx > 0 else "",
                    )
                )

                # Mark this position as having an error
                pos = context.word_positions[idx]
                context.existing_errors[pos] = ET_SYNTAX_ERROR
                context.existing_suggestions[pos] = [suggestion]
                context.existing_confidences[pos] = effective_confidence

        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError) as e:
            self.logger.error(f"Error in syntactic validation: {e}", exc_info=True)

        return errors

    def _add_duplicated_sentence_ending_errors(
        self,
        context: ValidationContext,
        errors: list[Error],
    ) -> None:
        """Detect duplicated sentence ending tokens merged into a single token."""
        for idx, token in enumerate(context.words):
            position = context.word_positions[idx]
            if context.is_name_mask[idx]:
                continue
            if position in context.existing_errors:
                continue

            suggestion = self._dedupe_sentence_ending_token(token)
            if suggestion is None:
                continue

            prev_word = context.words[idx - 1] if idx > 0 else ""
            errors.append(
                ContextError(
                    text=token,
                    position=position,
                    error_type=ET_SYNTAX_ERROR,
                    suggestions=[suggestion],
                    confidence=self.confidence,
                    probability=0.0,
                    prev_word=prev_word,
                )
            )
            context.existing_errors[position] = ET_SYNTAX_ERROR
            context.existing_suggestions[position] = [suggestion]
            context.existing_confidences[position] = self.confidence

    @staticmethod
    def _dedupe_sentence_ending_token(token: str) -> str | None:
        """Return corrected token when it is a duplicated sentence ending."""
        stripped = token.strip(_EDGE_PUNCT)
        if not stripped:
            return None
        for base in _DUPLICATED_ENDING_BASES:
            doubled = base + base
            if stripped == doubled:
                return token.replace(doubled, base, 1)
        return None

    def priority(self) -> int:
        """
        Return strategy execution priority.

        Returns:
            20 (runs after tone disambiguation, before statistical methods)
        """
        return 20

    def __repr__(self) -> str:
        """String representation."""
        enabled = "enabled" if self.syntactic_rule_checker else "disabled"
        return (
            f"SyntacticValidationStrategy(priority={self.priority()}, {enabled}, "
            f"confidence={self.confidence})"
        )
