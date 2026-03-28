"""
Question Structure Validation Strategy.

This strategy validates question sentence structure, detecting missing
question particles and incorrect question patterns.

Priority: 40 (runs after POS validation)
"""

from __future__ import annotations

from myspellchecker.core.constants import ET_QUESTION_STRUCTURE
from myspellchecker.core.response import ContextError, Error
from myspellchecker.core.validation_strategies.base import ValidationContext, ValidationStrategy
from myspellchecker.grammar.patterns import (
    COMPLETIVE_ENDINGS,
    NEGATIVE_ENDINGS,
    SECOND_PERSON_PRONOUNS,
    detect_malformed_question_ending,
    detect_sentence_type,
    get_question_completion_suggestions,
    has_question_particle_context,
    has_question_word_context,
    is_question_word,
    is_second_person_modal_future_question,
)
from myspellchecker.utils.logging_utils import get_logger


class QuestionStructureValidationStrategy(ValidationStrategy):
    """
    Question structure validation strategy.

    This strategy detects questions that are missing proper question
    particles. In Myanmar language, questions typically end with
    particles like လား, သလား, လဲ, ရဲ့လား.

    Common question structure errors:
    - Question word present but missing question particle
    - Statement ending (တယ်, သည်) used instead of question particle

    Priority: 40 (runs after POS validation, before n-gram checking)
    - Requires sentence type detection
    - Higher priority than statistical methods
    - Confidence: 0.7 (moderate - some questions don't require particles)

    Example:
        >>> strategy = QuestionStructureValidationStrategy()
        >>> context = ValidationContext(
        ...     sentence="ဘယ်မှာ သွား တယ်",  # "Where go [statement]" - should be question
        ...     words=["ဘယ်မှာ", "သွား", "တယ်"],
        ...     word_positions=[0, 18, 30]
        ... )
        >>> errors = strategy.validate(context)
        # Suggests: လား, သလား, လဲ instead of တယ်
    """

    def __init__(self, confidence: float = 0.7):
        """
        Initialize question structure validation strategy.

        Args:
            confidence: Confidence score for question structure errors (default: 0.7).
        """
        self.confidence = confidence
        self.logger = get_logger(__name__)

    def validate(self, context: ValidationContext) -> list[Error]:
        """
        Validate question sentence structure.

        Args:
            context: Validation context with sentence information.

        Returns:
            List of ContextError objects for question structure issues.
        """
        if len(context.words) < 2:
            return []

        errors: list[Error] = []

        try:
            last_idx = len(context.words) - 1
            last_word = context.words[last_idx]
            last_position = context.word_positions[last_idx]

            # Fix 2a: Don't count question words that are segmentation fragments.
            # When a misspelled word like ဆာဘာ is segmented into ဆာ + ဘာ,
            # ဘာ ("what") appears as a question word but is really a fragment.
            # Detect by: (a) existing error at position, OR (b) character-adjacent
            # to previous word (no space gap = same original token).
            effective_words = list(context.words)
            for i, word in enumerate(context.words):
                if not is_question_word(word):
                    continue
                if context.word_positions[i] in context.existing_errors:
                    effective_words[i] = ""
                    continue
                # Character-adjacent to previous word → segmentation fragment
                if i > 0:
                    prev_end = context.word_positions[i - 1] + len(context.words[i - 1])
                    if prev_end == context.word_positions[i]:
                        effective_words[i] = ""

            # Detect sentence type using effective_words (with fragments masked)
            sentence_type = detect_sentence_type(effective_words)
            context.sentence_type = sentence_type  # Store for other strategies

            has_question_word = has_question_word_context(effective_words)
            has_second_person_modal = is_second_person_modal_future_question(effective_words)

            # Split malformed ending form: ... ရဲ့ လဲ
            if (
                len(context.words) >= 2
                and context.words[-2] in {"ရဲ့", "ရဲ"}
                and context.words[-1] == "လဲ"
                and context.word_positions[-2] not in context.existing_errors
            ):
                malformed_text = context.words[-2] + context.words[-1]
                malformed_suggestions = get_question_completion_suggestions(
                    malformed_text,
                    context.words,
                    prefer_yes_no=not has_question_word,
                    phrase_first=False,
                )
                if malformed_suggestions:
                    errors.append(
                        ContextError(
                            text=malformed_text,
                            position=context.word_positions[-2],
                            error_type=ET_QUESTION_STRUCTURE,
                            suggestions=malformed_suggestions,
                            confidence=self.confidence,
                            probability=0.0,
                            prev_word=context.words[-3] if len(context.words) > 2 else "",
                        )
                    )
                    qpos = context.word_positions[-2]
                    context.existing_errors[qpos] = ET_QUESTION_STRUCTURE
                    context.existing_suggestions[qpos] = malformed_suggestions
                    context.existing_confidences[qpos] = self.confidence
                    return errors

            malformed_ending = detect_malformed_question_ending(last_word)
            if malformed_ending and last_position not in context.existing_errors:
                malformed_suggestions = get_question_completion_suggestions(
                    last_word,
                    context.words,
                    prefer_yes_no=not has_question_word,
                    phrase_first=False,
                )
                if malformed_suggestions:
                    errors.append(
                        ContextError(
                            text=last_word,
                            position=last_position,
                            error_type=ET_QUESTION_STRUCTURE,
                            suggestions=malformed_suggestions,
                            confidence=self.confidence,
                            probability=0.0,
                            prev_word=context.words[-2] if len(context.words) > 1 else "",
                        )
                    )
                    context.existing_errors[last_position] = ET_QUESTION_STRUCTURE
                    context.existing_suggestions[last_position] = malformed_suggestions
                    context.existing_confidences[last_position] = self.confidence
                    return errors

            has_question_context = sentence_type == "question" or has_second_person_modal

            # For non-question sentences, only check completive implicit questions.
            if not has_question_context:
                self._check_implicit_question(context, errors)
                return errors

            # Check if question has proper ending particle
            has_question_particle = has_question_particle_context(
                context.words,
                has_question_word=has_question_word,
            )

            # If question words present but no question particle, report error
            if not has_question_particle:
                rewrite_target = last_word
                if (
                    has_question_word
                    and len(context.words) >= 2
                    and is_question_word(context.words[-2])
                ):
                    # Phrase-first rewrite for split question word + predicate
                    # patterns (e.g., "ဘယ် သွား" -> "ဘယ်သွားလဲ").
                    rewrite_target = context.words[-2] + last_word
                question_suggestions = get_question_completion_suggestions(
                    rewrite_target,
                    context.words,
                    prefer_yes_no=has_second_person_modal,
                    phrase_first=False,
                )
                if question_suggestions and last_position not in context.existing_errors:
                    errors.append(
                        ContextError(
                            text=last_word,
                            position=last_position,
                            error_type=ET_QUESTION_STRUCTURE,
                            suggestions=question_suggestions,
                            confidence=self.confidence,
                            probability=0.0,  # Rule-based doesn't use n-gram probability
                            prev_word=context.words[-2] if len(context.words) > 1 else "",
                        )
                    )

                    # Mark this position as having an error
                    context.existing_errors[last_position] = ET_QUESTION_STRUCTURE
                    context.existing_suggestions[last_position] = question_suggestions
                    context.existing_confidences[last_position] = self.confidence

        except (RuntimeError, ValueError, KeyError, IndexError, AttributeError) as e:
            self.logger.error(f"Error in question structure validation: {e}", exc_info=True)

        return errors

    def _check_implicit_question(self, context: ValidationContext, errors: list[Error]) -> None:
        """Detect implicit questions: 2nd-person pronoun + completive ending."""
        words = context.words

        # Guard: need at least 2 words
        if len(words) < 2:
            return

        # Guard: skip if any explicit question word present (handled above)
        if has_question_word_context(words):
            return

        # Guard: skip if already has a question particle.
        if has_question_particle_context(words, has_question_word=False):
            return

        # Guard: skip negative patterns (e.g., "မင်း မစားဘူး" is a valid statement)
        if any(word in NEGATIVE_ENDINGS for word in words):
            return

        # Check 1: 2nd-person pronoun in subject position (first or second word)
        has_second_person = False
        for i in range(min(2, len(words))):
            if words[i] in SECOND_PERSON_PRONOUNS:
                has_second_person = True
                break
        if not has_second_person:
            return

        # Check 2: sentence ends with a completive ending
        last_word = words[-1]

        # Check standalone completive ending
        matched_ending = None
        if last_word in COMPLETIVE_ENDINGS:
            matched_ending = last_word
        else:
            # Check enclitic completive (e.g., "ဖတ်ပြီးပြီ" — verb+completive)
            for ending in sorted(COMPLETIVE_ENDINGS, key=len, reverse=True):
                if last_word.endswith(ending) and len(last_word) > len(ending):
                    matched_ending = ending
                    break

        if not matched_ending:
            return

        # Position of the completive ending within the last word
        last_position = context.word_positions[-1]
        if matched_ending == last_word:
            error_position = last_position
        else:
            # Enclitic: position at the ending part within the word
            error_position = last_position + len(last_word) - len(matched_ending)

        # Skip if this position already has an error
        if error_position in context.existing_errors:
            return

        # Lower confidence for implicit questions — 2nd-person + completive
        # can be valid statements ("You have arrived"), not just questions.
        implicit_conf = min(self.confidence, 0.55)
        errors.append(
            ContextError(
                text=matched_ending,
                position=error_position,
                error_type=ET_QUESTION_STRUCTURE,
                suggestions=[matched_ending + "လား"],
                confidence=implicit_conf,
                probability=0.0,
                prev_word=words[-2] if len(words) > 1 else "",
            )
        )
        context.existing_errors[error_position] = ET_QUESTION_STRUCTURE
        context.existing_suggestions[error_position] = [matched_ending + "လား"]
        context.existing_confidences[error_position] = implicit_conf
        # Also claim the word-level position for dedup when error is sub-word
        if error_position != last_position:
            context.existing_errors[last_position] = ET_QUESTION_STRUCTURE

    def priority(self) -> int:
        """
        Return strategy execution priority.

        Returns:
            40 (runs after POS validation, before statistical n-gram checking)
        """
        return 40

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"QuestionStructureValidationStrategy(priority={self.priority()}, "
            f"confidence={self.confidence})"
        )
