"""
Myanmar Negation Pattern Validator.

This module implements validation for Myanmar negation patterns.
Myanmar negation follows specific patterns:
    - Standard: မ + verb_stem + ဘူး (don't/didn't)
    - Polite: မ + verb_stem + ပါဘူး (politely don't)
    - Prohibition: မ + verb_stem + နဲ့ (Don't!)
    - Formal: မ + verb_stem + ပါ (written formal)

Features:
    - Detect negation patterns in text
    - Validate negation structure
    - Identify common negation typos
    - Check negation-particle agreement

Examples:
    - မသွားဘူး (don't go)
    - မစားချင်ဘူး (don't want to eat)
    - မလုပ်နဲ့ (Don't do!)
    - မရှိပါ (doesn't exist - formal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from myspellchecker.core.config.grammar_configs import NegationCheckerConfig
from myspellchecker.core.constants import ET_NEGATION_ERROR, ET_TYPO
from myspellchecker.core.response import GrammarError
from myspellchecker.grammar.config import get_grammar_config
from myspellchecker.utils.singleton import Singleton

__all__ = [
    "NegationChecker",
    "NegationError",
    "NegationInfo",
    "get_negation_checker",
    "is_negative_ending",
]

# Singleton registry for NegationChecker
_singleton: Singleton["NegationChecker"] = Singleton()

# Register constants
REGISTER_FORMAL = "formal"
REGISTER_COLLOQUIAL = "colloquial"


@dataclass
class NegationInfo:
    """
    Information about a negation pattern.

    Attributes:
        start_index: Index where negation starts.
        end_index: Index where negation ends.
        words: The words forming the negation pattern.
        pattern_type: Type of negation (standard, prohibition, etc.).
        verb: The main verb being negated.
        auxiliaries: Any auxiliary verbs in the pattern.
        ending: The negative ending particle.
        register: Formal or colloquial register.
    """

    start_index: int
    end_index: int
    words: list[str]
    pattern_type: str
    verb: str | None
    auxiliaries: list[str]
    ending: str | None
    register: str


@dataclass
class NegationError(GrammarError):
    """
    Represents a negation-related error.

    Extends GrammarError to integrate with the spell checker's error hierarchy.

    Attributes:
        text: The erroneous word (inherited from Error).
        position: Index of the error in the word list (inherited from Error).
        suggestions: List of suggested corrections (inherited from Error).
        error_type: Type of error (typo, missing_ending, invalid_pattern).
        confidence: Confidence score (0.0-1.0) (inherited from Error).
        reason: Human-readable explanation (inherited from GrammarError).
        word: Alias for 'text' (inherited from GrammarError).
        suggestion: First suggestion (inherited from GrammarError).
    """

    # Override default error_type
    error_type: str = field(default=ET_NEGATION_ERROR)


class NegationChecker:
    """
    Validates Myanmar negation patterns.

    This checker identifies:
    - Negation pattern typos
    - Missing negative endings
    - Incomplete negation patterns
    - Register-inconsistent negation

    Attributes:
        negation_prefix: The negation marker (မ).
        valid_endings: Set of valid negative endings.
        typo_map: Dictionary of common typos.
    """

    def __init__(
        self,
        config_path: str | None = None,
        checker_config: NegationCheckerConfig | None = None,
    ) -> None:
        """Initialize the negation checker."""
        self.config = get_grammar_config(config_path)
        self.checker_config = checker_config or NegationCheckerConfig()
        neg_config = self.config.negation_config

        self.negation_prefix = neg_config.get("prefix", "မ")

        self.negative_endings = neg_config.get("endings", {})
        self.valid_endings = set(self.negative_endings.keys())

        self.typo_map = neg_config.get("typo_map", {})
        self.auxiliaries = neg_config.get("auxiliaries", {})
        self.common_verbs = neg_config.get("common_verbs", {})
        self.register_map = neg_config.get("register_map", {})

    def starts_with_negation(self, word: str) -> bool:
        """
        Check if a word starts with the negation prefix.

        Args:
            word: Word to check.

        Returns:
            True if word starts with မ.

        Examples:
            >>> checker = NegationChecker()
            >>> checker.starts_with_negation("မသွား")
            True
            >>> checker.starts_with_negation("သွား")
            False
        """
        return word.startswith(self.negation_prefix) and len(word) > 1

    def is_negative_ending(self, word: str) -> bool:
        """
        Check if a word is a valid negative ending.

        Args:
            word: Word to check.

        Returns:
            True if word is a valid negative ending.

        Examples:
            >>> checker = NegationChecker()
            >>> checker.is_negative_ending("ဘူး")
            True
            >>> checker.is_negative_ending("တယ်")
            False
        """
        return word in self.valid_endings

    def get_ending_type(self, ending: str) -> tuple[str, str] | None:
        """
        Get the type and description of a negative ending.

        Args:
            ending: Negative ending word.

        Returns:
            Tuple of (type, description) or None.

        Examples:
            >>> checker = NegationChecker()
            >>> checker.get_ending_type("ဘူး")
            ('standard_negative', 'Colloquial negative ending')
        """
        if ending in self.negative_endings:
            info = self.negative_endings[ending]
            if isinstance(info, dict):
                return (str(info.get("type", "")), str(info.get("description", "")))
            return cast(tuple[str, str], info)
        return None

    def is_auxiliary(self, word: str) -> bool:
        """
        Check if a word is a negative auxiliary verb.

        Auxiliaries appear between the main verb and ending:
        မလုပ်ချင်ဘူး (don't want to do) - ချင် is auxiliary

        Args:
            word: Word to check.

        Returns:
            True if word is an auxiliary.

        Examples:
            >>> checker = NegationChecker()
            >>> checker.is_auxiliary("ချင်")
            True
            >>> checker.is_auxiliary("သွား")
            False
        """
        return word in self.auxiliaries

    def check_ending_typo(self, word: str) -> tuple[str, float] | None:
        """
        Check if a word is a typo of a negative ending.

        Args:
            word: Word to check.

        Returns:
            Tuple of (correction, confidence) or None.

        Examples:
            >>> checker = NegationChecker()
            >>> checker.check_ending_typo("ဘူ")
            ('ဘူး', 0.90)
        """
        if word in self.typo_map:
            correction = self.typo_map[word]
            return (correction, self.checker_config.typo_confidence)

        # Check for common patterns
        # Missing visarga (း)
        if word + "း" in self.valid_endings:
            return (word + "း", self.checker_config.missing_visarga_confidence)

        return None

    def detect_negation_pattern(
        self,
        words: list[str],
        start_index: int,
    ) -> NegationInfo | None:
        """
        Detect a negation pattern starting at the given index.

        Looks for patterns like:
        - မ + verb + ဘူး
        - မ + verb + auxiliary + ဘူး
        - မ + verb + နဲ့ (prohibition)

        Args:
            words: List of words.
            start_index: Index to start detection.

        Returns:
            NegationInfo if pattern found, None otherwise.

        Examples:
            >>> checker = NegationChecker()
            >>> info = checker.detect_negation_pattern(
            ...     ["မ", "သွား", "ဘူး"], 0
            ... )
            >>> info.pattern_type
            'standard_negative'
        """
        if start_index >= len(words):
            return None

        current_word = words[start_index]

        # Check if first word starts with negation
        if not self.starts_with_negation(current_word):
            # Check if current word is standalone negation prefix
            if current_word != self.negation_prefix:
                return None

        # Collect pattern words
        pattern_words = [current_word]
        verb = None
        auxiliaries_found: list[str] = []
        ending = None
        end_index = start_index

        # If the word is just the prefix, the verb is next
        if current_word == self.negation_prefix:
            if start_index + 1 < len(words):
                verb = words[start_index + 1]
                pattern_words.append(verb)
                end_index = start_index + 1
        else:
            # Verb is attached to prefix (e.g., မသွား)
            verb = current_word[len(self.negation_prefix) :]

        # Look for auxiliaries and ending
        search_start = end_index + 1
        for i in range(search_start, min(len(words), search_start + 3)):
            word = words[i]

            if self.is_negative_ending(word):
                ending = word
                pattern_words.append(word)
                end_index = i
                break
            elif self.is_auxiliary(word):
                auxiliaries_found.append(word)
                pattern_words.append(word)
                end_index = i
            else:
                # Check for typo
                typo_result = self.check_ending_typo(word)
                if typo_result:
                    ending = typo_result[0]  # The correction
                    pattern_words.append(word)
                    end_index = i
                    break
                # Unknown word - might be end of pattern
                break

        # Determine pattern type and register
        pattern_type = "incomplete"
        register = REGISTER_COLLOQUIAL  # Default

        if ending:
            ending_info = self.get_ending_type(ending)
            if ending_info:
                pattern_type = ending_info[0]

            # Check register
            full_pattern = "".join(pattern_words)
            if full_pattern in self.register_map:
                register = self.register_map[full_pattern]
            elif ending in {"ပါ", "နှင့်", "လျှင်"}:
                register = REGISTER_FORMAL

        return NegationInfo(
            start_index=start_index,
            end_index=end_index,
            words=pattern_words,
            pattern_type=pattern_type,
            verb=verb,
            auxiliaries=auxiliaries_found,
            ending=ending,
            register=register,
        )

    def validate_sequence(
        self,
        words: list[str],
    ) -> list[NegationError]:
        """
        Validate negation patterns in a word sequence.

        Checks for:
        1. Negative ending typos
        2. Missing negative endings
        3. Invalid negation patterns

        Args:
            words: List of words to validate.

        Returns:
            List of NegationError objects.

        Examples:
            >>> checker = NegationChecker()
            >>> errors = checker.validate_sequence(["မ", "သွား", "ဘူ"])
            >>> len(errors)
            1
            >>> errors[0].suggestion
            'ဘူး'
        """
        errors: list[NegationError] = []
        flagged_positions: set[int] = set()
        i = 0

        while i < len(words):
            word = words[i]

            # Check for standalone typos of negative endings
            if not self.is_negative_ending(word):
                typo_result = self.check_ending_typo(word)
                if typo_result:
                    # Check if this follows a negation pattern
                    if i > 0:
                        prev_word = words[i - 1]
                        if self.starts_with_negation(prev_word) or (
                            i > 1 and self.starts_with_negation(words[i - 2])
                        ):
                            correction, confidence = typo_result
                            errors.append(
                                NegationError(
                                    text=word,
                                    position=i,
                                    suggestions=[correction],
                                    error_type=ET_TYPO,
                                    confidence=confidence,
                                    reason=(f"Negative ending typo: {word} → {correction}"),
                                )
                            )
                            flagged_positions.add(i)
                            i += 1
                            continue

            # Detect negation patterns
            if self.starts_with_negation(word) or word == self.negation_prefix:
                pattern = self.detect_negation_pattern(words, i)

                if pattern:
                    # Check for typos within the pattern
                    for j, pattern_word in enumerate(pattern.words):
                        actual_pos = pattern.start_index + j
                        # Skip positions already flagged by standalone typo check
                        if actual_pos in flagged_positions:
                            continue
                        # Check if it's a typo ending (like ဘူ instead of ဘူး)
                        typo_result = self.check_ending_typo(pattern_word)
                        if typo_result:
                            correction, confidence = typo_result
                            errors.append(
                                NegationError(
                                    text=pattern_word,
                                    position=actual_pos,
                                    suggestions=[correction],
                                    error_type=ET_TYPO,
                                    confidence=confidence,
                                    reason=(f"Negative ending typo: {pattern_word} → {correction}"),
                                )
                            )
                            flagged_positions.add(actual_pos)
                        # Also check if it's a recognized "typo" ending type
                        elif self.is_negative_ending(pattern_word):
                            ending_info = self.get_ending_type(pattern_word)
                            if ending_info and "typo" in ending_info[0]:
                                # This is a documented typo ending
                                # Look up the correction from typo_map
                                if pattern_word in self.typo_map:
                                    correction = self.typo_map[pattern_word]
                                    errors.append(
                                        NegationError(
                                            text=pattern_word,
                                            position=actual_pos,
                                            suggestions=[correction],
                                            error_type=ET_TYPO,
                                            confidence=self.checker_config.typo_confidence,
                                            reason=(
                                                f"Negative ending typo: "
                                                f"{pattern_word} → {correction}"
                                            ),
                                        )
                                    )
                                    flagged_positions.add(actual_pos)

                    # Check for incomplete patterns
                    if pattern.pattern_type == "incomplete" and pattern.ending is None:
                        # Look ahead to see if there's a potential ending typo
                        if pattern.end_index + 1 < len(words):
                            next_word = words[pattern.end_index + 1]
                            typo_result = self.check_ending_typo(next_word)
                            if typo_result:
                                correction, confidence = typo_result
                                errors.append(
                                    NegationError(
                                        text=next_word,
                                        position=pattern.end_index + 1,
                                        suggestions=[correction],
                                        error_type=ET_TYPO,
                                        confidence=confidence,
                                        reason=(
                                            f"Negative ending typo: {next_word} → {correction}"
                                        ),
                                    )
                                )

                    # Skip to end of pattern
                    i = pattern.end_index + 1
                    continue

            i += 1

        return errors


# Module-level singleton
def get_negation_checker() -> NegationChecker:
    """
    Get the default NegationChecker singleton.

    Returns:
        NegationChecker instance.
    """
    return _singleton.get(NegationChecker)


def is_negative_ending(word: str) -> bool:
    """
    Convenience function to check if a word is a negative ending.

    Args:
        word: Word to check.

    Returns:
        True if word is a valid negative ending.
    """
    return get_negation_checker().is_negative_ending(word)
