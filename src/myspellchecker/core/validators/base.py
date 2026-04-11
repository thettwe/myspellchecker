"""Base Validator abstract class for the spell checking pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import (
    COMMON_PUNCTUATION,
    MYANMAR_PUNCTUATION,
    contains_myanmar,
)
from myspellchecker.core.response import Error
from myspellchecker.text.validator import validate_word


class Validator(ABC):
    """
    Abstract base class for spell checking validators.

    Validators are the core components of the spell checking pipeline,
    responsible for detecting specific types of errors at different
    validation levels (syllable, word, context).

    Each validator follows a consistent interface:
        - Accepts normalized text via ``validate(text)``
        - Returns a list of ``Error`` objects with position, suggestions, and confidence
        - Provides static utility methods for common checks

    Subclasses:
        - ``SyllableValidator``: Layer 1 - validates individual syllables
        - ``WordValidator``: Layer 2 - validates multi-syllable words
        - ``ContextValidator``: Layer 3 - validates word sequences (in context_validator.py)

    Attributes:
        config: SpellCheckerConfig instance with validation parameters.

    Example:
        >>> class CustomValidator(Validator):
        ...     def validate(self, text: str) -> list[Error]:
        ...         errors = []
        ...         # Custom validation logic
        ...         return errors
    """

    def __init__(self, config: SpellCheckerConfig):
        """
        Initialize the validator with configuration.

        Args:
            config: SpellCheckerConfig instance containing validation
                parameters such as thresholds, confidence levels, and
                feature flags.
        """
        self.config = config

    @abstractmethod
    def validate(self, text: str) -> list[Error]:
        """
        Validate the given text and return a list of errors.

        Args:
            text: The normalized text to validate.

        Returns:
            List of Error objects found in the text.
        """
        raise NotImplementedError

    def _filter_suggestions(self, suggestions: list[str]) -> list[str]:
        """
        Filter out invalid suggestions using text validator.

        This removes suggestions that are structurally invalid Myanmar text,
        including Zawgyi artifacts, malformed syllables, and invalid patterns.

        Respects the ``allow_extended_myanmar`` config flag when validating.

        Args:
            suggestions: List of suggestion strings to filter.

        Returns:
            Filtered list containing only valid suggestions.
        """
        if not suggestions:
            return suggestions
        allow_extended = self.config.validation.allow_extended_myanmar
        return [s for s in suggestions if validate_word(s, allow_extended_myanmar=allow_extended)]

    @staticmethod
    def is_punctuation(text: str) -> bool:
        """
        Check if text consists entirely of punctuation characters.

        Recognizes both Myanmar-specific punctuation (section marks, etc.)
        and common ASCII punctuation marks.

        Args:
            text: Text to check.

        Returns:
            True if all characters are punctuation, False otherwise.
            Returns False for empty strings.

        Example:
            >>> Validator.is_punctuation("။")  # Myanmar section mark
            True
            >>> Validator.is_punctuation("...")
            True
            >>> Validator.is_punctuation("မြန်မာ")
            False
        """
        if not text:
            return False
        return all(c in MYANMAR_PUNCTUATION or c in COMMON_PUNCTUATION for c in text)

    def _is_myanmar_with_config(self, text: str) -> bool:
        """
        Check if text contains any Myanmar script characters, respecting config.

        This instance method uses the ``allow_extended_myanmar`` config flag
        to determine which character ranges to consider as Myanmar.

        Args:
            text: Text to check.

        Returns:
            True if any character is Myanmar script (per config), False otherwise.
        """
        if not text:
            return False
        allow_extended = self.config.validation.allow_extended_myanmar
        return contains_myanmar(text, allow_extended=allow_extended)

    @staticmethod
    def _find_token_position(text: str, token: str, start: int) -> tuple[int | None, int]:
        """
        Find token position in text starting from given index.

        Handles merged tokens where the segmenter removed whitespace
        (e.g. segmenter produces "ကြေးဇူးတင်" from "ကြေးဇူး တင်").

        Args:
            text: The text to search in.
            token: The token to find.
            start: Starting index for the search.

        Returns:
            Tuple of (position, next_start) where:
            - position: Index where token was found, or None if not found
            - next_start: Updated start index for next search
        """
        if not token:
            return None, start

        # Direct match (fast path)
        idx = text.find(token, start)
        if idx != -1:
            return idx, idx + len(token)

        # Space-tolerant match: match token chars while skipping spaces in text.
        # This handles cases where the segmenter merges words across spaces.
        ti = start
        token_idx = 0
        match_start = None
        while ti < len(text) and token_idx < len(token):
            if text[ti] == " ":
                ti += 1
                continue
            if text[ti] == token[token_idx]:
                if match_start is None:
                    match_start = ti
                ti += 1
                token_idx += 1
            else:
                # Reset and try from next position
                if match_start is not None:
                    ti = match_start + 1
                    match_start = None
                else:
                    ti += 1
                token_idx = 0

        if token_idx == len(token) and match_start is not None:
            return match_start, ti

        return None, start
