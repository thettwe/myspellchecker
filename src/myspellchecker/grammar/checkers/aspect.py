"""
Myanmar Aspect Marker Detection and Validation.

This module implements aspect marker detection and validation for Myanmar verbs.
Myanmar aspect markers modify verbs to express temporal, modal, and aspectual meanings.

Features:
    - Detect aspect markers in verb phrases
    - Validate aspect marker sequences
    - Identify typos in aspect markers
    - Check for valid/invalid aspect combinations

Aspect Categories:
    - Completion: ပြီ, ပြီး (action completed)
    - Progressive: နေ (action ongoing)
    - Habitual: တတ် (habitual action)
    - Resultative: ထား (state maintained)
    - Directional: လာ, သွား (motion direction)
    - Desiderative: ချင် (desire/want)
    - Potential: နိုင်, ရ (ability/possibility)
    - Immediate: လိုက် (following action)
    - Experiential: ဖူး (past experience)
    - Accidental: မိ (unintentional action)
    - Exhaustive: ကုန် (completive/all used up)
    - Dismissive: ပစ် (forceful/away completion)

Examples:
    သွားပြီ (went - completed)
    စားနေ (eating - progressive)
    လာချင် (want to come - desiderative)
    ရေးဖူး (have written - experiential)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from myspellchecker.core.config import AspectCheckerConfig
from myspellchecker.core.constants import (
    ET_ASPECT_ERROR,
    ET_ASPECT_TYPO,
    ET_INCOMPLETE_ASPECT,
    ET_INVALID_SEQUENCE,
)
from myspellchecker.core.response import GrammarError
from myspellchecker.grammar.config import get_grammar_config

__all__ = [
    "AspectChecker",
    "AspectError",
    "AspectInfo",
    "AspectPattern",
]

# Default Aspect Checker configuration (module-level singleton)
_default_aspect_config = AspectCheckerConfig()


@dataclass
class AspectInfo:
    """
    Aspect marker information.

    Attributes:
        marker: The aspect marker word.
        category: The aspect category (completion, progressive, etc.).
        description: Human-readable description.
        can_combine: Whether it can combine with other markers.
        register: Register compatibility (neutral, formal, colloquial).
        is_final: Whether it typically appears at phrase end.
    """

    marker: str
    category: str
    description: str
    can_combine: bool
    register: str
    is_final: bool

    def __str__(self) -> str:
        """Return string representation."""
        return f"AspectInfo({self.marker}: {self.category})"


@dataclass
class AspectPattern:
    """
    Represents a detected aspect marker pattern in text.

    Attributes:
        start_index: Starting position in word list.
        end_index: Ending position in word list.
        markers: List of aspect markers found.
        categories: List of aspect categories.
        is_valid: Whether the combination is valid.
        confidence: Detection confidence (0.0-1.0).
    """

    start_index: int
    end_index: int
    markers: list[str]
    categories: list[str]
    is_valid: bool
    confidence: float

    def __str__(self) -> str:
        """Return string representation."""
        markers_str = " + ".join(self.markers)
        validity = "valid" if self.is_valid else "invalid"
        return f"AspectPattern({markers_str}: {validity})"


@dataclass
class AspectError(GrammarError):
    """
    Represents an aspect marker error.

    Extends GrammarError to integrate with the spell checker's error hierarchy.

    Attributes:
        text: The erroneous word (inherited from Error).
        position: Index of the error in the word list (inherited from Error).
        suggestions: List of suggested corrections (inherited from Error).
        error_type: Type of error (aspect_typo, invalid_sequence, incomplete_aspect).
        confidence: Confidence score (0.0-1.0) (inherited from Error).
        reason: Human-readable explanation (inherited from GrammarError).
        word: Alias for 'text' (inherited from GrammarError).
        suggestion: First suggestion (inherited from GrammarError).
    """

    # Override default error_type
    error_type: str = field(default=ET_ASPECT_ERROR)

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"AspectError(pos={self.position}, {self.error_type}: {self.text} → {self.suggestion})"
        )


class AspectChecker:
    """
    Validates aspect marker usage in Myanmar text.

    This checker identifies:
    - Typos in aspect markers
    - Invalid aspect marker sequences
    - Incomplete aspect constructions
    - Valid aspect combinations
    """

    def __init__(
        self,
        config_path: str | None = None,
        aspect_config: AspectCheckerConfig | None = None,
    ):
        """
        Initialize the aspect checker with marker data.

        Args:
            config_path: Path to grammar/aspects config.
            aspect_config: AspectCheckerConfig for confidence settings.
        """
        self.config = get_grammar_config(config_path)
        self.aspect_config = aspect_config or _default_aspect_config

        self.markers: dict[str, dict[str, Any]] = {}
        self.marker_set: set[str] = set()
        self.typo_map: dict[str, str] = {}
        self.valid_combinations: dict[tuple[str, str], str] = {}
        self.invalid_sequences: dict[tuple[str, str], str] = {}
        self.final_markers: set[str] = set()
        self.non_final_markers: set[str] = set()

        self._load_from_config()

    def _load_from_config(self) -> None:
        """Load aspect data from configuration."""
        asp_config = self.config.aspects_config
        if not asp_config:
            from myspellchecker.utils.logging_utils import get_logger

            get_logger(__name__).warning(
                "Aspect checker config is empty — checker will be inactive"
            )
            return

        # Load markers
        if "markers" in asp_config:
            for m in asp_config["markers"]:
                marker = m["marker"]
                self.markers[marker] = m
                self.marker_set.add(marker)

                if m.get("is_final"):
                    self.final_markers.add(marker)
                # If explicitly marked as not final, or if it's sequential like 'ပြီး'
                # Logic: if is_final is False, add to non_final
                if m.get("is_final") is False:
                    self.non_final_markers.add(marker)

        # Load combinations
        if "combinations" in asp_config:
            for item in asp_config["combinations"]:
                seq = tuple(item["sequence"])
                if len(seq) == 2:
                    self.valid_combinations[seq] = item.get("description", "")

        # Load invalid sequences
        if "invalid_sequences" in asp_config:
            for item in asp_config["invalid_sequences"]:
                seq = tuple(item["sequence"])
                if len(seq) == 2:
                    self.invalid_sequences[seq] = item.get("reason", "")

        # Load typos
        if "typos" in asp_config:
            for item in asp_config["typos"]:
                self.typo_map[item["incorrect"]] = item["correct"]

    def is_aspect_marker(self, word: str) -> bool:
        """
        Check if a word is an aspect marker.

        Args:
            word: The word to check.

        Returns:
            True if the word is a valid aspect marker.
        """
        return word in self.marker_set

    def is_aspect_typo(self, word: str) -> bool:
        """
        Check if a word is a common aspect marker typo.

        Args:
            word: The word to check.

        Returns:
            True if the word is a known aspect typo.
        """
        return word in self.typo_map

    def get_typo_correction(self, word: str) -> str | None:
        """
        Get the correct form for an aspect marker typo.

        Args:
            word: The misspelled word.

        Returns:
            The correct spelling, or None if not a known typo.
        """
        return self.typo_map.get(word)

    def get_aspect_info(self, word: str) -> AspectInfo | None:
        """
        Get detailed information about an aspect marker.

        Args:
            word: The aspect marker to look up.

        Returns:
            AspectInfo if the word is an aspect marker, None otherwise.
        """
        if word not in self.markers:
            return None

        m_data = self.markers[word]

        return AspectInfo(
            marker=word,
            category=m_data.get("category", "unknown"),
            description=m_data.get("description", ""),
            can_combine=m_data.get("can_combine", True),
            register=m_data.get("register", "neutral"),
            is_final=m_data.get("is_final", False),
        )

    def get_invalid_reason(self, first: str, second: str) -> str | None:
        """
        Get reason why a combination is invalid.

        Args:
            first: The first aspect marker.
            second: The second aspect marker.

        Returns:
            Reason for invalidity, or None if not a known invalid sequence.
        """
        return self.invalid_sequences.get((first, second))

    def validate_sequence(self, words: list[str]) -> list[AspectError]:
        """
        Validate aspect marker usage in a word sequence.

        This method checks for:
        - Aspect marker typos
        - Invalid aspect marker sequences
        - Incomplete aspect constructions

        Args:
            words: List of words to validate.

        Returns:
            List of AspectError objects for any issues found.
        """
        errors: list[AspectError] = []
        n = len(words)
        i = 0

        while i < n:
            word = words[i]

            # Check for aspect marker typos
            if self.is_aspect_typo(word):
                correction = self.get_typo_correction(word)
                if correction:
                    info = self.get_aspect_info(correction)
                    category_desc = info.category if info else "aspect marker"
                    errors.append(
                        AspectError(
                            text=word,
                            position=i,
                            suggestions=[correction],
                            error_type=ET_ASPECT_TYPO,
                            confidence=self.aspect_config.high_confidence,
                            reason=(
                                f"'{word}' appears to be a typo for "
                                f"{category_desc} marker '{correction}'"
                            ),
                        )
                    )

            # Check for invalid sequences
            if self.is_aspect_marker(word) or self.is_aspect_typo(word):
                current = self.get_typo_correction(word) if self.is_aspect_typo(word) else word

                # Look at next word
                if i + 1 < n:
                    next_word = words[i + 1]
                    if self.is_aspect_marker(next_word) or self.is_aspect_typo(next_word):
                        next_marker = (
                            self.get_typo_correction(next_word)
                            if self.is_aspect_typo(next_word)
                            else next_word
                        )

                        if current and next_marker:
                            invalid_reason = self.get_invalid_reason(current, next_marker)
                            if invalid_reason:
                                errors.append(
                                    AspectError(
                                        text=f"{word} {next_word}",
                                        position=i,
                                        suggestions=["Review aspect marker sequence"],
                                        error_type=ET_INVALID_SEQUENCE,
                                        confidence=self.aspect_config.medium_confidence,
                                        reason=f"Invalid sequence: {invalid_reason}",
                                    )
                                )

                # Check for non-final markers at end of sentence
                current_info = self.get_aspect_info(current) if current else None
                if current_info and current in self.non_final_markers:
                    # Check if this is at end of phrase (no more meaningful words)
                    remaining = words[i + 1 :] if i + 1 < n else []
                    remaining_meaningful = [w for w in remaining if w not in {"။", "၊", ""}]
                    if not remaining_meaningful:
                        errors.append(
                            AspectError(
                                text=word,
                                position=i,
                                suggestions=[f"Add following verb or ending after '{word}'"],
                                error_type=ET_INCOMPLETE_ASPECT,
                                confidence=self.aspect_config.low_confidence,
                                reason=(
                                    f"Aspect marker '{word}' typically requires a following element"
                                ),
                            )
                        )

            i += 1

        return errors
