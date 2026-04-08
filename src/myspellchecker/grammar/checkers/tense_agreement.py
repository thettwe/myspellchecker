"""
Myanmar Tense-Time Agreement Validator.

This module validates that aspectual particles (sentence-final markers)
agree with temporal adverbials in Myanmar sentences.

When a temporal adverb indicates a specific tense (past or future),
the sentence-final aspectual particle must match that tense. For example:

    Correct:   မနေ့က သွားခဲ့တယ်   (yesterday + past marker)
    Incorrect: မနေ့က သွားမယ်       (yesterday + future marker)

    Correct:   မနက်ဖြန် သွားမယ်    (tomorrow + future marker)
    Incorrect: မနက်ဖြန် သွားခဲ့တယ်  (tomorrow + past marker)

Features:
    - Detect temporal adverbs and classify their tense
    - Detect aspectual particles and classify their tense
    - Flag tense mismatches with confidence scores
    - Suggest the correct aspect marker for the detected time context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from myspellchecker.core.config.grammar_configs import TenseAgreementCheckerConfig
from myspellchecker.core.constants import ET_TENSE_MISMATCH
from myspellchecker.core.response import GrammarError
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "TenseAgreementChecker",
    "TenseAgreementError",
]

DEFAULT_TENSE_MARKERS_PATH = Path(__file__).parent.parent.parent / "rules" / "tense_markers.yaml"

# Default configuration (module-level singleton)
_default_checker_config = TenseAgreementCheckerConfig()


@dataclass
class TenseAgreementError(GrammarError):
    """
    Represents a tense-time agreement error.

    Raised when a temporal adverb and an aspectual particle in the same
    sentence indicate incompatible tenses.

    Attributes:
        text: The incorrect aspect marker (inherited from Error).
        position: Index of the incorrect marker in the word list (inherited).
        suggestions: Correct aspect marker(s) for the detected time context.
        error_type: Always ET_TENSE_MISMATCH.
        confidence: Confidence score from YAML rules (inherited).
        reason: Human-readable explanation (inherited from GrammarError).
        time_adverb: The temporal adverb that set the expected tense.
        detected_tense: The tense class signalled by the adverb ("past"/"future").
    """

    error_type: str = field(default=ET_TENSE_MISMATCH)
    time_adverb: str = ""
    detected_tense: str = ""

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"TenseAgreementError(pos={self.position}, "
            f"{self.detected_tense}: {self.time_adverb} vs {self.text} "
            f"-> {self.suggestion})"
        )


class TenseAgreementChecker:
    """
    Validates tense-time agreement in Myanmar sentences.

    This checker scans a word sequence for temporal adverbs and
    aspectual particles, then flags incompatible combinations where
    the adverb signals one tense and the particle signals another.
    """

    def __init__(
        self,
        config_path: str | None = None,
        checker_config: TenseAgreementCheckerConfig | None = None,
    ) -> None:
        """
        Initialize the tense agreement checker.

        Args:
            config_path: Path to tense_markers.yaml. If None, uses default.
            checker_config: Confidence settings. If None, uses defaults.
        """
        self.checker_config = checker_config or _default_checker_config

        # Lookup: time_adverb -> tense_class ("past" or "future")
        self.adverb_to_tense: dict[str, str] = {}

        # Lookup: aspect_marker -> tense_class ("past" or "future")
        self.marker_to_tense: dict[str, str] = {}

        # Incompatibility rules: tense_class -> {incompatible markers}
        self.incompatible: dict[str, set[str]] = {}

        # Incompatibility confidence: tense_class -> confidence
        self.incompatible_confidence: dict[str, float] = {}

        # Incompatibility descriptions: tense_class -> description
        self.incompatible_descriptions: dict[str, str] = {}

        # SFP correction maps for generating suggestions
        self.sfp_past_to_future: dict[str, str] = {}
        self.sfp_future_to_past: dict[str, str] = {}

        self._load_config(config_path)

    def _load_config(self, config_path: str | None) -> None:
        """Load tense agreement rules from YAML configuration."""
        yaml_path = Path(config_path) if config_path else DEFAULT_TENSE_MARKERS_PATH

        if not yaml_path.exists():
            logger.warning(
                "Tense markers config not found at %s — checker will be inactive",
                yaml_path,
            )
            return

        try:
            with open(yaml_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            logger.warning("Failed to load tense markers config from %s: %s", yaml_path, e)
            return

        if not config:
            return

        self._parse_agreement_rules(config)
        self._parse_sfp_corrections(config)

    def _parse_agreement_rules(self, config: dict[str, Any]) -> None:
        """Parse the tense_agreement_rules section from loaded YAML."""
        rules = config.get("tense_agreement_rules")
        if not rules:
            logger.debug("No tense_agreement_rules section — checker will be inactive")
            return

        # Build adverb -> tense lookups
        for adverb in rules.get("past_time_adverbs", []):
            self.adverb_to_tense[adverb] = "past"

        for adverb in rules.get("future_time_adverbs", []):
            self.adverb_to_tense[adverb] = "future"

        # Build marker -> tense lookups
        for marker in rules.get("past_aspect_markers", []):
            self.marker_to_tense[marker] = "past"

        for marker in rules.get("future_aspect_markers", []):
            self.marker_to_tense[marker] = "future"

        # Build incompatibility map
        for pair in rules.get("incompatible_pairs", []):
            time_class = pair.get("time_class", "")
            aspects = pair.get("incompatible_aspects", [])
            confidence = pair.get("confidence", self.checker_config.default_confidence)
            description = pair.get("description", "")

            self.incompatible[time_class] = set(aspects)
            self.incompatible_confidence[time_class] = confidence
            self.incompatible_descriptions[time_class] = description

    def _parse_sfp_corrections(self, config: dict[str, Any]) -> None:
        """Parse sfp_corrections for suggestion generation."""
        sfp = config.get("sfp_corrections", {})
        self.sfp_past_to_future = sfp.get("past_present_to_future", {})
        self.sfp_future_to_past = sfp.get("future_to_past_present", {})

    def is_time_adverb(self, word: str) -> bool:
        """
        Check if a word is a temporal adverb.

        Also supports prefix matching — if the word starts with a known
        adverb, it is treated as belonging to that tense class.

        Args:
            word: The word to check.

        Returns:
            True if the word is a temporal adverb.
        """
        if word in self.adverb_to_tense:
            return True
        # Prefix match: e.g. "လာမယ့်နှစ်တွင်" contains prefix "လာမယ့်နှစ်"
        return any(word.startswith(adverb) for adverb in self.adverb_to_tense)

    def get_adverb_tense(self, word: str) -> str | None:
        """
        Get the tense class for a temporal adverb.

        Args:
            word: The word to look up.

        Returns:
            Tense class ("past" or "future"), or None if not a time adverb.
        """
        if word in self.adverb_to_tense:
            return self.adverb_to_tense[word]
        # Prefix match: longest match wins
        best_match: str | None = None
        best_len = 0
        for adverb, tense in self.adverb_to_tense.items():
            if word.startswith(adverb) and len(adverb) > best_len:
                best_match = tense
                best_len = len(adverb)
        return best_match

    def is_aspect_marker(self, word: str) -> bool:
        """
        Check if a word is a known aspectual particle.

        Args:
            word: The word to check.

        Returns:
            True if the word is a known aspect marker.
        """
        return word in self.marker_to_tense

    def get_marker_tense(self, word: str) -> str | None:
        """
        Get the tense class for an aspect marker.

        Args:
            word: The marker to look up.

        Returns:
            Tense class ("past" or "future"), or None if not a known marker.
        """
        return self.marker_to_tense.get(word)

    def _get_suggestions(self, marker: str, expected_tense: str) -> list[str]:
        """
        Suggest correct aspect markers given the expected tense.

        Uses the sfp_corrections mappings to find the correct marker
        for the detected time context.

        Args:
            marker: The incorrect aspect marker found.
            expected_tense: The tense expected by the temporal adverb.

        Returns:
            List of suggested correct markers.
        """
        if expected_tense == "past":
            # Marker is future but should be past -> use future_to_past map
            correction = self.sfp_future_to_past.get(marker)
            if correction:
                return [correction]
        elif expected_tense == "future":
            # Marker is past but should be future -> use past_to_future map
            correction = self.sfp_past_to_future.get(marker)
            if correction:
                return [correction]

        return []

    def validate_sequence(self, words: list[str]) -> list[TenseAgreementError]:
        """
        Validate tense-time agreement in a word sequence.

        Scans the word list for temporal adverbs and aspect markers,
        then checks whether they agree in tense. If a past adverb
        co-occurs with a future aspect marker (or vice versa), a
        TenseAgreementError is produced.

        Args:
            words: List of segmented words to validate.

        Returns:
            List of TenseAgreementError objects for any mismatches found.
        """
        if not self.adverb_to_tense or not self.incompatible:
            return []

        errors: list[TenseAgreementError] = []

        # Phase 1: Scan for temporal adverbs to determine expected tense
        detected_adverbs: list[tuple[int, str, str]] = []  # (index, word, tense)
        for i, word in enumerate(words):
            tense = self.get_adverb_tense(word)
            if tense is not None:
                detected_adverbs.append((i, word, tense))

        if not detected_adverbs:
            return []

        # Phase 2: Scan for aspect markers and check compatibility
        for i, word in enumerate(words):
            if not self.is_aspect_marker(word):
                continue

            marker_tense = self.get_marker_tense(word)
            if marker_tense is None:
                continue

            # Check against each detected adverb
            for _adv_idx, adv_word, adv_tense in detected_adverbs:
                # Only flag if there is a tense conflict
                if adv_tense == marker_tense:
                    continue

                incompatible_set = self.incompatible.get(adv_tense, set())
                if word not in incompatible_set:
                    continue

                confidence = self.incompatible_confidence.get(
                    adv_tense, self.checker_config.default_confidence
                )
                description = self.incompatible_descriptions.get(adv_tense, "")
                suggestions = self._get_suggestions(word, adv_tense)

                errors.append(
                    TenseAgreementError(
                        text=word,
                        position=i,
                        suggestions=suggestions,
                        confidence=confidence,
                        reason=(
                            f"Tense mismatch: temporal adverb '{adv_word}' "
                            f"({adv_tense}) conflicts with aspect marker "
                            f"'{word}' — {description}"
                        ),
                        time_adverb=adv_word,
                        detected_tense=adv_tense,
                    )
                )
                # One error per marker is enough (use first conflicting adverb)
                break

        return errors
