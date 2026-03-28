"""
Myanmar Classifier System Validator.

This module implements validation for Myanmar numeral + classifier patterns.
Myanmar uses numeral classifiers similar to Chinese/Japanese/Korean.

Pattern: Numeral + Classifier + Noun
Examples:
    - သုံး ယောက် (3 persons) - ယောက် = classifier for people
    - ငါး ကောင် (5 animals) - ကောင် = classifier for animals
    - နှစ် အုပ် (2 books) - အုပ် = classifier for books

Features:
    - Classifier recognition and validation
    - Numeral pattern detection (digits and words)
    - Classifier typo correction
    - Category-based noun agreement validation
"""

from __future__ import annotations

from dataclasses import dataclass, field

from myspellchecker.core.config import ClassifierCheckerConfig
from myspellchecker.core.constants import (
    ET_AGREEMENT,
    ET_CLASSIFIER_ERROR,
    ET_TYPO,
    MYANMAR_NUMERAL_WORDS,
    MYANMAR_NUMERALS,
)
from myspellchecker.core.response import GrammarError
from myspellchecker.grammar.config import get_grammar_config
from myspellchecker.utils.singleton import Singleton

# Default Classifier Checker configuration (module-level singleton)
_default_classifier_config = ClassifierCheckerConfig()

__all__ = [
    "ClassifierChecker",
    "ClassifierError",
    "get_classifier_checker",
    "is_classifier",
    "is_numeral",
]

# Singleton registry for ClassifierChecker
_singleton: Singleton["ClassifierChecker"] = Singleton()


@dataclass
class ClassifierError(GrammarError):
    """
    Represents a classifier-related error.

    Extends GrammarError to integrate with the spell checker's error hierarchy.

    Attributes:
        text: The erroneous word (inherited from Error).
        position: Index of the error in the word list (inherited from Error).
        suggestions: List of suggested corrections (inherited from Error).
        error_type: Type of error (typo, agreement, missing, invalid_pattern).
        confidence: Confidence score (0.0-1.0) (inherited from Error).
        reason: Human-readable explanation (inherited from GrammarError).
        word: Alias for 'text' (inherited from GrammarError).
        suggestion: First suggestion (inherited from GrammarError).
    """

    # Override default error_type
    error_type: str = field(default=ET_CLASSIFIER_ERROR)


class ClassifierChecker:
    """
    Validates Myanmar numeral + classifier patterns.

    This checker identifies:
    - Classifier typos (e.g., ယေက် → ယောက်)
    - Missing classifiers after numerals
    - Incorrect classifier-noun agreement
    - Invalid classifier usage patterns

    Attributes:
        classifiers: Set of valid classifier words.
        numeral_words: Set of numeral words (written form).
        classifier_map: Dictionary mapping classifiers to their categories.
    """

    # Context-specific typo overrides near numerals.
    # e.g., ငါး ကိုင် ရှိတယ် -> ငါး ကောင် ရှိတယ်
    _CLASSIFIER_TYPO_OVERRIDES: dict[str, tuple[str, float]] = {
        "ကိုင်": ("ကောင်", 0.92),
    }

    # Canonicalized classifier forms for suggestion output.
    # Keep recognition broad, but prefer standard spelling in corrections.
    _CLASSIFIER_SUGGESTION_CANONICAL: dict[str, str] = {
        "ခြောင်း": "ချောင်း",
    }

    def __init__(
        self,
        config_path: str | None = None,
        classifier_config: ClassifierCheckerConfig | None = None,
    ) -> None:
        """
        Initialize the classifier checker.

        Args:
            config_path: Optional path to grammar config.
            classifier_config: ClassifierCheckerConfig for confidence settings.
        """
        self.config = get_grammar_config(config_path)
        self.classifier_config = classifier_config or _default_classifier_config

        self.numeral_words = set(MYANMAR_NUMERAL_WORDS.keys())
        self.numerals = MYANMAR_NUMERALS

        # Load classifiers from config
        self.classifiers: set[str] = set()
        self.classifier_map: dict[str, tuple[str, str, list[str]]] = {}
        self._load_classifiers()

        # Precompute single-char-deletion variants for O(1) typo lookup.
        # Maps each deletion variant -> the original classifier.
        self._deletion_to_classifier: dict[str, str] = {}
        for clf in self.classifiers:
            for j in range(len(clf)):
                variant = clf[:j] + clf[j + 1 :]
                # First match wins (consistent with previous linear scan)
                if variant not in self._deletion_to_classifier:
                    self._deletion_to_classifier[variant] = clf

    def _load_classifiers(self) -> None:
        """Load classifier definitions from configuration.

        Note: Some classifiers (e.g., လုံး) appear in multiple categories.
        We preserve the first occurrence as the primary category to maintain
        consistency with expected behavior and linguistic convention.
        """
        for category, items in self.config.classifiers.items():
            for item in items:
                word = item.get("word")
                if word:
                    self.classifiers.add(word)
                    # Only add to map if not already present (preserve first occurrence)
                    # This handles classifiers that appear in multiple categories
                    if word not in self.classifier_map:
                        # format: classifier -> (category, description, examples)
                        self.classifier_map[word] = (
                            category,
                            item.get("description", ""),
                            item.get("examples", []),
                        )

    def _canonicalize_classifier_suggestion(self, classifier: str) -> str:
        """Normalize classifier suggestion to preferred canonical surface form."""
        return self._CLASSIFIER_SUGGESTION_CANONICAL.get(classifier, classifier)

    def is_numeral(self, word: str) -> bool:
        """
        Check if a word is a numeral (digit or word form).

        Args:
            word: Word to check.

        Returns:
            True if the word is a numeral.

        Examples:
            >>> checker = ClassifierChecker()
            >>> checker.is_numeral("သုံး")
            True
            >>> checker.is_numeral("၃")
            True
            >>> checker.is_numeral("စာအုပ်")
            False
        """
        # Check if it's a written numeral word
        if word in self.numeral_words:
            return True

        # Check if it's a digit string
        return all(c in self.numerals for c in word) and len(word) > 0

    def is_classifier(self, word: str) -> bool:
        """
        Check if a word is a valid classifier.

        Args:
            word: Word to check.

        Returns:
            True if the word is a classifier.

        Examples:
            >>> checker = ClassifierChecker()
            >>> checker.is_classifier("ယောက်")
            True
            >>> checker.is_classifier("လူ")
            False
        """
        return word in self.classifiers

    def get_classifier_category(self, classifier: str) -> str | None:
        """
        Get the category of a classifier.

        Args:
            classifier: Classifier word.

        Returns:
            Category string or None if not found.

        Examples:
            >>> checker = ClassifierChecker()
            >>> checker.get_classifier_category("ယောက်")
            'people'
            >>> checker.get_classifier_category("ကောင်")
            'animals'
        """
        if classifier in self.classifier_map:
            return self.classifier_map[classifier][0]
        return None

    def check_classifier_typo(self, word: str) -> tuple[str, float] | None:
        """
        Check if a word is a typo of a classifier.

        Args:
            word: Word to check.

        Returns:
            Tuple of (correction, confidence) or None if not a typo.

        Examples:
            >>> checker = ClassifierChecker()
            >>> checker.check_classifier_typo("ယေက်")
            ('ယောက်', 0.90)
        """
        override = self._CLASSIFIER_TYPO_OVERRIDES.get(word)
        if override is not None:
            return override

        # Check config for corrections
        correction_info = self.config.get_word_correction(word)
        if correction_info:
            correction = correction_info["correction"]
            # Ensure the correction is actually a classifier
            if correction in self.classifiers:
                correction = self._canonicalize_classifier_suggestion(correction)
                confidence = correction_info.get("confidence", 0.90)
                return (correction, confidence)

        # Check for common patterns
        # Missing visarga (း)
        if word + "း" in self.classifiers:
            return (self._canonicalize_classifier_suggestion(word + "း"), 0.85)

        # Missing anusvara (ံ) or other single-character insertion
        # Guard: skip very short words (< 2 Myanmar chars) -- single consonants
        # like ဆ (fold/times, freq=45K) are common content words that spuriously
        # match classifiers by edit-distance-1 (e.g., ဆ→ဆူ "pagoda classifier")
        if len(word) >= 2:
            match = self._deletion_to_classifier.get(word)
            if match is not None:
                return (self._canonicalize_classifier_suggestion(match), 0.80)

        return None

    def validate_sequence(
        self,
        words: list[str],
    ) -> list[ClassifierError]:
        """
        Validate classifier usage in a word sequence.

        Checks for:
        1. Classifier typos
        2. Missing classifiers after numerals
        3. Invalid classifier patterns
        4. Classifier-noun agreement (when context available)

        Args:
            words: List of words to validate.

        Returns:
            List of ClassifierError objects.

        Examples:
            >>> checker = ClassifierChecker()
            >>> errors = checker.validate_sequence(["သုံး", "ယေက်"])
            >>> len(errors)
            1
            >>> errors[0].suggestion
            'ယောက်'
        """
        errors: list[ClassifierError] = []
        flagged_positions: set[int] = set()

        for i, word in enumerate(words):
            # Check 1: Classifier typos — only when adjacent to a numeral
            # to avoid false positives on common particles (e.g., ပါ → ပါး)
            near_numeral = (i > 0 and self.is_numeral(words[i - 1])) or (
                i + 1 < len(words) and self.is_numeral(words[i + 1])
            )
            if not self.is_classifier(word) and near_numeral:
                typo_result = self.check_classifier_typo(word)
                if typo_result:
                    correction, confidence = typo_result
                    # Preceded by numeral gets higher confidence
                    if i > 0 and self.is_numeral(words[i - 1]):
                        confidence = min(
                            confidence + self.classifier_config.context_boost,
                            self.classifier_config.max_confidence,
                        )

                    errors.append(
                        ClassifierError(
                            text=word,
                            position=i,
                            suggestions=[correction],
                            error_type=ET_TYPO,
                            confidence=confidence,
                            reason=f"Classifier typo: {word} → {correction}",
                        )
                    )
                    flagged_positions.add(i)
                    continue

            # Check 2: Numeral without classifier
            if self.is_numeral(word):
                # Check if next word is a classifier or noun (not verb/particle)
                if i + 1 < len(words) and i + 1 not in flagged_positions:
                    next_word = words[i + 1]
                    if not self.is_classifier(next_word):
                        # Check if it might be a typo
                        typo_result = self.check_classifier_typo(next_word)
                        if typo_result:
                            correction, confidence = typo_result
                            errors.append(
                                ClassifierError(
                                    text=next_word,
                                    position=i + 1,
                                    suggestions=[correction],
                                    error_type=ET_TYPO,
                                    confidence=confidence,
                                    reason=(
                                        f"After numeral '{word}', expected: "
                                        f"{next_word} → {correction}"
                                    ),
                                )
                            )
                            flagged_positions.add(i + 1)
                        # Note: We don't flag missing classifiers as errors since
                        # sometimes numerals can be used without classifiers in
                        # certain contexts (e.g., dates, prices, math)

            # Check 3: Classifier-noun semantic agreement
            # Myanmar word order: Noun + Numeral + Classifier (e.g., လူ သုံး ယောက်)
            if self.is_numeral(word) and i + 1 < len(words):
                classifier = words[i + 1]
                if self.is_classifier(classifier):
                    # Check noun before numeral (Myanmar order: N-NUM-CLF)
                    if i > 0:
                        potential_noun = words[i - 1]
                        agreement_error = self.check_agreement(
                            classifier=classifier,
                            noun=potential_noun,
                        )
                        if agreement_error:
                            # Full-span: position at numeral, text = numeral+classifier
                            agreement_error.position = i
                            agreement_error.text = word + classifier
                            correct_clf = (
                                agreement_error.suggestions[0]
                                if agreement_error.suggestions
                                else classifier
                            )
                            agreement_error.suggestions = [word + correct_clf]
                            errors.append(agreement_error)

        return errors

    def get_compatible_classifiers(self, noun: str) -> list[str]:
        """
        Get classifiers compatible with a given noun.

        This is based on the example nouns defined in CLASSIFIER_MAP.

        Args:
            noun: Noun word.

        Returns:
            List of compatible classifier words.

        Examples:
            >>> checker = ClassifierChecker()
            >>> classifiers = checker.get_compatible_classifiers("လူ")
            >>> "ယောက်" in classifiers
            True
        """
        compatible: list[str] = []
        seen: set[str] = set()

        for classifier, (_category, _description, example_nouns) in self.classifier_map.items():
            if noun in example_nouns:
                canonical = self._canonicalize_classifier_suggestion(classifier)
                if canonical in seen:
                    continue
                seen.add(canonical)
                compatible.append(canonical)

        return compatible

    def check_agreement(
        self,
        classifier: str,
        noun: str | None = None,
    ) -> ClassifierError | None:
        """
        Check if classifier agrees with the noun (if provided).

        Args:
            classifier: Classifier word.
            noun: Optional noun word.

        Returns:
            ClassifierError if agreement issue found, None otherwise.

        Examples:
            >>> checker = ClassifierChecker()
            >>> # Using people classifier for animals - agreement error
            >>> error = checker.check_agreement("ယောက်", "ခွေး")
            >>> error is not None  # ခွေး (dog) should use ကောင် not ယောက်
            True
        """
        if not noun:
            return None

        if not self.is_classifier(classifier):
            return None

        # Get compatible classifiers for the noun
        compatible = self.get_compatible_classifiers(noun)

        if compatible and classifier not in compatible:
            suggested = compatible[0]
            category = self.get_classifier_category(classifier)
            noun_category = self.get_classifier_category(suggested)

            return ClassifierError(
                text=classifier,
                position=-1,  # Will be updated by caller
                suggestions=[suggested],
                error_type=ET_AGREEMENT,
                confidence=self.classifier_config.inanimate_classifier_confidence,
                reason=(
                    f"Classifier '{classifier}' ({category}) may not agree "
                    f"with noun '{noun}'. Consider using '{suggested}' ({noun_category})."
                ),
            )

        return None


# Module-level singleton
def get_classifier_checker() -> ClassifierChecker:
    """
    Get the default ClassifierChecker singleton.

    Returns:
        ClassifierChecker instance.
    """
    return _singleton.get(ClassifierChecker)


def is_classifier(word: str) -> bool:
    """
    Convenience function to check if a word is a classifier.

    Args:
        word: Word to check.

    Returns:
        True if the word is a classifier.
    """
    return get_classifier_checker().is_classifier(word)


def is_numeral(word: str) -> bool:
    """
    Convenience function to check if a word is a numeral.

    Args:
        word: Word to check.

    Returns:
        True if the word is a numeral.
    """
    return get_classifier_checker().is_numeral(word)
