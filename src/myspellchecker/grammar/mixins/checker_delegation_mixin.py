"""Checker delegation mixin for SyntacticRuleChecker.

Provides wrapper methods that delegate to specialized grammar checker
instances (classifier, negation, register, aspect, compound, merged
word).

Extracted from ``engine.py`` to reduce file size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myspellchecker.core.config import GrammarEngineConfig
    from myspellchecker.grammar.checkers.aspect import AspectChecker
    from myspellchecker.grammar.checkers.classifier import ClassifierChecker
    from myspellchecker.grammar.checkers.compound import CompoundChecker
    from myspellchecker.grammar.checkers.merged_word import MergedWordChecker
    from myspellchecker.grammar.checkers.negation import NegationChecker
    from myspellchecker.grammar.checkers.register import RegisterChecker
    from myspellchecker.grammar.config import GrammarRuleConfig


# Pure-particle POS tags -- words tagged exclusively as these (with no
# N/V/ADJ tag) are genuine particles and should not be treated as
# classifier typos.  Words like (PART|N) keep the N tag that
# marks them as classifiers, so they pass through.
_PURE_PARTICLE_POS_TAGS = frozenset({"PPM", "PART", "P", "SFP", "CONJ"})
_CONTENT_POS_TAGS = frozenset({"N", "V", "ADJ", "ADV", "TN"})


class CheckerDelegationMixin:
    """Mixin providing delegating wrapper methods to specialized checkers.

    Each method collects errors from its corresponding checker instance
    and returns them in the standard ``(index, word, suggestion, confidence)``
    tuple format.
    """

    # --- Type stubs for attributes provided by SyntacticRuleChecker ---
    config: "GrammarRuleConfig"
    grammar_config: "GrammarEngineConfig"
    aspect_checker: "AspectChecker"
    classifier_checker: "ClassifierChecker"
    compound_checker: "CompoundChecker"
    merged_word_checker: "MergedWordChecker"
    negation_checker: "NegationChecker"
    register_checker: "RegisterChecker"

    def _check_classifiers(
        self, words: list[str], pos_tags: list[str | None] | None = None
    ) -> list[tuple[int, str, str, float]]:
        """
        Check for classifier-related errors in the word sequence.

        Validates:
        1. Classifier typos (e.g., ယေက် -> ယောက်)
        2. Classifier patterns after numerals
        3. Classifier-noun agreement (when detectable)

        Args:
            words: List of words in the sentence.
            pos_tags: Optional POS tags for filtering known particles.

        Returns:
            List of (index, error_word, suggestion, confidence) tuples.
        """
        errors: list[tuple[int, str, str, float]] = []

        # Use the classifier checker to validate
        classifier_errors = self.classifier_checker.validate_sequence(words)

        for error in classifier_errors:
            # Skip classifier typo suggestions for words that are pure
            # particles (e.g., PPM should not be corrected to classifier).
            # Words with content tags (N, V, etc.) alongside particle tags
            # are kept -- classifiers like (PART|N) need checking.
            if pos_tags and error.position < len(pos_tags):
                pos = pos_tags[error.position]
                if pos:
                    word_tags = set(pos.split("|"))
                    is_particle = bool(word_tags & _PURE_PARTICLE_POS_TAGS)
                    has_content = bool(word_tags & _CONTENT_POS_TAGS)
                    if is_particle and not has_content:
                        continue
            errors.append((error.position, error.word, error.suggestion, error.confidence))

        return errors

    def _check_negation(self, words: list[str]) -> list[tuple[int, str, str, float]]:
        """
        Check for negation-related errors in the word sequence.

        Validates:
        1. Negative ending typos (e.g., ဘူ -> ဘူး)
        2. Incomplete negation patterns
        3. Register-consistent negation

        Args:
            words: List of words in the sentence.

        Returns:
            List of (index, error_word, suggestion, confidence) tuples.
        """
        errors: list[tuple[int, str, str, float]] = []

        # Use the negation checker to validate
        negation_errors = self.negation_checker.validate_sequence(words)

        for error in negation_errors:
            errors.append((error.position, error.word, error.suggestion, error.confidence))

        return errors

    def _check_register(self, words: list[str]) -> list[tuple[int, str, str, float]]:
        """
        Check for register consistency issues in the word sequence.

        Validates:
        1. Mixed formal/colloquial register usage
        2. Suggests register-consistent alternatives

        Args:
            words: List of words in the sentence.

        Returns:
            List of (index, error_word, suggestion, confidence) tuples.
        """
        errors: list[tuple[int, str, str, float]] = []

        # Use the register checker to validate
        register_errors = self.register_checker.validate_sequence(words)

        for error in register_errors:
            errors.append((error.position, error.word, error.suggestion, error.confidence))

        return errors

    def _check_aspect(
        self, words: list[str], pos_tags: list[str | None] | None = None
    ) -> list[tuple[int, str, str, float]]:
        """
        Check for aspect marker errors in the word sequence.

        Validates:
        1. Aspect marker typos (e.g., ပြိ -> ပြီ)
        2. Invalid aspect marker sequences
        3. Incomplete aspect constructions

        Args:
            words: List of words in the sentence.
            pos_tags: Optional POS tags for context filtering.

        Returns:
            List of (index, error_word, suggestion, confidence) tuples.
        """
        errors: list[tuple[int, str, str, float]] = []

        # Use the aspect checker to validate
        aspect_errors = self.aspect_checker.validate_sequence(words)

        for error in aspect_errors:
            # Context guard for aspect typos: aspect markers follow verbs.
            # Words like ခဲ (solidify/hard, freq=16K) are valid content words
            # that only become aspect typos in post-verbal position.
            # Skip if previous word is primarily a noun (N first in multi-POS
            # like N|V for "blood") -- the word is functioning as a noun,
            # and the "aspect typo" is really a content verb/adjective.
            if (
                error.error_type == "aspect_typo"
                and pos_tags
                and error.position > 0
                and error.position < len(pos_tags)
            ):
                prev_pos = pos_tags[error.position - 1]
                if prev_pos:
                    primary_tag = prev_pos.split("|")[0]
                    if primary_tag == "N":
                        continue

            errors.append((error.position, error.word, error.suggestion, error.confidence))

        return errors

    def _check_compound(self, words: list[str]) -> list[tuple[int, str, str, float]]:
        """
        Check for compound word errors in the word sequence.

        Validates:
        1. Compound typos (e.g., ပန်ခြံ -> ပန်းခြံ)
        2. Incomplete reduplications
        3. Invalid compound formations

        Args:
            words: List of words in the sentence.

        Returns:
            List of (index, error_word, suggestion, confidence) tuples.
        """
        errors: list[tuple[int, str, str, float]] = []

        # Use the compound checker to validate
        compound_errors = self.compound_checker.validate_sequence(words)

        for error in compound_errors:
            errors.append((error.position, error.word, error.suggestion, error.confidence))

        return errors

    def _check_merged_words(
        self, words: list[str], pos_tags: list[str | None]
    ) -> list[tuple[int, str, str, float]]:
        """
        Check for words that may have been incorrectly merged by the segmenter.

        Detects cases where a valid dictionary word (e.g., "play")
        is actually a mis-merge of a particle + verb (e.g., subject marker + eat)
        based on surrounding POS context.

        Requires three-way evidence:
        1. The word is in the known ambiguous-merge set
        2. The preceding word has a noun-like POS tag
        3. The following word is a clause continuation marker

        Args:
            words: List of words in the sentence.
            pos_tags: List of POS tags corresponding to words.

        Returns:
            List of (index, error_word, suggestion, confidence) tuples.
        """
        errors: list[tuple[int, str, str, float]] = []

        # Use the merged word checker with POS tags
        merged_errors = self.merged_word_checker.validate_sequence(words, pos_tags)

        for error in merged_errors:
            errors.append((error.position, error.word, error.suggestion, error.confidence))

        return errors
