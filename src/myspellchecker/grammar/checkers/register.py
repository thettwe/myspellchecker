"""
Myanmar Register Detection and Validation.

This module implements register (formal/colloquial) detection for Myanmar text.
Myanmar has distinct written and spoken registers, and mixing them is a common
stylistic error in modern Myanmar writing.

Features:
    - Detect formal vs colloquial register of words
    - Identify mixed register usage in sentences
    - Suggest register-consistent alternatives
    - Calculate register consistency score

Examples:
    Formal (literary/written):   သူသည် စာအုပ် ဖတ်သည်။
    Colloquial (spoken):         သူ စာအုပ် ဖတ်တယ်။
    Mixed (error):               သူသည် စာအုပ် ဖတ်တယ်။ (formal subject + colloquial ending)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from myspellchecker.core.config import RegisterCheckerConfig
from myspellchecker.core.constants import ET_MIXED_REGISTER, ET_REGISTER_ERROR
from myspellchecker.core.response import GrammarError
from myspellchecker.grammar.config import get_grammar_config

__all__ = [
    "RegisterChecker",
    "RegisterError",
    "RegisterInfo",
]

# Default Register Checker configuration (module-level singleton)
_default_register_config = RegisterCheckerConfig()

# Constants for internal use
REGISTER_FORMAL = "formal"
REGISTER_COLLOQUIAL = "colloquial"
REGISTER_NEUTRAL = "neutral"

# Register pair types that are safe for suffix-based detection.
# These particles genuinely attach to verbs as endings (e.g., "သွားပါသည်").
# Standalone particles (conjunctions, case markers, conditionals, etc.) are
# excluded to avoid false positives on nouns — e.g., "ဆရာသည်" (teacher + topic)
# should NOT be flagged as formal just because it ends with "သည်".
# "possessive" (၏) is included because ၏ also functions as a formal statement
# ending (e.g., ပေး၏ = "gave"), not just a possessive marker.
_SUFFIX_ELIGIBLE_TYPES: frozenset[str] = frozenset(
    {
        "statement_ending",
        "polite_statement",
        "future",
        "polite_future",
        "question",
        "future_relative",
        "possessive",
    }
)


@dataclass
class RegisterInfo:
    """
    Register information for a word.

    Attributes:
        word: The word being analyzed.
        register: The register type (formal, colloquial, neutral).
        formal_form: The formal equivalent (if available).
        colloquial_form: The colloquial equivalent (if available).
    """

    word: str
    register: str
    formal_form: str | None
    colloquial_form: str | None

    def is_formal(self) -> bool:
        """Check if word is formal register."""
        return self.register == REGISTER_FORMAL

    def is_colloquial(self) -> bool:
        """Check if word is colloquial register."""
        return self.register == REGISTER_COLLOQUIAL


@dataclass
class RegisterError(GrammarError):
    """
    Represents a register-related error or inconsistency.

    Extends GrammarError to integrate with the spell checker's error hierarchy.

    Attributes:
        text: The erroneous word (inherited from Error).
        position: Index of the error in the word list (inherited from Error).
        suggestions: List of suggested corrections (inherited from Error).
        error_type: Type of error (mixed_register, wrong_register).
        confidence: Confidence score (0.0-1.0) (inherited from Error).
        reason: Human-readable explanation (inherited from GrammarError).
        detected_register: The register detected for this word.
        expected_register: The expected register based on context.
        word: Alias for 'text' (inherited from GrammarError).
        suggestion: First suggestion (inherited from GrammarError).
    """

    # Override default error_type
    error_type: str = field(default=ET_REGISTER_ERROR)

    # Register-specific fields
    detected_register: str = ""
    expected_register: str = ""


class RegisterChecker:
    """
    Validates register consistency in Myanmar text.

    This checker identifies:
    - Mixed register usage (formal + colloquial in same sentence)
    - Suggests register-consistent alternatives
    - Calculates overall register consistency score

    Attributes:
        formal_words: Set of formal-only words.
        colloquial_words: Set of colloquial-only words.
        neutral_words: Set of neutral words.
    """

    def __init__(
        self,
        config_path: str | None = None,
        register_config: RegisterCheckerConfig | None = None,
        provider: object | None = None,
    ) -> None:
        """
        Initialize the register checker.

        Args:
            config_path: Optional path to grammar/register config.
            register_config: RegisterCheckerConfig for confidence settings.
            provider: Optional DictionaryProvider for compound word lookup.
                When provided, suffix-based register detection will skip
                words that are valid dictionary entries (compound nouns like
                တပ်မတော်သည် should not trigger register detection on သည်).
        """
        self.config = get_grammar_config(config_path)
        self.register_cfg = register_config or _default_register_config
        self._provider = provider

        # Initialize sets and maps
        self.formal_words: set[str] = set()
        self.colloquial_words: set[str] = set()
        self.neutral_words: set[str] = set()
        self.register_pairs: dict[str, str] = {}  # formal -> colloquial
        self.colloquial_to_formal: dict[str, str] = {}  # colloquial -> formal
        # word -> (register, formal_form, colloquial_form)
        self.word_register_map: dict[str, tuple[str, str | None, str | None]] = {}
        # Words that participate in register detection/validation.
        # Non-ending particles (location, conjunction, etc.) are excluded.
        self._register_detection_words: set[str] = set()
        # Vocabulary pairs: colloquial/slang words with formal equivalents.
        # Used for vocabulary-level register detection in formal sentences.
        # Sorted longest-first for greedy matching of multi-word patterns.
        self._vocabulary_pairs: list[tuple[str, str]] = []  # (colloquial, formal_equivalent)
        self._vocab_exempt_phrases: frozenset[str] = frozenset()

        self._load_from_config()

    def _load_from_config(self) -> None:
        """Load register data from configuration."""
        reg_config = self.config.register_config

        # Load vocab lists
        if "formal_words" in reg_config:
            self.formal_words.update(reg_config["formal_words"])

        if "colloquial_words" in reg_config:
            self.colloquial_words.update(reg_config["colloquial_words"])

        if "neutral_words" in reg_config:
            self.neutral_words.update(reg_config["neutral_words"])

        # Load suffix-eligible types from YAML config, fall back to hardcoded.
        # Only sentence-ending types (statement, future, question) are eligible —
        # standalone particles (conjunctions, case markers) must not be used as
        # suffixes or they'll misclassify nouns (e.g., "ဆရာသည်" flagged as formal).
        suffix_eligible_types: frozenset[str]
        if "suffix_eligible_types" in reg_config:
            suffix_eligible_types = frozenset(reg_config["suffix_eligible_types"])
        else:
            suffix_eligible_types = _SUFFIX_ELIGIBLE_TYPES
        suffix_eligible_words: set[str] = set()

        # Load register pairs
        if "register_pairs" in reg_config:
            for pair in reg_config["register_pairs"]:
                formal = pair.get("formal")
                colloquial = pair.get("colloquial")

                if formal and colloquial:
                    self.register_pairs[formal] = colloquial
                    self.colloquial_to_formal[colloquial] = formal

                    self.formal_words.add(formal)
                    self.colloquial_words.add(colloquial)

                    # Track suffix eligibility by pair type
                    pair_type = pair.get("type", "")
                    if pair_type in suffix_eligible_types:
                        suffix_eligible_words.add(formal)
                        suffix_eligible_words.add(colloquial)

        # Build comprehensive word map
        for word in self.formal_words:
            colloquial_eq = self.register_pairs.get(word)
            self.word_register_map[word] = (REGISTER_FORMAL, word, colloquial_eq)

        for word in self.colloquial_words:
            formal_eq = self.colloquial_to_formal.get(word)
            self.word_register_map[word] = (REGISTER_COLLOQUIAL, formal_eq, word)

        for word in self.neutral_words:
            self.word_register_map[word] = (REGISTER_NEUTRAL, word, word)

        # Pre-sorted suffix list for suffix-based register detection.
        # Sorted longest-first so "ပါသည်" matches before "သည်".
        # Only includes words from sentence-ending register pair types to prevent
        # false positives on standalone particles used as case markers or conjunctions.
        self._register_suffixes: list[tuple[str, str, str | None, str | None]] = []
        for suffix, (register, formal_form, colloquial_form) in sorted(
            self.word_register_map.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if register != REGISTER_NEUTRAL and suffix in suffix_eligible_words:
                self._register_suffixes.append((suffix, register, formal_form, colloquial_form))

        # Build set of words significant for register detection/validation.
        # Only sentence endings (suffix-eligible pairs) and standalone vocab words
        # participate — non-ending particles (location, conjunction, plural, etc.)
        # are excluded since they're universally acceptable in both registers.
        # e.g., "မှာ" (location) mixes freely with formal endings like "သည်".
        self._register_detection_words = set(suffix_eligible_words)
        if "formal_words" in reg_config:
            self._register_detection_words.update(reg_config["formal_words"])
        if "colloquial_words" in reg_config:
            self._register_detection_words.update(reg_config["colloquial_words"])

        # Load vocabulary pairs (informal/slang words with formal equivalents).
        # These are used for vocabulary-level register mismatch detection:
        # when the sentence is detected as formal but contains informal vocabulary,
        # or when the sentence is colloquial but contains formal vocabulary.
        # Sorted longest-first for greedy substring matching.
        vocab_pairs: list[tuple[str, str]] = []
        if "vocabulary_pairs" in reg_config:
            for pair in reg_config["vocabulary_pairs"]:
                colloquial = pair.get("colloquial", "")
                formal_eq = pair.get("formal_equivalent", "")
                if colloquial and formal_eq:
                    vocab_pairs.append((colloquial, formal_eq))
                    # Also register these colloquial words in the word map
                    # so they contribute to register detection.
                    self.colloquial_words.add(colloquial)
                    self.word_register_map[colloquial] = (
                        REGISTER_COLLOQUIAL,
                        formal_eq,
                        colloquial,
                    )
                    self._register_detection_words.add(colloquial)
        # Sort longest-first so multi-word patterns match before single words.
        # e.g., "တော်တော်မိုက်" matches before "မိုက်".
        self._vocabulary_pairs = sorted(vocab_pairs, key=lambda x: len(x[0]), reverse=True)

        # Load vocabulary exempt phrases — widely-used colloquial patterns
        # tolerated even in formal text (e.g., "တော်တော်မိုက်").
        exempt: list[str] = []
        if "vocabulary_exempt" in reg_config:
            for phrase in reg_config["vocabulary_exempt"]:
                if isinstance(phrase, str) and phrase:
                    exempt.append(phrase)
        self._vocab_exempt_phrases = frozenset(exempt)

    def get_register(self, word: str) -> RegisterInfo:
        """
        Get the register information for a word.

        Args:
            word: Word to analyze.

        Returns:
            RegisterInfo object with register details.

        Examples:
            >>> checker = RegisterChecker()
            >>> info = checker.get_register("သည်")
            >>> info.register
            'formal'
            >>> info = checker.get_register("တယ်")
            >>> info.register
            'colloquial'
        """
        if word in self.word_register_map:
            register, formal_form, colloquial_form = self.word_register_map[word]
            return RegisterInfo(
                word=word,
                register=register,
                formal_form=formal_form,
                colloquial_form=colloquial_form,
            )

        # Suffix-based matching for verb+particle tokens (e.g., "သွားပါသည်")
        # Myanmar register markers are always suffixes/postpositions.
        # Pre-sorted longest-first so "ပါသည်" matches before "သည်".
        # Skip suffix matching for lexicalized compound nouns (e.g., တပ်မတော်သည်
        # where သည် is part of the compound noun, not a register marker).
        # Only high-frequency compounds (>=1000) are truly lexicalized.
        # Low-frequency entries (e.g., ပြန်လာခဲ့တယ် at 153) are segmentation
        # artifacts and should still undergo register suffix detection.
        is_lexicalized_compound = False
        if self._provider and hasattr(self._provider, "get_word_frequency"):
            word_freq = self._provider.get_word_frequency(word)
            if isinstance(word_freq, (int, float)) and word_freq >= 1000:
                is_lexicalized_compound = True

        if not is_lexicalized_compound:
            for suffix, register, formal_form, colloquial_form in self._register_suffixes:
                if len(suffix) < len(word) and word.endswith(suffix):
                    # Build full-word suggestions by replacing the suffix with its
                    # register pair, so suggestions show "သွားသည်" instead of bare "သည်".
                    stem = word[: -len(suffix)]
                    if register == REGISTER_FORMAL:
                        # Word is formal; build colloquial full form
                        pair = self.register_pairs.get(suffix)
                        full_colloquial = stem + pair if pair else colloquial_form
                        full_formal = word
                    else:
                        # Word is colloquial; build formal full form
                        pair = self.colloquial_to_formal.get(suffix)
                        full_formal = stem + pair if pair else formal_form
                        full_colloquial = word
                    return RegisterInfo(
                        word=word,
                        register=register,
                        formal_form=full_formal,
                        colloquial_form=full_colloquial,
                    )

        # Default to neutral if unknown
        return RegisterInfo(
            word=word,
            register=REGISTER_NEUTRAL,
            formal_form=word,
            colloquial_form=word,
        )

    def is_formal(self, word: str) -> bool:
        """Check if a word is formal register."""
        return word in self.formal_words

    def is_colloquial(self, word: str) -> bool:
        """Check if a word is colloquial register."""
        return word in self.colloquial_words

    def is_neutral(self, word: str) -> bool:
        """Check if a word is neutral (usable in both registers)."""
        return word in self.neutral_words or (
            word not in self.formal_words and word not in self.colloquial_words
        )

    def detect_sentence_register(self, words: list[str]) -> tuple[str, float, list[RegisterInfo]]:
        """
        Detect the predominant register of a sentence.

        Analyzes all words and determines if the sentence is primarily
        formal, colloquial, or mixed.

        Args:
            words: List of words in the sentence.

        Returns:
            Tuple of (predominant_register, consistency_score, word_infos)
            - predominant_register: 'formal', 'colloquial', or 'mixed'
            - consistency_score: 0.0-1.0 (1.0 = perfectly consistent)
            - word_infos: List of RegisterInfo for each register-significant word

        Examples:
            >>> checker = RegisterChecker()
            >>> register, score, infos = checker.detect_sentence_register(
            ...     ["သူ", "သည်", "စာအုပ်", "ဖတ်", "တယ်"]
            ... )
            >>> register
            'mixed'
            >>> score < 1.0
            True
        """
        formal_count = 0
        colloquial_count = 0
        word_infos: list[RegisterInfo] = []

        for word in words:
            # Skip non-ending particles (location, conjunction, etc.) that are
            # exact matches in word_register_map but not register-significant.
            # Suffix-matched words (e.g., "သွားသည်") pass through since suffixes
            # are already filtered to sentence-ending types.
            if word in self.word_register_map and word not in self._register_detection_words:
                continue

            info = self.get_register(word)
            if info.register == REGISTER_FORMAL:
                formal_count += 1
                word_infos.append(info)
            elif info.register == REGISTER_COLLOQUIAL:
                colloquial_count += 1
                word_infos.append(info)
            # Neutral words don't affect the count

        total_register_words = formal_count + colloquial_count

        if total_register_words == 0:
            # No register-significant words
            return (REGISTER_NEUTRAL, 1.0, word_infos)

        # Determine predominant register
        if formal_count > 0 and colloquial_count > 0:
            # Mixed register
            # Consistency score = dominant / total
            dominant = max(formal_count, colloquial_count)
            consistency = dominant / total_register_words
            return ("mixed", consistency, word_infos)
        elif formal_count > 0:
            return (REGISTER_FORMAL, 1.0, word_infos)
        else:
            return (REGISTER_COLLOQUIAL, 1.0, word_infos)

    def validate_sequence(self, words: list[str]) -> list[RegisterError]:
        """
        Validate register consistency in a word sequence.

        Checks for:
        1. Mixed register usage (particle-level)
        2. Vocabulary-level register mismatches (slang in formal text)
        3. Suggests corrections to maintain consistency

        Args:
            words: List of words to validate.

        Returns:
            List of RegisterError objects.

        Examples:
            >>> checker = RegisterChecker()
            >>> errors = checker.validate_sequence(["သူ", "သည်", "စာအုပ်", "ဖတ်", "တယ်"])
            >>> len(errors) > 0  # Mixed register detected
            True
        """
        errors: list[RegisterError] = []

        # First pass: detect predominant register
        predominant, consistency, word_infos = self.detect_sentence_register(words)

        if predominant == "mixed":
            # Find which words don't match the majority
            formal_count = sum(1 for w in word_infos if w.is_formal())
            colloquial_count = sum(1 for w in word_infos if w.is_colloquial())

            # Determine expected register (majority wins)
            if formal_count >= colloquial_count:
                expected = REGISTER_FORMAL
            else:
                expected = REGISTER_COLLOQUIAL

            # Find offending words (only register-significant ones)
            for i, word in enumerate(words):
                # Skip non-ending particles — same filter as detect_sentence_register
                if word in self.word_register_map and word not in self._register_detection_words:
                    continue

                info = self.get_register(word)

                if info.register == REGISTER_NEUTRAL:
                    continue

                if info.register != expected:
                    # This word doesn't match the expected register
                    if expected == REGISTER_FORMAL:
                        suggestion = info.formal_form or word
                        reason = (
                            f"Mixed register: '{word}' is colloquial, "
                            f"but sentence is predominantly formal. "
                            f"Consider using '{suggestion}'."
                        )
                    else:
                        suggestion = info.colloquial_form or word
                        reason = (
                            f"Mixed register: '{word}' is formal, "
                            f"but sentence is predominantly colloquial. "
                            f"Consider using '{suggestion}'."
                        )

                    errors.append(
                        RegisterError(
                            text=word,
                            position=i,
                            suggestions=[suggestion],
                            error_type=ET_MIXED_REGISTER,
                            confidence=self.register_cfg.register_mismatch_confidence,
                            reason=reason,
                            detected_register=info.register,
                            expected_register=expected,
                        )
                    )

        # Second pass: vocabulary-level scanning for informal words in formal text.
        # This catches cases where slang/informal vocabulary words appear as
        # substrings within tokens that the word-level detection misses.
        # e.g., "တော်တော်မိုက်နေသည်" contains slang "မိုက်" but the token
        # as a whole might not be in the word_register_map.
        if self._vocabulary_pairs and predominant in (REGISTER_FORMAL, "mixed"):
            error_positions: set[int] = {e.position for e in errors}

            # Phase 1: Multi-token vocabulary matching.  The segmenter often
            # splits multi-word slang (e.g., "တော်တော်ဂွမ်း" → "တော်တော်"
            # + "ဂွမ်း").  Check adjacent-word concatenations against
            # vocabulary pairs.  If matched, replace any individual first-pass
            # errors at those positions with a single combined-span error.
            multi_covered: set[int] = set()
            for i in range(len(words)):
                if i in multi_covered:
                    continue
                for span_len in (4, 3, 2):
                    if i + span_len > len(words):
                        continue
                    combined = "".join(words[i : i + span_len])
                    for colloquial_term, formal_eq in self._vocabulary_pairs:
                        if colloquial_term == combined:
                            span_positions = set(range(i, i + span_len))
                            # Remove individual first-pass errors at these
                            # positions — the combined error supersedes them.
                            errors = [e for e in errors if e.position not in span_positions]
                            reason = (
                                f"Register mismatch: '{colloquial_term}' "
                                f"is informal/slang in formal text. "
                                f"Consider using '{formal_eq}'."
                            )
                            errors.append(
                                RegisterError(
                                    text=combined,
                                    position=i,
                                    suggestions=[formal_eq],
                                    error_type=ET_REGISTER_ERROR,
                                    confidence=(self.register_cfg.register_mismatch_confidence),
                                    reason=reason,
                                    detected_register=REGISTER_COLLOQUIAL,
                                    expected_register=REGISTER_FORMAL,
                                )
                            )
                            multi_covered.update(span_positions)
                            error_positions.update(span_positions)
                            break
                    if i in multi_covered:
                        break

            # Phase 2: Single-token substring matching.
            for i, word in enumerate(words):
                if i in error_positions:
                    continue
                for colloquial_term, formal_eq in self._vocabulary_pairs:
                    if colloquial_term in word and word not in self.formal_words:
                        # Skip when the token contains an exempt phrase that
                        # encompasses the matched colloquial term.
                        if self._vocab_exempt_phrases and any(
                            ep in word for ep in self._vocab_exempt_phrases
                        ):
                            continue
                        reason = (
                            f"Register mismatch: '{colloquial_term}' in '{word}' "
                            f"is informal/slang in formal text. "
                            f"Consider using '{formal_eq}'."
                        )
                        errors.append(
                            RegisterError(
                                text=word,
                                position=i,
                                suggestions=[formal_eq],
                                error_type=ET_REGISTER_ERROR,
                                confidence=self.register_cfg.register_mismatch_confidence,
                                reason=reason,
                                detected_register=REGISTER_COLLOQUIAL,
                                expected_register=REGISTER_FORMAL,
                            )
                        )
                        error_positions.add(i)
                        break  # One error per word position

        return errors
