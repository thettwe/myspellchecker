"""
Tests for Myanmar Negation Pattern System.

Tests the NegationChecker functionality:
- Negation prefix detection
- Negative ending validation
- Negation pattern detection
- Typo correction
- Register-aware negation
"""

from myspellchecker.grammar.checkers.negation import (
    REGISTER_COLLOQUIAL,
    NegationChecker,
    NegationError,
    NegationInfo,
    get_negation_checker,
    is_negative_ending,
)


class TestNegationCheckerInitialization:
    """Tests for NegationChecker initialization."""

    def test_initialization(self):
        """NegationChecker initializes with proper data structures."""
        checker = NegationChecker()
        assert checker.negation_prefix == "မ"
        assert checker.valid_endings is not None
        assert checker.negative_endings is not None
        assert checker.typo_map is not None

    def test_singleton_pattern(self):
        """get_negation_checker returns singleton instance."""
        checker1 = get_negation_checker()
        checker2 = get_negation_checker()
        assert checker1 is checker2


class TestNegationPrefixDetection:
    """Tests for negation prefix detection."""

    def test_starts_with_negation(self):
        """Words starting with negation prefix are detected."""
        checker = NegationChecker()

        # Words with negation prefix
        assert checker.starts_with_negation("မသွား")  # don't go
        assert checker.starts_with_negation("မစား")  # don't eat
        assert checker.starts_with_negation("မရှိ")  # doesn't exist
        assert checker.starts_with_negation("မလုပ်")  # don't do

    def test_not_negation(self):
        """Non-negated words are correctly identified."""
        checker = NegationChecker()

        # Regular words
        assert not checker.starts_with_negation("သွား")  # go
        assert not checker.starts_with_negation("စား")  # eat
        assert not checker.starts_with_negation("ရှိ")  # exist

    def test_standalone_prefix_not_negation(self):
        """Standalone prefix (မ) alone is not considered negation start."""
        checker = NegationChecker()

        # Single character is prefix but too short
        assert not checker.starts_with_negation("မ")


class TestNegativeEndingDetection:
    """Tests for negative ending detection."""

    def test_standard_negative_endings(self):
        """Standard negative endings are detected."""
        checker = NegationChecker()

        assert checker.is_negative_ending("ဘူး")  # standard
        assert checker.is_negative_ending("ပါဘူး")  # polite
        assert checker.is_negative_ending("ပါ")  # formal

    def test_prohibition_endings(self):
        """Prohibition endings are detected."""
        checker = NegationChecker()

        assert checker.is_negative_ending("နဲ့")  # Don't!
        assert checker.is_negative_ending("ပါနဲ့")  # Please don't!
        assert checker.is_negative_ending("နှင့်")  # formal prohibition

    def test_question_endings(self):
        """Question negative endings are detected."""
        checker = NegationChecker()

        assert checker.is_negative_ending("ဘူးလား")  # isn't it?

    def test_non_negative_endings(self):
        """Non-negative endings are correctly identified."""
        checker = NegationChecker()

        assert not checker.is_negative_ending("တယ်")  # statement
        assert not checker.is_negative_ending("သည်")  # formal statement
        assert not checker.is_negative_ending("မယ်")  # future

    def test_convenience_function(self):
        """is_negative_ending convenience function works."""
        assert is_negative_ending("ဘူး")
        assert not is_negative_ending("တယ်")


class TestEndingTypeDetection:
    """Tests for negative ending type detection."""

    def test_get_ending_type_standard(self):
        """Standard negative ending type is detected."""
        checker = NegationChecker()
        result = checker.get_ending_type("ဘူး")

        assert result is not None
        assert result[0] == "standard_negative"

    def test_get_ending_type_polite(self):
        """Polite negative ending type is detected."""
        checker = NegationChecker()
        result = checker.get_ending_type("ပါဘူး")

        assert result is not None
        assert result[0] == "polite_negative"

    def test_get_ending_type_formal(self):
        """Formal negative ending type is detected."""
        checker = NegationChecker()
        result = checker.get_ending_type("ပါ")

        assert result is not None
        assert result[0] == "formal_negative"

    def test_get_ending_type_prohibition(self):
        """Prohibition ending type is detected."""
        checker = NegationChecker()
        result = checker.get_ending_type("နဲ့")

        assert result is not None
        assert result[0] == "prohibition"

    def test_get_ending_type_invalid(self):
        """Invalid ending returns None."""
        checker = NegationChecker()
        result = checker.get_ending_type("တယ်")

        assert result is None


class TestAuxiliaryDetection:
    """Tests for negative auxiliary verb detection."""

    def test_auxiliary_verbs(self):
        """Auxiliary verbs are detected."""
        checker = NegationChecker()

        assert checker.is_auxiliary("ချင်")  # want
        assert checker.is_auxiliary("နိုင်")  # can
        assert checker.is_auxiliary("တတ်")  # habitually
        assert checker.is_auxiliary("ရ")  # permitted

    def test_non_auxiliary_verbs(self):
        """Non-auxiliary verbs are correctly identified."""
        checker = NegationChecker()

        # သွား is now classified as a directional auxiliary in negation contexts
        # (e.g., မရွေ့သွားဘူး = didn't move away)
        assert not checker.is_auxiliary("စား")  # eat — pure content verb
        assert not checker.is_auxiliary("ဘူး")  # negative ending, not auxiliary


class TestTypoDetection:
    """Tests for negative ending typo detection."""

    def test_missing_visarga(self):
        """Missing visarga typos are detected."""
        checker = NegationChecker()

        # ဘူ → ဘူး
        result = checker.check_ending_typo("ဘူ")
        assert result is not None
        assert result[0] == "ဘူး"
        assert result[1] >= 0.85

    def test_polite_missing_visarga(self):
        """Missing visarga in polite form is detected."""
        checker = NegationChecker()

        # ပါဘူ → ပါဘူး
        result = checker.check_ending_typo("ပါဘူ")
        assert result is not None
        assert result[0] == "ပါဘူး"

    def test_not_typo(self):
        """Non-typos return None."""
        checker = NegationChecker()

        result = checker.check_ending_typo("ဘူး")  # Already correct
        assert result is None

        result = checker.check_ending_typo("တယ်")  # Not a negative ending
        assert result is None


class TestNegationPatternDetection:
    """Tests for complete negation pattern detection."""

    def test_simple_negation(self):
        """Simple negation patterns are detected."""
        checker = NegationChecker()

        # မ + verb + ဘူး
        pattern = checker.detect_negation_pattern(["မသွား", "ဘူး"], 0)

        assert pattern is not None
        assert pattern.pattern_type == "standard_negative"
        assert pattern.verb == "သွား"
        assert pattern.ending == "ဘူး"

    def test_negation_with_auxiliary(self):
        """Negation with auxiliary is detected."""
        checker = NegationChecker()

        # မ + verb + auxiliary + ဘူး
        pattern = checker.detect_negation_pattern(["မလုပ်", "ချင်", "ဘူး"], 0)

        assert pattern is not None
        assert pattern.verb == "လုပ်"
        assert "ချင်" in pattern.auxiliaries
        assert pattern.ending == "ဘူး"

    def test_separated_negation(self):
        """Separated negation prefix is detected."""
        checker = NegationChecker()

        # မ (separate) + verb + ဘူး
        pattern = checker.detect_negation_pattern(["မ", "သွား", "ဘူး"], 0)

        assert pattern is not None
        assert pattern.verb == "သွား"
        assert pattern.ending == "ဘူး"

    def test_prohibition_pattern(self):
        """Prohibition patterns are detected."""
        checker = NegationChecker()

        pattern = checker.detect_negation_pattern(["မလုပ်", "နဲ့"], 0)

        assert pattern is not None
        assert pattern.pattern_type == "prohibition"
        assert pattern.ending == "နဲ့"

    def test_no_negation(self):
        """Non-negation returns None."""
        checker = NegationChecker()

        pattern = checker.detect_negation_pattern(["သွား", "တယ်"], 0)
        assert pattern is None


class TestSequenceValidation:
    """Tests for sequence validation."""

    def test_typo_in_sequence(self):
        """Typos in negation patterns are detected."""
        checker = NegationChecker()

        # မသွား + ဘူ (typo) → should suggest ဘူး
        errors = checker.validate_sequence(["မသွား", "ဘူ"])

        assert len(errors) > 0
        assert any(e.suggestion == "ဘူး" for e in errors)

    def test_valid_sequence_no_errors(self):
        """Valid sequences produce no errors."""
        checker = NegationChecker()

        errors = checker.validate_sequence(["မသွား", "ဘူး"])
        # This is a valid pattern, should have no typo errors
        typo_errors = [e for e in errors if e.error_type == "typo"]
        assert len(typo_errors) == 0

    def test_mixed_sentence(self):
        """Mixed sentences are handled correctly."""
        checker = NegationChecker()

        # Sentence with negation and other words
        words = ["သူ", "မသွား", "ဘူး", "ငါ", "သွား", "တယ်"]
        errors = checker.validate_sequence(words)

        # Should not have false positives
        for error in errors:
            assert error.word in words


class TestNegationInfo:
    """Tests for NegationInfo dataclass."""

    def test_negation_info_attributes(self):
        """NegationInfo has all expected attributes."""
        info = NegationInfo(
            start_index=0,
            end_index=2,
            words=["မ", "သွား", "ဘူး"],
            pattern_type="standard_negative",
            verb="သွား",
            auxiliaries=[],
            ending="ဘူး",
            register=REGISTER_COLLOQUIAL,
        )

        assert info.start_index == 0
        assert info.end_index == 2
        assert len(info.words) == 3
        assert info.pattern_type == "standard_negative"
        assert info.verb == "သွား"
        assert info.auxiliaries == []
        assert info.ending == "ဘူး"
        assert info.register == REGISTER_COLLOQUIAL


class TestNegationError:
    """Tests for NegationError dataclass."""

    def test_negation_error_attributes(self):
        """NegationError has all expected attributes."""
        error = NegationError(
            text="ဘူ",
            position=1,
            suggestions=["ဘူး"],
            error_type="typo",
            confidence=0.90,
            reason="Missing visarga",
        )

        assert error.position == 1
        assert error.word == "ဘူ"  # Backward compatibility property
        assert error.text == "ဘူ"
        assert error.error_type == "typo"
        assert error.suggestion == "ဘူး"  # Backward compatibility property
        assert error.suggestions == ["ဘူး"]
        assert error.confidence == 0.90
        assert "visarga" in error.reason.lower()


class TestDataConstantsIntegrity:
    """Tests for negation data constants."""

    def test_negation_prefix_defined(self):
        """NEGATION_PREFIX is correctly defined."""
        checker = NegationChecker()
        assert checker.negation_prefix == "မ"

    def test_negative_endings_populated(self):
        """NEGATIVE_ENDINGS has entries."""
        checker = NegationChecker()
        assert len(checker.negative_endings) > 0

    def test_valid_endings_set(self):
        """VALID_NEGATIVE_ENDINGS is a frozenset or set."""
        checker = NegationChecker()
        assert isinstance(checker.valid_endings, (set, frozenset))
        assert len(checker.valid_endings) > 0

    def test_typo_map_populated(self):
        """NEGATIVE_TYPO_MAP has entries."""
        checker = NegationChecker()
        assert len(checker.typo_map) > 0
        # Check that typo corrections point to valid endings
        for _typo, correction in checker.typo_map.items():
            assert correction in checker.valid_endings or correction == "ဘူး"

    def test_auxiliaries_populated(self):
        """NEGATIVE_AUXILIARIES has entries."""
        checker = NegationChecker()
        assert len(checker.auxiliaries) > 0
        assert "ချင်" in checker.auxiliaries
        assert "နိုင်" in checker.auxiliaries

    def test_common_verbs_populated(self):
        """COMMON_NEGATED_VERBS has entries."""
        checker = NegationChecker()
        assert len(checker.common_verbs) > 0
        assert "သွား" in checker.common_verbs
        assert "လုပ်" in checker.common_verbs


class TestNegationEdgeCases:
    """Tests for edge cases."""

    def test_empty_word_list(self):
        """Empty word list is handled gracefully."""
        checker = NegationChecker()

        errors = checker.validate_sequence([])
        assert errors == []

    def test_single_word(self):
        """Single word handling."""
        checker = NegationChecker()

        # Single negated word without ending
        pattern = checker.detect_negation_pattern(["မသွား"], 0)
        # Single negated word without ending is detected as incomplete pattern
        assert pattern is not None
        assert pattern.pattern_type == "incomplete"

    def test_out_of_bounds_index(self):
        """Out of bounds index returns None."""
        checker = NegationChecker()

        pattern = checker.detect_negation_pattern(["မသွား"], 5)
        assert pattern is None

    def test_negation_at_end(self):
        """Negation at end of sentence."""
        checker = NegationChecker()

        words = ["သူ", "က", "မလာ"]
        pattern = checker.detect_negation_pattern(words, 2)
        # Negation at end without ending is detected as incomplete
        assert pattern is not None
        assert pattern.pattern_type == "incomplete"
