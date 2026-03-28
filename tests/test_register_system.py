"""
Tests for Myanmar Register Detection System.

Tests the RegisterChecker functionality:
- Register detection (formal, colloquial, neutral)
- Mixed register validation
- Register-consistent suggestions
- Register consistency scoring
"""

from myspellchecker.grammar.checkers.register import (
    REGISTER_COLLOQUIAL,
    REGISTER_FORMAL,
    REGISTER_NEUTRAL,
    RegisterChecker,
    RegisterError,
    RegisterInfo,
)


class TestRegisterCheckerInitialization:
    """Tests for RegisterChecker initialization."""

    def test_initialization(self):
        """RegisterChecker initializes with proper data structures."""
        checker = RegisterChecker()
        assert checker.formal_words is not None
        assert checker.colloquial_words is not None
        assert checker.neutral_words is not None
        assert checker.register_pairs is not None
        assert checker.colloquial_to_formal is not None


class TestRegisterDetection:
    """Tests for individual word register detection."""

    def test_formal_word_detection(self):
        """Formal words are correctly identified."""
        checker = RegisterChecker()

        # Formal statement endings
        assert checker.is_formal("သည်")
        assert checker.is_formal("ပါသည်")
        assert checker.is_formal("မည်")
        assert checker.is_formal("ပါမည်")

        # Formal particles
        assert checker.is_formal("၏")
        assert checker.is_formal("နှင့်")
        assert checker.is_formal("သို့")
        assert checker.is_formal("များ")

    def test_colloquial_word_detection(self):
        """Colloquial words are correctly identified."""
        checker = RegisterChecker()

        # Colloquial statement endings
        assert checker.is_colloquial("တယ်")
        assert checker.is_colloquial("ပါတယ်")
        assert checker.is_colloquial("မယ်")
        assert checker.is_colloquial("ပါမယ်")

        # Colloquial particles
        assert checker.is_colloquial("ရဲ့")
        assert checker.is_colloquial("နဲ့")
        assert checker.is_colloquial("တွေ")
        # Note: ကို is neutral (used in both registers)

    def test_neutral_word_detection(self):
        """Neutral words are correctly identified."""
        checker = RegisterChecker()

        # Common words without register marking
        assert checker.is_neutral("စာအုပ်")  # book
        assert checker.is_neutral("ဖတ်")  # read
        assert checker.is_neutral("သွား")  # go

    def test_get_register_formal(self):
        """get_register returns correct info for formal words."""
        checker = RegisterChecker()
        info = checker.get_register("သည်")

        assert isinstance(info, RegisterInfo)
        assert info.word == "သည်"
        assert info.register == REGISTER_FORMAL
        assert info.formal_form == "သည်"
        assert info.colloquial_form is not None  # Should have တယ်

    def test_get_register_colloquial(self):
        """get_register returns correct info for colloquial words."""
        checker = RegisterChecker()
        info = checker.get_register("တယ်")

        assert isinstance(info, RegisterInfo)
        assert info.word == "တယ်"
        assert info.register == REGISTER_COLLOQUIAL
        assert info.formal_form is not None  # Should have သည်
        assert info.colloquial_form == "တယ်"

    def test_get_register_neutral(self):
        """get_register returns neutral for unmarked words."""
        checker = RegisterChecker()
        info = checker.get_register("စာအုပ်")

        assert info.word == "စာအုပ်"
        assert info.register == REGISTER_NEUTRAL
        assert info.formal_form == "စာအုပ်"
        assert info.colloquial_form == "စာအုပ်"


class TestRegisterInfo:
    """Tests for RegisterInfo dataclass."""

    def test_is_formal(self):
        """RegisterInfo.is_formal() works correctly."""
        info = RegisterInfo(
            word="သည်",
            register=REGISTER_FORMAL,
            formal_form="သည်",
            colloquial_form="တယ်",
        )
        assert info.is_formal()
        assert not info.is_colloquial()

    def test_is_colloquial(self):
        """RegisterInfo.is_colloquial() works correctly."""
        info = RegisterInfo(
            word="တယ်",
            register=REGISTER_COLLOQUIAL,
            formal_form="သည်",
            colloquial_form="တယ်",
        )
        assert info.is_colloquial()
        assert not info.is_formal()

    def test_neutral_register(self):
        """Neutral register word has neither formal nor colloquial flag."""
        info = RegisterInfo(
            word="စာ",
            register=REGISTER_NEUTRAL,
            formal_form="စာ",
            colloquial_form="စာ",
        )
        assert not info.is_formal()
        assert not info.is_colloquial()


class TestSentenceRegisterDetection:
    """Tests for sentence-level register detection."""

    def test_formal_sentence(self):
        """Correctly detects purely formal sentences."""
        checker = RegisterChecker()
        words = ["သူ", "သည်", "စာအုပ်", "ဖတ်", "သည်"]

        register, score, infos = checker.detect_sentence_register(words)

        assert register == REGISTER_FORMAL
        assert score == 1.0  # Perfect consistency

    def test_colloquial_sentence(self):
        """Correctly detects purely colloquial sentences."""
        checker = RegisterChecker()
        words = ["သူ", "စာအုပ်", "ဖတ်", "တယ်"]

        register, score, infos = checker.detect_sentence_register(words)

        assert register == REGISTER_COLLOQUIAL
        assert score == 1.0  # Perfect consistency

    def test_mixed_register_sentence(self):
        """Correctly detects mixed register sentences."""
        checker = RegisterChecker()
        words = ["သူ", "သည်", "စာအုပ်", "ဖတ်", "တယ်"]

        register, score, infos = checker.detect_sentence_register(words)

        assert register == "mixed"
        assert score < 1.0  # Imperfect consistency

    def test_neutral_only_sentence(self):
        """Neutral-only sentences return neutral register."""
        checker = RegisterChecker()
        words = ["စာအုပ်", "ကောင်း"]

        register, score, infos = checker.detect_sentence_register(words)

        assert register == REGISTER_NEUTRAL
        assert score == 1.0


class TestRegisterValidation:
    """Tests for register consistency validation."""

    def test_no_errors_for_consistent_formal(self):
        """No errors for consistently formal sentences."""
        checker = RegisterChecker()
        words = ["သူ", "သည်", "စာ", "ရေး", "သည်"]

        errors = checker.validate_sequence(words)
        assert len(errors) == 0

    def test_no_errors_for_consistent_colloquial(self):
        """No errors for consistently colloquial sentences."""
        checker = RegisterChecker()
        words = ["သူ", "စာ", "ရေး", "တယ်"]

        errors = checker.validate_sequence(words)
        assert len(errors) == 0

    def test_errors_for_mixed_register(self):
        """Errors reported for mixed register usage."""
        checker = RegisterChecker()
        # Formal subject marker + colloquial ending
        words = ["သူ", "သည်", "စာ", "ဖတ်", "တယ်"]

        errors = checker.validate_sequence(words)

        # Should flag either the formal or colloquial word
        assert len(errors) > 0
        assert all(isinstance(e, RegisterError) for e in errors)

    def test_error_has_position(self):
        """Register errors include correct position."""
        checker = RegisterChecker()
        words = ["သူ", "သည်", "ဖတ်", "တယ်"]

        errors = checker.validate_sequence(words)

        if errors:
            for error in errors:
                assert 0 <= error.position < len(words)

    def test_error_has_suggestion(self):
        """Register errors include correction suggestions."""
        checker = RegisterChecker()
        words = ["သူ", "သည်", "ဖတ်", "တယ်"]

        errors = checker.validate_sequence(words)

        if errors:
            for error in errors:
                assert error.suggestion is not None
                assert len(error.suggestion) > 0

    def test_error_has_reason(self):
        """Register errors include human-readable reasons."""
        checker = RegisterChecker()
        words = ["သူ", "သည်", "ဖတ်", "တယ်"]

        errors = checker.validate_sequence(words)

        if errors:
            for error in errors:
                assert error.reason is not None
                reason_lower = error.reason.lower()
                assert (
                    "register" in reason_lower
                    or "formal" in reason_lower
                    or "colloquial" in reason_lower
                )


class TestDataConstantsIntegrity:
    """Tests for register data constants."""

    def test_formal_words_set_exists(self):
        """FORMAL_ONLY_WORDS set is populated."""
        checker = RegisterChecker()
        assert len(checker.formal_words) > 0

    def test_colloquial_words_set_exists(self):
        """COLLOQUIAL_ONLY_WORDS set is populated."""
        checker = RegisterChecker()
        assert len(checker.colloquial_words) > 0

    def test_register_pairs_mapping(self):
        """REGISTER_PAIRS has valid mappings."""
        checker = RegisterChecker()
        assert len(checker.register_pairs) > 0
        for formal, colloquial in checker.register_pairs.items():
            assert isinstance(formal, str)
            assert isinstance(colloquial, str)

    def test_colloquial_to_formal_mapping(self):
        """COLLOQUIAL_TO_FORMAL has valid mappings."""
        checker = RegisterChecker()
        assert len(checker.colloquial_to_formal) > 0
        for colloquial, formal in checker.colloquial_to_formal.items():
            assert isinstance(colloquial, str)
            assert isinstance(formal, str)

    def test_bidirectional_mappings(self):
        """Register pairs are bidirectionally consistent."""
        checker = RegisterChecker()
        # Every formal→colloquial should have a colloquial→formal
        for formal, colloquial in checker.register_pairs.items():
            # Check reverse mapping exists
            assert colloquial in checker.colloquial_to_formal
            assert checker.colloquial_to_formal[colloquial] == formal


class TestRegisterEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_word_list(self):
        """Empty word list handles gracefully."""
        checker = RegisterChecker()

        register, score, infos = checker.detect_sentence_register([])
        assert register == REGISTER_NEUTRAL
        assert score == 1.0

        errors = checker.validate_sequence([])
        assert errors == []

    def test_single_word(self):
        """Single word sentences work correctly."""
        checker = RegisterChecker()

        # Single formal word
        register, score, _ = checker.detect_sentence_register(["သည်"])
        assert register == REGISTER_FORMAL

        # Single colloquial word
        register, score, _ = checker.detect_sentence_register(["တယ်"])
        assert register == REGISTER_COLLOQUIAL

    def test_unknown_word(self):
        """Unknown words are treated as neutral."""
        checker = RegisterChecker()
        info = checker.get_register("ညနှခပုက")  # Nonsense word

        assert info.register == REGISTER_NEUTRAL

    def test_multiple_formal_markers(self):
        """Multiple formal markers still detected as formal."""
        checker = RegisterChecker()
        words = ["သူ", "သည်", "စာ", "များ", "ဖတ်", "သည်"]

        register, score, _ = checker.detect_sentence_register(words)
        assert register == REGISTER_FORMAL
        assert score == 1.0
