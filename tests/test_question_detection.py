"""
Unit tests for question detection in the spell checker.

Tests the sentence type detection and question structure validation
via the QuestionStructureValidationStrategy.
"""

import pytest

from myspellchecker.core.validation_strategies import (
    QuestionStructureValidationStrategy,
    ValidationContext,
)
from myspellchecker.grammar.patterns import (
    QUESTION_PARTICLES,
    QUESTION_WORDS,
    detect_sentence_type,
    has_enclitic_question_particle,
    is_question_particle,
    is_question_word,
)


class TestQuestionWordsAndParticles:
    """Tests for the QUESTION_WORDS and QUESTION_PARTICLES sets."""

    def test_question_words_exists(self):
        """Test that QUESTION_WORDS is defined."""
        assert QUESTION_WORDS is not None
        assert isinstance(QUESTION_WORDS, frozenset)

    def test_question_particles_exists(self):
        """Test that QUESTION_PARTICLES is defined."""
        assert QUESTION_PARTICLES is not None
        assert isinstance(QUESTION_PARTICLES, frozenset)

    def test_question_words_contains_key_words(self):
        """Test that key question words are defined."""
        assert "ဘာ" in QUESTION_WORDS  # What
        assert "ဘယ်" in QUESTION_WORDS  # Which/where
        assert "ဘယ်လို" in QUESTION_WORDS  # How
        assert "ဘယ်သူ" in QUESTION_WORDS  # Who

    def test_question_particles_contains_key_particles(self):
        """Test that key question particles are defined."""
        assert "လား" in QUESTION_PARTICLES  # Yes/no question
        assert "သလား" in QUESTION_PARTICLES  # Yes/no question (alternative)
        assert "လဲ" in QUESTION_PARTICLES  # Wh-question
        assert "မလဲ" in QUESTION_PARTICLES  # Future wh-question


class TestQuestionUtilityFunctions:
    """Tests for question-related utility functions."""

    def test_is_question_word_true(self):
        """Test is_question_word returns True for question words."""
        assert is_question_word("ဘာ") is True
        assert is_question_word("ဘယ်") is True

    def test_is_question_word_false(self):
        """Test is_question_word returns False for non-question words."""
        assert is_question_word("စား") is False
        assert is_question_word("သွား") is False

    def test_is_question_particle_true(self):
        """Test is_question_particle returns True for question particles."""
        assert is_question_particle("လား") is True
        assert is_question_particle("လဲ") is True

    def test_is_question_particle_false(self):
        """Test is_question_particle returns False for non-question particles."""
        assert is_question_particle("တယ်") is False
        assert is_question_particle("သည်") is False


class TestEncliticQuestionParticles:
    """Tests for enclitic question particle detection.

    In Myanmar, question particles often attach directly to verbs without spaces.
    Example: "သွားလား" (go+question = did [you] go?)
    """

    def test_has_enclitic_standalone_particle(self):
        """Test that standalone particles are detected."""
        assert has_enclitic_question_particle("လား") is True
        assert has_enclitic_question_particle("လဲ") is True
        assert has_enclitic_question_particle("မလဲ") is True

    def test_has_enclitic_attached_particle(self):
        """Test that verb+particle combinations are detected."""
        # သွားလား = go + question particle = "did [you] go?"
        assert has_enclitic_question_particle("သွားလား") is True
        # လာမလား = come + future + question = "will [you] come?"
        assert has_enclitic_question_particle("လာမလား") is True
        # စားသလား = eat + question = "did [you] eat?"
        assert has_enclitic_question_particle("စားသလား") is True
        # ဘာလုပ်မလဲ = what + do + future + question = "what will [you] do?"
        assert has_enclitic_question_particle("ဘာလုပ်မလဲ") is True

    def test_has_enclitic_false_for_non_question(self):
        """Test that non-question words return False."""
        assert has_enclitic_question_particle("သွားတယ်") is False  # statement ending
        assert has_enclitic_question_particle("စားပြီ") is False  # completion
        assert has_enclitic_question_particle("လုပ်နေ") is False  # progressive

    def test_has_enclitic_empty_string(self):
        """Test that empty string returns False."""
        assert has_enclitic_question_particle("") is False

    def test_has_enclitic_particle_only(self):
        """Test that particle alone (no verb prefix) is detected."""
        # Should return True because it's a valid question particle
        assert has_enclitic_question_particle("လား") is True


class TestDetectSentenceType:
    """Tests for the detect_sentence_type function."""

    def test_empty_words_returns_unknown(self):
        """Test that empty word list returns 'unknown'."""
        assert detect_sentence_type([]) == "unknown"

    def test_question_particle_at_end(self):
        """Test detection of question by ending particle."""
        # "ဘာ" (what) + "လဲ" (question particle)
        words = ["ဘာ", "လဲ"]
        assert detect_sentence_type(words) == "question"

    def test_question_word_in_sentence(self):
        """Test detection of question by question word."""
        # "ဘယ်သူ" (who) + "လာ" (come) + "မလဲ" (future question)
        words = ["ဘယ်သူ", "လာ", "မလဲ"]
        assert detect_sentence_type(words) == "question"

    def test_question_particle_near_end(self):
        """Test detection of question particle near end (within 3 words)."""
        words = ["သူ", "လာ", "လား", ""]
        # Note: detect_sentence_type checks last 3 words for question particles
        # The "လား" at index 2 (from end: index 2) should be found
        assert detect_sentence_type(words) == "question"

    def test_statement_sentence(self):
        """Test detection of statement sentence."""
        # "သူ" (he) + "စား" (eat) + "တယ်" (statement ending)
        words = ["သူ", "စား", "တယ်"]
        assert detect_sentence_type(words) == "statement"

    def test_question_word_without_particle(self):
        """Test that question words trigger question type even without particle."""
        # "ဘာ" (what) alone is still detected as question
        words = ["ဘာ", "လုပ်", "နေ"]
        assert detect_sentence_type(words) == "question"

    def test_negative_indefinite_not_question(self):
        """Test that negative indefinite constructions are detected as statements.

        In Myanmar, question words + "မှ" suffix + negative verb pattern
        indicates negative indefinite (nobody, nothing, nowhere), NOT a question.
        Example: "ဘယ်သူမှ မလာဘူး" = "Nobody came" (statement, not question)
        """
        # "Nobody came" - negative indefinite, should be STATEMENT
        words = ["ဘယ်သူမှ", "မလာ", "ဘူး"]
        assert detect_sentence_type(words) == "statement"

        # "I don't know anything" - negative indefinite, should be STATEMENT
        words = ["ဘာမှ", "မသိ", "ဘူး"]
        assert detect_sentence_type(words) == "statement"

        # "It's nowhere" - negative indefinite, should be STATEMENT
        words = ["ဘယ်မှာမှ", "မရှိ", "ဘူး"]
        assert detect_sentence_type(words) == "statement"

    def test_regular_question_with_same_words(self):
        """Test that regular questions with question words are still detected.

        When question words appear without the negative indefinite pattern,
        they should still be detected as questions.
        """
        # "Who came?" - regular question, should be QUESTION
        words = ["ဘယ်သူ", "လာ", "မလဲ"]
        assert detect_sentence_type(words) == "question"

        # "What is it?" - regular question with particle, should be QUESTION
        words = ["ဘာ", "လဲ"]
        assert detect_sentence_type(words) == "question"

    def test_enclitic_question_particle_detection(self):
        """Test that enclitic question particles (attached to verbs) are detected.

        In Myanmar, question particles often attach directly to verbs without spaces.
        Example: "သွားလား" (go+question = did [you] go?)
        """
        # Enclitic particle at end - should be detected as question
        # "Did you go?" with particle attached: "သွားလား"
        words = ["သူ", "သွားလား"]
        assert detect_sentence_type(words) == "question"

        # "Will you come?" with particle attached: "လာမလား"
        words = ["သူ", "လာမလား"]
        assert detect_sentence_type(words) == "question"

        # Complex sentence with enclitic particle
        words = ["ဒီနေ့", "သူ", "လာမလား"]
        assert detect_sentence_type(words) == "question"


class TestQuestionStructureValidationStrategy:
    """Tests for QuestionStructureValidationStrategy."""

    def test_strategy_priority(self):
        """Test that strategy has correct priority."""
        strategy = QuestionStructureValidationStrategy()
        assert strategy.priority() == 40

    def test_strategy_repr(self):
        """Test strategy string representation."""
        strategy = QuestionStructureValidationStrategy()
        assert "QuestionStructureValidationStrategy" in repr(strategy)

    def test_validate_proper_question(self):
        """Test that properly formed questions don't generate errors."""
        strategy = QuestionStructureValidationStrategy()

        # "ဘာ" (what) + "လဲ" (question particle) - proper question
        context = ValidationContext(
            words=["ဘာ", "လဲ"],
            word_positions=[0, 3],
            sentence="ဘာလဲ",
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)
        assert errors == []

    def test_validate_missing_particle(self):
        """Test detection of question with missing question particle."""
        strategy = QuestionStructureValidationStrategy()

        # "ဘာ" (what) + "လုပ်" (do) + "တယ်" (statement ending)
        # This should flag an error because it has question word but statement ending
        context = ValidationContext(
            words=["ဘာ", "လုပ်", "တယ်"],
            word_positions=[0, 3, 6],
            sentence="ဘာလုပ်တယ်",
            is_name_mask=[False, False, False],
        )

        errors = strategy.validate(context)

        assert len(errors) == 1
        assert errors[0].error_type == "question_structure"
        assert errors[0].text == "တယ်"
        assert "လား" in errors[0].suggestions

    def test_validate_single_word(self):
        """Test that single word doesn't generate errors."""
        strategy = QuestionStructureValidationStrategy()

        context = ValidationContext(
            words=["ဘာ"],
            word_positions=[0],
            sentence="ဘာ",
            is_name_mask=[False],
        )

        errors = strategy.validate(context)
        assert errors == []

    def test_validate_statement(self):
        """Test that statements don't generate question errors."""
        strategy = QuestionStructureValidationStrategy()

        # Regular statement: "သူ" (he) + "စား" (eat) + "တယ်" (statement)
        context = ValidationContext(
            words=["သူ", "စား", "တယ်"],
            word_positions=[0, 3, 6],
            sentence="သူစားတယ်",
            is_name_mask=[False, False, False],
        )

        errors = strategy.validate(context)
        assert errors == []

    def test_validate_polite_ending(self):
        """Test detection with polite statement ending."""
        strategy = QuestionStructureValidationStrategy()

        # "ဘာ" (what) + "လုပ်" (do) + "ပါတယ်" (polite statement ending)
        context = ValidationContext(
            words=["ဘာ", "လုပ်", "ပါတယ်"],
            word_positions=[0, 3, 6],
            sentence="ဘာလုပ်ပါတယ်",
            is_name_mask=[False, False, False],
        )

        errors = strategy.validate(context)

        assert len(errors) == 1
        assert errors[0].text == "ပါတယ်"

    def test_validate_enclitic_question_particle(self):
        """Test that enclitic question particles don't generate errors.

        When question particle is attached to verb (enclitic form like "သွားလား"),
        the strategy should recognize this as a proper question and NOT flag it.
        """
        strategy = QuestionStructureValidationStrategy()

        # "သူ" (he) + "သွားလား" (go+question = did he go?)
        # This is a proper question with enclitic particle
        context = ValidationContext(
            words=["သူ", "သွားလား"],
            word_positions=[0, 6],
            sentence="သူသွားလား",
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)
        assert errors == []  # Should not generate errors

    def test_validate_enclitic_with_question_word(self):
        """Test question word + enclitic particle combination.

        "ဘယ်သူ" (who) + "လာမလား" (come+future+question = who will come?)
        """
        strategy = QuestionStructureValidationStrategy()

        context = ValidationContext(
            words=["ဘယ်သူ", "လာမလား"],
            word_positions=[0, 18],
            sentence="ဘယ်သူလာမလား",
            is_name_mask=[False, False],
        )

        errors = strategy.validate(context)
        assert errors == []  # Proper question, no errors


class TestContextValidatorQuestionIntegration:
    """Integration tests for question detection via ContextValidator with strategies."""

    @pytest.fixture
    def spellchecker(self):
        """Create a SpellChecker instance for integration testing."""
        try:
            from myspellchecker import SpellChecker

            return SpellChecker.create_default()
        except Exception as e:
            pytest.skip(f"SpellChecker not available: {e}")

    def test_context_validator_has_strategies(self, spellchecker):
        """Test that ContextValidator has validation strategies."""
        context_validator = spellchecker.context_validator
        assert context_validator is not None
        assert hasattr(context_validator, "strategies")
        assert isinstance(context_validator.strategies, list)

    def test_question_strategy_in_validator(self, spellchecker):
        """Test that QuestionStructureValidationStrategy is included in strategies."""
        context_validator = spellchecker.context_validator
        strategy_types = [type(s).__name__ for s in context_validator.strategies]
        assert "QuestionStructureValidationStrategy" in strategy_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
