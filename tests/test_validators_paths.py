from unittest.mock import MagicMock, patch

import pytest

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.context_validator import ContextValidator
from myspellchecker.core.response import SyllableError, WordError
from myspellchecker.core.validators import (
    SyllableValidator,
    Validator,
    WordValidator,
)


# Mixin for mocking _is_myanmar_with_config
class MockIsMyanmarMixin:
    @pytest.fixture(autouse=True)
    def mock_is_myanmar(self):
        with patch.object(Validator, "_is_myanmar_with_config", return_value=True):
            yield


class TestSyllableValidatorCoverage(MockIsMyanmarMixin):
    @pytest.fixture
    def validator_setup(self):
        config = SpellCheckerConfig()
        segmenter = MagicMock()
        provider = MagicMock()
        symspell = MagicMock()
        rule_validator = MagicMock()
        validator = SyllableValidator(config, segmenter, provider, symspell, rule_validator)
        return validator, segmenter, provider, symspell, rule_validator

    def test_validate_rule_failure(self, validator_setup):
        validator, segmenter, provider, symspell, rule_validator = validator_setup
        segmenter.segment_syllables.return_value = ["invalid"]
        rule_validator.validate.return_value = False  # Fails rule check
        provider.is_valid_word.return_value = False  # Also fails word fallback

        # Mock symspell lookup
        suggestion = MagicMock()
        suggestion.term = "valid"
        symspell.lookup.return_value = [suggestion]

        errors = validator.validate("invalid")
        assert len(errors) == 1
        assert isinstance(errors[0], SyllableError)
        assert errors[0].text == "invalid"

    def test_validate_dict_failure(self, validator_setup):
        validator, segmenter, provider, symspell, rule_validator = validator_setup
        segmenter.segment_syllables.return_value = ["unknown"]
        rule_validator.validate.return_value = True  # Passes rule check
        provider.is_valid_syllable.return_value = False  # Fails dict check
        provider.is_valid_word.return_value = False  # Also fails word check

        suggestion = MagicMock()
        suggestion.term = "known"
        symspell.lookup.return_value = [suggestion]

        errors = validator.validate("unknown")
        assert len(errors) == 1
        assert errors[0].text == "unknown"

    def test_validate_skip_empty_punctuation(self, validator_setup):
        validator, segmenter, _, _, _ = validator_setup
        segmenter.segment_syllables.return_value = ["", "...", "valid"]
        # Make "valid" actually valid to avoid errors
        validator.repository.is_valid_syllable.return_value = True
        validator.repository.get_syllable_frequency.return_value = (
            100  # High frequency to pass threshold
        )
        validator.syllable_rule_validator.validate.return_value = True

        errors = validator.validate("...valid")
        assert len(errors) == 0

    def test_validate_find_fallback(self, validator_setup):
        validator, segmenter, _, _, _ = validator_setup
        # Text: "ab a"
        # Segments: "a", "b", "a"
        segmenter.segment_syllables.return_value = ["a", "b", "a"]
        validator.repository.is_valid_syllable.return_value = True
        validator.repository.get_syllable_frequency.return_value = (
            100  # High frequency to pass threshold
        )
        validator.syllable_rule_validator.validate.return_value = True

        # This tests the idx finding loop logic implicitly
        errors = validator.validate("ab a")
        assert len(errors) == 0


class TestWordValidatorCoverage(MockIsMyanmarMixin):
    @pytest.fixture
    def validator_setup(self):
        config = SpellCheckerConfig()
        segmenter = MagicMock()
        provider = MagicMock()
        symspell = MagicMock()
        # WordValidator expects word_repository and syllable_repository
        validator = WordValidator(config, segmenter, provider, provider, symspell)
        return validator, segmenter, provider, symspell

    def test_validate_word_error_compound_lookup(self, validator_setup):
        validator, segmenter, provider, symspell = validator_setup
        segmenter.segment_words.return_value = ["wrongword"]
        segmenter.segment_syllables.return_value = ["wrong", "word"]
        provider.is_valid_word.return_value = False

        # Mock compound lookup returning a valid split
        # format: (terms_list, distance, count) ?? check symspell source if possible,
        # usually returns object or tuple. Validator expects:
        # top_split = compound_suggestions[0] -> tuple likely?
        # Validator accesses top_split[1] (distance) and top_split[0] (terms)
        symspell.lookup_compound.return_value = [(["wrong", "word"], 1)]

        suggestion = MagicMock()
        suggestion.term = "rightword"
        symspell.lookup.return_value = [suggestion]

        errors = validator.validate("wrongword")
        # If split matches original, it continues to error creation
        assert len(errors) == 1
        assert isinstance(errors[0], WordError)

    def test_validate_word_error_compound_lookup_skipped(self, validator_setup):
        validator, segmenter, provider, symspell = validator_setup
        segmenter.segment_words.return_value = ["typo"]
        provider.is_valid_word.return_value = False

        # Mock compound lookup returning DIFFERENT split (valid correction found via compound)
        # Wait, logic says:
        # if top_split[1] == 0 and "".join(top_split[0]) != word: continue
        # Meaning if we found a perfect split that is different from input,
        # we treat input as potentially valid split error??
        # actually "continue" means skip error creation?
        # Wait. "continue" in the loop over words means "move to next word".
        # So if symspell finds a perfect compound split that resolves to DIFFERENT text,
        # it implies the input word might be just a concatenation error,
        # but logic skips reporting it as WordError?
        # OR it assumes the segmenter made a mistake and we shouldn't flag it as a
        # specific WordError here?
        # This logic is a bit ambiguous without context, but let's test coverage.

        # Mock compound lookup returning SAME split (valid correction found via compound)
        # Compound validation now requires 3-tuple (parts, distance, score) and
        # at least one part must be a valid word. With mocked is_valid_word=False,
        # the compound check fails and a WordError is created.

        symspell.lookup_compound.return_value = [(["ty", "po"], 0)]
        # 2-tuple fails the len >= 3 check in _is_valid_compound

        errors = validator.validate("typo")
        assert len(errors) == 1

    def test_validate_does_not_skip_segment_merge_when_strong_candidate_exists(
        self, validator_setup
    ):
        validator, segmenter, provider, symspell = validator_setup
        word = "ကြိးကြပ်"
        segmenter.segment_words.return_value = [word]
        segmenter.segment_syllables.return_value = ["ကြိး", "ကြပ်"]

        def _is_valid(token: str) -> bool:
            return token in {"ကြိး", "ကြပ်", "ကြီးကြပ်"}

        provider.is_valid_word.side_effect = _is_valid
        provider.get_word_frequency.side_effect = lambda token: (
            9000 if token == "ကြီးကြပ်" else (281 if token == "ကြိး" else 0)
        )

        symspell.lookup_compound.return_value = []
        strong_candidate = MagicMock()
        strong_candidate.term = "ကြီးကြပ်"
        strong_candidate.edit_distance = 1
        strong_candidate.frequency = 9000
        symspell.lookup.return_value = [strong_candidate]

        errors = validator.validate(word)

        assert len(errors) == 1
        assert isinstance(errors[0], WordError)
        assert errors[0].text == word

    def test_validate_keeps_segment_merge_skip_without_strong_candidate(self, validator_setup):
        validator, segmenter, provider, symspell = validator_setup
        word = "ကြိးကြပ်"
        segmenter.segment_words.return_value = [word]
        segmenter.segment_syllables.return_value = ["ကြိး", "ကြပ်"]

        provider.is_valid_word.side_effect = lambda token: token in {"ကြိး", "ကြပ်"}
        provider.get_word_frequency.return_value = 0

        symspell.lookup_compound.return_value = []
        symspell.lookup.return_value = []

        errors = validator.validate(word)

        assert errors == []

    def test_validate_does_not_accept_compound_when_strong_candidate_exists(self, validator_setup):
        validator, segmenter, provider, symspell = validator_setup
        word = "အရေးကြိး"
        segmenter.segment_words.return_value = [word]
        segmenter.segment_syllables.return_value = ["အ", "ရေး", "ကြိး"]

        def _is_valid(token: str) -> bool:
            return token in {"အ", "ရေး", "ကြိး", "အရေး", "အရေးကြီး"}

        provider.is_valid_word.side_effect = _is_valid
        provider.get_word_frequency.side_effect = lambda token: 55247 if token == "အရေးကြီး" else 1

        # Compound check would normally accept this split.
        symspell.lookup_compound.return_value = [(["အရေး", "ကြိး"], 0, 1)]
        strong_candidate = MagicMock()
        strong_candidate.term = "အရေးကြီး"
        strong_candidate.edit_distance = 1
        strong_candidate.frequency = 55247
        symspell.lookup.return_value = [strong_candidate]

        errors = validator.validate(word)

        assert len(errors) == 1
        assert isinstance(errors[0], WordError)
        assert errors[0].text == word

    def test_validate_accepts_compound_when_no_strong_candidate(self, validator_setup):
        validator, segmenter, provider, symspell = validator_setup
        word = "အရေးကြိး"
        segmenter.segment_words.return_value = [word]
        segmenter.segment_syllables.return_value = ["အ", "ရေး", "ကြိး"]

        provider.is_valid_word.side_effect = lambda token: token in {"အ", "ရေး", "ကြိး", "အရေး"}
        provider.get_word_frequency.return_value = 1

        symspell.lookup_compound.return_value = [(["အရေး", "ကြိး"], 0, 1)]
        # No strong candidate to override compound acceptance.
        symspell.lookup.return_value = []

        errors = validator.validate(word)

        assert errors == []


class TestContextValidatorCoverage(MockIsMyanmarMixin):
    """
    Tests for ContextValidator (Strategy Pattern Orchestrator).

    Note: ContextValidator has been refactored from monolithic implementation
    to strategy-based orchestrator. These tests now test the orchestration
    behavior rather than specific validation logic (which is in strategies).
    """

    @pytest.fixture
    def validator_setup(self):
        """Set up a ContextValidator with mock strategies."""
        config = SpellCheckerConfig(use_context_checker=True, use_ner=True, use_phonetic=True)
        segmenter = MagicMock()
        name_heuristic = MagicMock()

        # Create mock strategies
        mock_strategy = MagicMock()
        mock_strategy.priority.return_value = 50
        mock_strategy.validate.return_value = []

        validator = ContextValidator(
            config, segmenter, strategies=[mock_strategy], name_heuristic=name_heuristic
        )
        return validator, segmenter, name_heuristic, mock_strategy

    def test_validate_with_strategies(self, validator_setup):
        """Test that validator calls strategies during validation."""
        validator, segmenter, name_heuristic, mock_strategy = validator_setup
        segmenter.segment_sentences.return_value = ["a b"]
        segmenter.segment_words.return_value = ["a", "b"]
        name_heuristic.analyze_sentence.return_value = [False, False]

        errors = validator.validate("a b")
        # Strategy should have been called
        mock_strategy.validate.assert_called()
        assert len(errors) == 0

    def test_validate_skip_names(self, validator_setup):
        """Test that validation skips words marked as names."""
        validator, segmenter, name_heuristic, mock_strategy = validator_setup
        segmenter.segment_sentences.return_value = ["U Ba"]
        segmenter.segment_words.return_value = ["U", "Ba"]
        # Mark "Ba" as name
        name_heuristic.analyze_sentence.return_value = [False, True]

        errors = validator.validate("U Ba")
        assert len(errors) == 0

    def test_validate_empty_strategies(self):
        """Test that validator returns empty list when no strategies configured."""
        config = SpellCheckerConfig()
        segmenter = MagicMock()

        # No strategies
        validator = ContextValidator(config, segmenter, strategies=[])

        segmenter.segment_sentences.return_value = ["a b c"]
        segmenter.segment_words.return_value = ["a", "b", "c"]

        errors = validator.validate("a b c")
        assert len(errors) == 0


class TestValidatorStaticMethods:
    def test_validate_is_myanmar_text_helper(self):
        # Test is_myanmar_text helper from constants (replaces removed Validator.is_myanmar)
        from myspellchecker.core.constants import is_myanmar_text

        assert is_myanmar_text("က") is True
        assert is_myanmar_text("a") is False
        assert is_myanmar_text("") is False

    def test_validate_is_punctuation_check(self):
        # Test static method
        assert Validator.is_punctuation("။") is True
        assert Validator.is_punctuation("a") is False
