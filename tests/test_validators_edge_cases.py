from unittest.mock import Mock, patch

import pytest

from myspellchecker.algorithms import NgramContextChecker, SymSpell
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.context_validator import ContextValidator
from myspellchecker.core.validators import SyllableValidator, WordValidator
from myspellchecker.segmenters import Segmenter


class TestValidatorsExtra:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = SpellCheckerConfig()
        self.mock_segmenter = Mock(spec=Segmenter)
        self.mock_provider = Mock()  # Implements both SyllableRepository and WordRepository
        self.mock_symspell = Mock(spec=SymSpell)
        self.mock_context_checker = Mock(spec=NgramContextChecker)

    def test_syllable_validator_particle_typo_path(self):
        validator = SyllableValidator(
            self.config, self.mock_segmenter, self.mock_provider, self.mock_symspell
        )

        # Mock segmenter to return a known particle typo "တယ"
        self.mock_segmenter.segment_syllables.return_value = ["တယ"]
        # Mock provider to say it's NOT in dictionary
        self.mock_provider.is_valid_syllable.return_value = False
        self.mock_provider.is_valid_word.return_value = False

        errors = validator.validate("တယ")
        assert len(errors) == 1
        assert errors[0].error_type == "particle_typo"
        assert errors[0].suggestions == ["တယ်"]

    def test_word_validator_context_reranking(self):
        """Test that WordValidator uses suggestion_strategy for suggestions."""
        from myspellchecker.algorithms.ranker import SuggestionData
        from myspellchecker.algorithms.suggestion_strategy import SuggestionResult

        # Create a mock suggestion strategy
        mock_strategy = Mock()

        # Configure strategy to return suggestions with context-aware ranking
        # "သွား" should be first because it has higher context probability
        suggestion_data = [
            SuggestionData(term="သွား", edit_distance=1, frequency=100, source="symspell"),
            SuggestionData(term="ရှိ", edit_distance=1, frequency=50, source="symspell"),
        ]
        mock_result = SuggestionResult(
            suggestions=suggestion_data,
            strategy_name="composite",
        )
        mock_strategy.suggest.return_value = mock_result

        # Mock symspell lookups to return empty lists (no matches)
        self.mock_symspell.lookup_compound.return_value = []
        self.mock_symspell.lookup.return_value = []

        validator = WordValidator(
            self.config,
            self.mock_segmenter,
            self.mock_provider,
            self.mock_provider,
            self.mock_symspell,
            self.mock_context_checker,
            suggestion_strategy=mock_strategy,
        )

        # Setup scenario: "သူ ကျောင်း" where "ကျောင်း" is OOV (for test purposes)
        self.mock_segmenter.segment_words.return_value = ["သူ", "ကျောင်း"]
        self.mock_segmenter.segment_syllables.return_value = ["ကျောင်း"]

        def is_valid_side_effect(w):
            return w == "သူ"

        self.mock_provider.is_valid_word.side_effect = is_valid_side_effect

        errors = validator.validate("သူ ကျောင်း")
        assert len(errors) == 1
        # Verify "သွား" is first suggestion (pre-ranked by strategy)
        assert errors[0].suggestions[0] == "သွား"
        # Verify strategy was called
        mock_strategy.suggest.assert_called()

    def test_word_validator_oov_recovery(self):
        """Test that WordValidator uses suggestion_strategy for OOV recovery."""
        from myspellchecker.algorithms.ranker import SuggestionData
        from myspellchecker.algorithms.suggestion_strategy import SuggestionResult

        # Create a mock suggestion strategy that returns morphology-based suggestion
        mock_strategy = Mock()
        suggestion_data = [
            SuggestionData(
                term="ဆရာဝန်",
                edit_distance=0,
                frequency=100,
                source="morphology",
            ),
        ]
        mock_result = SuggestionResult(
            suggestions=suggestion_data,
            strategy_name="morphology",
        )
        mock_strategy.suggest.return_value = mock_result

        # Mock symspell lookups to return empty lists (no matches)
        self.mock_symspell.lookup_compound.return_value = []
        self.mock_symspell.lookup.return_value = []

        validator = WordValidator(
            self.config,
            self.mock_segmenter,
            self.mock_provider,
            self.mock_provider,
            self.mock_symspell,
            suggestion_strategy=mock_strategy,
        )

        # Mock "ဆရာဝန်များ" (Doctors) which is OOV but has root "ဆရာဝန်"
        word = "ဆရာဝန်များ"
        self.mock_segmenter.segment_words.return_value = [word]
        # FIX: Mock segment_syllables to return a list (so len() works)
        self.mock_segmenter.segment_syllables.return_value = ["ဆရာ", "ဝန်", "များ"]
        self.mock_provider.is_valid_word.return_value = False
        self.mock_provider.is_valid_syllable.return_value = False

        errors = validator.validate(word)
        assert len(errors) == 1
        # Root should be suggested via strategy
        assert "ဆရာဝန်" in errors[0].suggestions
        # Verify strategy was called
        mock_strategy.suggest.assert_called()

    def test_context_validator_skip_conditions(self):
        """Test ContextValidator skip conditions with strategy pattern."""
        # Create mock strategy
        mock_strategy = Mock()
        mock_strategy.priority.return_value = 50
        mock_strategy.validate.return_value = []

        validator = ContextValidator(self.config, self.mock_segmenter, strategies=[mock_strategy])

        # Case 1: Empty strategies returns empty list (tested via separate validator)
        validator_no_strategies = ContextValidator(self.config, self.mock_segmenter, strategies=[])
        assert validator_no_strategies.validate("some text") == []

        # Case 2: Empty sentences
        self.mock_segmenter.segment_sentences.return_value = [""]
        self.mock_segmenter.segment_words.return_value = []
        assert validator.validate("") == []

    @patch("myspellchecker.core.validators.base.validate_word", return_value=True)
    def test_filter_suggestions(self, mock_val):
        validator = SyllableValidator(
            self.config, self.mock_segmenter, self.mock_provider, self.mock_symspell
        )
        res = validator._filter_suggestions(["valid", "invalid"])
        assert res == ["valid", "invalid"]

        mock_val.side_effect = [True, False]
        res = validator._filter_suggestions(["valid", "invalid"])
        assert res == ["valid"]
