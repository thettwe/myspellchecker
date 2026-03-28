"""Tests for the MorphemeSuggestionStrategy."""

from unittest.mock import MagicMock

import pytest

from myspellchecker.algorithms.morpheme_suggestion_strategy import (
    MorphemeSuggestionStrategy,
)


@pytest.fixture
def mock_symspell():
    """Create a mock SymSpell."""
    symspell = MagicMock()
    symspell.lookup.return_value = []
    return symspell


@pytest.fixture
def mock_compound_resolver():
    """Create a mock CompoundResolver with a segmenter."""
    resolver = MagicMock()
    resolver.segmenter = MagicMock()
    return resolver


@pytest.fixture
def mock_reduplication_engine():
    """Create a mock ReduplicationEngine with a segmenter."""
    engine = MagicMock()
    engine.segmenter = MagicMock()
    return engine


def _dict_check(valid_words):
    return lambda word: word in valid_words


class TestMorphemeCompoundCorrection:
    """Test compound morpheme correction."""

    def test_correct_right_morpheme(self, mock_symspell, mock_compound_resolver):
        """When left part is valid and right has a typo, correct the right part."""
        mock_compound_resolver.segmenter.segment_syllables.return_value = ["ကျောင်း", "သာ"]

        # SymSpell corrects "သာ" to "သား"
        correction = MagicMock()
        correction.term = "သား"
        correction.distance = 1
        correction.count = 300
        mock_symspell.lookup.return_value = [correction]

        strategy = MorphemeSuggestionStrategy(
            compound_resolver=mock_compound_resolver,
            reduplication_engine=None,
            symspell=mock_symspell,
            dictionary_check=_dict_check({"ကျောင်း", "သား", "ကျောင်းသား"}),
        )

        result = strategy.suggest("ကျောင်းသာ")

        assert len(result.suggestions) > 0
        assert result.suggestions[0].term == "ကျောင်းသား"
        assert result.suggestions[0].source == "morpheme"

    def test_correct_left_morpheme(self, mock_symspell, mock_compound_resolver):
        """When right part is valid and left has a typo, correct the left part."""
        mock_compound_resolver.segmenter.segment_syllables.return_value = ["ကျောင်", "သား"]

        # SymSpell corrects "ကျောင်" to "ကျောင်း"
        correction = MagicMock()
        correction.term = "ကျောင်း"
        correction.distance = 1
        correction.count = 200
        mock_symspell.lookup.return_value = [correction]

        strategy = MorphemeSuggestionStrategy(
            compound_resolver=mock_compound_resolver,
            reduplication_engine=None,
            symspell=mock_symspell,
            dictionary_check=_dict_check({"ကျောင်း", "သား", "ကျောင်းသား"}),
        )

        result = strategy.suggest("ကျောင်သား")

        assert len(result.suggestions) > 0
        assert result.suggestions[0].term == "ကျောင်းသား"

    def test_both_parts_invalid_no_suggestion(self, mock_symspell, mock_compound_resolver):
        """When both parts are invalid, no morpheme suggestions."""
        mock_compound_resolver.segmenter.segment_syllables.return_value = ["abc", "def"]

        strategy = MorphemeSuggestionStrategy(
            compound_resolver=mock_compound_resolver,
            reduplication_engine=None,
            symspell=mock_symspell,
            dictionary_check=_dict_check(set()),  # Nothing valid
        )

        result = strategy.suggest("abcdef")

        assert len(result.suggestions) == 0

    def test_both_parts_valid_no_suggestion(self, mock_symspell, mock_compound_resolver):
        """When both parts are valid, morpheme strategy doesn't apply."""
        mock_compound_resolver.segmenter.segment_syllables.return_value = ["ကျောင်း", "သား"]

        strategy = MorphemeSuggestionStrategy(
            compound_resolver=mock_compound_resolver,
            reduplication_engine=None,
            symspell=mock_symspell,
            dictionary_check=_dict_check({"ကျောင်း", "သား"}),
        )

        result = strategy.suggest("ကျောင်းသား")

        assert len(result.suggestions) == 0


class TestMorphemeReduplicationCorrection:
    """Test reduplication correction."""

    def test_correct_incomplete_reduplication(self, mock_symspell, mock_reduplication_engine):
        """When one half of reduplication has a typo, suggest the corrected form."""
        # "ကောင်းကောင်" → should suggest "ကောင်းကောင်း"
        mock_reduplication_engine.segmenter.segment_syllables.return_value = ["ကောင်း", "ကောင်"]

        strategy = MorphemeSuggestionStrategy(
            compound_resolver=None,
            reduplication_engine=mock_reduplication_engine,
            symspell=mock_symspell,
            dictionary_check=_dict_check({"ကောင်း"}),
        )

        result = strategy.suggest("ကောင်းကောင်")

        assert len(result.suggestions) > 0
        assert result.suggestions[0].term == "ကောင်းကောင်း"
        assert result.suggestions[0].source == "morpheme"

    def test_already_valid_reduplication_no_suggestion(
        self, mock_symspell, mock_reduplication_engine
    ):
        """When both halves are already identical, no suggestion needed."""
        mock_reduplication_engine.segmenter.segment_syllables.return_value = ["ကောင်း", "ကောင်း"]

        strategy = MorphemeSuggestionStrategy(
            compound_resolver=None,
            reduplication_engine=mock_reduplication_engine,
            symspell=mock_symspell,
            dictionary_check=_dict_check({"ကောင်း"}),
        )

        result = strategy.suggest("ကောင်းကောင်း")

        assert len(result.suggestions) == 0

    def test_odd_syllable_count_skipped(self, mock_symspell, mock_reduplication_engine):
        """Odd number of syllables can't be a simple reduplication."""
        mock_reduplication_engine.segmenter.segment_syllables.return_value = ["a", "b", "c"]

        strategy = MorphemeSuggestionStrategy(
            compound_resolver=None,
            reduplication_engine=mock_reduplication_engine,
            symspell=mock_symspell,
            dictionary_check=_dict_check(set()),
        )

        result = strategy.suggest("abc")

        assert len(result.suggestions) == 0


class TestMorphemeStrategyMetadata:
    """Test strategy metadata."""

    def test_strategy_name(self, mock_symspell):
        strategy = MorphemeSuggestionStrategy(
            compound_resolver=None,
            reduplication_engine=None,
            symspell=mock_symspell,
            dictionary_check=_dict_check(set()),
        )
        assert strategy.name == "morpheme"

    def test_no_context_support(self, mock_symspell):
        strategy = MorphemeSuggestionStrategy(
            compound_resolver=None,
            reduplication_engine=None,
            symspell=mock_symspell,
            dictionary_check=_dict_check(set()),
        )
        assert strategy.supports_context() is False

    def test_no_engines_returns_empty(self, mock_symspell):
        """With no engines, should return empty result."""
        strategy = MorphemeSuggestionStrategy(
            compound_resolver=None,
            reduplication_engine=None,
            symspell=mock_symspell,
            dictionary_check=_dict_check(set()),
        )

        result = strategy.suggest("anything")
        assert len(result.suggestions) == 0
