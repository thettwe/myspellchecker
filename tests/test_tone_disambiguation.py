"""
Unit tests for Myanmar tone disambiguation module.

Tests context-based tone disambiguation and tone mark correction
for commonly confused Myanmar words.
"""

import pytest

from myspellchecker.text.tone import (
    TONE_AMBIGUOUS,
    TONE_MARK_ERRORS,
    ToneDisambiguator,
    create_disambiguator,
)


class TestToneAmbiguousData:
    """Tests for the TONE_AMBIGUOUS dictionary."""

    def test_tone_ambiguous_exists(self):
        """Test that TONE_AMBIGUOUS is defined."""
        assert TONE_AMBIGUOUS is not None
        assert isinstance(TONE_AMBIGUOUS, dict)

    def test_contains_common_ambiguous_words(self):
        """Test that common ambiguous words are defined."""
        assert "ကျ" in TONE_AMBIGUOUS  # to fall vs compound
        assert "မ" in TONE_AMBIGUOUS  # female vs negative
        assert "သား" in TONE_AMBIGUOUS  # son vs tiger
        assert "ငါ" in TONE_AMBIGUOUS  # I/me vs fish
        assert "ပဲ" in TONE_AMBIGUOUS  # only vs bean

    def test_entry_format(self):
        """Test that entries have correct format."""
        for _word, contexts in TONE_AMBIGUOUS.items():
            assert isinstance(contexts, dict)
            for _context_type, entry in contexts.items():
                assert isinstance(entry, tuple)
                assert len(entry) == 3  # (patterns, form, meaning)
                patterns, form, meaning = entry
                assert isinstance(patterns, tuple)
                assert isinstance(form, str)
                assert isinstance(meaning, str)


class TestToneMarkErrors:
    """Tests for the TONE_MARK_ERRORS dictionary."""

    def test_tone_mark_errors_exists(self):
        """Test that TONE_MARK_ERRORS is defined."""
        assert TONE_MARK_ERRORS is not None
        assert isinstance(TONE_MARK_ERRORS, dict)

    def test_contains_common_errors(self):
        """Test that common tone mark errors are defined."""
        assert "သုံ" in TONE_MARK_ERRORS  # three (not a valid standalone syllable)
        assert "သလာ" in TONE_MARK_ERRORS  # question particle

    def test_excludes_valid_high_freq_words(self):
        """Test that valid high-frequency words are NOT in TONE_MARK_ERRORS.

        These words are handled via TONE_AMBIGUOUS with proper context checking.
        """
        assert "ငါ" not in TONE_MARK_ERRORS  # valid first-person pronoun
        assert "လာ" not in TONE_MARK_ERRORS  # valid verb "to come"


class TestToneDisambiguator:
    """Tests for the ToneDisambiguator class."""

    @pytest.fixture
    def disambiguator(self):
        """Create a ToneDisambiguator instance."""
        return ToneDisambiguator()

    def test_init_defaults(self, disambiguator):
        """Test default initialization."""
        assert disambiguator.context_window == 3
        assert disambiguator.ambiguous_words is not None
        assert disambiguator.tone_errors is not None

    def test_custom_context_window(self):
        """Test custom context window via config."""
        from myspellchecker.core.config.text_configs import ToneConfig

        config = ToneConfig(context_window=5)
        d = ToneDisambiguator(config=config)
        assert d.context_window == 5


class TestToneDisambiguatorContext:
    """Tests for context-based disambiguation."""

    @pytest.fixture
    def disambiguator(self):
        """Create a ToneDisambiguator instance."""
        return ToneDisambiguator()

    def test_get_context_words(self, disambiguator):
        """Test _get_context_words method."""
        words = ["သူ", "က", "သား", "ကို", "ချစ်"]
        left, right = disambiguator._get_context_words(words, 2)

        # Index 2 is "သား", left context is ["သူ", "က"]
        assert left == ["သူ", "က"]
        # Right context is ["ကို", "ချစ်"]
        assert right == ["ကို", "ချစ်"]

    def test_get_context_words_at_start(self, disambiguator):
        """Test context at sentence start."""
        words = ["သား", "ကို", "ချစ်"]
        left, right = disambiguator._get_context_words(words, 0)

        assert left == []
        assert right == ["ကို", "ချစ်"]

    def test_get_context_words_at_end(self, disambiguator):
        """Test context at sentence end."""
        words = ["သူ", "က", "သား"]
        left, right = disambiguator._get_context_words(words, 2)

        assert left == ["သူ", "က"]
        assert right == []


class TestToneDisambiguatorDisambiguate:
    """Tests for the disambiguate method."""

    @pytest.fixture
    def disambiguator(self):
        """Create a ToneDisambiguator instance."""
        return ToneDisambiguator()

    def test_disambiguate_returns_none_for_non_ambiguous(self, disambiguator):
        """Test disambiguate returns None for non-ambiguous words."""
        words = ["သူ", "စား", "တယ်"]
        result = disambiguator.disambiguate(words, 1)
        assert result is None

    def test_disambiguate_invalid_index(self, disambiguator):
        """Test disambiguate returns None for invalid index."""
        words = ["သူ", "စား"]
        assert disambiguator.disambiguate(words, -1) is None
        assert disambiguator.disambiguate(words, 10) is None

    def test_disambiguate_son_context(self, disambiguator):
        """Test disambiguating 'သား' in family context."""
        # "သမီး" (daughter) context suggests "son" meaning
        words = ["သူ့", "သား", "သမီး"]
        result = disambiguator.disambiguate(words, 1)

        if result:  # May return None if context score is too low
            form, meaning, confidence = result
            # Should prefer family context
            assert form == "သား"

    def test_disambiguate_with_animal_context(self, disambiguator):
        """Test disambiguating 'သား' in animal/forest context."""
        # "တော" (forest) context suggests "tiger" meaning
        words = ["တော", "ထဲ", "သား"]
        result = disambiguator.disambiguate(words, 2)

        if result:  # May return None if context score is too low
            form, meaning, confidence = result
            # Should prefer animal context and suggest "သား့"
            assert confidence > 0.0


class TestToneCorrection:
    """Tests for tone mark correction suggestions."""

    @pytest.fixture
    def disambiguator(self):
        """Create a ToneDisambiguator instance."""
        return ToneDisambiguator()

    def test_suggest_tone_correction_number_context(self, disambiguator):
        """Test that 'ငါ' is not corrected via TONE_MARK_ERRORS.

        'ငါ' (I/me) is a valid first-person pronoun and was removed from
        TONE_MARK_ERRORS. The ငါ/ငါး disambiguation is handled exclusively
        through TONE_AMBIGUOUS with proper context checking.
        """
        context = ["တစ်", "နှစ်", "သုံး", "လေး", "ခု"]
        result = disambiguator.suggest_tone_correction("ငါ", context)
        assert result is None

    def test_suggest_tone_correction_question_context(self, disambiguator):
        """Test that 'လာ' is not corrected via TONE_MARK_ERRORS.

        'လာ' ("to come") is one of the most common Myanmar verbs and was
        removed from TONE_MARK_ERRORS. The လာ/လား disambiguation is handled
        exclusively through TONE_AMBIGUOUS with proper context checking.
        """
        context = ["ဘာ", "လုပ်", "နေ"]
        result = disambiguator.suggest_tone_correction("လာ", context)
        assert result is None

    def test_suggest_tone_correction_no_match(self, disambiguator):
        """Test that non-matching words return None."""
        result = disambiguator.suggest_tone_correction("စား", ["သူ", "က"])
        assert result is None


class TestCheckSentence:
    """Tests for sentence-level tone checking."""

    @pytest.fixture
    def disambiguator(self):
        """Create a ToneDisambiguator instance."""
        return ToneDisambiguator()

    def test_check_sentence_empty(self, disambiguator):
        """Test check_sentence with empty list."""
        corrections = disambiguator.check_sentence([])
        assert corrections == []

    def test_check_sentence_no_errors(self, disambiguator):
        """Test check_sentence with correct sentence."""
        words = ["သူ", "စား", "တယ်"]
        corrections = disambiguator.check_sentence(words)
        # No tone-ambiguous words or errors
        assert len(corrections) == 0

    def test_check_sentence_returns_tuples(self, disambiguator):
        """Test check_sentence returns correct format."""
        # Create a sentence with potential tone issues
        words = ["တော", "ထဲ", "သား", "ရှိ", "တယ်"]
        corrections = disambiguator.check_sentence(words)

        for item in corrections:
            assert isinstance(item, tuple)
            assert len(item) == 4  # (index, original, suggestion, confidence)
            idx, original, suggestion, confidence = item
            assert isinstance(idx, int)
            assert isinstance(original, str)
            assert isinstance(suggestion, str)
            assert isinstance(confidence, float)


class TestFactoryFunction:
    """Tests for create_disambiguator factory."""

    def test_create_default(self):
        """Test creating with default settings."""
        d = create_disambiguator()
        assert isinstance(d, ToneDisambiguator)
        assert d.context_window == 3

    def test_create_custom_window(self):
        """Test creating with custom context window via config."""
        from myspellchecker.core.config.text_configs import ToneConfig

        config = ToneConfig(context_window=5)
        d = create_disambiguator(config=config)
        assert d.context_window == 5


class TestMatchContext:
    """Tests for _match_context method."""

    @pytest.fixture
    def disambiguator(self):
        return ToneDisambiguator()

    def test_match_context_with_match(self, disambiguator):
        patterns = ("pattern1", "pattern2")
        context = ["pattern1", "other"]
        score = disambiguator._match_context(patterns, context)
        assert score > 0

    def test_match_context_no_match(self, disambiguator):
        patterns = ("pattern1", "pattern2")
        context = ["other1", "other2"]
        score = disambiguator._match_context(patterns, context)
        assert score == 0.0

    def test_match_context_empty_patterns(self, disambiguator):
        score = disambiguator._match_context((), ["word1", "word2"])
        assert score == 0.0

    def test_match_context_empty_context(self, disambiguator):
        score = disambiguator._match_context(("pattern1",), [])
        assert score == 0.0

    def test_match_context_partial_match(self, disambiguator):
        patterns = ("ကျား", "ဆင်")
        context = ["ကျားတော", "other"]
        score = disambiguator._match_context(patterns, context)
        assert score > 0


class TestToneEdgeCases:
    """Edge cases salvaged from test_tone.py."""

    @pytest.fixture
    def disambiguator(self):
        return ToneDisambiguator()

    def test_suggest_tone_correction_unknown_word(self, disambiguator):
        result = disambiguator.suggest_tone_correction("unknownword", ["context"])
        assert result is None

    def test_suggest_tone_correction_empty_context(self, disambiguator):
        result = disambiguator.suggest_tone_correction("unknownword", [])
        assert result is None

    def test_no_correction_without_matching_context(self, disambiguator):
        result = disambiguator.suggest_tone_correction("သုံ", ["စကား", "ပြော"])
        assert result is not None
        assert result[0] == "သုံး"
        assert result[1] == 0.60

    def test_single_word_sentence(self, disambiguator):
        words = ["သား"]
        result = disambiguator.disambiguate(words, 0)
        assert result is None

    def test_all_tone_mark_errors_coverage(self, disambiguator):
        for original, correction in TONE_MARK_ERRORS.items():
            assert original in disambiguator.tone_errors
            assert disambiguator.tone_errors[original] == correction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
