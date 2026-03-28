"""Extended tests for text/morphology.py to boost coverage."""

import pytest


class TestNumeralDetection:
    """Test numeral detection functions."""

    def test_is_numeral_word_myanmar_digits(self):
        """Test is_numeral_word with Myanmar digits."""
        from myspellchecker.text.morphology import is_numeral_word

        assert is_numeral_word("၁၂၃") is True
        assert is_numeral_word("၀") is True
        assert is_numeral_word("၉") is True

    def test_is_numeral_word_numeral_words(self):
        """Test is_numeral_word with numeral words."""
        from myspellchecker.text.morphology import is_numeral_word

        assert is_numeral_word("တစ်") is True
        assert is_numeral_word("နှစ်") is True
        assert is_numeral_word("သုံး") is True

    def test_is_numeral_word_non_numeral(self):
        """Test is_numeral_word with non-numeral words."""
        from myspellchecker.text.morphology import is_numeral_word

        assert is_numeral_word("") is False
        assert is_numeral_word("စား") is False
        assert is_numeral_word("မြန်မာ") is False

    def test_get_numeral_pos_guess_digits(self):
        """Test get_numeral_pos_guess with digits returns NUM tag."""
        from myspellchecker.text.morphology import get_numeral_pos_guess

        guess = get_numeral_pos_guess("၁၂၃")
        assert guess is not None
        assert guess.tag == "NUM"
        assert guess.confidence == 0.99

    def test_get_numeral_pos_guess_numeral_word(self):
        """Test get_numeral_pos_guess with numeral word returns NUM tag."""
        from myspellchecker.text.morphology import get_numeral_pos_guess

        guess = get_numeral_pos_guess("တစ်")
        assert guess is not None
        assert guess.tag == "NUM"
        assert guess.confidence == 0.95

    def test_get_numeral_pos_guess_non_numeral(self):
        """Test get_numeral_pos_guess with non-numeral word returns None."""
        from myspellchecker.text.morphology import get_numeral_pos_guess

        guess = get_numeral_pos_guess("စား")
        assert guess is None


class TestMorphologyAnalyzer:
    """Test MorphologyAnalyzer class."""

    def test_guess_pos_empty(self):
        """Test guessing POS with empty word returns empty set."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()
        result = analyzer.guess_pos("")
        assert result == set()

    def test_guess_pos_best_numeral(self):
        """Test best POS guessing with numeral returns NUM."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()
        result = analyzer.guess_pos_best("၁၂၃")
        assert result == "NUM"

    def test_guess_pos_ranked_empty(self):
        """Test ranked POS guessing with empty word returns empty list."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()
        result = analyzer.guess_pos_ranked("")
        assert result == []

    def test_guess_pos_multi_numeral(self):
        """Test guess_pos_multi with numeral includes NUM tag."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()
        tags, confidence, source = analyzer.guess_pos_multi("တစ်")
        assert "NUM" in tags
        assert isinstance(confidence, float)
        assert isinstance(source, str)

    def test_analyze_word(self):
        """Test word analysis returns correct original."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()
        analysis = analyzer.analyze_word("မြန်မာ")
        assert analysis.original == "မြန်မာ"

    def test_analyze_word_empty(self):
        """Test word analysis with empty word."""
        from myspellchecker.text.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()
        analysis = analyzer.analyze_word("")
        assert analysis.original == ""


class TestTextNormalization:
    """Test text normalization utilities."""

    def test_normalize_zero_width(self):
        """Test zero-width character removal."""
        from myspellchecker.text.normalize import c_remove_zero_width

        text_with_zw = "test\u200btest"
        result = c_remove_zero_width(text_with_zw)
        assert "\u200b" not in result

    def test_is_myanmar_text(self):
        """Test is_myanmar_text function."""
        from myspellchecker.text.normalize import is_myanmar_text

        assert is_myanmar_text("မြန်မာ") is True
        assert is_myanmar_text("Hello") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
