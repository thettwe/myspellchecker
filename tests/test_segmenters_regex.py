"""Tests for segmenters/regex.py."""

import pytest

from myspellchecker.core.exceptions import TokenizationError


class TestRegexSegmenter:
    """Test RegexSegmenter class."""

    def test_segment_words_empty_raises(self):
        """Test segmenting empty string raises ValueError."""
        from myspellchecker.segmenters.regex import RegexSegmenter

        segmenter = RegexSegmenter()
        with pytest.raises(TokenizationError, match="empty"):
            segmenter.segment_words("")

    def test_segment_words_not_implemented(self):
        """Test segment_words raises NotImplementedError for Myanmar text."""
        from myspellchecker.segmenters.regex import RegexSegmenter

        segmenter = RegexSegmenter()
        with pytest.raises(NotImplementedError, match="does not support word segmentation"):
            segmenter.segment_words("မြန်မာပြည်")

    def test_segment_syllables_empty_raises(self):
        """Test segmenting empty string to syllables raises ValueError."""
        from myspellchecker.segmenters.regex import RegexSegmenter

        segmenter = RegexSegmenter()
        with pytest.raises(TokenizationError, match="empty"):
            segmenter.segment_syllables("")

    def test_segment_and_tag_not_implemented(self):
        """Test segment_and_tag raises NotImplementedError."""
        from myspellchecker.segmenters.regex import RegexSegmenter

        segmenter = RegexSegmenter()
        with pytest.raises(NotImplementedError):
            segmenter.segment_and_tag("မြန်မာ")

    def test_validate_input_with_none(self):
        """Test _validate_input raises TypeError for non-string."""
        from myspellchecker.segmenters.regex import RegexSegmenter

        segmenter = RegexSegmenter()
        with pytest.raises(TypeError, match="Expected str"):
            segmenter.segment_words(None)


class TestPunctuationSegmentation:
    """Tests for punctuation splitting during syllable segmentation."""

    def test_punctuation_splitting(self):
        from myspellchecker.segmenters.regex import RegexSegmenter

        segmenter = RegexSegmenter()
        tokens = segmenter.segment_syllables("များ၊")

        assert "များ၊" not in tokens, "Punctuation '၊' stuck to syllable 'များ'"
        assert "များ" in tokens
        assert "၊" in tokens

    def test_multiple_punctuation(self):
        from myspellchecker.segmenters.regex import RegexSegmenter

        segmenter = RegexSegmenter()
        tokens = segmenter.segment_syllables("ဟုတ်။ကဲ့။")

        assert "ဟုတ်" in tokens
        assert "။" in tokens
        assert "ကဲ့" in tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
