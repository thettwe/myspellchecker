"""Edge case tests for segmenters/default.py targeting uncovered lines."""

from unittest.mock import patch

import pytest

from myspellchecker.core.exceptions import TokenizationError


class TestDefaultSegmenterInit:
    """Tests for DefaultSegmenter initialization edge cases."""

    def test_init_with_unsupported_engine(self):
        """Test init raises ValueError for unsupported engine."""
        from myspellchecker.segmenters.default import DefaultSegmenter

        with pytest.raises(TokenizationError, match="Unsupported word_engine"):
            DefaultSegmenter(word_engine="invalid_engine")

    def test_init_with_crf_engine(self):
        """Test init with CRF engine."""
        from myspellchecker.segmenters.default import DefaultSegmenter

        try:
            segmenter = DefaultSegmenter(word_engine="crf")
            assert segmenter.word_engine == "crf"
        except (ImportError, RuntimeError):
            # CRF model may not be available in test environment
            pass


class TestDefaultSegmenterValidation:
    """Tests for input validation in DefaultSegmenter."""

    def test_validate_input_non_string(self):
        """Test _validate_input raises TypeError for non-string."""
        from myspellchecker.segmenters.default import DefaultSegmenter

        segmenter = DefaultSegmenter()
        with pytest.raises(TypeError, match="Expected str"):
            segmenter._validate_input(123)

    def test_validate_input_empty_string(self):
        """Test _validate_input raises ValueError for empty string."""
        from myspellchecker.segmenters.default import DefaultSegmenter

        segmenter = DefaultSegmenter()
        with pytest.raises(TokenizationError, match="empty"):
            segmenter._validate_input("")

    def test_validate_input_whitespace_only(self):
        """Test _validate_input raises ValueError for whitespace."""
        from myspellchecker.segmenters.default import DefaultSegmenter

        segmenter = DefaultSegmenter()
        with pytest.raises(TokenizationError, match="empty"):
            segmenter._validate_input("   ")


class TestLoadCustomDictionary:
    """Tests for load_custom_dictionary method."""

    def test_load_custom_dictionary_with_tokenizer(self):
        """Test load_custom_dictionary with tokenizer that supports it."""
        from myspellchecker.segmenters.default import DefaultSegmenter

        segmenter = DefaultSegmenter()

        if segmenter.word_tokenizer and hasattr(segmenter.word_tokenizer, "add_custom_words"):
            segmenter.load_custom_dictionary(["custom_word"])
        else:
            # Just verify method doesn't crash
            segmenter.load_custom_dictionary(["custom_word"])


class TestSegmentSentences:
    """Tests for segment_sentences method."""

    def test_segment_sentences_basic(self):
        """Test segment_sentences with basic text."""
        from myspellchecker.segmenters.default import DefaultSegmenter

        segmenter = DefaultSegmenter()
        result = segmenter.segment_sentences("ပထမစာကြောင်း။ ဒုတိယစာကြောင်း။")

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_segment_sentences_no_separator(self):
        """Test segment_sentences with no sentence separator."""
        from myspellchecker.segmenters.default import DefaultSegmenter

        segmenter = DefaultSegmenter()
        result = segmenter.segment_sentences("စာကြောင်းတစ်ခု")

        assert isinstance(result, list)
        assert len(result) == 1

    def test_segment_sentences_multiple(self):
        """Test segment_sentences with multiple sentences."""
        from myspellchecker.segmenters.default import DefaultSegmenter

        segmenter = DefaultSegmenter()
        text = "ပထမ။ ဒုတိယ။ တတိယ။"
        result = segmenter.segment_sentences(text)

        assert isinstance(result, list)
        assert len(result) >= 2


class TestSegmentSyllables:
    """Tests for segment_syllables method."""

    def test_segment_syllables_basic(self):
        """Test segment_syllables with basic text."""
        from myspellchecker.segmenters.default import DefaultSegmenter

        segmenter = DefaultSegmenter()
        result = segmenter.segment_syllables("မြန်မာ")

        assert isinstance(result, list)
        assert len(result) >= 1


class TestRegexSegmenter:
    """Tests for internal regex segmenter."""

    def test_regex_segmenter_is_initialized(self):
        """Test regex_segmenter is always initialized."""
        from myspellchecker.segmenters.default import DefaultSegmenter

        segmenter = DefaultSegmenter()
        assert segmenter.regex_segmenter is not None


class TestWordTokenizerFailure:
    """Tests for WordTokenizer initialization failure."""

    @patch("myspellchecker.segmenters.default.WordTokenizer")
    def test_word_tokenizer_init_failure(self, mock_tokenizer_class):
        """Test handling of WordTokenizer initialization failure."""
        from myspellchecker.segmenters.default import DefaultSegmenter

        mock_tokenizer_class.side_effect = RuntimeError("Init failed")

        # DefaultSegmenter uses lazy init, so construction succeeds
        segmenter = DefaultSegmenter(word_engine="myword")
        # Trigger lazy init which raises RuntimeError (re-raised after logging)
        with pytest.raises(RuntimeError, match="Init failed"):
            segmenter._ensure_word_segmenter_initialized()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
