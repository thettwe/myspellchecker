import logging
import sys
from typing import List
from unittest.mock import MagicMock

import pytest

from myspellchecker.core.exceptions import MissingDependencyError, TokenizationError
from myspellchecker.segmenters.default import DefaultSegmenter
from myspellchecker.segmenters.regex import RegexSegmenter


# Mock myTokenize for testing fallback
class MockWordTokenizer:
    def __init__(self, engine: str = "CRF"):
        self.engine = engine

    def tokenize(self, text: str) -> List[str]:
        return text.split()  # Simple split for mock


class TestRegexSegmenter:
    def test_segment_syllables_basic(self):
        segmenter = RegexSegmenter()

        # Test 1: Simple sequence of consonants
        assert segmenter.segment_syllables("ကခဂဃင") == ["က", "ခ", "ဂ", "ဃ", "င"]

        # Test 2: Consonant + Vowel
        assert segmenter.segment_syllables("ကာကီ") == ["ကာ", "ကီ"]

        # Test 3: Syllable with medial
        # "Myan" (Ma+Ya+Na+Asat) is one syllable.
        assert segmenter.segment_syllables("မြန်မာ") == ["မြန်", "မာ"]

        # Test 4: Complex syllable
        # "Kyaung" (Ka+Ya+Au+Nga+Asat+Tone) is one syllable.
        assert segmenter.segment_syllables("ကျောင်း") == ["ကျောင်း"]

        # Test 5: Mixed text
        # New behavior groups non-Myanmar chars including whitespace
        assert segmenter.segment_syllables("Hello မြန်မာ123") == ["Hello ", "မြန်", "မာ", "123"]

    def test_segment_syllables_empty_input(self):
        segmenter = RegexSegmenter()
        with pytest.raises(TokenizationError, match="Text cannot be empty"):
            segmenter.segment_syllables("")
        with pytest.raises(TokenizationError, match="Text cannot be empty"):
            segmenter.segment_syllables("   ")
        with pytest.raises(TypeError, match="Expected str"):
            segmenter.segment_syllables(123)  # type: ignore

    def test_segment_words_raises_not_implemented(self):
        segmenter = RegexSegmenter()
        text = "မြန်မာစာ"
        # RegexSegmenter.segment_words raises NotImplementedError.
        # Users should use DefaultSegmenter for word segmentation.
        with pytest.raises(NotImplementedError, match="does not support word segmentation"):
            segmenter.segment_words(text)


class TestDefaultSegmenter:
    # Use caplog to check for warnings/errors
    @pytest.fixture(autouse=True)
    def caplog_fixture(self, caplog):
        caplog.set_level(logging.WARNING)

    def test_default_segmenter_syllables_uses_regex(self, monkeypatch):
        # DefaultSegmenter should always use RegexSegmenter for syllables
        with monkeypatch.context() as m:
            # Simulate myTokenize being present, so DefaultSegmenter tries to import it
            mock_mytokenize_module = MagicMock()
            mock_mytokenize_module.WordTokenizer = MockWordTokenizer
            m.setitem(sys.modules, "myTokenize", mock_mytokenize_module)

            segmenter = DefaultSegmenter()
            text = "ကခဂဃင"
            assert segmenter.segment_syllables(text) == ["က", "ခ", "ဂ", "ဃ", "င"]

    def test_default_segmenter_words_with_mytokenize(self, monkeypatch):
        # Directly inject a mock word_tokenizer that splits by space
        segmenter = DefaultSegmenter()
        mock_tokenizer = MockWordTokenizer()
        segmenter.word_tokenizer = mock_tokenizer
        segmenter._word_tokenizer_initialized = True

        text = "မြန်မာ စာပေ"
        words = segmenter.segment_words(text)
        assert words == ["မြန်မာ", "စာပေ"]  # Mock splits by space

    def test_default_segmenter_words_without_mytokenize(self):
        # Simulate word_tokenizer init failure by setting it to None and marking as initialized
        segmenter = DefaultSegmenter()
        segmenter.word_tokenizer = None
        segmenter._word_tokenizer_initialized = True  # Prevent lazy re-init

        text = "မြန်မာ စာပေ"
        with pytest.raises(MissingDependencyError, match="Word segmentation is not available"):
            segmenter.segment_words(text)

    def test_default_segmenter_syllables_without_mytokenize_still_works(self):
        # Syllable segmentation should work even if myTokenize is unavailable
        # (not used for syllables)
        segmenter = DefaultSegmenter()
        text = "ကခဂဃင"
        assert segmenter.segment_syllables(text) == ["က", "ခ", "ဂ", "ဃ", "င"]

    def test_default_segmenter_unsupported_word_engine_raises_error(self, monkeypatch, caplog):
        # Test the ValueError for unsupported engine
        with monkeypatch.context() as m:
            # For this test, myTokenize should be available for the ValueError
            # to be raised internally
            mock_mytokenize_module = MagicMock()
            mock_mytokenize_module.WordTokenizer = MockWordTokenizer
            m.setitem(sys.modules, "myTokenize", mock_mytokenize_module)

            with pytest.raises(TokenizationError, match="Unsupported word_engine"):
                DefaultSegmenter(word_engine="unsupported")
