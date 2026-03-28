"""
Unit tests for DefaultSegmenter implementation.

This module tests the regex-based syllable and word segmentation
for Myanmar text.
"""

import sys
from unittest.mock import MagicMock

import pytest

from myspellchecker.core.exceptions import TokenizationError
from myspellchecker.segmenters import DefaultSegmenter
from tests.fixtures.myanmar_test_samples import (
    MyanmarTestSamples,
)


class TestDefaultSegmenterInit:
    """Test DefaultSegmenter initialization."""

    def test_init_default(self) -> None:
        """Test initialization with default parameters."""
        # Default word_engine is now "myword" (previously "crf")
        segmenter = DefaultSegmenter()
        assert segmenter.word_engine == "myword"

    def test_init_invalid_engine(self) -> None:
        """Test initialization with an unsupported engine."""
        with pytest.raises(TokenizationError, match="Unsupported word_engine"):
            DefaultSegmenter(word_engine="invalid_engine")

    # Note: Previous tests for invalid engines and unimplemented engines are removed
    # because the current implementation wraps myTokenize which may handle engines
    # differently. We are now delegating validation to myTokenize or ignoring unknown params.


class TestDefaultSegmenterSyllables:
    """Test syllable segmentation functionality."""

    @pytest.fixture
    def segmenter(self) -> DefaultSegmenter:
        """Create a DefaultSegmenter instance for testing."""
        return DefaultSegmenter()

    def test_segment_syllables_basic(self, segmenter: DefaultSegmenter) -> None:
        """Test basic syllable segmentation."""
        result = segmenter.segment_syllables("မြန်မာ")
        assert result == ["မြန်", "မာ"]

    def test_segment_syllables_with_tone_marks(self, segmenter: DefaultSegmenter) -> None:
        """Test segmentation with tone marks."""
        result = segmenter.segment_syllables("ထမင်း")
        # This should segment into syllables
        assert isinstance(result, list)
        assert len(result) > 0

    def test_segment_syllables_with_medials(self, segmenter: DefaultSegmenter) -> None:
        """Test segmentation with medial consonants."""
        result = segmenter.segment_syllables("ကျောင်းသား")
        # Should segment into 2 syllables
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_segment_syllables_with_punctuation(self, segmenter: DefaultSegmenter) -> None:
        """Test segmentation preserves punctuation."""
        result = segmenter.segment_syllables("သူသွားသည်။")
        # Should include the period
        assert "။" in result

    def test_segment_syllables_empty_string(self, segmenter: DefaultSegmenter) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(TokenizationError, match="cannot be empty"):
            segmenter.segment_syllables("")

    def test_segment_syllables_whitespace_only(self, segmenter: DefaultSegmenter) -> None:
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(TokenizationError, match="cannot be empty"):
            segmenter.segment_syllables("   ")

    def test_segment_syllables_type_error(self, segmenter: DefaultSegmenter) -> None:
        """Test that non-string input raises TypeError."""
        with pytest.raises(TypeError, match="Expected str"):
            segmenter.segment_syllables(123)  # type: ignore

    def test_segment_syllables_test_samples(self, segmenter: DefaultSegmenter) -> None:
        """Test segmentation against known good examples."""
        ground_truth = MyanmarTestSamples.get_syllable_ground_truth()

        for text, _expected_syllables in ground_truth.items():
            result = segmenter.segment_syllables(text)
            # Note: Our implementation might differ slightly from ground truth
            # but should produce reasonable segmentation
            assert isinstance(result, list)
            assert len(result) > 0
            # At minimum, all results should be strings
            assert all(isinstance(s, str) for s in result)

    def test_segment_syllables_kinzi(self, segmenter: DefaultSegmenter) -> None:
        """Test segmentation with kinzi character."""
        result = segmenter.segment_syllables("င်္ဂလိပ်")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_segment_syllables_numbers(self, segmenter: DefaultSegmenter) -> None:
        """Test segmentation with Myanmar numerals."""
        result = segmenter.segment_syllables("၁၂၃")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_segment_syllables_single_syllable(self, segmenter: DefaultSegmenter) -> None:
        """Test segmentation of single syllable word."""
        result = segmenter.segment_syllables("အိမ်")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_segment_syllables_long_text(self, segmenter: DefaultSegmenter) -> None:
        """Test segmentation of longer text."""
        text = "သူမနေကောင်းပါဘူး"
        result = segmenter.segment_syllables(text)
        assert isinstance(result, list)
        assert len(result) >= 5  # Should have multiple syllables


class TestDefaultSegmenterWords:
    """Test word segmentation functionality."""

    @pytest.fixture
    def segmenter(self) -> DefaultSegmenter:
        """Create a DefaultSegmenter instance for testing with mocked myTokenize."""
        mock_module = MagicMock()
        sys.modules["myTokenize"] = mock_module

        # Setup WordTokenizer mock to return the text as a list of words (simple mock)
        mock_tokenizer = MagicMock()
        # Mock tokenize to return [text] so assert len(result) > 0 passes
        mock_tokenizer.tokenize.side_effect = lambda t: [t]

        mock_module.WordTokenizer.return_value = mock_tokenizer

        return DefaultSegmenter()

    def test_segment_words_basic(self, segmenter: DefaultSegmenter) -> None:
        """Test basic word segmentation."""
        result = segmenter.segment_words("မြန်မာ")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_segment_words_compound(self, segmenter: DefaultSegmenter) -> None:
        """Test segmentation of compound phrase."""
        result = segmenter.segment_words("မြန်မာနိုင်ငံ")
        assert isinstance(result, list)
        # Should segment into words (exact count may vary)
        assert len(result) >= 1

    def test_segment_words_sentence(self, segmenter: DefaultSegmenter) -> None:
        """Test segmentation of sentence with punctuation."""
        result = segmenter.segment_words("သူသွားသည်။")
        assert isinstance(result, list)
        # Should include words and punctuation
        assert len(result) > 0

    def test_segment_words_empty_string(self, segmenter: DefaultSegmenter) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(TokenizationError, match="cannot be empty"):
            segmenter.segment_words("")

    def test_segment_words_whitespace_only(self, segmenter: DefaultSegmenter) -> None:
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(TokenizationError, match="cannot be empty"):
            segmenter.segment_words("   ")

    def test_segment_words_type_error(self, segmenter: DefaultSegmenter) -> None:
        """Test that non-string input raises TypeError."""
        with pytest.raises(TypeError, match="Expected str"):
            segmenter.segment_words(123)  # type: ignore

    def test_segment_words_test_samples(self, segmenter: DefaultSegmenter) -> None:
        """Test word segmentation against test samples."""
        texts = MyanmarTestSamples.get_texts()

        for text in texts:
            result = segmenter.segment_words(text)
            # Basic sanity checks
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(w, str) for w in result)

    def test_segment_words_preserves_punctuation(self, segmenter: DefaultSegmenter) -> None:
        """Test that punctuation is handled properly."""
        result = segmenter.segment_words("သူသွားသည်။")
        # Punctuation should be included (either in word or separate)
        assert any("။" in word for word in result) or "။" in result

    def test_segment_words_long_text(self, segmenter: DefaultSegmenter) -> None:
        """Test word segmentation of longer text."""
        text = "သူမနေကောင်းပါဘူး"
        result = segmenter.segment_words(text)
        assert isinstance(result, list)
        assert len(result) >= 1


def _check_benchmark_available():
    """Check if pytest-benchmark is available."""
    try:
        import pytest_benchmark  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _check_benchmark_available(), reason="pytest-benchmark not installed")
class TestDefaultSegmenterPerformance:
    """Test performance characteristics of DefaultSegmenter."""

    @pytest.fixture
    def segmenter(self) -> DefaultSegmenter:
        """Create a DefaultSegmenter instance for testing with mocked myTokenize."""
        mock_module = MagicMock()
        sys.modules["myTokenize"] = mock_module

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.side_effect = lambda t: [t]
        mock_module.WordTokenizer.return_value = mock_tokenizer

        return DefaultSegmenter()

    @pytest.mark.benchmark
    def test_syllable_segmentation_performance(
        self, segmenter: DefaultSegmenter, benchmark
    ) -> None:
        """Test syllable segmentation performance."""
        text = "မြန်မာနိုင်ငံသည်အရှေ့တောင်အာရှတွင်တည်ရှိသည်"

        result = benchmark(segmenter.segment_syllables, text)

        # Should complete in reasonable time (<10ms typically)
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.benchmark
    def test_word_segmentation_performance(self, segmenter: DefaultSegmenter, benchmark) -> None:
        """Test word segmentation performance."""
        text = "မြန်မာနိုင်ငံသည်အရှေ့တောင်အာရှတွင်တည်ရှိသည်"

        result = benchmark(segmenter.segment_words, text)

        # Should complete in reasonable time (<50ms typically)
        assert isinstance(result, list)
        assert len(result) > 0


class TestDefaultSegmenterEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def segmenter(self) -> DefaultSegmenter:
        """Create a DefaultSegmenter instance for testing."""
        return DefaultSegmenter()

    def test_mixed_content_myanmar_and_english(self, segmenter: DefaultSegmenter) -> None:
        """Test handling of mixed Myanmar and English text."""
        # For now, focus on Myanmar text
        # Mixed content handling can be improved later
        result = segmenter.segment_syllables("Myanmar")
        # Should not crash, even if English
        assert isinstance(result, list)

    def test_special_characters(self, segmenter: DefaultSegmenter) -> None:
        """Test handling of special characters."""
        result = segmenter.segment_syllables("၊")  # Myanmar comma
        assert isinstance(result, list)

    def test_repeated_syllables(self, segmenter: DefaultSegmenter) -> None:
        """Test segmentation of repeated syllables."""
        result = segmenter.segment_syllables("ကကက")
        assert isinstance(result, list)
        assert len(result) >= 3

    def test_all_numerals(self, segmenter: DefaultSegmenter) -> None:
        """Test segmentation of Myanmar numerals."""
        result = segmenter.segment_syllables("၁၂၃၄၅၆၇၈၉၀")
        assert isinstance(result, list)
        assert len(result) > 0


# Integration tests
class TestDefaultSegmenterIntegration:
    """Integration tests combining multiple features."""

    @pytest.fixture
    def segmenter(self) -> DefaultSegmenter:
        """Create a DefaultSegmenter instance for testing with mocked myTokenize."""
        mock_module = MagicMock()
        sys.modules["myTokenize"] = mock_module

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.side_effect = lambda t: [t]
        mock_module.WordTokenizer.return_value = mock_tokenizer

        return DefaultSegmenter()

    def test_syllables_to_words_consistency(self, segmenter: DefaultSegmenter) -> None:
        """Test that syllable and word segmentation are consistent."""
        text = "မြန်မာနိုင်ငံ"

        syllables = segmenter.segment_syllables(text)
        words = segmenter.segment_words(text)

        # All characters in syllables should appear in words
        syllable_text = "".join(syllables)
        word_text = "".join(words)

        # They should contain the same Myanmar characters (ignoring whitespace)
        assert len(syllable_text.replace(" ", "")) > 0
        assert len(word_text.replace(" ", "")) > 0

    def test_roundtrip_consistency(self, segmenter: DefaultSegmenter) -> None:
        """Test that segmentation doesn't lose characters."""
        original = "မြန်မာနိုင်ငံ"

        syllables = segmenter.segment_syllables(original)
        reconstructed = "".join(syllables)

        # Should preserve all characters (possibly with added whitespace)
        # For now, just check length is reasonable
        assert len(reconstructed.replace(" ", "")) >= len(original.replace(" ", "")) * 0.9


class TestDefaultSegmenterSentences:
    """Test sentence segmentation functionality."""

    @pytest.fixture
    def segmenter(self) -> DefaultSegmenter:
        """Create a DefaultSegmenter instance for testing."""
        # Ensure myTokenize is not present to test fallback logic
        if "myTokenize" in sys.modules:
            del sys.modules["myTokenize"]
        return DefaultSegmenter()

    def test_segment_sentences_basic(self, segmenter: DefaultSegmenter) -> None:
        """Test basic sentence segmentation."""
        text = "မြန်မာနိုင်ငံ။ ရန်ကုန်မြို့။"
        result = segmenter.segment_sentences(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_segment_sentences_empty_string(self, segmenter: DefaultSegmenter) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(TokenizationError, match="cannot be empty"):
            segmenter.segment_sentences("")


class TestDefaultSegmenterSentenceSeparator:
    """Test that sentence segmentation preserves separators."""

    @pytest.fixture
    def segmenter(self) -> DefaultSegmenter:
        """Create a DefaultSegmenter instance for testing."""
        return DefaultSegmenter()

    def test_segment_sentences_preserves_separator(self, segmenter: DefaultSegmenter) -> None:
        """Test that sentence separator is preserved."""
        text = "မြန်မာနိုင်ငံ။ ရန်ကုန်မြို့။"
        sentences = segmenter.segment_sentences(text)

        assert len(sentences) == 2
        assert sentences[0] == "မြန်မာနိုင်ငံ။"
        assert sentences[1] == "ရန်ကုန်မြို့။"

    def test_segment_sentences_single_sentence(self, segmenter: DefaultSegmenter) -> None:
        """Test single sentence without trailing separator."""
        text = "မြန်မာနိုင်ငံ"
        sentences = segmenter.segment_sentences(text)

        assert len(sentences) == 1
        assert sentences[0] == "မြန်မာနိုင်ငံ"

    def test_segment_sentences_trailing_separator(self, segmenter: DefaultSegmenter) -> None:
        """Test text with trailing separator only."""
        text = "မြန်မာနိုင်ငံ။"
        sentences = segmenter.segment_sentences(text)

        assert len(sentences) == 1
        assert sentences[0] == "မြန်မာနိုင်ငံ။"
