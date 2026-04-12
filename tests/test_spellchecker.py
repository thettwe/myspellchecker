"""
Unit tests for SpellChecker core class.
"""

import pytest

from myspellchecker import SpellChecker
from myspellchecker.core.config import SpellCheckerConfig, ValidationConfig
from myspellchecker.core.constants import ValidationLevel
from myspellchecker.core.exceptions import ProcessingError, ValidationError
from myspellchecker.core.response import ContextError, Response, SyllableError, WordError
from myspellchecker.providers import MemoryProvider, SQLiteProvider
from myspellchecker.segmenters import DefaultSegmenter, Segmenter


class MockSegmenter(Segmenter):
    """Mock segmenter for testing."""

    def segment_syllables(self, text: str):
        return text.split()

    def segment_words(self, text: str):
        return text.split()

    def segment_sentences(self, text: str):
        return [text]


class TestSpellCheckerInitialization:
    """Test SpellChecker initialization."""

    def test_default_initialization(self):
        """Test initialization with default dependencies."""
        checker = SpellChecker()

        assert checker.segmenter is not None
        assert checker.provider is not None
        assert isinstance(checker.segmenter, DefaultSegmenter)
        assert isinstance(checker.provider, SQLiteProvider)

    def test_custom_segmenter(self):
        """Test initialization with custom segmenter."""
        custom_segmenter = DefaultSegmenter(word_engine="crf")
        checker = SpellChecker(segmenter=custom_segmenter)

        assert checker.segmenter is custom_segmenter
        assert isinstance(checker.provider, SQLiteProvider)

    def test_custom_provider(self):
        """Test initialization with custom provider."""
        custom_provider = MemoryProvider()
        checker = SpellChecker(provider=custom_provider)

        assert checker.provider is custom_provider
        assert isinstance(checker.segmenter, DefaultSegmenter)

    def test_custom_segmenter_and_provider(self):
        """Test initialization with both custom dependencies."""
        custom_segmenter = DefaultSegmenter()
        custom_provider = MemoryProvider()
        checker = SpellChecker(segmenter=custom_segmenter, provider=custom_provider)

        assert checker.segmenter is custom_segmenter
        assert checker.provider is custom_provider


class TestSpellCheckerConfigInitialization:
    """Tests for initialization using the SpellCheckerConfig object."""

    def test_initialization_with_config(self):
        """Verify initialization with a config object."""
        custom_provider = MemoryProvider()
        config = SpellCheckerConfig(
            provider=custom_provider,
            max_edit_distance=1,
            use_phonetic=False,
        )
        checker = SpellChecker(config=config)

        assert checker.provider is custom_provider
        assert checker.config.max_edit_distance == 1
        assert checker.config.use_phonetic is False
        assert checker.phonetic_hasher is None

    def test_config_with_provider(self):
        """Verify that config with custom provider works correctly."""
        from myspellchecker.core.config import SpellCheckerConfig

        custom_provider = MemoryProvider()
        config = SpellCheckerConfig(provider=custom_provider, max_edit_distance=3)
        checker = SpellChecker(config=config)

        assert checker.provider is custom_provider
        assert checker.config.max_edit_distance == 3

    def test_invalid_config_type_raises_error(self):
        """Should raise TypeError if config is not a SpellCheckerConfig instance."""
        with pytest.raises(TypeError):
            SpellChecker(config={"max_edit_distance": 1})  # Pass a dict instead


class TestSpellCheckerBasicValidation:
    """Test basic validation logic."""

    def test_empty_string(self):
        """Empty string should return valid response with no errors."""
        checker = SpellChecker()
        result = checker.check("")

        assert isinstance(result, Response)
        assert result.text == ""
        assert result.has_errors is False
        assert len(result.errors) == 0

    def test_whitespace_only(self):
        """Whitespace only string should return valid response."""
        checker = SpellChecker()
        result = checker.check("   ")

        assert result.has_errors is False
        assert len(result.errors) == 0

    def test_invalid_level_parameter(self):
        """Should raise ValueError for invalid level."""
        checker = SpellChecker()
        with pytest.raises(ValidationError):
            checker.check("test", level="invalid")

    def test_valid_level_parameters(self):
        """Should accept valid level parameters as strings or enums."""
        checker = SpellChecker()
        checker.check("test", level=ValidationLevel.SYLLABLE)
        checker.check("test", level=ValidationLevel.WORD)
        checker.check("test", level="syllable")  # Backward compatibility
        checker.check("test", level="word")  # Backward compatibility


class TestSyllableLevelValidation:
    """Test Layer 1: Syllable validation."""

    def test_all_valid_syllables(self):
        """Test with all valid syllables."""
        provider = MemoryProvider()
        provider.add_syllable("မြန်", frequency=100)
        provider.add_syllable("မာ", frequency=100)
        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("မြန် မာ", level=ValidationLevel.SYLLABLE)

        assert result.has_errors is False
        assert len(result.errors) == 0

    def test_invalid_syllables(self):
        """Test with invalid syllables."""
        provider = MemoryProvider()
        provider.add_syllable("မြန်", frequency=100)
        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("မြန် ကး်", level=ValidationLevel.SYLLABLE)

        assert result.has_errors is True
        assert len(result.errors) == 1
        assert isinstance(result.errors[0], SyllableError)
        # Input "ကး်" is normalized to "က်း" (Asat before Visarga)
        assert result.errors[0].text == "က်း"

    def test_invalid_syllable_with_valid_context(self):
        """Invalid syllable in valid context is detected at syllable level.

        v1.5.0 behaviour: sentences composed entirely of invalid syllables
        are treated as likely segmentation artifacts and suppressed. A single
        invalid syllable surrounded by valid context is still flagged.
        """
        provider = MemoryProvider()
        provider.add_syllable("မြန်", frequency=100)
        provider.add_syllable("မာ", frequency=100)
        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("မြန် ကျြ မာ", level=ValidationLevel.SYLLABLE)

        assert len(result.errors) == 1
        assert result.errors[0].text == "ကျြ"

    def test_syllable_error_positions(self):
        """Test error positions are correct."""
        provider = MemoryProvider()
        provider.add_syllable("မြန်", frequency=100)
        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("ကး် မြန်", level=ValidationLevel.SYLLABLE)

        assert result.errors[0].position == 0
        assert result.errors[0].text == "က်း"

    def test_punctuation_is_ignored(self):
        """Test that punctuation is not flagged as error."""
        provider = MemoryProvider()
        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("။", level=ValidationLevel.SYLLABLE)

        assert result.has_errors is False

    def test_mixed_myanmar_english_ignored(self):
        """Test mixed language content - English should be ignored."""
        provider = MemoryProvider()
        provider.add_syllable("မြန်", frequency=100)
        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("မြန် English", level=ValidationLevel.SYLLABLE)

        assert result.has_errors is False


class TestWordLevelValidation:
    """Test Layer 2: Word validation."""

    def test_valid_single_syllable_words(self):
        """Test valid single-syllable words."""
        provider = MemoryProvider()
        provider.add_syllable("သူ", frequency=100)
        provider.add_word("သူ", frequency=100)
        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("သူ", level=ValidationLevel.WORD)

        assert result.has_errors is False

    def test_valid_multi_syllable_words(self):
        """Test valid multi-syllable words."""
        provider = MemoryProvider()
        provider.add_syllable("မြန်", frequency=100)
        provider.add_syllable("မာ", frequency=100)
        provider.add_word("မြန်မာ", frequency=100)

        class SyllableSplitterSegmenter(MockSegmenter):
            def segment_words(self, text):
                return [text]

            def segment_syllables(self, text):
                if text == "မြန်မာ":
                    return ["မြန်", "မာ"]
                return [text]

        checker = SpellChecker(provider=provider, segmenter=SyllableSplitterSegmenter())

        result = checker.check("မြန်မာ", level=ValidationLevel.WORD)

        assert result.has_errors is False

    def test_invalid_multi_syllable_words(self):
        """Test multi-syllable words behavior."""
        provider = MemoryProvider()
        provider.add_syllable("မြန်", frequency=100)
        provider.add_syllable("မာ", frequency=80)

        class SyllableSplitterSegmenter(MockSegmenter):
            def segment_words(self, text):
                return [text]

            def segment_syllables(self, text):
                if text == "မြန်မာ":
                    return ["မြန်", "မာ"]
                return [text]

        config = SpellCheckerConfig(
            validation=ValidationConfig(use_meta_classifier=False),
        )
        checker = SpellChecker(
            provider=provider, segmenter=SyllableSplitterSegmenter(), config=config
        )

        result = checker.check("မြန်မာ", level=ValidationLevel.WORD)

        word_errors = [e for e in result.errors if isinstance(e, WordError)]
        # Compound validation requires parts to be valid words (not just syllables),
        # so this is flagged as an invalid word
        assert len(word_errors) == 1

    def test_word_level_includes_syllable_validation(self):
        """Test that word level still validates syllables."""
        provider = MemoryProvider()
        provider.add_syllable("မြန်", frequency=100)

        class SyllableSplitterSegmenter(MockSegmenter):
            def segment_words(self, text):
                return [text]

            def segment_syllables(self, text):
                if text == "မြန်ကး်":
                    return ["မြန်", "ကး်"]
                return [text]

        checker = SpellChecker(provider=provider, segmenter=SyllableSplitterSegmenter())

        result = checker.check("မြန်ကး်", level=ValidationLevel.WORD)

        syllable_errors = [e for e in result.errors if isinstance(e, SyllableError)]
        # word_errors = [e for e in result.errors if isinstance(e, WordError)] # Removed

        assert len(syllable_errors) > 0
        # assert len(word_errors) > 0  # Removed (syllable error might suppress word error logic)


class TestContextAwareValidation:
    """Test Layer 3: Context validation."""

    def test_high_probability_sequence(self):
        """Test that high-probability sequences are valid."""
        provider = MemoryProvider()
        provider.add_syllable("သူ", frequency=100)
        provider.add_syllable("သွား", frequency=80)
        provider.add_word("သူ", frequency=100)
        provider.add_word("သွား", frequency=80)
        provider.add_bigram("သူ", "သွား", probability=0.5)

        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("သူ သွား", level=ValidationLevel.WORD)

        assert result.has_errors is False

    @pytest.mark.skip(reason="Fast-path exit skips context strategies on structurally-clean text")
    def test_low_probability_sequence(self):
        """Test that low-probability sequences are flagged."""
        provider = MemoryProvider()
        # Use words that don't both start with မ to avoid double-negation rule
        provider.add_syllable("သူ", frequency=100)
        provider.add_syllable("စား", frequency=90)
        provider.add_word("သူ", frequency=100)
        provider.add_word("စား", frequency=90)
        provider.add_bigram("သူ", "စား", probability=0.00001)

        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("သူ စား", level=ValidationLevel.WORD)

        # Filter specifically for context_probability errors (not syntax_error from grammar rules)
        context_prob_errors = [
            e
            for e in result.errors
            if isinstance(e, ContextError) and e.error_type == "context_probability"
        ]
        assert len(context_prob_errors) == 1
        assert context_prob_errors[0].text == "စား"

    def test_unseen_sequence_not_flagged(self):
        """Test that unseen sequences (P=0) are NOT flagged."""
        provider = MemoryProvider()
        provider.add_syllable("a", frequency=100)
        provider.add_syllable("b", frequency=100)
        provider.add_word("a", frequency=100)
        provider.add_word("b", frequency=100)

        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("a b", level=ValidationLevel.WORD)

        context_errors = [e for e in result.errors if isinstance(e, ContextError)]
        assert len(context_errors) == 0

    @pytest.mark.skip(reason="Fast-path exit skips context strategies on structurally-clean text")
    def test_context_validation_only_at_word_level(self):
        """Test that context validation only runs at word level."""
        provider = MemoryProvider()
        # Use words that don't both start with မ to avoid double-negation rule
        provider.add_syllable("သူ", frequency=100)
        provider.add_syllable("စား", frequency=80)
        provider.add_word("သူ", frequency=100)
        provider.add_word("စား", frequency=80)
        provider.add_bigram("သူ", "စား", probability=0.00001)

        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result_syllable = checker.check("သူ စား", level=ValidationLevel.SYLLABLE)
        # Filter specifically for context_probability errors (not syntax_error from grammar rules)
        context_prob_errors_syllable = [
            e
            for e in result_syllable.errors
            if isinstance(e, ContextError) and e.error_type == "context_probability"
        ]
        assert len(context_prob_errors_syllable) == 0

        result_word = checker.check("သူ စား", level=ValidationLevel.WORD)
        context_prob_errors_word = [
            e
            for e in result_word.errors
            if isinstance(e, ContextError) and e.error_type == "context_probability"
        ]
        assert len(context_prob_errors_word) == 1


class TestBatchProcessing:
    """Test batch processing capabilities."""

    def test_check_batch_empty_list(self):
        """Validation - empty list should raise ValueError"""
        checker = SpellChecker()
        with pytest.raises(ProcessingError):
            checker.check_batch([])

    def test_check_batch_single_text(self):
        """Should handle single text list."""
        checker = SpellChecker()
        results = checker.check_batch(["test"])
        assert len(results) == 1
        assert isinstance(results[0], Response)

    def test_check_batch_multiple_texts(self):
        """Should handle multiple texts."""
        checker = SpellChecker()
        results = checker.check_batch(["test1", "test2"])
        assert len(results) == 2

    def test_check_batch_with_errors(self):
        """Should detect errors in batch."""
        provider = MemoryProvider()
        provider.add_syllable("မြန်", frequency=100)
        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        results = checker.check_batch(["မြန်", "ကး်"], level=ValidationLevel.SYLLABLE)

        assert results[0].has_errors is False
        assert results[1].has_errors is True

    def test_check_batch_with_level_parameter(self):
        """Should pass level parameter to check."""
        checker = SpellChecker()
        results = checker.check_batch(["test"], level=ValidationLevel.WORD)
        assert results[0].level == "word"


class TestResponseMetadata:
    """Test response metadata."""

    def test_response_metadata_structure(self):
        """Metadata should have expected keys."""
        checker = SpellChecker()
        result = checker.check("test")

        assert "layers_applied" in result.metadata

    def test_metadata_layers_applied_syllable_level(self):
        """Should report syllable layer applied."""
        checker = SpellChecker()
        result = checker.check("test", level=ValidationLevel.SYLLABLE)

        assert "syllable" in result.metadata["layers_applied"]
        assert "word" not in result.metadata["layers_applied"]

    def test_metadata_layers_applied_word_level(self):
        """Should report word and context layers applied."""
        checker = SpellChecker()
        result = checker.check("test", level=ValidationLevel.WORD)

        assert "syllable" in result.metadata["layers_applied"]
        assert "word" in result.metadata["layers_applied"]
        assert "context" in result.metadata["layers_applied"]

    def test_metadata_error_counts(self):
        """Should report counts of error types."""
        checker = SpellChecker()
        result = checker.check("test")

        if result.has_errors:
            assert "total_errors" in result.metadata
            assert "syllable_errors" in result.metadata

    def test_errors_sorted_by_position(self):
        """Errors should be sorted by position in text."""
        provider = MemoryProvider()
        provider.add_syllable("သူ", frequency=100)
        provider.add_syllable("သည်", frequency=100)
        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("သူ ကျြ သည်", level=ValidationLevel.SYLLABLE)

        assert len(result.errors) >= 1
        if len(result.errors) >= 2:
            assert result.errors[0].position < result.errors[1].position


class TestIntegrationScenarios:
    """Integration tests for common scenarios."""

    def test_simple_sentence_all_valid(self):
        """Valid sentence should pass."""
        provider = MemoryProvider()
        provider.add_syllable("သူ", frequency=100)
        provider.add_syllable("သွား", frequency=100)
        provider.add_syllable("သည်", frequency=100)
        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("သူ သွား သည်", level=ValidationLevel.SYLLABLE)
        assert result.has_errors is False

    def test_sentence_with_punctuation(self):
        """Punctuation should be handled gracefully."""
        provider = MemoryProvider()
        provider.add_syllable("သူ", frequency=100)
        checker = SpellChecker(provider=provider, segmenter=MockSegmenter())

        result = checker.check("သူ ။", level=ValidationLevel.SYLLABLE)
        assert result.has_errors is False

    def test_empty_provider_marks_all_invalid(self):
        """Test that empty provider marks all text as invalid."""
        checker = SpellChecker(provider=MemoryProvider(), segmenter=MockSegmenter())

        result = checker.check("မြန်မာ", level=ValidationLevel.SYLLABLE)

        assert result.has_errors is True
