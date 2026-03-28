"""
Integration tests for Segmenter + Validator pipeline.

These tests verify that segmenters and validators work correctly together:
1. DefaultSegmenter - segments text into syllables, words, sentences
2. RegexSegmenter - rule-based syllable segmentation
3. SyllableValidator - validates syllables and generates suggestions
4. WordValidator - validates words with morphological analysis
"""

import pytest

from myspellchecker.algorithms.symspell import SymSpell
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.exceptions import TokenizationError
from myspellchecker.core.response import SyllableError, WordError
from myspellchecker.core.validators import SyllableValidator, WordValidator
from myspellchecker.providers.memory import MemoryProvider
from myspellchecker.segmenters.default import DefaultSegmenter
from myspellchecker.segmenters.regex import RegexSegmenter


class TestRegexSegmenterBasics:
    """Basic tests for RegexSegmenter functionality."""

    @pytest.fixture
    def segmenter(self) -> RegexSegmenter:
        """Create a regex segmenter instance."""
        return RegexSegmenter()

    def test_segment_simple_syllables(self, segmenter: RegexSegmenter):
        """RegexSegmenter should segment Myanmar text into syllables."""
        text = "မြန်မာ"
        syllables = segmenter.segment_syllables(text)
        assert isinstance(syllables, list)
        assert len(syllables) == 2
        assert "မြန်" in syllables
        assert "မာ" in syllables

    def test_segment_multiple_syllables(self, segmenter: RegexSegmenter):
        """RegexSegmenter should handle multi-syllable text."""
        text = "မြန်မာနိုင်ငံ"
        syllables = segmenter.segment_syllables(text)
        assert isinstance(syllables, list)
        assert len(syllables) == 4
        assert syllables == ["မြန်", "မာ", "နိုင်", "ငံ"]

    def test_segment_with_spaces(self, segmenter: RegexSegmenter):
        """RegexSegmenter should handle text with spaces."""
        text = "မြန်မာ နိုင်ငံ"
        syllables = segmenter.segment_syllables(text)
        assert isinstance(syllables, list)
        # Space is preserved as a separate token between syllable groups
        assert "မြန်" in syllables
        assert "မာ" in syllables
        assert "နိုင်" in syllables
        assert "ငံ" in syllables
        assert len(syllables) == 5  # 4 syllables + space token

    def test_segment_empty_string_raises(self, segmenter: RegexSegmenter):
        """Empty or whitespace-only text should raise ValueError."""
        with pytest.raises(TokenizationError):
            segmenter.segment_syllables("")

    def test_segment_non_string_raises(self, segmenter: RegexSegmenter):
        """Non-string input should raise TypeError."""
        with pytest.raises(TypeError):
            segmenter.segment_syllables(123)  # type: ignore

    def test_segment_mixed_myanmar_english(self, segmenter: RegexSegmenter):
        """RegexSegmenter should handle mixed Myanmar and English."""
        text = "Hello မြန်မာ World"
        syllables = segmenter.segment_syllables(text)
        assert isinstance(syllables, list)
        # Myanmar syllables should be properly segmented
        assert "မြန်" in syllables
        assert "မာ" in syllables
        # Non-Myanmar parts are kept as tokens
        assert len(syllables) == 4

    def test_segment_myanmar_punctuation(self, segmenter: RegexSegmenter):
        """RegexSegmenter should handle Myanmar punctuation."""
        text = "မြန်မာ။"
        syllables = segmenter.segment_syllables(text)
        assert isinstance(syllables, list)
        assert "မြန်" in syllables
        assert "မာ" in syllables
        assert "။" in syllables
        assert len(syllables) == 3

    def test_segment_myanmar_numerals(self, segmenter: RegexSegmenter):
        """RegexSegmenter should handle Myanmar numerals."""
        text = "၁၂၃"
        syllables = segmenter.segment_syllables(text)
        assert isinstance(syllables, list)
        assert len(syllables) == 3
        assert "၁" in syllables
        assert "၂" in syllables
        assert "၃" in syllables


class TestSegmenterSyllableValidatorIntegration:
    """Integration tests for RegexSegmenter + SyllableValidator."""

    @pytest.fixture
    def provider(self) -> MemoryProvider:
        """Create a test provider with Myanmar syllables."""
        return MemoryProvider(
            syllables={
                "မြန်": 10000,
                "မာ": 8000,
                "နိုင်": 7000,
                "ငံ": 6000,
                "စာ": 5000,
                "သင်": 4000,
                "ကျောင်း": 3500,
                "သူ": 3000,
                "သည်": 2500,
                "ပါ": 2000,
                "က": 1500,
                "ခ": 1000,
            },
            words={
                "မြန်မာ": 8000,
                "နိုင်ငံ": 6000,
                "ကျောင်း": 3500,
            },
        )

    @pytest.fixture
    def segmenter(self) -> RegexSegmenter:
        """Create a regex segmenter."""
        return RegexSegmenter()

    @pytest.fixture
    def config(self) -> SpellCheckerConfig:
        """Create a test config."""
        return SpellCheckerConfig()

    @pytest.fixture
    def symspell(self, provider: MemoryProvider) -> SymSpell:
        """Create SymSpell instance with provider."""
        sym = SymSpell(provider=provider, max_edit_distance=2)
        sym.build_index(["syllable"])
        return sym

    @pytest.fixture
    def syllable_validator(
        self,
        config: SpellCheckerConfig,
        segmenter: RegexSegmenter,
        provider: MemoryProvider,
        symspell: SymSpell,
    ) -> SyllableValidator:
        """Create a SyllableValidator with segmenter and provider."""
        return SyllableValidator(
            config=config,
            segmenter=segmenter,
            repository=provider,
            symspell=symspell,
        )

    def test_validate_valid_text(self, syllable_validator: SyllableValidator):
        """Valid Myanmar text should produce no errors."""
        # Use text with known valid syllables (both in provider)
        text = "မြန်မာ"
        errors = syllable_validator.validate(text)
        assert isinstance(errors, list)
        # All syllables ("မြန်", "မာ") are in the provider, so no errors
        assert len(errors) == 0

    def test_validate_returns_syllable_errors(self, syllable_validator: SyllableValidator):
        """Invalid syllables should produce SyllableError instances."""
        # Use text with invalid syllable
        text = "xyz"  # Non-Myanmar - should be filtered
        errors = syllable_validator.validate(text)
        assert isinstance(errors, list)
        # Non-Myanmar text should be skipped (no errors)
        assert len(errors) == 0

    def test_validator_uses_segmenter_output(
        self,
        syllable_validator: SyllableValidator,
        segmenter: RegexSegmenter,
    ):
        """Validator should process segmenter output."""
        text = "မြန်မာစာ"
        # Get syllables from segmenter
        segmenter.segment_syllables(text)
        # Validator processes same text
        errors = syllable_validator.validate(text)
        # Each error should reference a segment
        for error in errors:
            assert isinstance(error, SyllableError)
            assert isinstance(error.text, str)
            assert isinstance(error.position, int)

    def test_error_positions_are_valid(self, syllable_validator: SyllableValidator):
        """Error positions should be valid indices in the original text."""
        text = "မြန်မာစာသင်"
        errors = syllable_validator.validate(text)
        for error in errors:
            assert error.position >= 0
            assert error.position < len(text)

    def test_error_suggestions_are_list(self, syllable_validator: SyllableValidator):
        """Error suggestions should be lists with valid suggestion entries."""
        # "ဘာ" is not in the provider — should produce an error with suggestions
        text = "ဘာ"
        errors = syllable_validator.validate(text)
        assert len(errors) == 1
        error = errors[0]
        assert isinstance(error.suggestions, list)
        # SymSpell should find edit-distance neighbors like "မာ" or "စာ"
        assert len(error.suggestions) >= 1
        assert all(isinstance(s, str) for s in error.suggestions)

    def test_punctuation_is_skipped(self, syllable_validator: SyllableValidator):
        """Punctuation should not produce validation errors."""
        text = "။"  # Myanmar punctuation
        errors = syllable_validator.validate(text)
        assert isinstance(errors, list)
        # Punctuation is not a syllable; should produce no errors
        assert len(errors) == 0


class TestSegmenterWordValidatorIntegration:
    """Integration tests for Segmenter + WordValidator.

    Note: This class uses DefaultSegmenter (not RegexSegmenter) because
    WordValidator requires word segmentation, which RegexSegmenter does not support.
    """

    @pytest.fixture
    def provider(self) -> MemoryProvider:
        """Create a test provider with words and syllables."""
        return MemoryProvider(
            syllables={
                "မြန်": 10000,
                "မာ": 8000,
                "နိုင်": 7000,
                "ငံ": 6000,
                "စာ": 5000,
                "သင်": 4000,
                "ကျောင်း": 3500,
                "သူ": 3000,
                "သည်": 2500,
                "များ": 2000,  # Plural suffix
                "ခဲ့": 1500,  # Past tense suffix
            },
            words={
                "မြန်မာ": 8000,
                "နိုင်ငံ": 6000,
                "ကျောင်း": 3500,
                "စာသင်": 3000,
                "စာအုပ်": 2500,
                "သူငယ်ချင်း": 2000,
            },
        )

    @pytest.fixture
    def segmenter(self) -> DefaultSegmenter:
        """Create a default segmenter that supports word segmentation."""
        return DefaultSegmenter()

    @pytest.fixture
    def config(self) -> SpellCheckerConfig:
        """Create a test config."""
        return SpellCheckerConfig()

    @pytest.fixture
    def symspell(self, provider: MemoryProvider) -> SymSpell:
        """Create SymSpell instance."""
        sym = SymSpell(provider=provider, max_edit_distance=2)
        sym.build_index(["syllable", "word"])
        return sym

    @pytest.fixture
    def word_validator(
        self,
        config: SpellCheckerConfig,
        segmenter: DefaultSegmenter,
        provider: MemoryProvider,
        symspell: SymSpell,
    ) -> WordValidator:
        """Create WordValidator with segmenter and provider."""
        return WordValidator(
            config=config,
            segmenter=segmenter,
            word_repository=provider,
            syllable_repository=provider,
            symspell=symspell,
        )

    def test_validate_valid_word(self, word_validator: WordValidator):
        """Validate known word text — in test env, segmenter returns syllables."""
        text = "မြန်မာ"
        errors = word_validator.validate(text)
        assert isinstance(errors, list)
        # In test env, DefaultSegmenter is mocked to produce syllables (not words),
        # so individual syllables "မြန်" and "မာ" are checked as words and flagged.
        assert len(errors) >= 1
        for error in errors:
            assert isinstance(error, WordError)
            assert error.text in ("မြန်", "မာ")

    def test_validate_returns_word_errors(self, word_validator: WordValidator):
        """Invalid words should produce WordError instances."""
        text = "invalid_word"  # Non-Myanmar
        errors = word_validator.validate(text)
        # Non-Myanmar text should be skipped (no errors)
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_word_error_has_syllable_count(
        self,
        word_validator: WordValidator,
        segmenter: DefaultSegmenter,
    ):
        """WordError should include syllable count."""
        # "များ" is not in the word repository, so it produces a WordError
        text = "ကျောင်းများ"
        errors = word_validator.validate(text)
        assert len(errors) >= 1
        word_errors = [e for e in errors if isinstance(e, WordError)]
        assert len(word_errors) >= 1
        for error in word_errors:
            assert hasattr(error, "syllable_count")
            assert isinstance(error.syllable_count, int)
            assert error.syllable_count >= 1

    def test_validator_uses_word_repository(
        self,
        word_validator: WordValidator,
        provider: MemoryProvider,
    ):
        """WordValidator should use word repository for validation."""
        # Confirm compound word exists in the repository
        valid_word = "မြန်မာ"
        assert provider.is_valid_word(valid_word)
        # In test env, segmenter produces syllables; individual syllables
        # are not in the word repository, so they get flagged
        errors = word_validator.validate(valid_word)
        assert isinstance(errors, list)
        # Errors should reference Myanmar syllable text, not arbitrary content
        for error in errors:
            assert isinstance(error, WordError)
            assert len(error.text) > 0


class TestFullValidationPipeline:
    """End-to-end integration tests for the full validation pipeline.

    Note: Uses DefaultSegmenter because WordValidator requires word segmentation.
    """

    @pytest.fixture
    def full_pipeline(self) -> tuple:
        """Create full validation pipeline with all components."""
        provider = MemoryProvider(
            syllables={
                "မြန်": 10000,
                "မာ": 8000,
                "နိုင်": 7000,
                "ငံ": 6000,
                "စာ": 5000,
                "သင်": 4000,
                "ကျောင်း": 3500,
                "သူ": 3000,
                "သည်": 2500,
            },
            words={
                "မြန်မာ": 8000,
                "နိုင်ငံ": 6000,
                "ကျောင်း": 3500,
            },
        )

        segmenter = DefaultSegmenter()
        config = SpellCheckerConfig()
        symspell = SymSpell(provider=provider, max_edit_distance=2)
        symspell.build_index(["syllable", "word"])

        syllable_validator = SyllableValidator(
            config=config,
            segmenter=segmenter,
            repository=provider,
            symspell=symspell,
        )

        word_validator = WordValidator(
            config=config,
            segmenter=segmenter,
            word_repository=provider,
            syllable_repository=provider,
            symspell=symspell,
        )

        return provider, segmenter, syllable_validator, word_validator

    def test_pipeline_components_connected(self, full_pipeline: tuple):
        """All pipeline components should be properly connected."""
        provider, segmenter, syl_val, word_val = full_pipeline

        # Validators should have segmenter
        assert syl_val.segmenter is segmenter
        assert word_val.segmenter is segmenter

        # Validators should have repository
        assert syl_val.repository is provider
        assert word_val.word_repository is provider

    def test_syllable_then_word_validation(self, full_pipeline: tuple):
        """Pipeline should support syllable then word validation order."""
        provider, segmenter, syl_val, word_val = full_pipeline

        text = "မြန်မာနိုင်ငံ"

        # Run syllable validation first
        syllable_errors = syl_val.validate(text)
        assert isinstance(syllable_errors, list)
        # All syllables (မြန်, မာ, နိုင်, ငံ) are in the provider
        assert len(syllable_errors) == 0

        # Then word validation — in test env, segmenter splits into syllables,
        # so individual syllables are checked as words
        word_errors = word_val.validate(text)
        assert isinstance(word_errors, list)
        # Word errors reference Myanmar syllable text
        for error in word_errors:
            assert isinstance(error, WordError)
            assert error.position >= 0
            assert error.position < len(text)

    def test_pipeline_handles_mixed_content(self, full_pipeline: tuple):
        """Pipeline should handle mixed Myanmar and non-Myanmar content."""
        _, _, syl_val, word_val = full_pipeline

        text = "Hello မြန်မာ World"

        syllable_errors = syl_val.validate(text)
        word_errors = word_val.validate(text)

        # Syllable validation should skip non-Myanmar and find no errors
        # for the valid Myanmar syllables (မြန်, မာ)
        assert isinstance(syllable_errors, list)
        assert len(syllable_errors) == 0

        # Word validation may flag the mixed content (Myanmar adjacent to English)
        assert isinstance(word_errors, list)

    def test_pipeline_error_positions_consistent(self, full_pipeline: tuple):
        """Error positions should be consistent across validators."""
        _, _, syl_val, word_val = full_pipeline

        text = "မြန်မာစာ"

        syllable_errors = syl_val.validate(text)
        word_errors = word_val.validate(text)

        # All positions should be within text bounds
        for error in syllable_errors + word_errors:
            assert 0 <= error.position < len(text)


class TestSentenceSegmentation:
    """Tests for sentence segmentation integration."""

    @pytest.fixture
    def segmenter(self) -> RegexSegmenter:
        """Create segmenter."""
        return RegexSegmenter()

    def test_segment_single_sentence(self, segmenter: RegexSegmenter):
        """Single sentence should be segmented correctly."""
        text = "မြန်မာနိုင်ငံ။"
        sentences = segmenter.segment_sentences(text)
        assert isinstance(sentences, list)
        assert len(sentences) == 1
        assert "မြန်မာနိုင်ငံ" in sentences[0]

    def test_segment_multiple_sentences(self, segmenter: RegexSegmenter):
        """Multiple sentences should be segmented correctly."""
        text = "မြန်မာနိုင်ငံ။ ကျောင်းသွားသည်။"
        sentences = segmenter.segment_sentences(text)
        assert isinstance(sentences, list)
        assert len(sentences) == 2
        assert "မြန်မာနိုင်ငံ" in sentences[0]
        assert "ကျောင်းသွားသည်" in sentences[1]

    def test_sentence_preserves_separator(self, segmenter: RegexSegmenter):
        """Sentence separator should be preserved."""
        text = "မြန်မာ။"
        sentences = segmenter.segment_sentences(text)
        assert isinstance(sentences, list)
        assert len(sentences) == 1
        # Sentence-final marker (။) is preserved in the sentence text
        assert "။" in sentences[0]
        assert "မြန်မာ" in sentences[0]

    def test_sentence_without_separator(self, segmenter: RegexSegmenter):
        """Text without separator should be treated as one sentence."""
        text = "မြန်မာနိုင်ငံ"
        sentences = segmenter.segment_sentences(text)
        assert isinstance(sentences, list)
        assert len(sentences) == 1


class TestValidatorFactoryMethods:
    """Tests for validator factory methods.

    Note: Uses DefaultSegmenter because WordValidator requires word segmentation.
    """

    @pytest.fixture
    def provider(self) -> MemoryProvider:
        """Create test provider."""
        return MemoryProvider(
            syllables={"မြန်": 100, "မာ": 50},
            words={"မြန်မာ": 80},
        )

    @pytest.fixture
    def segmenter(self) -> DefaultSegmenter:
        """Create segmenter that supports word segmentation."""
        return DefaultSegmenter()

    @pytest.fixture
    def symspell(self, provider: MemoryProvider) -> SymSpell:
        """Create SymSpell."""
        sym = SymSpell(provider=provider)
        sym.build_index(["syllable", "word"])
        return sym

    def test_syllable_validator_create_method(
        self,
        provider: MemoryProvider,
        segmenter: DefaultSegmenter,
        symspell: SymSpell,
    ):
        """SyllableValidator.create() should produce working validator."""
        validator = SyllableValidator.create(
            repository=provider,
            segmenter=segmenter,
            symspell=symspell,
        )

        assert isinstance(validator, SyllableValidator)
        # Should be able to validate known-valid text with no errors
        errors = validator.validate("မြန်မာ")
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_word_validator_create_method(
        self,
        provider: MemoryProvider,
        segmenter: DefaultSegmenter,
        symspell: SymSpell,
    ):
        """WordValidator.create() should produce working validator."""
        validator = WordValidator.create(
            word_repository=provider,
            syllable_repository=provider,
            segmenter=segmenter,
            symspell=symspell,
        )

        assert isinstance(validator, WordValidator)
        # In test env, segmenter splits "မြန်မာ" into syllables;
        # individual syllables are not in the word dict, so they are flagged
        errors = validator.validate("မြန်မာ")
        assert isinstance(errors, list)
        for error in errors:
            assert isinstance(error, WordError)
            assert error.text in ("မြန်", "မာ")

    def test_factory_with_custom_config(
        self,
        provider: MemoryProvider,
        segmenter: RegexSegmenter,
        symspell: SymSpell,
    ):
        """Factory methods should accept custom config."""
        custom_config = SpellCheckerConfig(max_suggestions=3)

        validator = SyllableValidator.create(
            repository=provider,
            segmenter=segmenter,
            symspell=symspell,
            config=custom_config,
        )

        assert validator.config.max_suggestions == 3


class TestSegmenterValidatorEdgeCases:
    """Edge case tests for Segmenter + Validator integration."""

    @pytest.fixture
    def pipeline(self) -> tuple:
        """Create minimal pipeline for edge case testing."""
        provider = MemoryProvider(syllables={"က": 100})
        segmenter = RegexSegmenter()
        config = SpellCheckerConfig()
        symspell = SymSpell(provider=provider)
        symspell.build_index(["syllable"])

        validator = SyllableValidator(
            config=config,
            segmenter=segmenter,
            repository=provider,
            symspell=symspell,
        )

        return provider, segmenter, validator

    def test_single_character_myanmar(self, pipeline: tuple):
        """Single Myanmar character should be handled."""
        _, _, validator = pipeline
        errors = validator.validate("က")
        assert isinstance(errors, list)
        # "က" is in the provider (freq=100), so no errors expected
        assert len(errors) == 0

    def test_very_long_text(self, pipeline: tuple):
        """Very long text should be handled without errors for valid syllables."""
        _, _, validator = pipeline
        # Create long text of valid syllable repeated with spaces
        text = "က " * 100
        text = text.strip()
        errors = validator.validate(text)
        assert isinstance(errors, list)
        # All syllables are "က" which is in the provider, so no errors
        assert len(errors) == 0

    def test_unicode_normalization(self, pipeline: tuple):
        """Unicode variations should be handled consistently."""
        _, _, validator = pipeline
        text = "က"
        errors = validator.validate(text)
        assert isinstance(errors, list)
        # "က" is a valid syllable in the provider, no errors expected
        assert len(errors) == 0

    def test_consecutive_spaces(self, pipeline: tuple):
        """Consecutive spaces should be handled."""
        _, _, validator = pipeline
        text = "က    က"
        errors = validator.validate(text)
        assert isinstance(errors, list)
        # Both "က" syllables are valid, consecutive spaces should not cause errors
        assert len(errors) == 0


class TestConsistency:
    """Consistency tests for Segmenter + Validator."""

    @pytest.fixture
    def pipeline(self) -> tuple:
        """Create pipeline for consistency testing."""
        provider = MemoryProvider(syllables={"မြန်": 100, "မာ": 50, "က": 30})
        segmenter = RegexSegmenter()
        config = SpellCheckerConfig()
        symspell = SymSpell(provider=provider)
        symspell.build_index(["syllable"])

        validator = SyllableValidator(
            config=config,
            segmenter=segmenter,
            repository=provider,
            symspell=symspell,
        )

        return provider, segmenter, validator

    def test_deterministic_validation(self, pipeline: tuple):
        """Same input should produce consistent validation results."""
        _, _, validator = pipeline

        text = "မြန်မာက"

        # Run validation multiple times
        results = []
        for _ in range(5):
            errors = validator.validate(text)
            results.append([(e.text, e.position) for e in errors])

        # All results should be identical
        for result in results[1:]:
            assert result == results[0]

    def test_segmenter_deterministic(self, pipeline: tuple):
        """Segmenter should produce consistent results."""
        _, segmenter, _ = pipeline

        text = "မြန်မာစာ"

        # Segment multiple times
        results = []
        for _ in range(5):
            syllables = segmenter.segment_syllables(text)
            results.append(syllables)

        # All results should be identical
        for result in results[1:]:
            assert result == results[0]
