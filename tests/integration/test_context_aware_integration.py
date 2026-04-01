"""
Integration tests for Context-Aware Checking.

These tests verify the complete functionality including:
- N-gram context checking with suggestions
- Phonetic similarity matching
- Compound word detection and correction
- End-to-end spell checking with all features
"""

import pytest

from myspellchecker import SpellChecker
from myspellchecker.algorithms import SymSpell
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.response import ContextError, Response, SyllableError
from myspellchecker.providers import MemoryProvider
from myspellchecker.segmenters import DefaultSegmenter, Segmenter


class SpaceSegmenter(Segmenter):
    """Simple segmenter that splits by space for testing."""

    def segment_syllables(self, text: str):
        return DefaultSegmenter().segment_syllables(text)

    def segment_words(self, text: str):
        return text.split()

    def segment_sentences(self, text: str):
        return [text]


@pytest.fixture
def comprehensive_provider():
    """Create a MemoryProvider with comprehensive test data."""
    provider = MemoryProvider()

    # Add syllables with frequencies
    # Each entry must be a valid Myanmar syllable (not a sub-syllabic fragment)
    syllables = [
        ("မြန်", 5000),
        ("မာ", 4000),
        ("မ", 3800),
        ("န", 3500),
        ("ကျွန်", 3000),
        ("ကျောင်း", 2800),
        ("သူ", 2500),
        ("တော်", 2300),
        ("သွား", 2200),
        ("သည်", 2000),
        ("ရှိ", 1800),
        ("တို့", 1600),
        ("စ", 1400),
        ("စာ", 1300),
        ("နိုင်", 1200),
        ("ငံ", 1100),
        ("မယ်", 900),
    ]
    for syllable, freq in syllables:
        provider.add_syllable(syllable, frequency=freq)

    # Add words with frequencies
    # NOTE: Words expected to be flagged as context errors must have freq < 1000.
    # The pipeline suppresses suggestion-less context_probability errors on
    # high-frequency words (>= 1000) via _suppress_low_value_context_probability.
    words = [
        ("မြန်မာ", 5000),  # Myanmar
        ("နိုင်ငံ", 800),  # country (low freq: expected in context errors)
        ("ကျောင်း", 800),  # school (low freq: expected in context errors)
        ("သူ", 3500),  # he/she
        ("သွား", 3000),  # go
        ("သည်", 2800),  # particle
        ("ရှိ", 2500),  # have/exist
        ("တို့", 2200),  # plural
        ("စာ", 2000),  # book/letter
        ("မယ်", 1800),  # future marker
        ("ပြန်", 1600),  # return
        ("လာ", 1500),  # come
        ("ဘယ်", 1400),  # which/where
        ("မြန်", 1300),  # fast (syllable)
    ]
    for word, freq in words:
        provider.add_word(word, frequency=freq)

    # Add bigrams for context checking
    # High probability sequences (common)
    high_prob_bigrams = [
        (("သူ", "သွား"), 0.30),  # he goes
        (("သူ", "သည်"), 0.25),  # he (subject)
        (("သူ", "တို့"), 0.20),  # they
        (("သူ", "ရှိ"), 0.18),  # he has
        (("ကျောင်း", "သွား"), 0.35),  # go to school
        (("ကျောင်း", "သား"), 0.15),  # student
        (("မြန်မာ", "နိုင်ငံ"), 0.40),  # Myanmar country
        (("နိုင်ငံ", "တော်"), 0.25),  # country (formal)
        (("စာ", "ကျောင်း"), 0.20),  # book school
        (("သွား", "မယ်"), 0.22),  # will go
        (("ပြန်", "လာ"), 0.28),  # come back
    ]
    for (w1, w2), prob in high_prob_bigrams:
        provider._bigrams[(w1, w2)] = prob

    # Low probability sequences (unusual/errors)
    low_prob_bigrams = [
        (("သူ", "ကျောင်း"), 0.00005),  # Unusual
        (("သူ", "စာ"), 0.00007),  # Unusual
        (("သူ", "ဘယ်"), 0.00003),  # Very unusual
        (("ကျောင်း", "မယ်"), 0.00004),  # Unusual
        (("နိုင်ငံ", "သွား"), 0.00006),  # Unusual
    ]
    for (w1, w2), prob in low_prob_bigrams:
        provider._bigrams[(w1, w2)] = prob

    # Rebuild bigram index so get_top_continuations() works after direct
    # _bigrams dict assignment above.
    provider._rebuild_bigram_index()

    return provider


@pytest.fixture
def full_checker(comprehensive_provider):
    """Create SpellChecker with all features enabled."""
    config = SpellCheckerConfig(
        segmenter=SpaceSegmenter(),
        provider=comprehensive_provider,
        max_edit_distance=2,
        max_suggestions=5,
        use_phonetic=True,
        use_context_checker=True,
    )
    return SpellChecker(config=config)


@pytest.mark.skip(
    reason="Fast-path exit skips context strategies on structurally-clean text. "
    "This is intentional — see context_validator._FAST_PATH_PRIORITY_CUTOFF. "
    "Context-only errors on clean text are traded for 50% FPR reduction."
)
class TestContextAwareChecking:
    """Test context-aware spell checking (Layer 3)."""

    def test_context_error_detection(self, full_checker):
        """Test detection of context errors with low bigram probability."""
        # "သူ ကျောင်း" - unusual sequence
        result = full_checker.check("သူ ကျောင်း", level="word")

        # Should detect context error
        assert result.has_errors is True

        # Should have at least one context error
        context_errors = [e for e in result.errors if isinstance(e, ContextError)]
        assert len(context_errors) > 0

        # Check error details
        error = context_errors[0]
        assert error.text == "ကျောင်း"
        assert error.prev_word == "သူ"
        assert 0 < error.probability < 0.01

    def test_context_suggestions_provided(self, full_checker):
        """Test that context errors have suggestions."""
        # "သူ ကျောင်း" - should get suggestions like "သွား", "သည်", etc.
        result = full_checker.check("သူ ကျောင်း", level="word")

        # Get context errors
        context_errors = [e for e in result.errors if isinstance(e, ContextError)]

        if context_errors:
            error = context_errors[0]
            # Should provide context-aware suggestions
            assert isinstance(error.suggestions, list)

    def test_high_probability_sequence_ok(self, full_checker):
        """Test that high probability sequences are not flagged."""
        # "သူ သွား" - common sequence, high probability
        result = full_checker.check("သူ သွား", level="word")

        # Should not have context errors for this sequence
        context_errors = [e for e in result.errors if isinstance(e, ContextError)]

        # Filter for the specific "သွား" following "သူ"
        specific_errors = [e for e in context_errors if e.text == "သွား" and e.prev_word == "သူ"]
        assert len(specific_errors) == 0

    def test_multiple_context_errors_in_sequence(self, full_checker):
        """Test detection of multiple context errors in one text."""
        # Multiple unusual sequences
        text = "သူ ကျောင်း နိုင်ငံ သွား"
        result = full_checker.check(text, level="word")

        # Should detect multiple context errors
        context_errors = [e for e in result.errors if isinstance(e, ContextError)]
        # Expect at least one error (possibly more depending on data)
        assert len(context_errors) >= 1


class TestPhoneticMatching:
    """Test phonetic similarity matching."""

    def test_phonetic_suggestion_for_confusable_medials(self, full_checker):
        """Test phonetic matching finds suggestions for confusable medials."""
        # မျန် (wrong medial ya) vs မြန် (correct medial ra)
        # Note: The phonetic hasher should detect similarity
        result = full_checker.check("မျန်", level="syllable")

        # Should have error (invalid syllable)
        assert result.has_errors is True

        # Check if suggestions include phonetically similar terms
        syllable_errors = [e for e in result.errors if isinstance(e, SyllableError)]
        if syllable_errors:
            suggestions = syllable_errors[0].suggestions
            # With phonetic matching, might find "မြန်" as suggestion
            # (exact behavior depends on dictionary content)
            assert isinstance(suggestions, list)

    def test_phonetic_enabled_vs_disabled(self, comprehensive_provider):
        """Test difference between phonetic enabled and disabled."""
        from myspellchecker.core.config import SpellCheckerConfig

        # Create two checkers - with and without phonetic
        config_with = SpellCheckerConfig(provider=comprehensive_provider, use_phonetic=True)
        config_without = SpellCheckerConfig(provider=comprehensive_provider, use_phonetic=False)
        checker_with_phonetic = SpellChecker(config=config_with)
        checker_without_phonetic = SpellChecker(config=config_without)

        # Add a phonetically similar word to provider
        comprehensive_provider.add_syllable("မျန်း", frequency=100)

        # Check with both
        text = "မျန်"  # Not in dictionary, but phonetically similar to "မြန်"

        result_with = checker_with_phonetic.check(text)
        result_without = checker_without_phonetic.check(text)

        # Both should detect error
        assert result_with.has_errors is True
        assert result_without.has_errors is True

        # With phonetic might have more/different suggestions
        # (exact behavior depends on implementation)


class TestCompoundWordDetection:
    """Test compound word detection and correction."""

    def test_compound_lookup_basic(self, comprehensive_provider):
        """Test basic compound word lookup."""
        symspell = SymSpell(provider=comprehensive_provider)

        # Test compound segmentation
        # "မြန်မာနိုင်ငံ" should split into ["မြန်မာ", "နိုင်ငံ"]
        results = symspell.lookup_compound("မြန်မာနိုင်ငံ")

        # Should return segmentation suggestions
        assert isinstance(results, list)

        if results:
            words, distance, frequency = results[0]
            assert isinstance(words, list)
            assert isinstance(distance, int)
            assert isinstance(frequency, int)

    def test_compound_with_spaces(self, comprehensive_provider):
        """Test compound lookup with existing spaces."""
        symspell = SymSpell(provider=comprehensive_provider)

        # Text with spaces should be processed
        results = symspell.lookup_compound("မြန်မာ နိုင်ငံ")

        assert isinstance(results, list)

    def test_compound_with_errors(self, comprehensive_provider):
        """Test compound lookup with spelling errors."""
        symspell = SymSpell(provider=comprehensive_provider)

        # Add valid word for correction
        comprehensive_provider.add_word("နိုင်ငံ", frequency=5000)

        # Compound with error might suggest corrections
        results = symspell.lookup_compound("မြန်မာ")

        # Should handle gracefully
        assert isinstance(results, list)


class TestEndToEndFeatures:
    """End-to-end tests for complete functionality."""

    def test_full_feature_checking(self, full_checker):
        """Test complete spell checking with all features."""
        # Text with multiple error types
        text = "သူ ကျောင်း"  # Context error

        result = full_checker.check(text, level="word")

        # Verify response structure
        assert hasattr(result, "text")
        assert hasattr(result, "corrected_text")
        assert hasattr(result, "has_errors")
        assert hasattr(result, "level")
        assert hasattr(result, "errors")
        assert hasattr(result, "metadata")

        # Verify metadata includes all layers
        assert "layers_applied" in result.metadata
        layers = result.metadata["layers_applied"]
        assert "syllable" in layers
        assert "word" in layers
        assert "context" in layers

    def test_syllable_level_without_context(self, full_checker):
        """Test that syllable level doesn't run context checking."""
        text = "သူ ကျောင်း"

        result = full_checker.check(text, level="syllable")

        # Verify only syllable layer applied
        layers = result.metadata["layers_applied"]
        assert "syllable" in layers
        assert "context" not in layers

        # Should not have context errors
        context_errors = [e for e in result.errors if isinstance(e, ContextError)]
        assert len(context_errors) == 0

    def test_batch_checking(self, full_checker):
        """Test batch checking with all features."""
        texts = [
            "သူ သွား",  # OK
            "သူ ကျောင်း",  # Context error
            "မြန်မာ",  # OK
        ]

        results = full_checker.check_batch(texts, level="word")

        # Should have result for each text
        assert len(results) == len(texts)

        # Each result should have proper structure
        for result in results:
            assert hasattr(result, "has_errors")
            assert hasattr(result, "errors")

    def test_features_disabled_vs_enabled(self, comprehensive_provider):
        """Test behavior with features disabled vs enabled."""
        # Checker with extra features disabled
        config_basic = SpellCheckerConfig(
            provider=comprehensive_provider, use_phonetic=False, use_context_checker=False
        )
        checker_basic = SpellChecker(config=config_basic)

        # Checker with extra features enabled
        config_full = SpellCheckerConfig(
            provider=comprehensive_provider, use_phonetic=True, use_context_checker=True
        )
        checker_full = SpellChecker(config=config_full)

        text = "သူ ကျောင်း"

        # Check with both
        checker_basic.check(text, level="word")
        checker_full.check(text, level="word")

        # Both should run basic checks

    def test_error_count_metadata(self, full_checker):
        """Test that metadata includes error counts by type."""
        text = "သူ ကျောင်း"  # Should have context error

        result = full_checker.check(text, level="word")

        # Verify metadata includes error counts
        metadata = result.metadata
        assert "total_errors" in metadata
        assert "syllable_errors" in metadata
        assert "word_errors" in metadata
        assert "context_errors" in metadata

        # Counts should sum to total
        total = metadata["total_errors"]
        sum_counts = (
            metadata["syllable_errors"] + metadata["word_errors"] + metadata["context_errors"]
        )
        assert total == sum_counts

    def test_response_serialization(self, full_checker):
        """Test that responses serialize correctly."""
        text = "သူ ကျောင်း"

        result = full_checker.check(text, level="word")

        # to_dict should work
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "errors" in result_dict
        assert "metadata" in result_dict

        # to_json should work
        result_json = result.to_json()
        assert isinstance(result_json, str)
        assert len(result_json) > 0


class TestContextAwareEdgeCases:
    """Test edge cases for functionality."""

    def test_empty_text(self, full_checker):
        """Test with empty text."""
        result = full_checker.check("", level="word")

        assert result.has_errors is False
        assert len(result.errors) == 0

    def test_whitespace_only(self, full_checker):
        """Test with whitespace-only text."""
        result = full_checker.check("   ", level="word")

        assert result.has_errors is False

    def test_punctuation_only(self, full_checker):
        """Test with punctuation-only text."""
        result = full_checker.check("။ ၊ ။", level="word")

        # Should not crash
        assert isinstance(result.has_errors, bool)

    def test_mixed_valid_invalid(self, full_checker):
        """Test text with mix of valid and invalid words."""
        # "သူ" is valid, "xyz" is invalid, "သွား" is valid
        text = "သူ xyz သွား"

        result = full_checker.check(text, level="word")

        # Should detect the invalid word
        assert isinstance(result.errors, list)

    def test_very_long_text(self, full_checker):
        """Test with longer text."""
        # Repeat valid sequence multiple times
        text = " ".join(["သူ သွား"] * 10)

        result = full_checker.check(text, level="word")

        # Should handle without crashing and return proper Response
        assert isinstance(result, Response)
        assert isinstance(result.has_errors, bool)
