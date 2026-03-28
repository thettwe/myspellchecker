import pytest

from myspellchecker import SpellChecker
from myspellchecker.providers import MemoryProvider
from myspellchecker.text.normalize import normalize


@pytest.fixture
def regression_checker():
    """
    Fixture for regression tests with a MemoryProvider.
    Pre-populated with specific words needed for the test cases.
    """
    provider = MemoryProvider()

    # Words for test_regression_valid_words
    # "ငါ့လို", "ငါပျော်မယ်", "ပျော်"
    # Need constituent syllables and words
    valid_data = ["ငါ", "ငါ့", "လို", "ငါ့လို", "ပျော်", "မယ်", "ငါပျော်မယ်"]
    for w in valid_data:
        provider.add_syllable(w, frequency=100)  # High frequency to pass threshold
        provider.add_word(w, frequency=100)

    # Add bigram probabilities for valid combinations
    provider.add_bigram("ငါ့", "လို", 0.1)  # Valid combination
    provider.add_bigram("ငါ", "ပျော်", 0.1)
    provider.add_bigram("ပျော်", "မယ်", 0.1)

    # Words for test_regression_invalid_words
    # Target words for corrections
    targets = ["လူမျိုး", "မုန့်", "နူး", "လုပ်", "နေ", "လုပ်နေ"]
    for w in targets:
        provider.add_syllable(w)
        provider.add_word(w)

    # Specific component parts for segmentation logic
    provider.add_syllable("လူ", frequency=100)
    provider.add_syllable("မျိုး", frequency=100)

    # Disable context checking since we don't have comprehensive n-gram data
    # Also disable colloquial detection to test pure word validation
    from myspellchecker.core.config import SpellCheckerConfig, ValidationConfig

    config = SpellCheckerConfig(
        provider=provider,
        use_context_checker=False,
        validation=ValidationConfig(colloquial_strictness="off"),
    )
    c = SpellChecker(config=config)
    return c


def test_regression_valid_words(regression_checker):
    """Test valid words/phrases that were previously flagged as errors."""
    valid_phrases = [
        "ငါ့လို",  # Low probability bigram, should pass with new threshold
        "ငါပျော်မယ်",  # Valid sentence "I will be happy", confused with "tell"
        "ပျော်",  # Valid syllable/word
    ]

    for text in valid_phrases:
        result = regression_checker.check(text, level="word")
        assert not result.has_errors, (
            f"Valid phrase '{text}' incorrectly flagged as error: {result.errors}"
        )


def test_regression_invalid_words(regression_checker):
    """Test invalid words/phrases that were previously missed."""
    invalid_phrases = [
        ("လူမြိုး", "လူမြိုး"),  # Typo for လူမျိုး (Input: လူမြိုး -> Normalized: လူမြိုး)
        ("မူန်ူး", "မူန်ူး"),  # Corrupt syllable (Asat usage) - now detected as whole
        # Note: "လုပ်နီ" was removed — "နီ" is now recognized as a valid word
        # by the segmentation model (it means "red/near" in Myanmar), so the
        # compound check passes with edit_distance=0.
    ]

    for text, error_segment_raw in invalid_phrases:
        # Debug segmentation

        result = regression_checker.check(text, level="word")
        assert result.has_errors, f"Invalid phrase '{text}' failed to flag error."

        # Check if the text itself has an error OR if a specific segment has an error
        # With updated segmentation behavior, the error might be on the full text
        # or on a normalized segment
        if text == "မူန်ူး":
            # The entire string is now treated as invalid syllable
            expected_error_segment = normalize("မူန်ူး")
        elif text == "လူမြိုး":
            # "လူမြိုး" segments to "လူ", "မြိုး". We expect error on "မြိုး".
            expected_error_segment = normalize("မြိုး")
        else:
            expected_error_segment = normalize(error_segment_raw)

        # Check if specific segment is flagged
        found = any(e.text == expected_error_segment for e in result.errors)

        assert found, (
            f"Expected error segment '{expected_error_segment}' not found in {result.errors}"
        )


def test_normalization_rendering(regression_checker):
    """Test that suggestions return canonical normalization order (I before U)."""
    # 'လူမြိုး' -> Suggests 'မျိုး'
    # 'မျိုး' should be 102D 102F (I U), not 102F 102D (U I)

    result = regression_checker.check("လူမြိုး", level="word")
    # Find error for 'မြုိး'
    error = next(
        (e for e in result.errors if "မျိုး" in e.suggestions or "မျုိး" in e.suggestions), None
    )

    if error:
        # Check top suggestion
        top = error.suggestions[0]
        # Normalize it (should be stable)
        from myspellchecker.text.normalize import normalize

        assert top == normalize(top), "Suggestion is not normalized?"

        # Check codepoints if it contains both I and U
        if "\u102d" in top and "\u102f" in top:
            i_idx = top.find("\u102d")
            u_idx = top.find("\u102f")
            assert i_idx < u_idx, (
                f"Canonical ordering failed: {top} (I index {i_idx} must be < U index {u_idx})"
            )
