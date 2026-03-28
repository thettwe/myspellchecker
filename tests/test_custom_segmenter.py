"""
Integration tests for Custom Segmenter support.

Verifies that the SpellChecker correctly accepts and utilizes
a user-provided Segmenter implementation.
"""

from typing import List

import pytest
from pydantic import ValidationError

from myspellchecker import SpellChecker
from myspellchecker.providers import MemoryProvider
from myspellchecker.segmenters import Segmenter


class DashSegmenter(Segmenter):
    """
    A simple test segmenter that splits syllables by '-' and words by ' '.
    """

    def segment_syllables(self, text: str) -> List[str]:
        # Remove spaces for syllable processing if mixed
        clean_text = text.replace(" ", "-")
        return [s for s in clean_text.split("-") if s]

    def segment_words(self, text: str) -> List[str]:
        return [w for w in text.split(" ") if w]

    def segment_sentences(self, text: str) -> List[str]:
        return [text]


@pytest.fixture
def custom_checker():
    # Create a provider with data matching our custom segmentation format
    provider = MemoryProvider()
    provider.add_syllable("မြန်", frequency=100)
    provider.add_syllable("ပေ", frequency=100)  # "ပေ" has no ha_htoe confusion pair
    provider.add_word("မြန်-ပေ", frequency=50)

    from myspellchecker.core.config import SpellCheckerConfig

    segmenter = DashSegmenter()
    config = SpellCheckerConfig(segmenter=segmenter, provider=provider, use_phonetic=False)
    return SpellChecker(config=config)


class TestCustomSegmenterIntegration:
    def test_custom_syllable_segmentation(self, custom_checker):
        """Test that custom syllable segmentation logic is used."""
        # Input "မြန်-ပေ" -> Syllables ["မြန်", "ပေ"]
        # Both are in dictionary -> Valid

        result = custom_checker.check("မြန်-ပေ", level="syllable")
        assert result.has_errors is False

    def test_custom_segmentation_error_detection(self, custom_checker):
        """Test detecting errors with custom segmenter."""
        # "မြန်-ဂ" -> Syllables ["မြန်", "ဂ"]
        # "ဂ" not in dict -> Error

        result = custom_checker.check("မြန်-ဂ", level="syllable")
        assert result.has_errors is True
        assert len(result.errors) == 1
        assert result.errors[0].text == "ဂ"

    def test_custom_word_segmentation(self, custom_checker):
        """Test that custom word segmentation logic is used."""
        # Input "မြန်-ပေ" -> Word ["မြန်-ပေ"] (split by space, none here)
        # "မြန်-ပေ" is in dict -> Valid

        result = custom_checker.check("မြန်-ပေ", level="word")
        assert result.has_errors is False

    def test_integrity_of_flow(self):
        """Verify the interface contract enforcement."""
        # If a user provides an object that is NOT a Segmenter subclass
        with pytest.raises(ValidationError):
            SpellChecker(segmenter=object())  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
