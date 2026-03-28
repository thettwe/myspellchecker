"""Unit tests for text/validator.py — is_truncated_word() and get_truncation_candidates()."""

import pytest


class TestTruncationDetection:
    """Tests for frequency-based truncation detection."""

    def test_is_truncated_word_detects_truncation(self):
        """Test is_truncated_word detects obvious truncations."""
        from myspellchecker.text.validator import is_truncated_word

        # Mock frequency lookup
        freq_data = {
            "ချိန": 83,
            "ချိန်": 179977,
            "အောင": 209,
            "အောင်": 392646,
        }

        def get_freq(w):
            return freq_data.get(w, 0)

        # Test truncated word detection
        is_trunc, complete = is_truncated_word("ချိန", get_freq)
        assert is_trunc is True
        assert complete == "ချိန်"

        is_trunc, complete = is_truncated_word("အောင", get_freq)
        assert is_trunc is True
        assert complete == "အောင်"

    def test_is_truncated_word_respects_pali_whitelist(self):
        """Test is_truncated_word doesn't flag Pali words."""
        from myspellchecker.text.validator import is_truncated_word

        # Mock frequency lookup - even if complete form exists
        freq_data = {
            "ဒေသ": 436426,
            "ဒေသ်": 100,  # Hypothetical complete form
            "ပထမ": 96325,
            "ပထမ်": 50,
        }

        def get_freq(w):
            return freq_data.get(w, 0)

        # Pali words should NOT be flagged as truncated
        is_trunc, _ = is_truncated_word("ဒေသ", get_freq)
        assert is_trunc is False

        is_trunc, _ = is_truncated_word("ပထမ", get_freq)
        assert is_trunc is False

    def test_is_truncated_word_skips_non_consonant_endings(self):
        """Test is_truncated_word skips words not ending with consonants."""
        from myspellchecker.text.validator import is_truncated_word

        freq_data = {"မြန်မာ": 1000000}

        def get_freq(w):
            return freq_data.get(w, 0)

        # Word ending with vowel sign - should not be checked
        is_trunc, _ = is_truncated_word("မြန်မာ", get_freq)
        assert is_trunc is False

    def test_is_truncated_word_threshold(self):
        """Test is_truncated_word respects frequency threshold."""
        from myspellchecker.text.validator import is_truncated_word

        # Low ratio - should NOT be flagged
        freq_data = {
            "တစ": 1000,
            "တစ်": 5000,  # Only 5x more frequent, below default 100x threshold
        }

        def get_freq(w):
            return freq_data.get(w, 0)

        is_trunc, _ = is_truncated_word("တစ", get_freq)
        assert is_trunc is False

        # But with lower threshold, should be flagged
        is_trunc, _ = is_truncated_word("တစ", get_freq, frequency_ratio_threshold=3)
        assert is_trunc is True

    def test_get_truncation_candidates(self):
        """Test get_truncation_candidates returns sorted candidates."""
        from myspellchecker.text.validator import get_truncation_candidates

        freq_data = {
            "ချိန": 83,
            "ချိန်": 179977,
            "အောင": 209,
            "အောင်": 392646,
            "မြန်မာ": 1000000,  # Not truncated
            "ဒေသ": 436426,  # Pali - not truncated
        }

        def get_freq(w):
            return freq_data.get(w, 0)

        words = [
            ("ချိန", 83),
            ("အောင", 209),
            ("မြန်မာ", 1000000),
            ("ဒေသ", 436426),
        ]

        candidates = get_truncation_candidates(words, get_freq)

        # Should find 2 truncation candidates
        assert len(candidates) == 2

        # Should be sorted by ratio descending
        # ချိန: 179977/83 ≈ 2168
        # အောင: 392646/209 ≈ 1879
        assert candidates[0][0] == "ချိန"
        assert candidates[1][0] == "အောင"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
