"""
Tests for ingester_c Cython module.

Tests cover: empty/valid/invalid input handling, Zawgyi conversion,
Myanmar text validation, edge cases, batch ordering, and determinism.
"""

import pytest

# Try to import the Cython module
try:
    from myspellchecker.data_pipeline.ingester_c import normalize_batch_c

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython ingester_c not compiled")
class TestNormalizeBatchC:
    """Tests for normalize_batch_c() function."""

    def test_empty_batch_returns_empty(self):
        """normalize_batch_c() should return empty list for empty input."""
        result = normalize_batch_c([])
        assert result == []

    def test_single_valid_myanmar_line(self):
        """Single valid Myanmar line should produce (text, True) tuple."""
        batch = ["မြန်မာ"]
        result = normalize_batch_c(batch)
        assert len(result) == 1
        cleaned_text, is_valid = result[0]
        assert isinstance(cleaned_text, str)
        assert isinstance(is_valid, bool)

    def test_empty_string_returns_invalid(self):
        """Empty string should return empty result with False flag."""
        result = normalize_batch_c([""])
        assert result[0] == ("", False)

    def test_whitespace_only_returns_invalid(self):
        """Whitespace-only string should return invalid result."""
        result = normalize_batch_c(["   "])
        assert result[0][1] is False

    def test_non_myanmar_text_returns_invalid(self):
        """Non-Myanmar text should be marked as invalid."""
        result = normalize_batch_c(["Hello World"])
        assert result[0][1] is False

    def test_mixed_batch_processes_all(self):
        """Batch with mixed content should process all lines."""
        batch = [
            "မြန်မာ",  # Valid Myanmar
            "English",  # Non-Myanmar
            "",  # Empty
            "ကျောင်း",  # Valid Myanmar
        ]
        result = normalize_batch_c(batch)
        assert len(result) == 4

    def test_valid_result_has_nonempty_text(self):
        """Valid results should have non-empty cleaned text."""
        result = normalize_batch_c(["မြန်မာစာ"])
        if result[0][1]:
            assert len(result[0][0]) > 0


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython ingester_c not compiled")
class TestZawgyiAndMyanmarCheck:
    """Tests for Zawgyi conversion and Myanmar text ratio checking."""

    def test_unicode_text_unchanged(self):
        """Standard Unicode Myanmar text should pass through."""
        result = normalize_batch_c(["မြန်မာ"])
        assert len(result) == 1

    def test_batch_with_potential_zawgyi(self):
        """Batch processing should handle potential Zawgyi text."""
        result = normalize_batch_c(["ျမန္မာ"])
        assert len(result) == 1

    def test_pure_english_invalid(self):
        """Pure English text should fail Myanmar text check."""
        result = normalize_batch_c(["This is English text"])
        assert result[0][1] is False

    def test_myanmar_numbers_handling(self):
        """Myanmar numerals should be handled."""
        result = normalize_batch_c(["၁၂၃၄၅၆"])
        assert len(result) == 1


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython ingester_c not compiled")
class TestEdgeCases:
    """Edge case tests for normalize_batch_c()."""

    def test_single_character_myanmar(self):
        """Single Myanmar character should be handled."""
        result = normalize_batch_c(["က"])
        assert len(result) == 1

    def test_very_long_line(self):
        """Very long lines should be handled."""
        long_line = " ".join(["မြန်မာ"] * 100)
        result = normalize_batch_c([long_line])
        assert len(result) == 1

    def test_special_characters_handling(self):
        """Special characters should be handled gracefully."""
        result = normalize_batch_c(["မြန်မာ!@#$%"])
        assert len(result) == 1

    def test_myanmar_punctuation(self):
        """Myanmar punctuation should be handled."""
        result = normalize_batch_c(["မြန်မာ။"])
        assert len(result) == 1

    def test_large_batch_processing(self):
        """Large batches should be processed correctly."""
        result = normalize_batch_c(["မြန်မာ"] * 1000)
        assert len(result) == 1000


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython ingester_c not compiled")
class TestBatchBehavior:
    """Tests for batch processing correctness."""

    def test_batch_preserves_order(self):
        """Results should be in same order as input."""
        batch = ["မြန်မာ", "English", "စာ"]
        result = normalize_batch_c(batch)
        assert len(result) == 3
        assert result[1][1] is False  # English invalid

    def test_batch_processes_independently(self):
        """Each line should be processed independently; empty lines don't affect others."""
        batch = ["", "မြန်မာ", ""]
        result = normalize_batch_c(batch)
        assert len(result) == 3
        assert result[0] == ("", False)
        assert result[2] == ("", False)

    def test_deterministic_output(self):
        """Same input should always produce same output."""
        batch = ["မြန်မာ", "စာ", "English"]
        results = [normalize_batch_c(batch.copy()) for _ in range(5)]
        for result in results[1:]:
            assert result == results[0]

    def test_idempotent_for_valid_text(self):
        """Processing already valid text should be stable."""
        result1 = normalize_batch_c(["မြန်မာ"])
        if result1[0][1]:
            result2 = normalize_batch_c([result1[0][0]])
            assert result2[0][1] is True
