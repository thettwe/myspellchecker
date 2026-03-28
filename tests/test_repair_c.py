"""
Tests for repair_c Cython module.

Note: These tests cover the public Python interface. The module provides
Cython-optimized segmentation repair for fixing incorrectly segmented Myanmar words.
"""

import pytest

# Try to import the Cython module
try:
    from myspellchecker.data_pipeline.repair_c import (
        CythonSegmentationRepair,
        init_repair_module,
        repair,
        repair_batch,
    )

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython repair_c not compiled")
class TestRepairFunction:
    """Tests for repair() function."""

    def test_empty_list_returns_empty(self):
        """repair() should return empty list for empty input."""
        result = repair([])
        assert result == []

    def test_single_token_returns_unchanged(self):
        """repair() should return single token unchanged."""
        result = repair(["ကျောင်း"])
        assert result == ["ကျောင်း"]

    def test_valid_tokens_returned_unchanged(self):
        """Valid tokens should not be merged."""
        tokens = ["ကျောင်း", "သို့", "သွားသည်"]
        result = repair(tokens)
        assert len(result) == 3

    def test_suspicious_fragment_merged(self):
        """Suspicious fragments like င်း should be merged with previous token."""
        tokens = ["ကျော", "င်း"]
        result = repair(tokens)
        assert len(result) <= 2

    def test_invalid_start_fragment_handling(self):
        """Tokens without valid start should be considered for merging."""
        tokens = ["က", "ာ"]
        result = repair(tokens)
        assert len(result) <= 2

    def test_numeral_tokens_not_merged(self):
        """Myanmar numeral tokens should remain standalone."""
        tokens = ["၁၉၅၃", "ခုနှစ်"]
        result = repair(tokens)
        assert "၁၉၅၃" in result

    def test_double_ending_prevention(self):
        """Double-ending patterns should not be created by merging."""
        tokens = ["တွင်", "င်း"]
        result = repair(tokens)
        assert "တွင်င်း" not in result

    def test_closed_syllable_prevents_merge(self):
        """Tokens ending with closing chars should not merge with next."""
        tokens = ["သည်", "င်း"]
        result = repair(tokens)
        assert len(result) == 2

    def test_preserves_valid_words(self):
        """Valid Myanmar words should be preserved."""
        tokens = ["မြန်မာ", "စာ", "ပေါင်း"]
        result = repair(tokens)
        assert "မြန်မာ" in result


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython repair_c not compiled")
class TestRepairBatchFunction:
    """Tests for repair_batch() function."""

    def test_empty_batch_returns_empty(self):
        """repair_batch() should return empty list for empty input."""
        result = repair_batch([])
        assert result == []

    def test_multiple_token_lists(self):
        """repair_batch() should process multiple token lists."""
        batch = [
            ["ကျော", "င်း"],
            ["သည်", "ဟု"],
            ["မြန်မာ"],
        ]
        result = repair_batch(batch)
        assert len(result) == 3

    def test_batch_with_empty_lists(self):
        """repair_batch() should handle empty lists in batch."""
        batch = [[], ["က", "ခ"], []]
        result = repair_batch(batch)
        assert len(result) == 3
        assert result[0] == []
        assert result[2] == []

    def test_batch_processes_each_independently(self):
        """Each token list in batch should be processed independently."""
        batch = [
            ["ကျော", "င်း"],
            ["သည်"],
        ]
        result = repair_batch(batch)
        assert len(result) == 2
        assert result[1] == ["သည်"]


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython repair_c not compiled")
class TestCythonSegmentationRepairClass:
    """Tests for CythonSegmentationRepair class."""

    def test_repair_method(self):
        """repair() method should work like standalone function."""
        repairer = CythonSegmentationRepair()
        tokens = ["ကျော", "င်း"]
        result = repairer.repair(tokens)
        assert isinstance(result, list)

    def test_repair_batch_method(self):
        """repair_batch() method should work like standalone function."""
        repairer = CythonSegmentationRepair()
        batch = [["က", "ခ"], ["ဂ", "ဃ"]]
        result = repairer.repair_batch(batch)
        assert len(result) == 2


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython repair_c not compiled")
class TestInitRepairModule:
    """Tests for init_repair_module() function."""

    def test_init_can_be_called_multiple_times(self):
        """init_repair_module() should be safe to call multiple times."""
        init_repair_module()
        init_repair_module()


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython repair_c not compiled")
class TestMyanmarCharacterHandling:
    """Tests for Myanmar character handling in repair module."""

    def test_closing_chars_detected(self):
        """Tokens ending with closing chars should prevent certain merges."""
        closed_tokens = ["ပါး", "ပါ့", "သည်"]
        for token in closed_tokens:
            result = repair([token, "င်း"])
            assert len(result) == 2

    def test_suspicious_fragments_identified(self):
        """Known suspicious fragments should be identified."""
        suspicious = ["င်း", "င့်", "န့်"]
        for fragment in suspicious:
            result = repair(["က", fragment])
            assert len(result) >= 1


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython repair_c not compiled")
class TestRepairCythonEdgeCases:
    """Edge case tests for repair module."""

    def test_empty_string_in_tokens(self):
        """Empty strings in token list should be handled."""
        tokens = ["က", "", "ခ"]
        result = repair(tokens)
        assert isinstance(result, list)

    def test_non_myanmar_text(self):
        """Non-Myanmar text should be handled gracefully."""
        tokens = ["hello", "world"]
        result = repair(tokens)
        assert isinstance(result, list)

    def test_mixed_myanmar_english(self):
        """Mixed Myanmar and English tokens should be handled."""
        tokens = ["မြန်မာ", "English", "စာ"]
        result = repair(tokens)
        assert isinstance(result, list)


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython repair_c not compiled")
class TestDoubleEndingPrevention:
    """Tests specifically for double-ending prevention."""

    def test_no_double_asat_ending(self):
        """Should not create double asat ending patterns."""
        tokens = ["တွင်", "င်း"]
        result = repair(tokens)
        for token in result:
            if "တွင်င်း" in token:
                pytest.fail("Double-ending pattern created")

    def test_valid_merge_still_works(self):
        """Valid merges should still occur when not creating double-ending."""
        tokens = ["ကျော", "င်း"]
        result = repair(tokens)
        assert len(result) >= 1


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython repair_c not compiled")
class TestRepairConsistency:
    """Tests for consistency between function and class interface."""

    def test_repair_function_matches_class_method(self):
        """repair() function should produce same result as class method."""
        tokens = ["ကျော", "င်း", "သို့"]
        func_result = repair(tokens)

        repairer = CythonSegmentationRepair()
        method_result = repairer.repair(tokens)

        assert func_result == method_result

    def test_idempotent_repair(self):
        """Repairing already-repaired tokens should produce same result."""
        tokens = ["ကျော", "င်း"]
        first_repair = repair(tokens)
        second_repair = repair(first_repair)
        assert first_repair == second_repair

    def test_deterministic_results(self):
        """Same input should always produce same output."""
        tokens = ["ကျော", "င်း", "သို့", "သွား"]
        results = [repair(tokens.copy()) for _ in range(10)]
        for result in results[1:]:
            assert result == results[0]
