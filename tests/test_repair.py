"""
Tests for the segmentation repair module.

Tests both the pure Python implementation (SegmentationRepair)
and the Cython implementation (CythonSegmentationRepair) when available.
"""

import pytest

from myspellchecker.data_pipeline.repair import SegmentationRepair


class TestSegmentationRepairBasic:
    """Test basic repair functionality."""

    def setup_method(self):
        self.repair = SegmentationRepair()

    def test_empty_tokens(self):
        """Empty list should return empty list."""
        assert self.repair.repair([]) == []

    def test_single_token(self):
        """Single token should be returned as-is."""
        assert self.repair.repair(["ကျောင်း"]) == ["ကျောင်း"]

    def test_valid_tokens_unchanged(self):
        """Valid tokens should not be merged."""
        tokens = ["ကျောင်း", "သည်", "ကောင်း"]
        assert self.repair.repair(tokens) == tokens

    def test_merge_split_syllable(self):
        """Split syllable should be merged back together."""
        # "ကျော" + "င်း" -> "ကျောင်း"
        tokens = ["ကျော", "င်း"]
        result = self.repair.repair(tokens)
        assert result == ["ကျောင်း"]

    def test_merge_multiple_fragments(self):
        """Multiple fragmented words should be repaired."""
        # "ကျော" + "င်း" + "သို့" -> ["ကျောင်း", "သို့"]
        tokens = ["ကျော", "င်း", "သို့"]
        result = self.repair.repair(tokens)
        assert result == ["ကျောင်း", "သို့"]


class TestDoubleEndingPrevention:
    """Test prevention of double-ending patterns."""

    def setup_method(self):
        self.repair = SegmentationRepair()

    def test_would_create_double_ending_method(self):
        """Test the _would_create_double_ending helper method."""
        # Closed prev + fragment -> True
        assert self.repair._would_create_double_ending("တွင်", "င်း") is True
        assert self.repair._would_create_double_ending("သော်", "င်း") is True
        assert self.repair._would_create_double_ending("သည်", "င့်") is True
        assert self.repair._would_create_double_ending("ပြီး", "န့်") is True

        # Open prev + fragment -> False (merge is OK)
        assert self.repair._would_create_double_ending("ကျော", "င်း") is False
        assert self.repair._would_create_double_ending("တွ", "င်း") is False

        # Closed prev + non-fragment -> False
        assert self.repair._would_create_double_ending("တွင်", "ကျောင်း") is False

        # Empty inputs -> False
        assert self.repair._would_create_double_ending("", "င်း") is False
        assert self.repair._would_create_double_ending("တွင်", "") is False
        assert self.repair._would_create_double_ending("", "") is False

    def test_prevent_double_ending_with_asat(self):
        """Don't merge when prev ends with asat (်) and current is fragment."""
        # "တွင်" + "င်း" should NOT become "တွင်င်း"
        tokens = ["တွင်", "င်း"]
        result = self.repair.repair(tokens)
        # Fragment should remain separate (will be filtered later)
        assert result == ["တွင်", "င်း"]

    def test_prevent_double_ending_with_visarga(self):
        """Don't merge when prev ends with visarga (း) and current is fragment."""
        # "ပြီး" + "င်း" should NOT become "ပြီးင်း"
        tokens = ["ပြီး", "င်း"]
        result = self.repair.repair(tokens)
        assert result == ["ပြီး", "င်း"]

    def test_prevent_double_ending_with_tone(self):
        """Don't merge when prev ends with tone mark (့) and current is fragment."""
        # "လို့" + "င်း" should NOT become "လို့င်း"
        tokens = ["လို့", "င်း"]
        result = self.repair.repair(tokens)
        assert result == ["လို့", "င်း"]

    def test_allow_valid_merge_with_open_syllable(self):
        """Allow merge when prev is open syllable (no closing char)."""
        # "ကျော" + "င်း" -> "ကျောင်း" (valid merge)
        tokens = ["ကျော", "င်း"]
        result = self.repair.repair(tokens)
        assert result == ["ကျောင်း"]

    def test_prevent_multiple_double_endings_in_sequence(self):
        """Multiple potential double-endings should all be prevented."""
        tokens = ["ကျောင်း", "တွင်", "င်း", "သည်", "န့်"]
        result = self.repair.repair(tokens)
        # All closed-syllable + fragment combinations should be rejected
        # Valid syllables like ကျောင်း and တွင် remain separate
        # Fragments after closed syllables (တွင် + င်း, သည် + န့်) are NOT merged
        assert result == ["ကျောင်း", "တွင်", "င်း", "သည်", "န့်"]


class TestNumeralHandling:
    """Test handling of Myanmar numerals in repair."""

    def setup_method(self):
        self.repair = SegmentationRepair()

    def test_numeral_tokens_not_merged(self):
        """Numeral tokens should not be merged with previous word."""
        tokens = ["နှစ်", "၁၉၅၃"]
        result = self.repair.repair(tokens)
        # Numerals should remain separate
        assert result == ["နှစ်", "၁၉၅၃"]

    def test_numeral_followed_by_word(self):
        """Numeral followed by word should remain separate."""
        tokens = ["၁၉၅၃", "ခုနှစ်"]
        result = self.repair.repair(tokens)
        assert result == ["၁၉၅၃", "ခုနှစ်"]


class TestCythonRepair:
    """Test Cython implementation when available."""

    @pytest.fixture(autouse=True)
    def setup_cython(self):
        """Try to import Cython repair."""
        try:
            from myspellchecker.data_pipeline.repair_c import (
                CythonSegmentationRepair,
                repair,
            )

            self.cython_repair = CythonSegmentationRepair()
            self.repair_func = repair
            self.cython_available = True
        except ImportError:
            self.cython_available = False
            pytest.skip("Cython repair module not available")

    def test_cython_basic_repair(self):
        """Test basic repair with Cython."""
        if not self.cython_available:
            pytest.skip("Cython not available")

        tokens = ["ကျော", "င်း"]
        result = self.cython_repair.repair(tokens)
        assert result == ["ကျောင်း"]

    def test_cython_prevents_double_ending(self):
        """Test double-ending prevention with Cython."""
        if not self.cython_available:
            pytest.skip("Cython not available")

        tokens = ["တွင်", "င်း"]
        result = self.cython_repair.repair(tokens)
        # Should NOT merge
        assert result == ["တွင်", "င်း"]

    def test_cython_function_directly(self):
        """Test the repair function directly."""
        if not self.cython_available:
            pytest.skip("Cython not available")

        tokens = ["တွင်", "င်း"]
        result = self.repair_func(tokens)
        assert result == ["တွင်", "င်း"]


class TestRepairEdgeCases:
    """Test edge cases in repair logic."""

    def setup_method(self):
        self.repair = SegmentationRepair()

    def test_fragment_at_start(self):
        """Fragment at start of list should be kept."""
        tokens = ["င်း", "ကျောင်း"]
        result = self.repair.repair(tokens)
        # First token is kept, second is separate valid word
        assert result == ["င်း", "ကျောင်း"]

    def test_all_fragments(self):
        """List of all fragments should remain as-is."""
        tokens = ["င်း", "င့်", "န့်"]
        result = self.repair.repair(tokens)
        # All fragments, first kept, others not merged due to closing chars
        assert result == ["င်း", "င့်", "န့်"]

    def test_long_sentence(self):
        """Test repair on a longer sentence."""
        tokens = ["ကျော", "င်း", "မှာ", "ပညာ", "သင်", "ကြ", "သည်"]
        result = self.repair.repair(tokens)
        # "ကျော" + "င်း" should merge
        assert "ကျောင်း" in result
        assert len(result) < len(tokens)
