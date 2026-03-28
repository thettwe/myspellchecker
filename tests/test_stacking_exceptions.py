"""Tests for STACKING_EXCEPTIONS constant.

Add missing stacking exception patterns for Pali/Sanskrit loanwords.

Validates that specific consonant stacking combinations are properly
defined for Pali/Sanskrit loanwords commonly found in Myanmar text.
"""

import pytest

from myspellchecker.core.constants import (
    CONSONANTS,
    STACKING_EXCEPTIONS,
)


class TestStackingExceptionsConstant:
    """Tests for the STACKING_EXCEPTIONS constant definition."""

    def test_stacking_exceptions_exists(self):
        """Test that STACKING_EXCEPTIONS is defined and non-empty."""
        assert STACKING_EXCEPTIONS is not None
        assert len(STACKING_EXCEPTIONS) > 0

    def test_stacking_exceptions_is_set(self):
        """Test that STACKING_EXCEPTIONS is a set for O(1) lookup."""
        assert isinstance(STACKING_EXCEPTIONS, (set, frozenset))

    def test_all_entries_are_tuples(self):
        """Test that all entries are tuples of two consonants."""
        for entry in STACKING_EXCEPTIONS:
            assert isinstance(entry, tuple), f"Entry {entry} is not a tuple"
            assert len(entry) == 2, f"Entry {entry} does not have 2 elements"

    def test_all_entries_are_consonants(self):
        """Test that all entries contain valid Myanmar consonants."""
        for upper, lower in STACKING_EXCEPTIONS:
            assert upper in CONSONANTS, f"Upper {repr(upper)} is not a consonant"
            assert lower in CONSONANTS, f"Lower {repr(lower)} is not a consonant"


class TestIssue630StackingAdditions:
    """Tests for the specific additions."""

    def test_cna_combination_present(self):
        """Test that cna (စ္န) combination is present."""
        cna = ("\u1005", "\u1014")  # Ca + Na
        assert cna in STACKING_EXCEPTIONS, "Missing cna (စ္န) combination"

    def test_tna_combination_present(self):
        """Test that tna (တ္န) combination is present - common in ratna (ရတ္န)."""
        tna = ("\u1010", "\u1014")  # Ta + Na
        assert tna in STACKING_EXCEPTIONS, "Missing tna (တ္န) combination for 'ratna'"

    def test_mha_combination_present(self):
        """Test that mha (မ္ဟ) combination is present - Brahmi loanwords."""
        mha = ("\u1019", "\u101f")  # Ma + Ha
        assert mha in STACKING_EXCEPTIONS, "Missing mha (မ္ဟ) combination"


class TestStackingExceptionsCategories:
    """Tests for various categories of stacking exceptions."""

    def test_gemination_patterns_present(self):
        """Test that same-consonant gemination patterns are present."""
        geminations = [
            ("\u1000", "\u1000"),  # kka
            ("\u1010", "\u1010"),  # tta
            ("\u1015", "\u1015"),  # ppa
            ("\u1019", "\u1019"),  # mma
            ("\u101c", "\u101c"),  # lla
        ]
        for pattern in geminations:
            assert pattern in STACKING_EXCEPTIONS, f"Missing gemination {pattern}"

    def test_aspirated_pairs_present(self):
        """Test that unaspirated + aspirated pairs are present."""
        aspirated_pairs = [
            ("\u1000", "\u1001"),  # kkha
            ("\u1010", "\u1011"),  # ttha
            ("\u1015", "\u1016"),  # ppha
        ]
        for pattern in aspirated_pairs:
            assert pattern in STACKING_EXCEPTIONS, f"Missing aspirated pair {pattern}"

    def test_nasal_combinations_present(self):
        """Test that nasal + stop combinations are present."""
        nasal_combos = [
            ("\u1014", "\u1010"),  # nta
            ("\u1014", "\u1012"),  # nda
            ("\u1019", "\u1015"),  # mpa
            ("\u1019", "\u1017"),  # mba
        ]
        for pattern in nasal_combos:
            assert pattern in STACKING_EXCEPTIONS, f"Missing nasal combo {pattern}"

    def test_ya_combinations_present(self):
        """Test that consonant + ya combinations are present."""
        ya_combos = [
            ("\u1000", "\u101a"),  # kya
            ("\u1010", "\u101a"),  # tya
            ("\u1019", "\u101a"),  # mya
        ]
        for pattern in ya_combos:
            assert pattern in STACKING_EXCEPTIONS, f"Missing ya combo {pattern}"

    def test_ra_combinations_present(self):
        """Test that consonant + ra combinations are present."""
        ra_combos = [
            ("\u1000", "\u101b"),  # kra
            ("\u1010", "\u101b"),  # tra
            ("\u1015", "\u101b"),  # pra
        ]
        for pattern in ra_combos:
            assert pattern in STACKING_EXCEPTIONS, f"Missing ra combo {pattern}"

    def test_retroflex_combinations_present(self):
        """Test that retroflex consonant combinations are present."""
        # ဏ (U+100F) with ဍ (U+100D) and ဌ (U+100C)
        retroflex_combos = [
            ("\u100f", "\u100d"),  # ṇḍa
            ("\u100f", "\u100c"),  # ṇṭha
        ]
        for pattern in retroflex_combos:
            assert pattern in STACKING_EXCEPTIONS, f"Missing retroflex combo {pattern}"


class TestStackingExceptionsCompleteness:
    """Tests for completeness of stacking exceptions."""

    def test_minimum_exception_count(self):
        """Test that we have a reasonable minimum number of exceptions."""
        # Based on Pali/Sanskrit phonotactics, we should have at least 50 patterns
        assert len(STACKING_EXCEPTIONS) >= 50, (
            f"Expected at least 50 stacking exceptions, got {len(STACKING_EXCEPTIONS)}"
        )

    def test_ha_as_lower_present(self):
        """Test that consonant + ha (as lower) combinations are present."""
        ha_lower = [
            ("\u1000", "\u101f"),  # kha (stacked)
            ("\u1010", "\u101f"),  # tha (stacked)
            ("\u1014", "\u101f"),  # nha
        ]
        for pattern in ha_lower:
            assert pattern in STACKING_EXCEPTIONS, f"Missing ha-lower combo {pattern}"

    def test_sa_combinations_present(self):
        """Test that sa (သ) combinations are present."""
        sa_combos = [
            ("\u101e", "\u1010"),  # sta
            ("\u101e", "\u1014"),  # sna
            ("\u101e", "\u1019"),  # sma
            ("\u101e", "\u1015"),  # spa
        ]
        for pattern in sa_combos:
            assert pattern in STACKING_EXCEPTIONS, f"Missing sa combo {pattern}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
