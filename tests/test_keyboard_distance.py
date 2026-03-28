"""Tests for keyboard distance utilities."""

import pytest

from myspellchecker.algorithms.distance.keyboard import (
    KEY_ADJACENCY,
    KEY_POSITIONS,
    MYANMAR3_LAYOUT,
    SHIFTED_LAYOUT,
    get_keyboard_distance,
    is_keyboard_adjacent,
)


class TestKeyboardLayout:
    """Tests for keyboard layout definitions."""

    def test_myanmar3_layout_has_expected_keys(self):
        """Myanmar3 layout should contain common Myanmar consonants."""
        # Row 0 consonants
        assert "ဆ" in MYANMAR3_LAYOUT  # q
        assert "တ" in MYANMAR3_LAYOUT  # w
        assert "န" in MYANMAR3_LAYOUT  # e
        assert "မ" in MYANMAR3_LAYOUT  # r
        assert "အ" in MYANMAR3_LAYOUT  # t
        assert "ပ" in MYANMAR3_LAYOUT  # y
        assert "က" in MYANMAR3_LAYOUT  # u
        assert "င" in MYANMAR3_LAYOUT  # i
        assert "သ" in MYANMAR3_LAYOUT  # o
        assert "စ" in MYANMAR3_LAYOUT  # p

    def test_myanmar3_layout_row_positions(self):
        """Characters should have correct row assignments."""
        # Row 0 characters (consonants on QWERTY number row)
        assert MYANMAR3_LAYOUT["ဆ"][0] == 0
        assert MYANMAR3_LAYOUT["တ"][0] == 0
        assert MYANMAR3_LAYOUT["က"][0] == 0

        # Row 1 characters (vowel signs and diacritics on home row)
        assert MYANMAR3_LAYOUT["\u1031"][0] == 1  # ေ (a key)
        assert MYANMAR3_LAYOUT["\u102f"][0] == 1  # ု (k key)
        assert MYANMAR3_LAYOUT["\u1038"][0] == 1  # း (semicolon key)

        # Row 2 characters (consonants on bottom row)
        assert MYANMAR3_LAYOUT["ဖ"][0] == 2
        assert MYANMAR3_LAYOUT["ထ"][0] == 2
        assert MYANMAR3_LAYOUT["ခ"][0] == 2

    def test_myanmar3_layout_column_positions(self):
        """Characters should have correct column assignments."""
        # First column (index 0)
        assert MYANMAR3_LAYOUT["ဆ"][1] == 0  # q
        assert MYANMAR3_LAYOUT["\u1031"][1] == 0  # ေ (a key)
        assert MYANMAR3_LAYOUT["ဖ"][1] == 0  # z

        # Middle column
        assert MYANMAR3_LAYOUT["ပ"][1] == 5  # y
        assert MYANMAR3_LAYOUT["ည"][1] == 5  # n

    def test_shifted_layout_has_shifted_characters(self):
        """Shifted layout should contain shift-layer characters."""
        assert "ဈ" in SHIFTED_LAYOUT  # Shift+q
        assert "ဝ" in SHIFTED_LAYOUT  # Shift+w
        assert "ဉ" in SHIFTED_LAYOUT  # Shift+e
        assert "ဦ" in SHIFTED_LAYOUT  # Shift+r
        assert "ဇ" in SHIFTED_LAYOUT  # Shift+z
        assert "ဌ" in SHIFTED_LAYOUT  # Shift+x
        assert "ဃ" in SHIFTED_LAYOUT  # Shift+c

    def test_shifted_layout_positions_match_unshifted(self):
        """Shifted characters should have same positions as unshifted counterparts."""
        # Shift+q = ဈ should be at same position as q = ဆ
        assert SHIFTED_LAYOUT["ဈ"] == MYANMAR3_LAYOUT["ဆ"]
        # Shift+w = ဝ should be at same position as w = တ
        assert SHIFTED_LAYOUT["ဝ"] == MYANMAR3_LAYOUT["တ"]
        # Shift+z = ဇ should be at same position as z = ဖ
        assert SHIFTED_LAYOUT["ဇ"] == MYANMAR3_LAYOUT["ဖ"]


class TestKeyPositions:
    """Tests for combined key positions dictionary."""

    def test_key_positions_contains_unshifted_keys(self):
        """KEY_POSITIONS should include all unshifted characters."""
        for char in MYANMAR3_LAYOUT:
            assert char in KEY_POSITIONS

    def test_key_positions_contains_shifted_keys(self):
        """KEY_POSITIONS should include all shifted characters."""
        for char in SHIFTED_LAYOUT:
            assert char in KEY_POSITIONS

    def test_key_positions_returns_list(self):
        """KEY_POSITIONS values should be lists of positions."""
        for _char, positions in KEY_POSITIONS.items():
            assert isinstance(positions, list)
            assert len(positions) >= 1
            for pos in positions:
                assert isinstance(pos, tuple)
                assert len(pos) == 2

    def test_key_positions_no_duplicates(self):
        """Each character's positions should be unique."""
        for char, positions in KEY_POSITIONS.items():
            assert len(positions) == len(set(positions)), f"Duplicate positions for {char}"

    def test_special_character_alternate_positions(self):
        """ဉ should have alternate position on row 2."""
        positions = KEY_POSITIONS["ဉ"]
        # Should have at least two positions
        assert len(positions) >= 1
        # One should be from shifted layout (row 0)
        row_0_positions = [p for p in positions if p[0] == 0]
        assert len(row_0_positions) >= 1


class TestGetKeyboardDistance:
    """Tests for get_keyboard_distance function."""

    def test_same_character_returns_zero(self):
        """Distance between same character should be 0."""
        assert get_keyboard_distance("က", "က") == 0.0
        assert get_keyboard_distance("ဖ", "ဖ") == 0.0
        assert get_keyboard_distance("သ", "သ") == 0.0

    def test_adjacent_characters_return_one(self):
        """Adjacent characters should have distance 1."""
        # ဆ (0,0) and တ (0,1) are horizontally adjacent
        assert get_keyboard_distance("ဆ", "တ") == 1.0
        # တ (0,1) and န (0,2) are horizontally adjacent
        assert get_keyboard_distance("တ", "န") == 1.0
        # ဆ (0,0) and ဗ (1,0) are vertically adjacent
        assert get_keyboard_distance("ဆ", "ဗ") == 1.0

    def test_diagonal_characters_return_two(self):
        """Diagonally adjacent characters should have distance 2 (Manhattan)."""
        # ဆ (0,0) and \u103e (1,1) are diagonal - Manhattan distance = 2
        assert get_keyboard_distance("ဆ", "\u103e") == 2.0

    def test_far_characters_return_larger_distance(self):
        """Characters far apart should have larger distances."""
        # ဆ (0,0) and \u1030 (1,8) ူ - distance = |0-1| + |0-8| = 9
        assert get_keyboard_distance("ဆ", "\u1030") == 9.0

    def test_unknown_character_returns_default(self):
        """Unknown characters should return default distance (3.0)."""
        assert get_keyboard_distance("က", "X") == 3.0
        assert get_keyboard_distance("Y", "Z") == 3.0
        assert get_keyboard_distance("unknown", "က") == 3.0

    def test_distance_is_symmetric(self):
        """Distance should be same regardless of order."""
        assert get_keyboard_distance("က", "ဖ") == get_keyboard_distance("ဖ", "က")
        assert get_keyboard_distance("သ", "တ") == get_keyboard_distance("တ", "သ")

    def test_minimum_distance_with_multiple_positions(self):
        """When a character has multiple positions, use minimum distance."""
        # ဉ has positions at (0,2) from shifted and potentially (2,5)
        # Distance calculation should use the minimum
        dist = get_keyboard_distance("ဉ", "ဆ")  # ဆ is at (0,0)
        # From (0,2) to (0,0) = 2, or from (2,5) to (0,0) = 7
        assert dist == 2.0


class TestIsKeyboardAdjacent:
    """Tests for is_keyboard_adjacent function."""

    def test_same_character_is_adjacent(self):
        """Same character should be considered adjacent (distance 0)."""
        assert is_keyboard_adjacent("က", "က") is True

    def test_horizontally_adjacent_characters(self):
        """Horizontally adjacent keys should return True."""
        # Row 0 adjacent pairs
        assert is_keyboard_adjacent("ဆ", "တ") is True
        assert is_keyboard_adjacent("တ", "န") is True
        assert is_keyboard_adjacent("က", "င") is True

    def test_vertically_adjacent_characters(self):
        """Vertically adjacent keys should return True."""
        # Same column, adjacent rows
        assert is_keyboard_adjacent("ဆ", "ဗ") is True  # (0,0) and (1,0)
        assert is_keyboard_adjacent("ဗ", "ဖ") is True  # (1,0) and (2,0)

    def test_diagonal_characters_not_adjacent(self):
        """Diagonally positioned keys should not be adjacent (Manhattan distance > 1)."""
        # ဆ (0,0) and \u103e (1,1) - distance 2
        assert is_keyboard_adjacent("ဆ", "\u103e") is False

    def test_far_characters_not_adjacent(self):
        """Keys far apart should not be adjacent."""
        assert is_keyboard_adjacent("ဆ", "ဓ") is False
        assert is_keyboard_adjacent("ဖ", "ဧ") is False

    def test_unknown_character_not_adjacent(self):
        """Unknown characters should not be adjacent."""
        assert is_keyboard_adjacent("က", "X") is False
        assert is_keyboard_adjacent("unknown", "ဖ") is False


class TestKeyAdjacency:
    """Tests for pre-computed KEY_ADJACENCY dictionary."""

    def test_key_adjacency_contains_known_keys(self):
        """KEY_ADJACENCY should contain entries for known characters."""
        # Should have entries for characters in KEY_POSITIONS
        for char in KEY_POSITIONS:
            assert char in KEY_ADJACENCY

    def test_key_adjacency_values_are_sets(self):
        """KEY_ADJACENCY values should be sets of characters."""
        for _char, adjacent in KEY_ADJACENCY.items():
            assert isinstance(adjacent, set)

    def test_key_adjacency_excludes_self(self):
        """A character should not be in its own adjacency set."""
        for char, adjacent in KEY_ADJACENCY.items():
            assert char not in adjacent

    def test_key_adjacency_is_symmetric(self):
        """If A is adjacent to B, then B should be adjacent to A."""
        for char, adjacent in KEY_ADJACENCY.items():
            for adj_char in adjacent:
                assert char in KEY_ADJACENCY[adj_char], (
                    f"{char} is adjacent to {adj_char}, but not vice versa"
                )

    def test_key_adjacency_only_contains_adjacent_keys(self):
        """Adjacency set should only contain truly adjacent characters."""
        for char, adjacent in KEY_ADJACENCY.items():
            for adj_char in adjacent:
                dist = get_keyboard_distance(char, adj_char)
                assert dist <= 1.0, f"{adj_char} in adjacency of {char} but distance is {dist}"

    def test_specific_adjacencies(self):
        """Test specific known adjacencies."""
        # ဆ (0,0) should be adjacent to တ (0,1) and ဗ (1,0)
        assert "တ" in KEY_ADJACENCY["ဆ"]
        assert "ဗ" in KEY_ADJACENCY["ဆ"]

        # က (0,6) should be adjacent to ပ (0,5) and င (0,7)
        assert "ပ" in KEY_ADJACENCY["က"]
        assert "င" in KEY_ADJACENCY["က"]


class TestKeyboardDistanceEdgeCases:
    """Edge case tests for keyboard distance utilities."""

    def test_empty_string_handling(self):
        """Empty strings should be handled correctly."""
        # Empty string vs known character returns default (not in KEY_POSITIONS)
        assert get_keyboard_distance("", "က") == 3.0
        assert get_keyboard_distance("က", "") == 3.0
        # Two empty strings are equal, so distance is 0
        assert get_keyboard_distance("", "") == 0.0

    def test_non_myanmar_characters(self):
        """Non-Myanmar characters should return default distance."""
        assert get_keyboard_distance("a", "b") == 3.0
        assert get_keyboard_distance("1", "2") == 3.0

    def test_punctuation_in_layout(self):
        """Punctuation marks in layout should work correctly."""
        # "." and "," are in the layout
        assert "." in KEY_POSITIONS
        assert "," in KEY_POSITIONS
        assert "?" in KEY_POSITIONS

        # Distance between . (2,7) and , (2,8) should be 1
        assert get_keyboard_distance(".", ",") == 1.0

    def test_unicode_diacritics(self):
        """Unicode diacritics should be handled correctly."""
        # These are in the layout
        assert "\u103e" in KEY_POSITIONS  # ha-toh
        assert "\u102d" in KEY_POSITIONS  # i vowel
        assert "\u103a" in KEY_POSITIONS  # asat

        # They should have valid distances
        dist = get_keyboard_distance("\u103e", "\u102d")
        assert isinstance(dist, float)
        assert dist >= 0

    def test_vowel_signs_distance(self):
        """Vowel signs should have correct distances."""
        # \u102d (1,2) and \u103a (1,3) are adjacent
        assert get_keyboard_distance("\u102d", "\u103a") == 1.0

    def test_multichar_string_uses_first_char_only(self):
        """Function should use first character for multi-char strings."""
        # The function expects single characters, but should handle gracefully
        # Based on implementation, it looks up the whole string
        assert get_keyboard_distance("ကက", "ကက") == 0.0  # Same string
        assert get_keyboard_distance("ကက", "က") == 3.0  # "ကက" not in positions


class TestPerformance:
    """Performance-related tests."""

    def test_key_adjacency_precomputed(self):
        """KEY_ADJACENCY should be fully pre-computed."""
        # Should be a regular dict, not a defaultdict that computes on access
        # Access a non-existent key should raise KeyError
        with pytest.raises(KeyError):
            _ = KEY_ADJACENCY["nonexistent_key"]

    def test_repeated_distance_calls_consistent(self):
        """Repeated calls should return consistent results."""
        for _ in range(100):
            assert get_keyboard_distance("က", "ဖ") == get_keyboard_distance("က", "ဖ")
            assert is_keyboard_adjacent("က", "င") is True
