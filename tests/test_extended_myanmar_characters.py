"""
Tests for Unicode Scope - Extended Myanmar Character Detection.

Tests cover:
1. MYANMAR_CORE_CHARS excludes Extended Core Block (U+1050-U+109F), Extended-A/B blocks
2. MYANMAR_EXTENDED_CORE_BLOCK contains U+1050-U+109F
3. EXTENDED_MYANMAR_CHARS combines all extended blocks
4. get_myanmar_char_set() behavior with both flag values
5. has_extended_myanmar_chars() detection (including Extended Core Block)
6. EXTENDED_MYANMAR_PATTERN regex integration
"""

from myspellchecker.core.constants import (
    ALL_MYANMAR_CHARS,
    EXTENDED_MYANMAR_CHARS,
    MYANMAR_CORE_CHARS,
    MYANMAR_EXTENDED_A_CHARS,
    MYANMAR_EXTENDED_B_CHARS,
    MYANMAR_EXTENDED_CORE_BLOCK,
    NON_STANDARD_CHARS,
    get_myanmar_char_set,
    has_extended_myanmar_chars,
)


class TestMyanmarCoreChars:
    """Tests for MYANMAR_CORE_CHARS character set."""

    def test_core_chars_excludes_extended_core_block(self) -> None:
        """MYANMAR_CORE_CHARS should not contain Extended Core Block (U+1050-U+109F)."""
        # Extended Core Block contains Shan/Mon/Karen additions
        for code_point in range(0x1050, 0x10A0):
            char = chr(code_point)
            assert char not in MYANMAR_CORE_CHARS, (
                f"Extended Core Block char U+{code_point:04X} should not be in MYANMAR_CORE_CHARS"
            )

    def test_core_chars_excludes_extended_a(self) -> None:
        """MYANMAR_CORE_CHARS should not contain Extended-A characters."""
        # Extended-A: U+AA60-U+AA7F (Shan, Khamti, Aiton, Phake, Pa'O Karen)
        for char in MYANMAR_EXTENDED_A_CHARS:
            assert char not in MYANMAR_CORE_CHARS, (
                f"Extended-A char U+{ord(char):04X} should not be in MYANMAR_CORE_CHARS"
            )

    def test_core_chars_excludes_extended_b(self) -> None:
        """MYANMAR_CORE_CHARS should not contain Extended-B characters."""
        # Extended-B: U+A9E0-U+A9FF (Shan, Pa'O)
        for char in MYANMAR_EXTENDED_B_CHARS:
            assert char not in MYANMAR_CORE_CHARS, (
                f"Extended-B char U+{ord(char):04X} should not be in MYANMAR_CORE_CHARS"
            )

    def test_core_chars_excludes_non_standard(self) -> None:
        """MYANMAR_CORE_CHARS should not contain non-standard characters."""
        # Non-standard chars (Mon/Shan specific in Core Block)
        for char in NON_STANDARD_CHARS:
            assert char not in MYANMAR_CORE_CHARS, (
                f"Non-standard char U+{ord(char):04X} should not be in MYANMAR_CORE_CHARS"
            )

    def test_core_chars_contains_main_block(self) -> None:
        """MYANMAR_CORE_CHARS should contain main Myanmar block characters."""
        # Test a few key consonants (Ka, Kha, Ga, Nga, etc.)
        burmese_consonants = "ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအ"
        for char in burmese_consonants:
            if char not in NON_STANDARD_CHARS:
                assert char in MYANMAR_CORE_CHARS, (
                    f"Burmese consonant {char} (U+{ord(char):04X}) should be in MYANMAR_CORE_CHARS"
                )

    def test_core_chars_is_proper_subset(self) -> None:
        """MYANMAR_CORE_CHARS should be a proper subset of ALL_MYANMAR_CHARS."""
        assert MYANMAR_CORE_CHARS < ALL_MYANMAR_CHARS, (
            "MYANMAR_CORE_CHARS should be a proper subset of ALL_MYANMAR_CHARS"
        )


class TestMyanmarExtendedCoreBlock:
    """Tests for MYANMAR_EXTENDED_CORE_BLOCK character set."""

    def test_extended_core_block_covers_correct_range(self) -> None:
        """MYANMAR_EXTENDED_CORE_BLOCK should contain U+1050-U+109F."""
        # Check all characters in the range
        for code_point in range(0x1050, 0x10A0):
            char = chr(code_point)
            assert char in MYANMAR_EXTENDED_CORE_BLOCK, (
                f"U+{code_point:04X} should be in MYANMAR_EXTENDED_CORE_BLOCK"
            )

    def test_extended_core_block_excludes_main_block(self) -> None:
        """MYANMAR_EXTENDED_CORE_BLOCK should not contain U+1000-U+104F."""
        for code_point in range(0x1000, 0x1050):
            char = chr(code_point)
            assert char not in MYANMAR_EXTENDED_CORE_BLOCK, (
                f"U+{code_point:04X} should not be in MYANMAR_EXTENDED_CORE_BLOCK"
            )

    def test_extended_core_block_is_separate_from_extended_a_b(self) -> None:
        """Extended Core Block should be distinct from Extended-A and Extended-B."""
        # No overlap with Extended-A
        overlap_a = MYANMAR_EXTENDED_CORE_BLOCK & MYANMAR_EXTENDED_A_CHARS
        assert len(overlap_a) == 0, "Extended Core Block should not overlap with Extended-A"

        # No overlap with Extended-B
        overlap_b = MYANMAR_EXTENDED_CORE_BLOCK & MYANMAR_EXTENDED_B_CHARS
        assert len(overlap_b) == 0, "Extended Core Block should not overlap with Extended-B"


class TestExtendedMyanmarChars:
    """Tests for EXTENDED_MYANMAR_CHARS character set."""

    def test_extended_chars_combines_all_extended_blocks(self) -> None:
        """EXTENDED_MYANMAR_CHARS should include Core Block, Extended-A, and Extended-B."""
        expected = MYANMAR_EXTENDED_CORE_BLOCK | MYANMAR_EXTENDED_A_CHARS | MYANMAR_EXTENDED_B_CHARS
        assert EXTENDED_MYANMAR_CHARS == expected

    def test_extended_chars_has_correct_ranges(self) -> None:
        """Extended chars should cover all three extended ranges."""
        # Check Extended Core Block (U+1050-U+109F)
        for code_point in range(0x1050, 0x10A0):
            assert chr(code_point) in EXTENDED_MYANMAR_CHARS, (
                f"U+{code_point:04X} should be in EXTENDED_MYANMAR_CHARS"
            )

        # Check Extended-A range (U+AA60-U+AA7F)
        for code_point in range(0xAA60, 0xAA80):
            assert chr(code_point) in EXTENDED_MYANMAR_CHARS, (
                f"U+{code_point:04X} should be in EXTENDED_MYANMAR_CHARS"
            )

        # Check Extended-B range (U+A9E0-U+A9FF)
        for code_point in range(0xA9E0, 0xAA00):
            assert chr(code_point) in EXTENDED_MYANMAR_CHARS, (
                f"U+{code_point:04X} should be in EXTENDED_MYANMAR_CHARS"
            )


class TestGetMyanmarCharSet:
    """Tests for get_myanmar_char_set() function."""

    def test_default_returns_core_chars(self) -> None:
        """get_myanmar_char_set() without args should return core chars only."""
        result = get_myanmar_char_set()
        assert result == MYANMAR_CORE_CHARS

    def test_allow_extended_false_returns_core_chars(self) -> None:
        """get_myanmar_char_set(allow_extended=False) should return core chars."""
        result = get_myanmar_char_set(allow_extended=False)
        assert result == MYANMAR_CORE_CHARS

    def test_allow_extended_true_returns_all_chars(self) -> None:
        """get_myanmar_char_set(allow_extended=True) should include extended chars."""
        result = get_myanmar_char_set(allow_extended=True)

        # Should include Extended-A chars
        for char in MYANMAR_EXTENDED_A_CHARS:
            assert char in result, (
                f"Extended-A char U+{ord(char):04X} should be included when allow_extended=True"
            )

        # Should include Extended-B chars
        for char in MYANMAR_EXTENDED_B_CHARS:
            assert char in result, (
                f"Extended-B char U+{ord(char):04X} should be included when allow_extended=True"
            )

    def test_allow_extended_true_includes_non_standard(self) -> None:
        """With allow_extended=True, non-standard chars should be INCLUDED.

        Non-standard chars (U+1022, U+1028, U+1033-U+1035) are Mon/Shan specific
        characters within the core block. When extended mode is enabled for
        non-Burmese Myanmar-script languages, these should be accepted.

        Fixed get_myanmar_char_set to include NON_STANDARD_CHARS
        when allow_extended=True.
        """
        result = get_myanmar_char_set(allow_extended=True)
        for char in NON_STANDARD_CHARS:
            assert char in result, (
                f"Non-standard char U+{ord(char):04X} should be included when allow_extended=True"
            )


class TestHasExtendedMyanmarChars:
    """Tests for has_extended_myanmar_chars() function."""

    def test_empty_string_returns_false(self) -> None:
        """Empty string should return False."""
        assert has_extended_myanmar_chars("") is False

    def test_core_myanmar_only_returns_false(self) -> None:
        """Text with only core Myanmar chars (U+1000-U+104F) should return False."""
        # Standard Burmese text
        text = "မြန်မာနိုင်ငံ"  # Myanmar country
        assert has_extended_myanmar_chars(text) is False

    def test_extended_core_block_char_returns_true(self) -> None:
        """Text containing Extended Core Block (U+1050-U+109F) chars should return True."""
        # These chars are for Shan/Mon/Karen, not Burmese
        # U+1050 is first char in Extended Core Block (Myanmar Letter Sha)
        text = "test\u1050text"
        assert has_extended_myanmar_chars(text) is True

    def test_extended_core_block_boundaries(self) -> None:
        """Test boundary characters of Extended Core Block range."""
        # First char (U+1050)
        assert has_extended_myanmar_chars("\u1050") is True
        # Last char (U+109F)
        assert has_extended_myanmar_chars("\u109f") is True
        # Just before range (U+104F) - should NOT be detected
        assert has_extended_myanmar_chars("\u104f") is False

    def test_extended_a_char_returns_true(self) -> None:
        """Text containing Extended-A chars should return True."""
        # U+AA60 is first char in Extended-A (Shan letter Kue)
        text = "test\uaa60text"
        assert has_extended_myanmar_chars(text) is True

    def test_extended_b_char_returns_true(self) -> None:
        """Text containing Extended-B chars should return True."""
        # U+A9E0 is first char in Extended-B (Myanmar Extended-B Letter E)
        text = "test\ua9e0text"
        assert has_extended_myanmar_chars(text) is True

    def test_mixed_extended_core_block_in_myanmar_text(self) -> None:
        """Mixed text with Extended Core Block should be detected."""
        # Burmese text with an Extended Core Block char inserted
        text = "မြန်\u1055မာ"  # Inserted Shan char from Core Block
        assert has_extended_myanmar_chars(text) is True

    def test_mixed_extended_a_in_myanmar_text(self) -> None:
        """Mixed text with Extended-A should be detected."""
        # Burmese text with an Extended-A char inserted
        text = "မြန်\uaa65မာ"  # Inserted Shan letter Khi
        assert has_extended_myanmar_chars(text) is True

    def test_mixed_extended_b_in_myanmar_text(self) -> None:
        """Mixed text with Extended-B should be detected."""
        # Burmese text with an Extended-B char inserted
        text = "မြန်\ua9e5မာ"  # Inserted Extended-B char
        assert has_extended_myanmar_chars(text) is True

    def test_boundary_chars_extended_a(self) -> None:
        """Test boundary characters of Extended-A range."""
        # First char (U+AA60)
        assert has_extended_myanmar_chars("\uaa60") is True
        # Last char (U+AA7F)
        assert has_extended_myanmar_chars("\uaa7f") is True
        # Just before range (U+AA5F) - should not be detected as Extended Myanmar
        assert has_extended_myanmar_chars("\uaa5f") is False

    def test_boundary_chars_extended_b(self) -> None:
        """Test boundary characters of Extended-B range."""
        # First char (U+A9E0)
        assert has_extended_myanmar_chars("\ua9e0") is True
        # Last char (U+A9FF)
        assert has_extended_myanmar_chars("\ua9ff") is True
        # Just before range (U+A9DF) - should not be detected as Extended Myanmar
        assert has_extended_myanmar_chars("\ua9df") is False


class TestExtendedMyanmarPatternIntegration:
    """Integration tests for Extended Myanmar pattern detection."""

    def test_extended_pattern_in_validator(self) -> None:
        """The EXTENDED_MYANMAR_PATTERN should include Extended-A/B ranges."""
        from myspellchecker.text.validator import EXTENDED_MYANMAR_PATTERN

        # Should match Extended-A chars
        extended_a_text = "\uaa60"  # First Extended-A char
        assert EXTENDED_MYANMAR_PATTERN.search(extended_a_text) is not None

        # Should match Extended-B chars
        extended_b_text = "\ua9e0"  # First Extended-B char
        assert EXTENDED_MYANMAR_PATTERN.search(extended_b_text) is not None

        # Should match original U+1050-U+109F range too
        original_range_text = "\u1050"
        assert EXTENDED_MYANMAR_PATTERN.search(original_range_text) is not None

    def test_core_myanmar_not_matched_by_extended_pattern(self) -> None:
        """Core Myanmar consonants (U+1000-U+104F) should not be flagged."""
        from myspellchecker.text.validator import EXTENDED_MYANMAR_PATTERN

        # Main consonants (U+1000-U+1020) should NOT match
        core_text = "ကခဂဃင"  # Ka, Kha, Ga, Gha, Nga
        for char in core_text:
            assert EXTENDED_MYANMAR_PATTERN.search(char) is None, (
                f"Core char {char} (U+{ord(char):04X}) shouldn't match EXTENDED_MYANMAR_PATTERN"
            )


class TestIsMyanmarTextHelper:
    """Tests for is_myanmar_text() shared helper function."""

    def test_is_myanmar_text_imported(self) -> None:
        """is_myanmar_text should be importable from constants."""
        from myspellchecker.core.constants import is_myanmar_text

        assert callable(is_myanmar_text)

    def test_empty_string_returns_false(self) -> None:
        """Empty string should return False."""
        from myspellchecker.core.constants import is_myanmar_text

        assert is_myanmar_text("") is False

    def test_core_myanmar_text_returns_true(self) -> None:
        """Text with core Myanmar chars should return True."""
        from myspellchecker.core.constants import is_myanmar_text

        text = "မြန်မာ"  # Myanmar
        assert is_myanmar_text(text) is True

    def test_english_only_returns_false(self) -> None:
        """English-only text should return False."""
        from myspellchecker.core.constants import is_myanmar_text

        assert is_myanmar_text("hello world") is False

    def test_extended_a_default_returns_false(self) -> None:
        """Extended-A chars should return False by default (not Myanmar)."""
        from myspellchecker.core.constants import is_myanmar_text

        # Only Extended-A char, no core Myanmar
        text = "\uaa60"  # Shan letter
        assert is_myanmar_text(text) is False

    def test_extended_a_allow_extended_returns_true(self) -> None:
        """Extended-A chars should return True when allow_extended=True."""
        from myspellchecker.core.constants import is_myanmar_text

        text = "\uaa60"  # Shan letter
        assert is_myanmar_text(text, allow_extended=True) is True

    def test_extended_b_default_returns_false(self) -> None:
        """Extended-B chars should return False by default (not Myanmar)."""
        from myspellchecker.core.constants import is_myanmar_text

        # Only Extended-B char, no core Myanmar
        text = "\ua9e0"
        assert is_myanmar_text(text) is False

    def test_extended_b_allow_extended_returns_true(self) -> None:
        """Extended-B chars should return True when allow_extended=True."""
        from myspellchecker.core.constants import is_myanmar_text

        text = "\ua9e0"
        assert is_myanmar_text(text, allow_extended=True) is True

    def test_mixed_core_and_extended_default(self) -> None:
        """Mixed text with core Myanmar should return True (core is detected)."""
        from myspellchecker.core.constants import is_myanmar_text

        text = "မြန်\uaa60မာ"  # Core Myanmar with Extended-A inserted
        # Core Myanmar chars are present, so should return True
        assert is_myanmar_text(text) is True

    def test_numbers_only_returns_false(self) -> None:
        """Number-only text should return False."""
        from myspellchecker.core.constants import is_myanmar_text

        assert is_myanmar_text("12345") is False
