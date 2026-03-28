"""Tests for correction_utils module."""

from myspellchecker.core.correction_utils import (
    build_orig_to_norm_map,
    filter_syllable_errors_in_valid_words,
    generate_corrected_text,
    remap_pre_norm_error,
)
from myspellchecker.core.response import SyllableError, WordError


class TestGenerateCorrectedText:
    """Tests for generate_corrected_text function."""

    def test_empty_errors(self) -> None:
        """Should return original text when no errors."""
        text = "hello world"
        result = generate_corrected_text(text, [])
        assert result == text

    def test_single_error(self) -> None:
        """Should apply single error correction."""
        text = "helo world"
        errors = [SyllableError(position=0, text="helo", suggestions=["hello"])]
        result = generate_corrected_text(text, errors)
        assert result == "hello world"

    def test_multiple_non_overlapping_errors(self) -> None:
        """Should apply all non-overlapping corrections."""
        text = "helo wrld"
        errors = [
            SyllableError(position=0, text="helo", suggestions=["hello"]),
            SyllableError(position=5, text="wrld", suggestions=["world"]),
        ]
        result = generate_corrected_text(text, errors)
        assert result == "hello world"

    def test_overlapping_errors_first_wins(self) -> None:
        """Should only apply first error when overlapping."""
        text = "abcdefgh"
        errors = [
            SyllableError(position=0, text="abcde", suggestions=["ABCDE"]),
            SyllableError(position=2, text="cdefg", suggestions=["CDEFG"]),  # Overlaps
        ]
        result = generate_corrected_text(text, errors)
        assert result == "ABCDEfgh"

    def test_error_without_suggestions(self) -> None:
        """Should skip errors without suggestions."""
        text = "hello world"
        errors = [SyllableError(position=0, text="hello", suggestions=[])]
        result = generate_corrected_text(text, errors)
        assert result == text

    def test_unsorted_errors(self) -> None:
        """Should handle unsorted error list."""
        text = "ab cd"
        errors = [
            SyllableError(position=3, text="cd", suggestions=["CD"]),
            SyllableError(position=0, text="ab", suggestions=["AB"]),
        ]
        result = generate_corrected_text(text, errors)
        assert result == "AB CD"

    def test_error_at_end(self) -> None:
        """Should handle error at end of text."""
        text = "hello world"
        errors = [SyllableError(position=6, text="world", suggestions=["WORLD"])]
        result = generate_corrected_text(text, errors)
        assert result == "hello WORLD"


class TestFilterSyllableErrorsInValidWords:
    """Tests for filter_syllable_errors_in_valid_words function."""

    def test_empty_errors(self) -> None:
        """Should return empty list for no errors."""
        result = filter_syllable_errors_in_valid_words("text", [], ["text"], {"text": True})
        assert result == []

    def test_syllable_error_in_valid_word_filtered(self) -> None:
        """Should filter syllable error that's inside valid word."""
        text = "abcdef"
        errors = [SyllableError(position=0, text="abc", suggestions=["ABC"])]
        words = ["abcdef"]
        validity_map = {"abcdef": True}
        result = filter_syllable_errors_in_valid_words(text, errors, words, validity_map)
        assert len(result) == 0

    def test_syllable_error_not_in_valid_word_kept(self) -> None:
        """Should keep syllable error that's not inside valid word."""
        text = "abc def"
        errors = [SyllableError(position=0, text="abc", suggestions=["ABC"])]
        words = ["abc", "def"]
        validity_map = {"abc": False, "def": True}
        result = filter_syllable_errors_in_valid_words(text, errors, words, validity_map)
        assert len(result) == 1

    def test_word_error_not_filtered(self) -> None:
        """Should never filter WordError (only SyllableError)."""
        text = "abcdef"
        errors = [WordError(position=0, text="abcdef", suggestions=["ABCDEF"])]
        words = ["abcdef"]
        validity_map = {"abcdef": True}
        result = filter_syllable_errors_in_valid_words(text, errors, words, validity_map)
        assert len(result) == 1

    def test_no_valid_words(self) -> None:
        """Should return all errors when no valid words."""
        text = "abc def"
        errors = [SyllableError(position=0, text="abc", suggestions=["ABC"])]
        words = ["abc", "def"]
        validity_map = {"abc": False, "def": False}
        result = filter_syllable_errors_in_valid_words(text, errors, words, validity_map)
        assert len(result) == 1

    def test_mixed_error_types(self) -> None:
        """Should properly filter mixed error types."""
        text = "abc def ghi"
        errors = [
            SyllableError(position=0, text="abc", suggestions=["ABC"]),  # In valid word
            WordError(position=4, text="def", suggestions=["DEF"]),  # Not filtered
            SyllableError(position=8, text="ghi", suggestions=["GHI"]),  # Not in valid word
        ]
        words = ["abc", "def", "ghi"]
        validity_map = {"abc": True, "def": True, "ghi": False}
        result = filter_syllable_errors_in_valid_words(text, errors, words, validity_map)
        # SyllableError at 0 filtered (in valid word "abc")
        # WordError at 4 kept (not filtered)
        # SyllableError at 8 kept (not in valid word)
        assert len(result) == 2

    def test_preserves_missing_visarga_suffix_inside_valid_word(self) -> None:
        """Keep high-value suffix root-cause corrections even in valid segmented tokens."""
        text = "မိုးရွာလို"
        errors = [
            SyllableError(position=4, text="လို", suggestions=["လို့"], error_type="invalid_syllable")
        ]
        words = ["မိုး", "ရွာ", "လို"]
        validity_map = {"မိုး": True, "ရွာ": True, "လို": True}

        result = filter_syllable_errors_in_valid_words(text, errors, words, validity_map)

        assert len(result) == 1
        assert result[0].text == "လို"
        assert result[0].suggestions[0] == "လို့"


class TestBuildOrigToNormMap:
    """Tests for build_orig_to_norm_map function."""

    def test_no_zero_width_chars(self) -> None:
        """Identity mapping when no zero-width characters present."""
        text = "abc"
        mapping = build_orig_to_norm_map(text)
        assert mapping == [0, 1, 2, 3]

    def test_single_zwsp(self) -> None:
        """ZWSP removed: positions after it shift left by 1."""
        original = "a\u200bb"
        mapping = build_orig_to_norm_map(original)
        # orig 0 -> 'a' -> norm 0
        # orig 1 -> ZWSP -> norm 1 (maps here but char removed)
        # orig 2 -> 'b' -> norm 1
        # sentinel -> 2
        assert mapping == [0, 1, 1, 2]

    def test_multiple_zero_width_chars(self) -> None:
        """Multiple ZW chars collapsed correctly."""
        original = "\u200ba\u200cb\u200dc"
        mapping = build_orig_to_norm_map(original)
        # pos 0: ZWSP -> norm 0 (removed)
        # pos 1: 'a'  -> norm 0
        # pos 2: ZWNJ -> norm 1 (removed)
        # pos 3: 'b'  -> norm 1
        # pos 4: ZWJ  -> norm 2 (removed)
        # pos 5: 'c'  -> norm 2
        # sentinel    -> 3
        assert mapping == [0, 0, 1, 1, 2, 2, 3]

    def test_bom_character(self) -> None:
        """BOM (U+FEFF) is also remapped."""
        original = "\ufeffabc"
        mapping = build_orig_to_norm_map(original)
        assert mapping == [0, 0, 1, 2, 3]

    def test_empty_string(self) -> None:
        """Empty string produces only the sentinel."""
        mapping = build_orig_to_norm_map("")
        assert mapping == [0]

    def test_myanmar_text_with_zwsp(self) -> None:
        """Myanmar text with ZWSP between syllables."""
        original = "\u1000\u200b\u1001"
        mapping = build_orig_to_norm_map(original)
        assert mapping == [0, 1, 1, 2]


class TestRemapPreNormError:
    """Tests for remap_pre_norm_error function."""

    def test_remap_shifts_position(self) -> None:
        """Error position should shift when ZW chars precede it."""
        original = "a\u200bbc"
        mapping = build_orig_to_norm_map(original)
        err = SyllableError(position=2, text="b", suggestions=["B"])
        remap_pre_norm_error(err, mapping)
        assert err.position == 1

    def test_remap_no_change_before_zw(self) -> None:
        """Positions before the ZW char are unchanged."""
        original = "ab\u200bc"
        mapping = build_orig_to_norm_map(original)
        err = SyllableError(position=0, text="a", suggestions=["A"])
        remap_pre_norm_error(err, mapping)
        assert err.position == 0

    def test_remap_out_of_bounds_ignored(self) -> None:
        """Position beyond map length is left unchanged."""
        mapping = [0, 1, 2]
        err = SyllableError(position=10, text="x", suggestions=["X"])
        remap_pre_norm_error(err, mapping)
        assert err.position == 10

    def test_remap_empty_map_ignored(self) -> None:
        """Empty offset map leaves position unchanged."""
        err = SyllableError(position=5, text="x", suggestions=["X"])
        remap_pre_norm_error(err, [])
        assert err.position == 5

    def test_remap_negative_position_ignored(self) -> None:
        """Negative position is left unchanged."""
        mapping = [0, 1, 2, 3]
        err = SyllableError(position=-1, text="x", suggestions=["X"])
        remap_pre_norm_error(err, mapping)
        assert err.position == -1
