"""Tests for the Python implementation of edit distance algorithms.

These tests specifically target the Python fallback implementations
to ensure coverage when Cython is not available.
"""

from unittest.mock import patch

# Import with Cython disabled to test Python implementations
with patch.dict("sys.modules", {"myspellchecker.utils.edit_distance_c": None}):
    # Force reload to use Python implementations
    import importlib

    import myspellchecker.algorithms.distance.edit_distance as ed_module

    importlib.reload(ed_module)

from myspellchecker.algorithms.distance.edit_distance import (
    MEDIAL_CONFUSIONS,
    MYANMAR_CONSONANTS,
    MYANMAR_MEDIALS,
    _are_medial_confusions,
    damerau_levenshtein_distance,
    levenshtein_distance,
    myanmar_syllable_edit_distance,
    tokenize_myanmar_syllable_units,
    weighted_damerau_levenshtein_distance,
)


class TestLevenshteinDistance:
    """Tests for levenshtein_distance function."""

    def test_identical_strings(self):
        """Identical strings have distance 0."""
        assert levenshtein_distance("hello", "hello") == 0
        assert levenshtein_distance("", "") == 0

    def test_empty_strings(self):
        """Empty string distance equals length of other string."""
        assert levenshtein_distance("", "hello") == 5
        assert levenshtein_distance("hello", "") == 5

    def test_single_insertion(self):
        """Single character insertion has distance 1."""
        assert levenshtein_distance("cat", "cats") == 1
        assert levenshtein_distance("at", "cat") == 1

    def test_single_deletion(self):
        """Single character deletion has distance 1."""
        assert levenshtein_distance("cats", "cat") == 1
        assert levenshtein_distance("cat", "at") == 1

    def test_single_substitution(self):
        """Single character substitution has distance 1."""
        assert levenshtein_distance("cat", "car") == 1
        assert levenshtein_distance("cat", "bat") == 1

    def test_multiple_operations(self):
        """Multiple operations have cumulative distance."""
        assert levenshtein_distance("kitten", "sitting") == 3
        assert levenshtein_distance("saturday", "sunday") == 3

    def test_case_sensitivity(self):
        """Distance is case-sensitive."""
        assert levenshtein_distance("Hello", "hello") == 1

    def test_myanmar_text(self):
        """Test with Myanmar characters."""
        assert levenshtein_distance("ကာ", "ကာ") == 0
        assert levenshtein_distance("ကာ", "ခါ") == 2  # Both chars different


class TestDamerauLevenshteinDistance:
    """Tests for damerau_levenshtein_distance function."""

    def test_identical_strings(self):
        """Identical strings have distance 0."""
        assert damerau_levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        """Empty string distance equals length of other string."""
        assert damerau_levenshtein_distance("", "hello") == 5
        assert damerau_levenshtein_distance("hello", "") == 5

    def test_single_transposition(self):
        """Adjacent character transposition has distance 1."""
        assert damerau_levenshtein_distance("ab", "ba") == 1
        assert damerau_levenshtein_distance("abc", "bac") == 1

    def test_transposition_vs_two_substitutions(self):
        """Transposition is cheaper than two substitutions."""
        # Without transposition: ca -> ac would be 2 substitutions
        # With transposition: ca -> ac is 1 transposition
        assert damerau_levenshtein_distance("ca", "ac") == 1

    def test_insertion_deletion_substitution(self):
        """Standard edit operations work correctly."""
        assert damerau_levenshtein_distance("cat", "cats") == 1
        assert damerau_levenshtein_distance("cats", "cat") == 1
        assert damerau_levenshtein_distance("cat", "car") == 1

    def test_complex_edits(self):
        """Complex edit sequences."""
        assert damerau_levenshtein_distance("kitten", "sitting") == 3

    def test_myanmar_text(self):
        """Test with Myanmar characters."""
        assert damerau_levenshtein_distance("ကာ", "ကာ") == 0


class TestWeightedDamerauLevenshteinDistance:
    """Tests for weighted_damerau_levenshtein_distance function."""

    def test_identical_strings(self):
        """Identical strings have distance 0."""
        dist = weighted_damerau_levenshtein_distance("hello", "hello")
        assert dist == 0.0

    def test_empty_strings(self):
        """Empty string distance equals length of other string."""
        assert weighted_damerau_levenshtein_distance("", "hello") == 5.0
        assert weighted_damerau_levenshtein_distance("hello", "") == 5.0

    def test_standard_substitution(self):
        """Standard substitution has cost 1.0."""
        dist = weighted_damerau_levenshtein_distance("a", "b")
        assert dist == 1.0

    def test_visual_similar_weight(self):
        """Visually similar characters have reduced cost."""
        # Visual similar pairs get visual_weight instead of 1.0
        dist = weighted_damerau_levenshtein_distance(
            "\u1000",
            "\u1001",  # Similar Myanmar consonants if in VISUAL_SIMILAR
            keyboard_weight=0.5,
            visual_weight=0.3,
        )
        # Result depends on VISUAL_SIMILAR mapping
        assert 0.0 <= dist <= 1.0

    def test_custom_weights(self):
        """Test custom weight parameters."""
        dist1 = weighted_damerau_levenshtein_distance(
            "abc",
            "abd",
            keyboard_weight=0.2,
            visual_weight=0.3,
        )
        dist2 = weighted_damerau_levenshtein_distance(
            "abc",
            "abd",
            keyboard_weight=0.8,
            visual_weight=0.9,
        )
        # Both should work, actual values depend on char similarities
        assert isinstance(dist1, float)
        assert isinstance(dist2, float)

    def test_transposition(self):
        """Transposition has default cost 0.7."""
        dist = weighted_damerau_levenshtein_distance("ab", "ba")
        assert dist == 0.7  # Default transposition_weight is 0.7


class TestTokenizeMyanmarSyllableUnits:
    """Tests for tokenize_myanmar_syllable_units function."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert tokenize_myanmar_syllable_units("") == []

    def test_single_consonant(self):
        """Single consonant is one unit."""
        assert tokenize_myanmar_syllable_units("က") == ["က"]

    def test_consonant_with_medials(self):
        """Consonant + medials grouped together."""
        # Ka + Ya-pin
        result = tokenize_myanmar_syllable_units("ကျ")
        assert result == ["ကျ"]

        # Ka + Ya-yit
        result = tokenize_myanmar_syllable_units("ကြ")
        assert result == ["ကြ"]

        # Ka + multiple medials
        result = tokenize_myanmar_syllable_units("ကျွ")
        assert result == ["ကျွ"]

    def test_vowels_separate(self):
        """Vowels are separate units."""
        # Ka + Aa vowel
        result = tokenize_myanmar_syllable_units("ကာ")
        assert result == ["က", "ာ"]

    def test_mixed_text(self):
        """Mixed consonants, medials, and vowels."""
        # Mya + n
        result = tokenize_myanmar_syllable_units("မြန်")
        assert len(result) >= 3
        assert result[0] == "မြ"  # Ma + Ra grouped

    def test_non_myanmar_chars(self):
        """Non-Myanmar characters are separate units."""
        result = tokenize_myanmar_syllable_units("abc")
        assert result == ["a", "b", "c"]

    def test_consecutive_consonants(self):
        """Consecutive consonants without medials."""
        result = tokenize_myanmar_syllable_units("ကခ")
        assert result == ["က", "ခ"]


class TestAreMedialConfusions:
    """Tests for _are_medial_confusions function."""

    def test_identical_units(self):
        """Identical units are not confusions."""
        assert not _are_medial_confusions("ကျ", "ကျ")

    def test_different_base_consonant(self):
        """Different base consonants are not confusions."""
        assert not _are_medial_confusions("ကျ", "ချ")

    def test_short_units(self):
        """Units shorter than 2 chars are not confusions."""
        assert not _are_medial_confusions("က", "ခ")
        assert not _are_medial_confusions("က", "ကျ")

    def test_ya_ra_confusion(self):
        """Ya-pin vs Ya-yit is a confusion pair."""
        # Ka + Ya vs Ka + Ra
        result = _are_medial_confusions("ကျ", "ကြ")
        assert result is True

    def test_wa_ha_confusion(self):
        """Wa-hswe vs Ha-htoe is a confusion pair."""
        # Ka + Wa vs Ka + Ha
        result = _are_medial_confusions("ကွ", "ကှ")
        assert result is True

    def test_non_confusion_medials(self):
        """Non-confusion medial pairs."""
        # Ka + Ya vs Ka + Wa - not a standard confusion pair
        result = _are_medial_confusions("ကျ", "ကွ")
        assert result is False


class TestMyanmarSyllableEditDistance:
    """Tests for myanmar_syllable_edit_distance function."""

    def test_identical_strings(self):
        """Identical strings have distance (0, 0.0)."""
        int_dist, weighted_dist = myanmar_syllable_edit_distance("ကာ", "ကာ")
        assert int_dist == 0
        assert weighted_dist == 0.0

    def test_empty_strings(self):
        """Empty string distance equals unit count of other string."""
        int_dist, weighted_dist = myanmar_syllable_edit_distance("", "ကာ")
        # "ကာ" tokenizes to ["က", "ာ"] = 2 units
        assert int_dist == 2
        assert weighted_dist == 2.0

        int_dist2, weighted_dist2 = myanmar_syllable_edit_distance("ကာ", "")
        assert int_dist2 == 2
        assert weighted_dist2 == 2.0

    def test_medial_confusion_reduced_weight(self):
        """Medial confusion has reduced weighted cost."""
        # မျ vs မြ - medial confusion
        s1 = "မျ"
        s2 = "မြ"
        int_dist, weighted_dist = myanmar_syllable_edit_distance(s1, s2)

        # Integer distance is 1 (one substitution)
        assert int_dist == 1
        # Weighted distance is 0.5 (reduced cost for medial confusion)
        assert weighted_dist == 0.5

    def test_non_confusion_substitution(self):
        """Non-confusion substitution has full cost."""
        s1 = "ကာ"  # Ka + Aa
        s2 = "ခါ"  # Kha + Aa
        int_dist, weighted_dist = myanmar_syllable_edit_distance(s1, s2)

        # Different consonants = 1 substitution
        # Note: tokenization affects this
        assert int_dist >= 1
        assert weighted_dist >= 1.0

    def test_transposition(self):
        """Transposition of units."""
        # Two-unit strings with transposition
        s1 = "ကခ"  # Ka, Kha
        s2 = "ခက"  # Kha, Ka
        int_dist, weighted_dist = myanmar_syllable_edit_distance(s1, s2)

        # Transposition cost is 1
        assert int_dist == 1
        assert weighted_dist == 1.0


class TestMedialConfusionConstants:
    """Tests for medial confusion constants."""

    def test_medial_confusions_defined(self):
        """MEDIAL_CONFUSIONS contains expected pairs."""
        assert isinstance(MEDIAL_CONFUSIONS, set)
        assert len(MEDIAL_CONFUSIONS) > 0

    def test_myanmar_medials_defined(self):
        """MYANMAR_MEDIALS contains expected characters."""
        assert isinstance(MYANMAR_MEDIALS, (set, frozenset))
        assert "\u103b" in MYANMAR_MEDIALS  # Ya-pin
        assert "\u103c" in MYANMAR_MEDIALS  # Ya-yit
        assert "\u103d" in MYANMAR_MEDIALS  # Wa-hswe
        assert "\u103e" in MYANMAR_MEDIALS  # Ha-htoe

    def test_myanmar_consonants_defined(self):
        """MYANMAR_CONSONANTS contains expected characters."""
        assert isinstance(MYANMAR_CONSONANTS, (set, frozenset))
        assert "\u1000" in MYANMAR_CONSONANTS  # Ka
        assert "\u1001" in MYANMAR_CONSONANTS  # Kha


class TestCaching:
    """Tests for LRU caching of edit distance functions."""

    def test_levenshtein_cache(self):
        """Levenshtein distance results are cached."""
        # Clear cache first
        levenshtein_distance.cache_clear()

        # First call
        levenshtein_distance("test", "test")
        info1 = levenshtein_distance.cache_info()

        # Second call with same args
        levenshtein_distance("test", "test")
        info2 = levenshtein_distance.cache_info()

        # Hits should increase
        assert info2.hits > info1.hits

    def test_damerau_cache(self):
        """Damerau-Levenshtein distance results are cached."""
        damerau_levenshtein_distance.cache_clear()

        damerau_levenshtein_distance("test", "test")
        info1 = damerau_levenshtein_distance.cache_info()

        damerau_levenshtein_distance("test", "test")
        info2 = damerau_levenshtein_distance.cache_info()

        assert info2.hits > info1.hits

    def test_weighted_cache(self):
        """Weighted Damerau-Levenshtein distance results are cached."""
        weighted_damerau_levenshtein_distance.cache_clear()

        weighted_damerau_levenshtein_distance("test", "test")
        info1 = weighted_damerau_levenshtein_distance.cache_info()

        weighted_damerau_levenshtein_distance("test", "test")
        info2 = weighted_damerau_levenshtein_distance.cache_info()

        assert info2.hits > info1.hits

    def test_myanmar_syllable_cache(self):
        """Myanmar syllable edit distance results are cached."""
        myanmar_syllable_edit_distance.cache_clear()

        myanmar_syllable_edit_distance("ကာ", "ကာ")
        info1 = myanmar_syllable_edit_distance.cache_info()

        myanmar_syllable_edit_distance("ကာ", "ကာ")
        info2 = myanmar_syllable_edit_distance.cache_info()

        assert info2.hits > info1.hits
