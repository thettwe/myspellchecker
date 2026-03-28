from myspellchecker.algorithms.distance.edit_distance import (
    damerau_levenshtein_distance,
    levenshtein_distance,
    weighted_damerau_levenshtein_distance,
)
from myspellchecker.algorithms.distance.keyboard import get_keyboard_distance, is_keyboard_adjacent


class TestEditDistance:
    """Test cases for edit distance algorithms."""

    def test_levenshtein_basic(self):
        """Test basic Levenshtein distance (insert, delete, sub)."""
        assert levenshtein_distance("kitten", "sitting") == 3
        assert levenshtein_distance("rosettacode", "raisethysword") == 8
        assert levenshtein_distance("abc", "abc") == 0
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("abc", "") == 3

    def test_levenshtein_myanmar(self):
        """Test Levenshtein with Myanmar characters."""
        # Substitution: က -> ခ
        assert levenshtein_distance("က", "ခ") == 1
        # Insertion: က -> ကာ
        assert levenshtein_distance("က", "ကာ") == 1
        # Deletion: ကာ -> က
        assert levenshtein_distance("ကာ", "က") == 1

    def test_damerau_levenshtein_transposition(self):
        """Test Damerau-Levenshtein specifically for transpositions."""
        # Transposition cost is 1 in DL, but 2 in standard Levenshtein (sub + sub)
        assert damerau_levenshtein_distance("ab", "ba") == 1
        assert levenshtein_distance("ab", "ba") == 2

        # Myanmar valid transposition: ကမ -> မက
        assert damerau_levenshtein_distance("ကမ", "မက") == 1

    def test_damerau_levenshtein_myanmar_typos(self):
        """Test DL distance on common Myanmar typos."""
        # Missing medial
        assert damerau_levenshtein_distance("မြန်", "မန်") == 1

        # Wrong medial (substitution)
        assert damerau_levenshtein_distance("မြန်", "မျန်") == 1

        # Extra tone (insertion)
        assert damerau_levenshtein_distance("ကား", "ကားး") == 1

        # Missing tone (deletion)
        assert damerau_levenshtein_distance("ကား", "ကာ") == 1

    def test_empty_strings(self):
        """Test edge cases with empty strings."""
        assert damerau_levenshtein_distance("", "") == 0
        assert damerau_levenshtein_distance("a", "") == 1
        assert damerau_levenshtein_distance("", "a") == 1


class TestKeyboard:
    def test_adjacency(self):
        # Standard Myanmar3 neighbors
        assert is_keyboard_adjacent("က", "င")  # u and i
        assert is_keyboard_adjacent("က", "ပ")  # u and y

        # Not neighbors
        assert not is_keyboard_adjacent("က", "မ")

    def test_distance(self):
        # Same char
        assert get_keyboard_distance("က", "က") == 0.0
        # Adjacent
        assert get_keyboard_distance("က", "င") == 1.0
        # Far
        assert get_keyboard_distance("က", "မ") > 1.0


class TestWeightedEditDistance:
    def test_basic_weighted(self):
        # Keyboard error: u (Ka) vs i (Nga)
        # Standard distance = 1
        # Weighted distance (keyboard=0.5) = 0.5
        dist = weighted_damerau_levenshtein_distance("က", "င", keyboard_weight=0.5)
        assert dist == 0.5

        # Non-keyboard error: u (Ka) vs r (Ma)
        dist2 = weighted_damerau_levenshtein_distance("က", "မ", keyboard_weight=0.5)
        assert dist2 == 1.0

    def test_visual_weighted(self):
        # Visual error: ိ (short i) vs ီ (long ii)
        # These are in MYANMAR_SUBSTITUTION_COSTS with cost 0.2
        dist = weighted_damerau_levenshtein_distance("ိ", "ီ", visual_weight=0.3)
        assert dist == 0.2  # Uses substitution cost from phonetic data

    def test_combined(self):
        # "ကိ" vs "ငီ"
        # က->င uses keyboard weight 0.5, ိ->ီ uses substitution cost 0.2
        # Total = 0.5 + 0.2 = 0.7
        dist = weighted_damerau_levenshtein_distance(
            "ကိ", "ငီ", keyboard_weight=0.5, visual_weight=0.5
        )
        assert dist == 0.7


class TestTranspositionWeighting:
    """Test cases for transposition weighting."""

    def test_transposition_default_weight(self):
        """Test default transposition weight (0.7)."""
        dist = weighted_damerau_levenshtein_distance("ab", "ba")
        assert dist == 0.7

    def test_transposition_custom_weight(self):
        """Test custom transposition weight."""
        dist = weighted_damerau_levenshtein_distance("ab", "ba", transposition_weight=0.5)
        assert dist == 0.5

    def test_myanmar_vowel_transposition(self):
        """Test Myanmar vowel ordering confusion (common error)."""
        # ာေ vs ော - common vowel ordering mistake
        dist = weighted_damerau_levenshtein_distance("ာေ", "ော")
        assert dist == 0.7  # Should use transposition, not 2 substitutions


class TestSyllableTokenization:
    """Test cases for Myanmar syllable tokenization."""

    def test_basic_consonant_medial(self):
        """Test basic consonant + medial grouping."""
        from myspellchecker.algorithms.distance.edit_distance import (
            tokenize_myanmar_syllable_units,
        )

        # Consonant + Ya-pin
        assert tokenize_myanmar_syllable_units("မျန်") == ["မျ", "န", "်"]

        # Consonant + Ya-yit
        assert tokenize_myanmar_syllable_units("မြန်") == ["မြ", "န", "်"]

        # Consonant + multiple medials
        assert tokenize_myanmar_syllable_units("မြွှ") == ["မြွှ"]

    def test_kinzi_handling(self):
        """Test Kinzi (င်္) handling - groups Kinzi with following consonant."""
        from myspellchecker.algorithms.distance.edit_distance import (
            tokenize_myanmar_syllable_units,
        )

        # "အင်္ဂလိပ်" (English) has Kinzi before ဂ
        # Kinzi (င်္) should be grouped with ဂ as a single unit
        result = tokenize_myanmar_syllable_units("အင်္ဂလိပ်")
        assert "င်္ဂ" in result  # Kinzi + Ga should be together
        assert result == ["အ", "င်္ဂ", "လ", "ိ", "ပ", "်"]

        # Test Kinzi with medials: င်္ + consonant + medial
        # "အင်္ကြိ" - Kinzi + Ka + Ya-yit + vowel
        result = tokenize_myanmar_syllable_units("အင်္ကြိ")
        assert "င်္ကြ" in result  # Kinzi + Ka + medial should be together
        assert result == ["အ", "င်္ကြ", "ိ"]

    def test_empty_and_non_myanmar(self):
        """Test edge cases."""
        from myspellchecker.algorithms.distance.edit_distance import (
            tokenize_myanmar_syllable_units,
        )

        assert tokenize_myanmar_syllable_units("") == []
        assert tokenize_myanmar_syllable_units("abc") == ["a", "b", "c"]
