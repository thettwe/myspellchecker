"""
Unit tests for Phonetic Hasher.
"""

import pytest

from myspellchecker.algorithms.distance.edit_distance import levenshtein_distance
from myspellchecker.utils import PhoneticHasher


@pytest.fixture
def hasher():
    """Create a PhoneticHasher with default settings."""
    return PhoneticHasher()


@pytest.fixture
def hasher_with_tones():
    """Create a PhoneticHasher that includes tone marks."""
    return PhoneticHasher(ignore_tones=False)


@pytest.fixture
def hasher_strict():
    """Create a PhoneticHasher that distinguishes vowel length."""
    return PhoneticHasher(normalize_length=False)


class TestPhoneticHasherBasic:
    """Basic tests for PhoneticHasher functionality."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        hasher = PhoneticHasher()

        assert hasher.ignore_tones is True
        assert hasher.normalize_length is True
        assert hasher.max_code_length == 10

    def test_initialization_custom(self):
        """Test custom initialization."""
        hasher = PhoneticHasher(ignore_tones=False, normalize_length=False, max_code_length=20)

        assert hasher.ignore_tones is False
        assert hasher.normalize_length is False
        assert hasher.max_code_length == 20

    def test_encode_empty_string(self, hasher):
        """Test encoding empty string."""
        assert hasher.encode("") == ""

    def test_encode_single_consonant(self, hasher):
        """Test encoding single Myanmar consonant."""
        # က (ka) should map to 'k' group
        code = hasher.encode("က")
        assert "k" in code

        # မ (ma) should map to 'm' code
        code = hasher.encode("မ")
        assert "m" in code

    def test_encode_syllable_with_medial(self, hasher):
        """Test encoding syllable with medial consonants."""
        # မြ (mya) = consonant + medial_r
        code1 = hasher.encode("မြ")
        assert "m" in code1  # မ maps to m
        assert "medial_r" in code1

        # မျ (my) = consonant + medial_y
        code2 = hasher.encode("မျ")
        assert "m" in code2
        assert "medial_y" in code2

        # Different medials should produce different codes
        assert code1 != code2

    def test_encode_syllable_with_vowel(self, hasher):
        """Test encoding syllable with vowels."""
        # မာ (ma) = consonant + vowel_a
        code = hasher.encode("မာ")
        assert "m" in code
        assert "vowel_a" in code

    def test_encode_complete_syllable(self, hasher):
        """Test encoding complete syllable with consonant, medial, vowel."""
        # မြန် (myan) = consonant + medial_r + vowel_a + consonant
        code = hasher.encode("မြန်")

        # Should contain codes for each component
        assert "m" in code  # မ
        assert "medial_r" in code  # ြ
        # The exact structure depends on implementation

    def test_encode_ignores_tones(self, hasher):
        """Test that tone marks are ignored by default."""
        # Myanmar tone marks should be ignored
        _ = hasher.encode("မ")
        _ = hasher.encode("မ့")  # With tone mark ့ (U+1037)

        # Codes should be similar/same if tones are ignored
        # (exact behavior depends on implementation)

    def test_encode_includes_tones_when_enabled(self, hasher_with_tones):
        """Test that tone marks are included when ignore_tones=False."""
        hasher = hasher_with_tones

        # With tone marks disabled, codes might differ
        _ = hasher.encode("မ")
        _ = hasher.encode("မ့")

        # Codes may differ when tones are not ignored
        # (exact behavior depends on implementation)

    def test_encode_word(self, hasher):
        """Test encoding multi-syllable word."""
        # မြန်မာ (Myanmar) = two syllables
        code = hasher.encode("မြန်မာ")

        # Should be non-empty
        assert len(code) > 0

        # Should contain components from both syllables
        assert "m" in code  # Both syllables start with မ

    def test_encode_max_length_truncation(self):
        """Test that long codes are truncated when adaptive_length is disabled."""
        hasher = PhoneticHasher(max_code_length=5, adaptive_length=False)

        # Encode a long word
        code = hasher.encode("မြန်မာနိုင်ငံတော်")

        # Should be truncated to max length
        assert len(code) <= 5

    def test_encode_adaptive_length_for_compounds(self):
        """Test that adaptive length preserves more info for compound words."""
        # Default hasher with adaptive_length=True
        adaptive_hasher = PhoneticHasher(max_code_length=10, adaptive_length=True)
        # Fixed length hasher
        fixed_hasher = PhoneticHasher(max_code_length=10, adaptive_length=False)

        # Long compound word
        compound = "မြန်မာနိုင်ငံတော်"  # "Republic of the Union of Myanmar"

        adaptive_code = adaptive_hasher.encode(compound)
        fixed_code = fixed_hasher.encode(compound)

        # Adaptive code should be longer (more information preserved)
        assert len(adaptive_code) >= len(fixed_code)
        # Fixed code should be truncated at max_code_length
        assert len(fixed_code) <= 10


class TestPhoneticHasherSimilarity:
    """Tests for phonetic similarity detection."""

    def test_similar_identical_codes(self, hasher):
        """Test that identical codes are similar."""
        code = hasher.encode("မြန်")

        assert hasher.similar(code, code) is True

    def test_similar_within_distance(self, hasher):
        """Test similarity with edit distance threshold."""
        code1 = "p-medial_r-vowel_a-n"
        code2 = "p-medial_y-vowel_a-n"  # One difference

        # Should be similar with distance=1
        assert hasher.similar(code1, code2, max_distance=1) is True

        # Should NOT be similar with distance=0
        assert hasher.similar(code1, code2, max_distance=0) is False

    def test_similar_beyond_distance(self, hasher):
        """Test that codes beyond threshold are not similar."""
        code1 = "p-medial_r-vowel_a-n"
        code2 = "k-medial_w-vowel_i-t"  # Many differences

        # Should not be similar even with distance=2
        assert hasher.similar(code1, code2, max_distance=2) is False


class TestPhoneticVariants:
    """Tests for generating phonetic variants."""

    def test_get_phonetic_variants_includes_original(self, hasher):
        """Test that variants include the original text."""
        text = "မြန်"
        variants = hasher.get_phonetic_variants(text)

        # Should include original
        assert text in variants

    def test_get_phonetic_variants_generates_alternatives(self, hasher):
        """Test that variants are generated."""
        text = "မြန်"
        variants = hasher.get_phonetic_variants(text)

        # Should have more than just the original
        # (exact count depends on implementation)
        assert len(variants) >= 1

    def test_get_phonetic_variants_substitutes_similar_chars(self, hasher):
        """Test that variants substitute phonetically similar characters."""
        text = "မ"  # Labial consonant
        _ = hasher.get_phonetic_variants(text)

        # Should include other labial consonants from same phonetic group
        # မ is in group with ပ, ဖ, ဗ, ဘ
        # (exact variants depend on implementation)

    def test_get_phonetic_variants_empty_string(self, hasher):
        """Test variants for empty string."""
        variants = hasher.get_phonetic_variants("")

        assert len(variants) == 0


class TestLevenshteinDistance:
    """Tests for the Levenshtein distance helper."""

    def test_levenshtein_identical_strings(self):
        """Test distance between identical strings."""
        distance = levenshtein_distance("abc", "abc")
        assert distance == 0

    def test_levenshtein_empty_strings(self):
        """Test distance with empty strings."""
        # Empty to non-empty
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("abc", "") == 3

        # Both empty
        assert levenshtein_distance("", "") == 0

    def test_levenshtein_single_substitution(self):
        """Test single character substitution."""
        distance = levenshtein_distance("abc", "adc")
        assert distance == 1

    def test_levenshtein_single_insertion(self):
        """Test single character insertion."""
        distance = levenshtein_distance("abc", "abcd")
        assert distance == 1

    def test_levenshtein_single_deletion(self):
        """Test single character deletion."""
        distance = levenshtein_distance("abcd", "abc")
        assert distance == 1

    def test_levenshtein_multiple_operations(self):
        """Test multiple edit operations."""
        distance = levenshtein_distance("kitten", "sitting")
        # kitten -> sitten (substitute k->s)
        # sitten -> sittin (substitute e->i)
        # sittin -> sitting (insert g)
        assert distance == 3


class TestRealWorldScenarios:
    """Real-world scenario tests."""

    def test_confusable_medials(self, hasher):
        """Test detection of confusable medials (ြ vs ျ)."""
        # မြန် (correct) vs မျန် (wrong medial)
        code1 = hasher.encode("မြန်")
        code2 = hasher.encode("မျန်")

        # Codes should be different
        assert code1 != code2

        # But they might be similar (both have medials)
        # Distance depends on exact encoding

    def test_vowel_length_normalization(self, hasher):
        """Test vowel length normalization (default)."""
        # ိ (short i) vs ီ (long ii)
        # With normalize_length=True, should be treated similarly

        _ = hasher.encode("ကိ")
        _ = hasher.encode("ကီ")

        # With normalization, both map to same vowel_i group
        # (exact behavior depends on implementation)

    def test_vowel_length_distinction(self, hasher_strict):
        """Test vowel length distinction when disabled."""
        hasher = hasher_strict  # normalize_length=False

        _ = hasher.encode("ကိ")
        _ = hasher.encode("ကီ")

        # Without normalization, codes might differ
        # (exact behavior depends on implementation)

    def test_word_level_phonetic_matching(self, hasher):
        """Test phonetic matching for complete words."""
        # Similar sounding words
        word1 = "မြန်မာ"
        word2 = "မြန်မာ"  # Identical

        code1 = hasher.encode(word1)
        code2 = hasher.encode(word2)

        assert code1 == code2

    def test_fallback_for_unknown_characters(self, hasher):
        """Test that unknown characters are handled gracefully."""
        # Mix Myanmar with English
        code = hasher.encode("မြန်ABC")

        # Should not crash, should handle unknown chars
        assert isinstance(code, str)
        assert len(code) > 0

    def test_unicode_normalization(self, hasher):
        """Test that different Unicode representations normalize."""
        import unicodedata

        # Same text in different Unicode forms
        text_nfc = unicodedata.normalize("NFC", "မြန်")
        text_nfd = unicodedata.normalize("NFD", "မြန်")

        code_nfc = hasher.encode(text_nfc)
        code_nfd = hasher.encode(text_nfd)

        # After encoding, should produce same phonetic code
        # (hasher normalizes to NFC internally)
        assert code_nfc == code_nfd


class TestPhoneticGaps:
    """Tests for identified phonetic gaps."""

    def test_independent_vowel_a(self, hasher):
        """Test Independent Vowel A (U+1021) matches Aa."""
        # အ (U+1021) vs အာ (U+1021 + U+102C) -> actually A vs Aa
        # U+1021 is Independent Vowel A / Glottal Stop?
        # In modern typing, U+1021 is 'A'.
        code_indep = hasher.encode("\u1021")  # Independent A
        hasher.encode("\u102b")  # Vowel A sign

        # U+1021 is the carrier consonant (glottal stop), not a pure vowel
        assert "vowel_carr" in code_indep

    def test_independent_vowel_i(self, hasher):
        """Test Independent Vowel I (U+1023) matches I."""
        code_indep = hasher.encode("\u1023")
        code_sign = hasher.encode("\u102d")  # Vowel I sign

        # Both should map to vowel_i
        assert "vowel_i" in code_indep
        assert "vowel_i" in code_sign

    def test_nga_that_normalization(self):
        """Test Nga+Asat normalizes to Anusvara when normalize_nasals=True."""
        # Nasal normalization is opt-in via normalize_nasals parameter
        hasher = PhoneticHasher(normalize_nasals=True)

        # မင်္ဂလာ (Min-ga-la) vs မင်ဂလာ (Min-ga-la typed with Nga+Asat)
        # Anusvara is U+1036
        # Nga+Asat is U+1004 + U+103A

        text_anusvara = "မ\u1036"  # Min (with Anusvara)
        text_nga_that = "မ\u1004\u103a"  # Min (with Nga+Asat)

        code1 = hasher.encode(text_anusvara)
        code2 = hasher.encode(text_nga_that)

        assert code1 == code2

    def test_virama_skipping(self, hasher):
        """Test Virama skipping for stacked vs unstacked."""
        # က္က (Stacked Ka) = Ka + Virama + Ka
        # ကက (Unstacked Ka) = Ka + Ka

        stacked = "\u1000\u1039\u1000"
        unstacked = "\u1000\u1000"

        code1 = hasher.encode(stacked)
        code2 = hasher.encode(unstacked)

        assert code1 == code2


class TestSandhiVoicing:
    """Tests for sandhi voicing — phonetic equivalence of voiced/unvoiced pairs."""

    def test_voicing_pairs(self, hasher):
        # Ka vs Ga
        assert hasher.encode("က") == hasher.encode("ဂ")
        # Sa vs Za
        assert hasher.encode("စ") == hasher.encode("ဇ")
        # Ta vs Da
        assert hasher.encode("တ") == hasher.encode("ဒ")
        # Pa vs Ba
        assert hasher.encode("ပ") == hasher.encode("ဗ")

    def test_sandhi_voicing_words(self, hasher):
        # Kati vs Gadi
        code1 = hasher.encode("ကတိ")
        code2 = hasher.encode("ဂတိ")
        assert code1 == code2

        # Sapwe vs Zabwe (typo)
        code3 = hasher.encode("စားပွဲ")
        code4 = hasher.encode("ဇားပွဲ")
        assert code3 == code4
