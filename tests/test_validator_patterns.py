"""Unit tests for text/validator.py — regex patterns and character set constants."""

import pytest


class TestCharacterSets:
    """Tests for character set constants."""

    # Note: EXTENDED_MYANMAR_CHARS was removed from validator.py
    # The canonical definition is in core/constants/myanmar_constants.py
    # and is tested in tests/test_extended_myanmar_characters.py

    def test_myanmar_consonants(self):
        """Test MYANMAR_CONSONANTS contains expected consonants."""
        from myspellchecker.text.validator import MYANMAR_CONSONANTS

        assert isinstance(MYANMAR_CONSONANTS, set)
        assert "က" in MYANMAR_CONSONANTS
        assert "ခ" in MYANMAR_CONSONANTS
        assert "ဂ" in MYANMAR_CONSONANTS
        assert "အ" in MYANMAR_CONSONANTS

    def test_myanmar_vowels(self):
        """Test MYANMAR_VOWELS contains expected vowels."""
        from myspellchecker.text.validator import MYANMAR_VOWELS

        assert isinstance(MYANMAR_VOWELS, set)
        assert "ါ" in MYANMAR_VOWELS
        assert "ာ" in MYANMAR_VOWELS
        assert "ိ" in MYANMAR_VOWELS
        assert "ု" in MYANMAR_VOWELS

    def test_myanmar_medials(self):
        """Test MYANMAR_MEDIALS contains expected medials."""
        from myspellchecker.text.validator import MYANMAR_MEDIALS

        assert isinstance(MYANMAR_MEDIALS, set)
        assert "ျ" in MYANMAR_MEDIALS
        assert "ြ" in MYANMAR_MEDIALS
        assert "ွ" in MYANMAR_MEDIALS
        assert "ှ" in MYANMAR_MEDIALS

    def test_myanmar_tones(self):
        """Test MYANMAR_TONES contains expected tone marks."""
        from myspellchecker.text.validator import MYANMAR_TONES

        assert isinstance(MYANMAR_TONES, set)
        assert "ံ" in MYANMAR_TONES
        assert "့" in MYANMAR_TONES
        assert "း" in MYANMAR_TONES

    def test_asat_constant(self):
        """Test ASAT constant."""
        from myspellchecker.core.constants import ASAT

        assert ASAT == "\u103a"

    def test_virama_constant(self):
        """Test VIRAMA constant."""
        from myspellchecker.core.constants import VIRAMA

        assert VIRAMA == "\u1039"

    def test_myanmar_digits(self):
        """Test MYANMAR_DIGITS contains expected digits."""
        from myspellchecker.text.validator import MYANMAR_DIGITS

        assert isinstance(MYANMAR_DIGITS, set)
        assert "၀" in MYANMAR_DIGITS
        assert "၁" in MYANMAR_DIGITS
        assert "၉" in MYANMAR_DIGITS

    def test_valid_starters(self):
        """Test VALID_STARTERS contains consonants, independent vowels, digits."""
        from myspellchecker.text.validator import VALID_STARTERS

        assert isinstance(VALID_STARTERS, set)
        # Consonants should be included
        assert "က" in VALID_STARTERS
        assert "အ" in VALID_STARTERS
        # Digits should be included
        assert "၀" in VALID_STARTERS

    def test_known_invalid_words(self):
        """Test KNOWN_INVALID_WORDS set."""
        from myspellchecker.text.validator import KNOWN_INVALID_WORDS

        assert isinstance(KNOWN_INVALID_WORDS, set)
        assert len(KNOWN_INVALID_WORDS) > 0
        assert "ပျှော်" in KNOWN_INVALID_WORDS
        assert "င်း" in KNOWN_INVALID_WORDS
        assert "်" in KNOWN_INVALID_WORDS

    def test_pali_whitelist(self):
        """Test VALID_PALI_BARE_ENDINGS whitelist."""
        from myspellchecker.text.validator import VALID_PALI_BARE_ENDINGS

        assert isinstance(VALID_PALI_BARE_ENDINGS, set)
        assert len(VALID_PALI_BARE_ENDINGS) > 50  # Should have many entries

        # Test single consonant particles
        assert "က" in VALID_PALI_BARE_ENDINGS
        assert "မ" in VALID_PALI_BARE_ENDINGS
        assert "အ" in VALID_PALI_BARE_ENDINGS

        # Test Pali/Sanskrit loanwords
        assert "ဒေသ" in VALID_PALI_BARE_ENDINGS  # region
        assert "ဌာန" in VALID_PALI_BARE_ENDINGS  # department
        assert "ပထမ" in VALID_PALI_BARE_ENDINGS  # first
        assert "ဒုတိယ" in VALID_PALI_BARE_ENDINGS  # second
        assert "အစိုးရ" in VALID_PALI_BARE_ENDINGS  # government
        assert "ဘဝ" in VALID_PALI_BARE_ENDINGS  # life

        # Test country names
        assert "အိန္ဒိယ" in VALID_PALI_BARE_ENDINGS  # India
        assert "ဥရောပ" in VALID_PALI_BARE_ENDINGS  # Europe


class TestPatternConstants:
    """Tests for regex pattern constants."""

    def test_extended_myanmar_pattern(self):
        """Test EXTENDED_MYANMAR_PATTERN matches extended range."""
        from myspellchecker.text.validator import EXTENDED_MYANMAR_PATTERN

        assert EXTENDED_MYANMAR_PATTERN.search("\u1050") is not None
        assert EXTENDED_MYANMAR_PATTERN.search("\u109f") is not None
        assert EXTENDED_MYANMAR_PATTERN.search("က") is None

    def test_zawgyi_ya_asat_pattern(self):
        """Test ZAWGYI_YA_ASAT_PATTERN matches ya+tone."""
        from myspellchecker.text.validator import ZAWGYI_YA_ASAT_PATTERN

        assert ZAWGYI_YA_ASAT_PATTERN.search("ငျး") is not None
        assert ZAWGYI_YA_ASAT_PATTERN.search("ကျ") is None

    def test_zawgyi_ya_terminal_pattern(self):
        """Test ZAWGYI_YA_TERMINAL_PATTERN matches ya at end."""
        from myspellchecker.text.validator import ZAWGYI_YA_TERMINAL_PATTERN

        assert ZAWGYI_YA_TERMINAL_PATTERN.search("ငျ") is not None
        assert ZAWGYI_YA_TERMINAL_PATTERN.search("ကျ") is None

    def test_asat_before_vowel_pattern(self):
        """Test ASAT_BEFORE_VOWEL_PATTERN matches asat before vowel."""
        from myspellchecker.text.validator import ASAT_BEFORE_VOWEL_PATTERN

        assert ASAT_BEFORE_VOWEL_PATTERN.search("်ု") is not None
        assert ASAT_BEFORE_VOWEL_PATTERN.search("ု်") is None

    def test_digit_tone_pattern(self):
        """Test DIGIT_TONE_PATTERN matches digit+tone."""
        from myspellchecker.text.validator import DIGIT_TONE_PATTERN

        assert DIGIT_TONE_PATTERN.search("၁့") is not None
        assert DIGIT_TONE_PATTERN.search("၁") is None

    def test_scrambled_asat_pattern(self):
        """Test SCRAMBLED_ASAT_PATTERN matches vowel+asat+vowel."""
        from myspellchecker.text.validator import SCRAMBLED_ASAT_PATTERN

        assert SCRAMBLED_ASAT_PATTERN.search("ိ်ု") is not None
        assert SCRAMBLED_ASAT_PATTERN.search("ို") is None

    def test_incomplete_o_vowel_pattern(self):
        """Test INCOMPLETE_O_VOWEL_PATTERN matches i+tone without u."""
        from myspellchecker.text.validator import INCOMPLETE_O_VOWEL_PATTERN

        assert INCOMPLETE_O_VOWEL_PATTERN.search("ိ့") is not None
        assert INCOMPLETE_O_VOWEL_PATTERN.search("ို့") is None

    def test_doubled_vowel_pattern(self):
        """Test DOUBLED_VOWEL_PATTERN matches doubled vowels."""
        from myspellchecker.text.validator import DOUBLED_VOWEL_PATTERN

        assert DOUBLED_VOWEL_PATTERN.search("ာာ") is not None
        assert DOUBLED_VOWEL_PATTERN.search("ာ") is None

    def test_doubled_medial_pattern(self):
        """Test DOUBLED_MEDIAL_PATTERN matches doubled medials."""
        from myspellchecker.text.validator import DOUBLED_MEDIAL_PATTERN

        assert DOUBLED_MEDIAL_PATTERN.search("ျျ") is not None
        assert DOUBLED_MEDIAL_PATTERN.search("ျ") is None

    def test_virama_at_end_pattern(self):
        """Test VIRAMA_AT_END_PATTERN matches virama at end."""
        from myspellchecker.text.validator import VIRAMA_AT_END_PATTERN

        assert VIRAMA_AT_END_PATTERN.search("က္") is not None
        assert VIRAMA_AT_END_PATTERN.search("္က") is None

    def test_invalid_vowel_sequence_pattern(self):
        """Test INVALID_VOWEL_SEQUENCE matches invalid sequences."""
        from myspellchecker.text.validator import INVALID_VOWEL_SEQUENCE

        assert INVALID_VOWEL_SEQUENCE.search("ိိ") is not None
        assert INVALID_VOWEL_SEQUENCE.search("ုူ") is not None
        assert INVALID_VOWEL_SEQUENCE.search("ါာ") is not None

    # ========================================================================
    # Phase 1 Pattern Tests
    # ========================================================================

    def test_pure_numeral_pattern(self):
        """Test PURE_NUMERAL_PATTERN matches pure numeral sequences."""
        from myspellchecker.text.validator import PURE_NUMERAL_PATTERN

        # Should match pure numerals
        assert PURE_NUMERAL_PATTERN.match("၆၉၀၀") is not None
        assert PURE_NUMERAL_PATTERN.match("၁") is not None
        assert PURE_NUMERAL_PATTERN.match("၀၁၂၃၄၅၆၇၈၉") is not None

        # Should not match mixed content
        assert PURE_NUMERAL_PATTERN.match("က၁") is None
        assert PURE_NUMERAL_PATTERN.match("၁က") is None
        assert PURE_NUMERAL_PATTERN.match("မြန်မာ") is None

    def test_doubled_consonant_pattern(self):
        """Test DOUBLED_CONSONANT_PATTERN matches doubled consonants."""
        from myspellchecker.text.validator import DOUBLED_CONSONANT_PATTERN

        # Should match doubled consonants
        assert DOUBLED_CONSONANT_PATTERN.match("ဆဆ") is not None
        assert DOUBLED_CONSONANT_PATTERN.match("အအ") is not None
        assert DOUBLED_CONSONANT_PATTERN.match("ကက") is not None
        assert DOUBLED_CONSONANT_PATTERN.match("ညည") is not None

        # Should not match different consonants
        assert DOUBLED_CONSONANT_PATTERN.match("ကခ") is None
        assert DOUBLED_CONSONANT_PATTERN.match("အက") is None

        # Should not match longer sequences
        assert DOUBLED_CONSONANT_PATTERN.match("ကကက") is None
        assert DOUBLED_CONSONANT_PATTERN.match("ကကခ") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
