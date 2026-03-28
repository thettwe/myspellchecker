"""Unit tests for text/validator.py — quality filters and segmentation fragments."""

import pytest


class TestQualityFilterFunctions:
    """Tests for quality filter functions."""

    # ========================================================================
    # is_fragment_pattern tests
    # ========================================================================

    def test_is_fragment_pattern_consonant_asat(self):
        """Test detection of consonant+asat only fragments."""
        from myspellchecker.text.validator import is_fragment_pattern

        # Fragment patterns
        is_frag, pattern = is_fragment_pattern("ဉ်")  # Consonant + asat
        assert is_frag is True
        assert pattern == "consonant_asat"

        is_frag, pattern = is_fragment_pattern("ည်")
        assert is_frag is True
        assert pattern == "consonant_asat"

    def test_is_fragment_pattern_consonant_tone(self):
        """Test detection of consonant+tone only fragments."""
        from myspellchecker.text.validator import is_fragment_pattern

        is_frag, pattern = is_fragment_pattern("မး")  # Consonant + visarga
        assert is_frag is True
        assert pattern == "consonant_tone"

        is_frag, pattern = is_fragment_pattern("က့")  # Consonant + tone
        assert is_frag is True
        assert pattern == "consonant_tone"

    def test_is_fragment_pattern_consonant_tone_asat(self):
        """Test detection of consonant+tone+asat fragments."""
        from myspellchecker.text.validator import is_fragment_pattern

        is_frag, pattern = is_fragment_pattern("န့်")
        assert is_frag is True
        assert pattern == "consonant_tone_asat"

        is_frag, pattern = is_fragment_pattern("င့်")
        assert is_frag is True
        assert pattern == "consonant_tone_asat"

    def test_is_fragment_pattern_double_ending(self):
        """Test detection of double-ending patterns."""
        from myspellchecker.text.validator import is_fragment_pattern

        is_frag, pattern = is_fragment_pattern("တွင်င်း")  # Double-ending
        assert is_frag is True
        assert pattern == "double_ending"

        is_frag, pattern = is_fragment_pattern("သည်င်း")
        assert is_frag is True
        assert pattern == "double_ending"

    def test_is_fragment_pattern_valid_words(self):
        """Test that valid words are not flagged as fragments."""
        from myspellchecker.text.validator import is_fragment_pattern

        # Valid words
        is_frag, pattern = is_fragment_pattern("ကျောင်း")
        assert is_frag is False
        assert pattern is None

        is_frag, pattern = is_fragment_pattern("မြန်မာ")
        assert is_frag is False
        assert pattern is None

        is_frag, pattern = is_fragment_pattern("တွင်")
        assert is_frag is False
        assert pattern is None

    def test_is_fragment_pattern_empty(self):
        """Test that empty input returns False."""
        from myspellchecker.text.validator import is_fragment_pattern

        is_frag, pattern = is_fragment_pattern("")
        assert is_frag is False
        assert pattern is None

    # ========================================================================
    # is_incomplete_word tests
    # ========================================================================

    def test_is_incomplete_word_medial_end(self):
        """Test detection of words ending with medial only."""
        from myspellchecker.text.validator import is_incomplete_word

        is_inc, pattern = is_incomplete_word("ကြ")  # Consonant + medial
        assert is_inc is True
        assert pattern == "medial_end"

        is_inc, pattern = is_incomplete_word("မျ")
        assert is_inc is True
        assert pattern == "medial_end"

    def test_is_incomplete_word_virama_end(self):
        """Test detection of words ending with stacking marker."""
        from myspellchecker.text.validator import is_incomplete_word

        is_inc, pattern = is_incomplete_word("န္")  # Consonant + virama
        assert is_inc is True
        assert pattern == "virama_end"

    def test_is_incomplete_word_consonant_medial_end(self):
        """Test detection of words ending with consonant+medial."""
        from myspellchecker.text.validator import is_incomplete_word

        is_inc, pattern = is_incomplete_word("မြန")  # Ends with bare consonant after medial
        assert is_inc is True
        # Could be either consonant_medial_end or medial_consonant_end

    def test_is_incomplete_word_pali_whitelist(self):
        """Test that Pali whitelist words are not flagged as incomplete."""
        from myspellchecker.text.validator import is_incomplete_word

        # Pali words with bare consonant endings are valid
        is_inc, pattern = is_incomplete_word("ဒေသ")  # region
        assert is_inc is False
        assert pattern is None

        is_inc, pattern = is_incomplete_word("ကာလ")  # time/period
        assert is_inc is False
        assert pattern is None

        is_inc, pattern = is_incomplete_word("ဌာန")  # department
        assert is_inc is False
        assert pattern is None

    def test_is_incomplete_word_valid_words(self):
        """Test that valid complete words are not flagged."""
        from myspellchecker.text.validator import is_incomplete_word

        is_inc, pattern = is_incomplete_word("ကျောင်း")
        assert is_inc is False
        assert pattern is None

        is_inc, pattern = is_incomplete_word("မြန်မာ")
        assert is_inc is False
        assert pattern is None

    def test_is_incomplete_word_empty(self):
        """Test that empty input returns False."""
        from myspellchecker.text.validator import is_incomplete_word

        is_inc, pattern = is_incomplete_word("")
        assert is_inc is False
        assert pattern is None

    # ========================================================================
    # is_quality_word tests
    # ========================================================================

    def test_is_quality_word_valid(self):
        """Test that valid words pass quality check."""
        from myspellchecker.text.validator import is_quality_word

        assert is_quality_word("ကျောင်း") is True
        assert is_quality_word("မြန်မာ") is True
        assert is_quality_word("တွင်") is True
        assert is_quality_word("သည်") is True

    def test_is_quality_word_fragments_rejected(self):
        """Test that fragments are rejected."""
        from myspellchecker.text.validator import is_quality_word

        assert is_quality_word("ဉ်") is False  # consonant+asat
        assert is_quality_word("မး") is False  # consonant+tone
        assert is_quality_word("န့်") is False  # consonant+tone+asat

    def test_is_quality_word_double_ending_rejected(self):
        """Test that double-ending patterns are rejected."""
        from myspellchecker.text.validator import is_quality_word

        assert is_quality_word("တွင်င်း") is False
        assert is_quality_word("သည်င်း") is False

    def test_is_quality_word_incomplete_rejected(self):
        """Test that incomplete words are rejected."""
        from myspellchecker.text.validator import is_quality_word

        assert is_quality_word("ကြ") is False  # medial end
        assert is_quality_word("န္") is False  # virama end

    def test_is_quality_word_pure_numeral_rejected(self):
        """Test that pure numerals are rejected."""
        from myspellchecker.text.validator import is_quality_word

        assert is_quality_word("၁၂၃၄") is False
        assert is_quality_word("၀") is False

    def test_is_quality_word_doubled_consonant_rejected(self):
        """Test that doubled consonants are rejected."""
        from myspellchecker.text.validator import is_quality_word

        assert is_quality_word("ကက") is False
        assert is_quality_word("ခခ") is False

    def test_is_quality_word_invalid_vowel_sequence_rejected(self):
        """Test that invalid ေါ sequence is rejected."""
        from myspellchecker.text.validator import is_quality_word

        assert is_quality_word("ေါင်း") is False  # Invalid ေါ

    def test_is_quality_word_empty_rejected(self):
        """Test that empty/whitespace is rejected."""
        from myspellchecker.text.validator import is_quality_word

        assert is_quality_word("") is False
        assert is_quality_word("   ") is False

    def test_is_quality_word_pali_valid(self):
        """Test that Pali whitelist words pass quality check."""
        from myspellchecker.text.validator import is_quality_word

        assert is_quality_word("ဒေသ") is True  # region
        assert is_quality_word("ကာလ") is True  # time/period
        assert is_quality_word("ဌာန") is True  # department

    # ========================================================================
    # get_quality_issues tests
    # ========================================================================

    def test_get_quality_issues_valid_word(self):
        """Test that valid words have no issues."""
        from myspellchecker.text.validator import get_quality_issues

        issues = get_quality_issues("ကျောင်း")
        assert issues == []

        issues = get_quality_issues("မြန်မာ")
        assert issues == []

    def test_get_quality_issues_fragment(self):
        """Test that fragments are detected."""
        from myspellchecker.text.validator import get_quality_issues

        issues = get_quality_issues("ဉ်")
        assert any("fragment" in issue[0] for issue in issues)

    def test_get_quality_issues_double_ending(self):
        """Test that double-ending is detected."""
        from myspellchecker.text.validator import get_quality_issues

        issues = get_quality_issues("တွင်င်း")
        assert any("double_ending" in str(issue) for issue in issues)

    def test_get_quality_issues_multiple_issues(self):
        """Test detection of multiple issues."""
        from myspellchecker.text.validator import get_quality_issues

        # Word with multiple problems
        issues = get_quality_issues("ကက")  # Doubled consonant
        assert len(issues) > 0
        assert any("doubled_consonant" in issue[0] for issue in issues)

    def test_get_quality_issues_pure_numeral(self):
        """Test that pure numerals are detected."""
        from myspellchecker.text.validator import get_quality_issues

        issues = get_quality_issues("၁၂၃")
        assert any("pure_numeral" in issue[0] for issue in issues)

    def test_get_quality_issues_empty(self):
        """Test that empty string has issues."""
        from myspellchecker.text.validator import get_quality_issues

        issues = get_quality_issues("")
        assert any("empty" in issue[0] for issue in issues)


class TestSegmentationFragmentDetection:
    """Tests for is_segmentation_fragment function.

    This function detects likely segmentation artifacts from word segmenters
    like myword. These are words that have been incorrectly split and should
    be filtered from the dictionary.
    """

    # ========================================================================
    # Rule 1: Bare consonant ending (without asat)
    # ========================================================================

    def test_bare_consonant_end_detected(self):
        """Test detection of words ending with bare consonant."""
        from myspellchecker.text.validator import is_segmentation_fragment

        # Words ending with consonants (U+1000-U+1021) without asat
        is_frag, issue = is_segmentation_fragment("ဒစ်ဂျစ်တယ")  # Missing final asat
        assert is_frag is True
        assert issue == "bare_consonant_end"

        is_frag, issue = is_segmentation_fragment("အောင")  # Missing asat
        assert is_frag is True
        assert issue == "bare_consonant_end"

    def test_bare_consonant_end_single_char_exceptions(self):
        """Test that single-char interjections (အ, ဟ) are allowed."""
        from myspellchecker.text.validator import is_segmentation_fragment

        # Single အ and ဟ are valid interjections
        is_frag, issue = is_segmentation_fragment("အ")
        assert is_frag is False
        assert issue is None

        is_frag, issue = is_segmentation_fragment("ဟ")
        assert is_frag is False
        assert issue is None

    def test_bare_consonant_end_multi_char_not_excepted(self):
        """Test that multi-char words ending with အ/ဟ are flagged."""
        from myspellchecker.text.validator import is_segmentation_fragment

        # Multi-char words ending with consonants should be flagged
        is_frag, issue = is_segmentation_fragment("ကအ")
        assert is_frag is True
        assert issue == "bare_consonant_end"

    # ========================================================================
    # Rule 2: Stacked consonant at start (္)
    # ========================================================================

    def test_stacked_start_detected(self):
        """Test detection of stacking marker at word start."""
        from myspellchecker.text.validator import is_segmentation_fragment

        # Stacking marker at position 0 (with proper ending to avoid bare_consonant_end)
        is_frag, issue = is_segmentation_fragment("္ကာ")
        assert is_frag is True
        assert issue == "stacked_start"

        # Stacking marker at position 1 (consonant + stacking) with proper ending
        is_frag, issue = is_segmentation_fragment("စ္ဆာ")
        assert is_frag is True
        assert issue == "stacked_start"

    # ========================================================================
    # Rule 3: Medial at start (ျ ြ ွ ှ)
    # ========================================================================

    def test_medial_start_detected(self):
        """Test detection of medial at word start."""
        from myspellchecker.text.validator import is_segmentation_fragment

        # Medials cannot start a word
        is_frag, issue = is_segmentation_fragment("ြမန်မာ")  # Ya-yit
        assert is_frag is True
        assert issue == "medial_start"

        is_frag, issue = is_segmentation_fragment("ျောက်")  # Ya-pin
        assert is_frag is True
        assert issue == "medial_start"

        is_frag, issue = is_segmentation_fragment("ွန်")  # Wa-hswe
        assert is_frag is True
        assert issue == "medial_start"

        is_frag, issue = is_segmentation_fragment("ှ")  # Ha-hto
        assert is_frag is True
        assert issue == "medial_start"

    # ========================================================================
    # Rule 4: Dependent vowel at start
    # ========================================================================

    def test_dependent_vowel_start_detected(self):
        """Test detection of dependent vowel at word start."""
        from myspellchecker.text.validator import is_segmentation_fragment

        # Dependent vowels cannot start a word
        # Use words with proper endings to avoid bare_consonant_end triggering first
        is_frag, issue = is_segmentation_fragment("ါကာ")  # tall aa at start
        assert is_frag is True
        assert issue == "dependent_vowel_start"

        is_frag, issue = is_segmentation_fragment("ိန်")  # i vowel at start
        assert is_frag is True
        assert issue == "dependent_vowel_start"

        is_frag, issue = is_segmentation_fragment("ေကာင်း")  # e vowel at start
        assert is_frag is True
        assert issue == "dependent_vowel_start"

        is_frag, issue = is_segmentation_fragment("ုံ")  # u vowel at start (with anusvara)
        assert is_frag is True
        assert issue == "dependent_vowel_start"

    # ========================================================================
    # Rule 5: Great Sa at start (ဿ)
    # ========================================================================

    def test_great_sa_start_detected(self):
        """Test detection of Great Sa at word start.

        Great Sa (ဿ, U+103F) is a Pali geminate consonant that only appears
        mid-word in legitimate words. Words starting with ဿ are always
        segmentation fragments.
        """
        from myspellchecker.text.validator import is_segmentation_fragment

        # ဿနာ is a fragment of ပြဿနာ (problem)
        is_frag, issue = is_segmentation_fragment("ဿနာ")
        assert is_frag is True
        assert issue == "great_sa_start"

    def test_great_sa_mid_word_valid(self):
        """Test that Great Sa mid-word is valid.

        Words with ဿ in the middle are legitimate Pali/Sanskrit loanwords.
        """
        from myspellchecker.text.validator import is_segmentation_fragment

        # ပြဿနာ (problem) - valid word with ဿ in middle
        is_frag, issue = is_segmentation_fragment("ပြဿနာ")
        assert is_frag is False
        assert issue is None

        # မနုဿ (human - Pali) - valid word ending with ဿ
        is_frag, issue = is_segmentation_fragment("မနုဿ")
        assert is_frag is False
        assert issue is None

    # ========================================================================
    # Rule 6: Asat + Anusvara sequence (်ံ)
    # ========================================================================

    def test_asat_anusvara_sequence_detected(self):
        """Test detection of asat+anusvara sequence.

        The sequence ်ံ is phonetically impossible in Myanmar:
        - Asat (်) closes a syllable, indicating no following vowel
        - Anusvara (ံ) nasalizes a vowel, requiring a vowel to exist

        These cannot co-occur and indicate OCR errors or garbage data.
        """
        from myspellchecker.text.validator import is_segmentation_fragment

        # Words containing ်ံ sequence
        is_frag, issue = is_segmentation_fragment("အ်ံအပ်")
        assert is_frag is True
        assert issue == "asat_anusvara_sequence"

        is_frag, issue = is_segmentation_fragment("က်ံစာ")
        assert is_frag is True
        assert issue == "asat_anusvara_sequence"

        # Just the sequence itself
        is_frag, issue = is_segmentation_fragment("်ံ")
        # Note: This also starts with dependent vowel, so that rule triggers first
        assert is_frag is True

    def test_valid_anusvara_usage(self):
        """Test that valid anusvara usage is not flagged.

        Valid: vowel + anusvara (e.g., ံု in ကျုံ)
        Invalid: asat + anusvara (်ံ)
        """
        from myspellchecker.text.validator import is_segmentation_fragment

        # Valid words with anusvara after vowel
        is_frag, issue = is_segmentation_fragment("ကုံး")  # valid ံ after ု
        assert is_frag is False
        assert issue is None

        is_frag, issue = is_segmentation_fragment("သံ")  # valid ံ
        assert is_frag is False
        assert issue is None

    # ========================================================================
    # Rule 7: Doubled independent vowel (e.g., ဤဤ, ဥဥ)
    # ========================================================================

    def test_doubled_independent_vowel_detected(self):
        """Test detection of doubled independent vowels.

        Two identical independent vowels as a 2-char word is always an OCR error.
        Independent vowels are standalone characters that represent vowel sounds.
        """
        from myspellchecker.text.validator import is_segmentation_fragment

        # Doubled independent vowels
        is_frag, issue = is_segmentation_fragment("ဤဤ")  # ii ii
        assert is_frag is True
        assert issue == "doubled_independent_vowel"

        is_frag, issue = is_segmentation_fragment("ဥဥ")  # u u
        assert is_frag is True
        assert issue == "doubled_independent_vowel"

        is_frag, issue = is_segmentation_fragment("ဧဧ")  # e e
        assert is_frag is True
        assert issue == "doubled_independent_vowel"

        is_frag, issue = is_segmentation_fragment("ဩဩ")  # o o
        assert is_frag is True
        assert issue == "doubled_independent_vowel"

    def test_single_independent_vowel_valid(self):
        """Test that single independent vowels are valid."""
        from myspellchecker.text.validator import is_segmentation_fragment

        # Single independent vowels can be valid words/particles
        is_frag, issue = is_segmentation_fragment("ဤ")  # "this"
        assert is_frag is False
        assert issue is None

        is_frag, issue = is_segmentation_fragment("ဥ")  # "that"
        assert is_frag is False
        assert issue is None

    def test_different_independent_vowels_valid(self):
        """Test that two different independent vowels are not flagged.

        The rule only catches identical doubled vowels, not different ones.
        """
        from myspellchecker.text.validator import is_segmentation_fragment

        # Different independent vowels together
        is_frag, issue = is_segmentation_fragment("ဤဥ")  # different vowels
        assert is_frag is False
        assert issue is None

    # ========================================================================
    # Valid words should not be flagged
    # ========================================================================

    def test_valid_words_not_flagged(self):
        """Test that valid Myanmar words are not flagged as fragments."""
        from myspellchecker.text.validator import is_segmentation_fragment

        valid_words = [
            "မြန်မာ",  # Myanmar
            "ကျောင်း",  # school
            "စာ",  # letter/writing
            "လူ",  # person
            "ရေ",  # water
            "ကောင်း",  # good
            "တွင်",  # in/at
            "သည်",  # verb ending
            "များ",  # plural marker
        ]

        for word in valid_words:
            is_frag, issue = is_segmentation_fragment(word)
            assert is_frag is False, f"Valid word '{word}' incorrectly flagged"
            assert issue is None

    def test_empty_string_not_flagged(self):
        """Test that empty string returns False (not a fragment)."""
        from myspellchecker.text.validator import is_segmentation_fragment

        is_frag, issue = is_segmentation_fragment("")
        assert is_frag is False
        assert issue is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
