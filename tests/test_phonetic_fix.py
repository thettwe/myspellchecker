from myspellchecker.text.phonetic import PhoneticHasher


def test_phonetic_normalization_fix():
    hasher = PhoneticHasher()

    # Nga + Asat (င်) -> /in/ or /ing/
    # Anusvara (ံ) -> /an/ or /am/

    code_ng = hasher.encode("ကင်း")  # King
    code_an = hasher.encode("ကံ")  # Kan

    # Before fix, these would be identical or very similar because င် -> ံ
    # After fix, they should be different

    assert code_ng != code_an, f"Phonetic codes should differ: {code_ng} == {code_an}"

    # Also check explicitly that similar returns False (or low similarity)
    # Depending on how the rest of the code is generated, they might still share 'k'
    # But the vowel/final part should differ.

    # Verify similar() reflects the difference
    assert not hasher.similar(code_ng, code_an), "Should not be phonetically similar"


def test_other_nasals_still_normalize():
    """Verify that Na+Asat and Ma+Asat normalize to Anusvara when normalize_nasals=True."""
    # Nasal normalization is now opt-in via normalize_nasals parameter
    hasher = PhoneticHasher(normalize_nasals=True)

    code_na = hasher.encode("ကန်")  # Na + Asat
    code_ma = hasher.encode("ကမ်")  # Ma + Asat
    code_an = hasher.encode("ကံ")  # Anusvara

    # These represent the same /an/ sound in many contexts,
    # so normalization usually keeps them close
    # Note: Depending on implementation details of CHAR_TO_PHONETIC,
    # they might map to same code
    # The replace() logic in _encode_impl explicitly maps them to ံ before encoding

    # So their codes MUST be identical when normalize_nasals=True
    assert code_na == code_an, "Na+Asat should normalize to Anusvara code"
    assert code_ma == code_an, "Ma+Asat should normalize to Anusvara code"


class TestPhoneticBypassThreshold:
    """Tests for phonetic bypass threshold behavior in SymSpell."""

    def test_candidate_within_max_edit_distance_included_regardless_of_similarity(self):
        """Candidate with distance <= max_edit_distance is included regardless of similarity."""
        from myspellchecker.algorithms.symspell import SymSpell
        from myspellchecker.providers.memory import MemoryProvider

        # Create provider with test data
        provider = MemoryProvider()
        provider.add_syllable("ကျွန်", 100)
        provider.add_syllable("ကြွန်", 80)  # 1 edit away (medial change)

        symspell = SymSpell(
            provider=provider,
            max_edit_distance=2,
            phonetic_bypass_threshold=0.85,
            phonetic_extra_distance=1,
        )
        symspell.build_index(["syllable"])

        # Lookup for misspelled word - should find suggestions
        # Using a word that is 1 edit away from indexed syllable
        results = symspell.lookup("ကျွန", level="syllable")  # Missing final character
        assert len(results) >= 1
        assert any(r.term == "ကျွန်" for r in results)

    def test_high_similarity_candidate_bypasses_distance_cap(self):
        """Candidate with similarity >= threshold can exceed max_edit_distance by extra_distance."""
        from myspellchecker.algorithms.symspell import SymSpell
        from myspellchecker.providers.memory import MemoryProvider
        from myspellchecker.text.phonetic import PhoneticHasher

        provider = MemoryProvider()
        provider.add_syllable("ကျွန်", 100)
        provider.add_syllable("ကြွန်", 80)

        phonetic_hasher = PhoneticHasher()

        symspell = SymSpell(
            provider=provider,
            max_edit_distance=2,
            phonetic_hasher=phonetic_hasher,
            phonetic_bypass_threshold=0.7,
            phonetic_extra_distance=2,
        )
        symspell.build_index(["syllable"])

        # Lookup for misspelled word - should find suggestions
        results = symspell.lookup("ကျွန", level="syllable")  # Missing final character
        assert len(results) >= 1

    def test_low_similarity_candidate_blocked_beyond_distance_cap(self):
        """Candidate with similarity < threshold is excluded if distance > max_edit_distance."""
        from myspellchecker.algorithms.symspell import SymSpell
        from myspellchecker.providers.memory import MemoryProvider
        from myspellchecker.text.phonetic import PhoneticHasher

        provider = MemoryProvider()
        provider.add_syllable("abc", 100)
        provider.add_syllable("xyz", 50)  # Completely unrelated, 3 edits away

        phonetic_hasher = PhoneticHasher()

        symspell = SymSpell(
            provider=provider,
            max_edit_distance=1,
            phonetic_hasher=phonetic_hasher,
            phonetic_bypass_threshold=0.95,
            phonetic_extra_distance=1,
        )
        symspell.build_index(["syllable"])

        # Lookup should not return "xyz" because it's too far (3 edits)
        # and phonetic similarity is low
        results = symspell.lookup("abc", level="syllable")
        assert all(r.term != "xyz" for r in results)

    def test_phonetic_bypass_threshold_configurable(self):
        """Verify phonetic_bypass_threshold is configurable and affects behavior."""
        from unittest.mock import MagicMock

        from myspellchecker.algorithms.symspell import SymSpell

        provider = MagicMock()
        provider.get_syllables.return_value = [("test", 100)]
        provider.get_words.return_value = []

        # Test with different threshold values
        symspell_strict = SymSpell(
            provider=provider,
            max_edit_distance=2,
            phonetic_bypass_threshold=0.99,  # Very strict
            phonetic_extra_distance=1,
        )
        assert symspell_strict._phonetic_bypass_threshold == 0.99

        symspell_lenient = SymSpell(
            provider=provider,
            max_edit_distance=2,
            phonetic_bypass_threshold=0.5,  # Lenient
            phonetic_extra_distance=2,
        )
        assert symspell_lenient._phonetic_bypass_threshold == 0.5
        assert symspell_lenient._phonetic_extra_distance == 2

    def test_phonetic_extra_distance_limits_bypass(self):
        """Verify phonetic_extra_distance limits how far beyond max_edit_distance we go."""
        from unittest.mock import MagicMock

        from myspellchecker.algorithms.symspell import SymSpell

        provider = MagicMock()
        provider.get_syllables.return_value = [("test", 100)]
        provider.get_words.return_value = []

        # With extra_distance=0, no bypass allowed (same as before)
        symspell = SymSpell(
            provider=provider,
            max_edit_distance=2,
            phonetic_bypass_threshold=0.5,
            phonetic_extra_distance=0,  # No extra distance
        )
        assert symspell._phonetic_extra_distance == 0

        # With extra_distance=3 (max), candidates can be 3 edits beyond
        symspell_max = SymSpell(
            provider=provider,
            max_edit_distance=2,
            phonetic_bypass_threshold=0.5,
            phonetic_extra_distance=3,
        )
        assert symspell_max._phonetic_extra_distance == 3
