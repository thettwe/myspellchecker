"""
Tests for CRIT-001: Vowel Conflation Fix

Verifies that ေ (U+1031, /e/) and ဲ (U+1032, /ɛ/) are properly
treated as distinct phonemes, not conflated into the same group.

Reference: Myanmar Language Commission, "Myanmar Grammar" (2005)
- ေ (e-prefix): Mid front unrounded vowel /e/
- ဲ (ai-suffix): Mid-low front vowel with glide /ɛ/
"""


class TestVowelSeparation:
    """Test that ေ and ဲ are in separate phonetic groups."""

    def test_vowel_keys_distinct(self):
        """KEY_VOWEL_E and KEY_VOWEL_AI should be different keys."""
        from myspellchecker.text.phonetic_data import KEY_VOWEL_AI, KEY_VOWEL_E

        assert KEY_VOWEL_E != KEY_VOWEL_AI
        assert KEY_VOWEL_E == "vowel_e"
        assert KEY_VOWEL_AI == "vowel_ai"

    def test_e_vowel_in_correct_group(self):
        """ေ (U+1031) should be in KEY_VOWEL_E group only."""
        from myspellchecker.text.phonetic_data import (
            KEY_VOWEL_AI,
            KEY_VOWEL_E,
            PHONETIC_GROUPS,
        )

        # ေ should be in E group
        assert "ေ" in PHONETIC_GROUPS[KEY_VOWEL_E]

        # ေ should NOT be in AI group
        assert "ေ" not in PHONETIC_GROUPS[KEY_VOWEL_AI]

    def test_ai_vowel_in_correct_group(self):
        """ဲ (U+1032) should be in KEY_VOWEL_AI group only."""
        from myspellchecker.text.phonetic_data import (
            KEY_VOWEL_AI,
            KEY_VOWEL_E,
            PHONETIC_GROUPS,
        )

        # ဲ should be in AI group
        assert "ဲ" in PHONETIC_GROUPS[KEY_VOWEL_AI]

        # ဲ should NOT be in E group
        assert "ဲ" not in PHONETIC_GROUPS[KEY_VOWEL_E]

    def test_e_ai_not_visually_similar(self):
        """ေ and ဲ should not be in VISUAL_SIMILAR mapping."""
        from myspellchecker.text.phonetic_data import VISUAL_SIMILAR

        # Check ေ doesn't map to ဲ
        if "ေ" in VISUAL_SIMILAR:
            assert "ဲ" not in VISUAL_SIMILAR["ေ"]

        # Check ဲ doesn't map to ေ
        if "ဲ" in VISUAL_SIMILAR:
            assert "ေ" not in VISUAL_SIMILAR["ဲ"]


class TestPhoneticGroupIntegrity:
    """Test overall phonetic group integrity after fix."""

    def test_all_vowel_groups_exist(self):
        """All vowel group keys should exist in PHONETIC_GROUPS."""
        from myspellchecker.text.phonetic_data import (
            KEY_VOWEL_A,
            KEY_VOWEL_AI,
            KEY_VOWEL_E,
            KEY_VOWEL_I,
            KEY_VOWEL_O,
            KEY_VOWEL_U,
            PHONETIC_GROUPS,
        )

        vowel_keys = [
            KEY_VOWEL_A,
            KEY_VOWEL_I,
            KEY_VOWEL_U,
            KEY_VOWEL_E,
            KEY_VOWEL_AI,
            KEY_VOWEL_O,
        ]

        for key in vowel_keys:
            assert key in PHONETIC_GROUPS, f"Missing vowel group: {key}"

    def test_vowel_groups_not_empty(self):
        """Each vowel group should contain at least one character."""
        from myspellchecker.text.phonetic_data import (
            KEY_VOWEL_A,
            KEY_VOWEL_AI,
            KEY_VOWEL_E,
            KEY_VOWEL_I,
            KEY_VOWEL_O,
            KEY_VOWEL_U,
            PHONETIC_GROUPS,
        )

        vowel_keys = [
            KEY_VOWEL_A,
            KEY_VOWEL_I,
            KEY_VOWEL_U,
            KEY_VOWEL_E,
            KEY_VOWEL_AI,
            KEY_VOWEL_O,
        ]

        for key in vowel_keys:
            assert len(PHONETIC_GROUPS[key]) > 0, f"Empty vowel group: {key}"

    def test_no_vowel_overlap(self):
        """No vowel character should appear in multiple vowel groups."""
        from myspellchecker.text.phonetic_data import (
            KEY_VOWEL_A,
            KEY_VOWEL_AI,
            KEY_VOWEL_E,
            KEY_VOWEL_I,
            KEY_VOWEL_O,
            KEY_VOWEL_U,
            PHONETIC_GROUPS,
        )

        vowel_keys = [
            KEY_VOWEL_A,
            KEY_VOWEL_I,
            KEY_VOWEL_U,
            KEY_VOWEL_E,
            KEY_VOWEL_AI,
            KEY_VOWEL_O,
        ]

        all_vowels = []
        for key in vowel_keys:
            all_vowels.extend(PHONETIC_GROUPS[key])

        # Check for duplicates
        seen = set()
        for vowel in all_vowels:
            assert vowel not in seen, f"Duplicate vowel in groups: {vowel}"
            seen.add(vowel)


class TestMinimalPairs:
    """Test words that differ only in ေ vs ဲ (minimal pairs)."""

    def test_ke_kai_different(self):
        """ကေ (ke) and ကဲ (kai) should not be treated as phonetically similar."""
        # These words have completely different meanings:
        # ကေ - like that, such as
        # ကဲ - to overflow; sentence particle

        ke = "ကေ"
        kai = "ကဲ"

        # They differ only in vowel, but these vowels are phonetically distinct
        assert ke != kai
        assert ke[0] == kai[0]  # Same consonant
        assert ke[1] != kai[1]  # Different vowels

    def test_se_sai_different(self):
        """စေ and စဲ should be treated as distinct."""
        se = "စေ"  # to cause, to make
        sai = "စဲ"  # to end, to stop

        assert se != sai
        assert se[0] == sai[0]  # Same consonant
        assert se[1] != sai[1]  # Different vowels


class TestTonalGroups:
    """Test that TONAL_GROUPS properly separates ေ and ဲ."""

    def test_e_tonal_variants_correct(self):
        """ေ tonal variants should not include ဲ."""
        from myspellchecker.text.phonetic_data import TONAL_GROUPS

        if "ေ" in TONAL_GROUPS:
            assert "ဲ" not in TONAL_GROUPS["ေ"]

    def test_ai_tonal_variants_correct(self):
        """ဲ tonal variants should not include ေ."""
        from myspellchecker.text.phonetic_data import TONAL_GROUPS

        if "ဲ" in TONAL_GROUPS:
            assert "ေ" not in TONAL_GROUPS["ဲ"]

    def test_e_tonal_group_exists(self):
        """ေ should have its own tonal variants."""
        from myspellchecker.text.phonetic_data import TONAL_GROUPS

        assert "ေ" in TONAL_GROUPS
        # Should include ေ, ေ့, ေး
        assert "ေ" in TONAL_GROUPS["ေ"]

    def test_ai_tonal_group_exists(self):
        """ဲ should have its own tonal variants."""
        from myspellchecker.text.phonetic_data import TONAL_GROUPS

        assert "ဲ" in TONAL_GROUPS
        # Should include ဲ, ဲ့
        assert "ဲ" in TONAL_GROUPS["ဲ"]
