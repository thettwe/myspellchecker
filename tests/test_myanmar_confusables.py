"""Tests for myanmar_confusables.py — confusable pair data and variant generation."""

from myspellchecker.core.myanmar_confusables import (
    ALL_MEDIALS,
    ASPIRATION_PAIRS,
    ASPIRATION_PAIRS_RERANK,
    ASPIRATION_SWAP_MAP,
    MEDIAL_SWAP_PAIRS,
    NASAL_PAIRS,
    STOP_CODA_PAIRS,
    TONE_MARK_PAIRS,
    VOWEL_LENGTH_PAIRS,
    _get_aspiration_variants,
    _get_kinzi_variants,
    _get_medial_variants,
    _get_nasal_variants,
    _get_stacking_variants,
    _get_stop_coda_variants,
    _replace_per_position,
    generate_myanmar_variants,
    is_aspirated_confusable,
    is_medial_confusable,
)

# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


class TestDataIntegrity:
    def test_aspiration_pairs_are_tuples_of_two_chars(self):
        for a, b in ASPIRATION_PAIRS:
            assert len(a) == 1 and len(b) == 1

    def test_medial_swap_pairs_contain_medial_chars(self):
        for a, b in MEDIAL_SWAP_PAIRS:
            assert a in ALL_MEDIALS
            assert b in ALL_MEDIALS

    def test_nasal_pairs_have_expected_count(self):
        assert len(NASAL_PAIRS) == 3

    def test_stop_coda_pairs_have_expected_count(self):
        assert len(STOP_CODA_PAIRS) == 3

    def test_tone_mark_pairs_cover_dot_below_and_visarga(self):
        for a, b in TONE_MARK_PAIRS:
            assert "\u1037" in (a, b) and "\u1038" in (a, b)

    def test_vowel_length_pairs_cover_short_and_long(self):
        assert len(VOWEL_LENGTH_PAIRS) == 2

    def test_aspiration_swap_map_is_bidirectional(self):
        for k, v in ASPIRATION_SWAP_MAP.items():
            assert ASPIRATION_SWAP_MAP[v] == k

    def test_aspiration_rerank_pairs_involve_myanmar_consonants(self):
        # RERANK pairs may differ from FULL pairs (e.g., ဗ↔ဘ vs ဖ↔ဘ),
        # but all characters should be Myanmar consonants (U+1000-U+1021).
        for a, b in ASPIRATION_PAIRS_RERANK:
            assert 0x1000 <= ord(a) <= 0x1021, f"'{a}' not a Myanmar consonant"
            assert 0x1000 <= ord(b) <= 0x1021, f"'{b}' not a Myanmar consonant"


# ---------------------------------------------------------------------------
# _replace_per_position
# ---------------------------------------------------------------------------


class TestReplacePerPosition:
    def test_single_occurrence_produces_one_variant(self):
        variants = set()
        _replace_per_position("ကခ", "က", "ခ", variants)
        assert "ခခ" in variants

    def test_two_occurrences_produce_two_variants(self):
        variants = set()
        _replace_per_position("ကကခ", "က", "ခ", variants)
        assert len(variants) == 2
        assert "ခကခ" in variants
        assert "ကခခ" in variants

    def test_no_match_produces_no_variants(self):
        variants = set()
        _replace_per_position("abc", "x", "y", variants)
        assert len(variants) == 0


# ---------------------------------------------------------------------------
# Variant generation functions
# ---------------------------------------------------------------------------


class TestAspirationVariants:
    def test_generates_aspirated_swap_for_ka(self):
        # က → ခ
        variants = _get_aspiration_variants("က")
        assert "ခ" in variants

    def test_bidirectional_swap(self):
        # ခ → က
        variants = _get_aspiration_variants("ခ")
        assert "က" in variants

    def test_original_word_not_in_variants(self):
        variants = _get_aspiration_variants("က")
        assert "က" not in variants


class TestMedialVariants:
    def test_medial_swap_ya_pin_to_ya_yit(self):
        # ကျ → ကြ (ya-pin to ya-yit)
        word = "က\u103b"  # ကျ
        variants = _get_medial_variants(word)
        assert "က\u103c" in variants  # ကြ

    def test_medial_insertion_ha_htoe(self):
        # မာ → မှာ (insert ha-htoe)
        word = "မာ"
        variants = _get_medial_variants(word)
        assert any("\u103e" in v for v in variants)

    def test_medial_deletion_ha_htoe(self):
        # မှာ → မာ (delete ha-htoe)
        word = "မ\u103eာ"  # မှာ
        variants = _get_medial_variants(word)
        assert "မာ" in variants


class TestNasalVariants:
    def test_na_that_to_ma_that(self):
        # န် → မ်
        word = "န\u103a"
        variants = _get_nasal_variants(word)
        assert "မ\u103a" in variants

    def test_empty_for_non_nasal_word(self):
        variants = _get_nasal_variants("က")
        assert len(variants) == 0


class TestStopCodaVariants:
    def test_ka_that_to_ta_that(self):
        # က် → တ်
        word = "က\u103a"
        variants = _get_stop_coda_variants(word)
        assert "တ\u103a" in variants


class TestStackingVariants:
    def test_asat_to_virama(self):
        # ဒ်ဓ → ဒ္ဓ (asat → virama)
        word = "ဒ\u103aဓ"
        variants = _get_stacking_variants(word)
        assert "ဒ\u1039ဓ" in variants

    def test_virama_to_asat(self):
        # ဒ္ဓ → ဒ်ဓ (virama → asat)
        word = "ဒ\u1039ဓ"
        variants = _get_stacking_variants(word)
        assert "ဒ\u103aဓ" in variants


# ---------------------------------------------------------------------------
# Confusable-check predicates
# ---------------------------------------------------------------------------


class TestIsAspiratedConfusable:
    def test_single_aspiration_diff_returns_true(self):
        assert is_aspirated_confusable("က", "ခ") is True

    def test_multiple_diffs_returns_false(self):
        assert is_aspirated_confusable("ကက", "ခခ") is False

    def test_same_word_returns_false(self):
        assert is_aspirated_confusable("က", "က") is False

    def test_different_lengths_returns_false(self):
        assert is_aspirated_confusable("က", "ကခ") is False

    def test_empty_strings_returns_false(self):
        assert is_aspirated_confusable("", "") is False


class TestIsMedialConfusable:
    def test_ya_pin_to_ya_yit_swap(self):
        # ကျ → ကြ
        assert is_medial_confusable("က\u103b", "က\u103c") is True

    def test_medial_insertion(self):
        # မာ → မှာ (ha-htoe inserted)
        assert is_medial_confusable("မာ", "မ\u103eာ") is True

    def test_non_medial_diff_returns_false(self):
        assert is_medial_confusable("ကို", "ခို") is False


# ---------------------------------------------------------------------------
# generate_myanmar_variants (public API)
# ---------------------------------------------------------------------------


class TestGenerateMyanmarVariants:
    def test_excludes_original_word(self):
        word = "ကို"
        variants = generate_myanmar_variants(word)
        assert word not in variants

    def test_generates_aspiration_variants(self):
        # ကို should have ခို as a variant
        variants = generate_myanmar_variants("ကို")
        assert any("ခ" in v for v in variants)

    def test_returns_set(self):
        variants = generate_myanmar_variants("မြန်မာ")
        assert isinstance(variants, set)

    def test_empty_word_returns_empty_set(self):
        variants = generate_myanmar_variants("")
        assert variants == set()

    def test_generates_kinzi_variants(self):
        # အင်္ဂလိပ် (English, with Kinzi) should have Kinzi-dropped variant
        variants = generate_myanmar_variants("အင်္ဂလိပ်")
        assert "အင်ဂလိပ်" in variants


# ---------------------------------------------------------------------------
# Kinzi variant generation
# ---------------------------------------------------------------------------


class TestKinziVariants:
    """Tests for _get_kinzi_variants()."""

    def test_kinzi_removal_english(self):
        """အင်္ဂလိပ် (with Kinzi) → အင်ဂလိပ် (Kinzi dropped)."""
        variants = _get_kinzi_variants("အင်္ဂလိပ်")
        assert "အင်ဂလိပ်" in variants

    def test_kinzi_insertion_english(self):
        """အင်ဂလိပ် (without Kinzi) → အင်္ဂလိပ် (Kinzi added)."""
        variants = _get_kinzi_variants("အင်ဂလိပ်")
        assert "အင်္ဂလိပ်" in variants

    def test_kinzi_removal_ship(self):
        """သင်္ဘော (ship, with Kinzi) → သင်ဘော."""
        variants = _get_kinzi_variants("သင်္ဘော")
        assert "သင်ဘော" in variants

    def test_kinzi_insertion_ship(self):
        """သင်ဘော (ship, without Kinzi) → သင်္ဘော."""
        variants = _get_kinzi_variants("သင်ဘော")
        assert "သင်္ဘော" in variants

    def test_kinzi_removal_auspiciousness(self):
        """မင်္ဂလာ (with Kinzi) → မင်ဂလာ."""
        variants = _get_kinzi_variants("မင်္ဂလာ")
        assert "မင်ဂလာ" in variants

    def test_kinzi_insertion_auspiciousness(self):
        """မင်ဂလာ (without Kinzi) → မင်္ဂလာ."""
        variants = _get_kinzi_variants("မင်ဂလာ")
        assert "မင်္ဂလာ" in variants

    def test_kinzi_singapore(self):
        """စင်္ကာပူ (Singapore) ↔ စင်ကာပူ — bidirectional."""
        assert "စင်ကာပူ" in _get_kinzi_variants("စင်္ကာပူ")
        assert "စင်္ကာပူ" in _get_kinzi_variants("စင်ကာပူ")

    def test_no_kinzi_in_plain_word(self):
        """မြန်မာ (Myanmar) has no Kinzi potential — empty result."""
        assert _get_kinzi_variants("မြန်မာ") == set()

    def test_no_kinzi_in_good(self):
        """ကောင်း (good) — ng+asat at end, no following consonant."""
        assert _get_kinzi_variants("ကောင်း") == set()

    def test_empty_word(self):
        """Empty string produces no variants."""
        assert _get_kinzi_variants("") == set()

    def test_kinzi_thingyan(self):
        """သင်္ကြန် (Thingyan) ↔ သင်ကြန်."""
        assert "သင်ကြန်" in _get_kinzi_variants("သင်္ကြန်")
        assert "သင်္ကြန်" in _get_kinzi_variants("သင်ကြန်")

    def test_kinzi_shirt(self):
        """အင်္ကျီ (shirt) ↔ အင်ကျီ."""
        assert "အင်ကျီ" in _get_kinzi_variants("အင်္ကျီ")
        assert "အင်္ကျီ" in _get_kinzi_variants("အင်ကျီ")

    def test_removal_is_bidirectional_with_insertion(self):
        """Removing Kinzi from X produces Y, inserting Kinzi into Y produces X."""
        correct = "သင်္ဘော"
        error = "သင်ဘော"
        assert error in _get_kinzi_variants(correct)
        assert correct in _get_kinzi_variants(error)
