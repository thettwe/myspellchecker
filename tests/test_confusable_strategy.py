"""Tests for confusable variant generation utilities.

The ConfusableVariantStrategy class has been removed; its logic is now
handled by NgramContextChecker.check_word_in_context().  These tests
cover the module-level functions that remain as re-exports.
"""

from unittest.mock import MagicMock, patch

from myspellchecker.core.validation_strategies.confusable_strategy import (
    _get_medial_variants,
    _get_nasal_variants,
    _get_stop_coda_variants,
    _is_aspirated_confusable,
    generate_confusable_variants,
)

# ===========================================================================
# Module-level function tests: generate_confusable_variants
# ===========================================================================


class TestGenerateConfusableVariants:
    """Tests for the generate_confusable_variants() function."""

    def test_returns_set_excluding_original(self):
        """Variants must not include the original word."""
        hasher = MagicMock()
        hasher.get_phonetic_variants.return_value = {"A", "B", "original"}
        hasher.get_tonal_variants.return_value = set()

        result = generate_confusable_variants("original", hasher)
        assert "original" not in result
        assert "A" in result
        assert "B" in result

    def test_combines_phonetic_tonal_medial_nasal(self):
        """Result is the union of all four variant sources minus the original."""
        hasher = MagicMock()
        hasher.get_phonetic_variants.return_value = {"phon1"}
        hasher.get_tonal_variants.return_value = {"tone1"}

        with (
            patch(
                "myspellchecker.core.myanmar_confusables._get_medial_variants",
                return_value={"med1"},
            ),
            patch(
                "myspellchecker.core.myanmar_confusables._get_nasal_variants",
                return_value={"nas1"},
            ),
        ):
            result = generate_confusable_variants("word", hasher)

        assert result == {"phon1", "tone1", "med1", "nas1"}

    def test_empty_when_all_sources_empty(self):
        """No variants when hasher and helpers return nothing."""
        hasher = MagicMock()
        hasher.get_phonetic_variants.return_value = set()
        hasher.get_tonal_variants.return_value = set()

        result = generate_confusable_variants("word", hasher)
        # Medial and nasal generators may still produce variants for
        # certain inputs, but for a plain ASCII input they won't.
        # The test just validates the function doesn't crash.
        assert isinstance(result, set)
        assert "word" not in result


# ===========================================================================
# Medial variant generation
# ===========================================================================


class TestGetMedialVariants:
    """Tests for _get_medial_variants()."""

    def test_ha_htoe_insertion(self):
        """Inserting ha-htoe (\u103e) after a consonant."""
        # Myanmar consonant ka (\u1000) + vowel aa (\u102c) = "ကာ"
        word = "\u1000\u102c"  # ကာ
        variants = _get_medial_variants(word)
        # Should contain a variant with ha-htoe inserted: ကှာ
        expected = "\u1000\u103e\u102c"
        assert expected in variants

    def test_ha_htoe_deletion(self):
        """Removing ha-htoe (\u103e) from a word."""
        # မှာ = \u1019\u103e\u102c
        word = "\u1019\u103e\u102c"
        variants = _get_medial_variants(word)
        # Should contain မာ (ha-htoe removed)
        expected = "\u1019\u102c"
        assert expected in variants

    def test_single_consonant_deletion_blocked(self):
        """Deleting medial from a 2-char word should not produce single consonant."""
        # မှ = \u1019\u103e (consonant + ha-htoe only, len=2)
        word = "\u1019\u103e"
        variants = _get_medial_variants(word)
        # Single consonant "\u1019" should NOT appear (len <= 1 check)
        assert "\u1019" not in variants

    def test_medial_swap_ya_yit_to_ya_pin(self):
        """Swap ျ (\u103b) to ြ (\u103c)."""
        # ပျော် = \u1015\u103b\u1031\u102c\u103a
        word = "\u1015\u103b\u1031\u102c\u103a"
        variants = _get_medial_variants(word)
        swapped = word.replace("\u103b", "\u103c")
        assert swapped in variants

    def test_medial_swap_ya_pin_to_ya_yit(self):
        """Swap ြ (\u103c) to ျ (\u103b)."""
        word = "\u1015\u103c\u1031\u102c\u103a"
        variants = _get_medial_variants(word)
        swapped = word.replace("\u103c", "\u103b")
        assert swapped in variants


# ===========================================================================
# Nasal variant generation
# ===========================================================================


class TestGetNasalVariants:
    """Tests for _get_nasal_variants()."""

    def test_na_that_to_ma_that(self):
        """န\u103a (\u1014\u103a) swapped to မ\u103a (\u1019\u103a)."""
        # ကန် = \u1000\u1014\u103a
        word = "\u1000\u1014\u103a"
        variants = _get_nasal_variants(word)
        assert "\u1000\u1019\u103a" in variants  # ကမ်

    def test_na_that_to_anusvara(self):
        """န\u103a (\u1014\u103a) swapped to anusvara (\u1036)."""
        word = "\u1000\u1014\u103a"
        variants = _get_nasal_variants(word)
        assert "\u1000\u1036" in variants  # ကံ

    def test_anusvara_to_na_that(self):
        """anusvara (\u1036) swapped to န\u103a (\u1014\u103a)."""
        word = "\u1000\u1036"  # ကံ
        variants = _get_nasal_variants(word)
        assert "\u1000\u1014\u103a" in variants  # ကန်

    def test_no_nasal_ending(self):
        """No nasal ending produces no variants."""
        word = "\u1000\u102c"  # ကာ
        variants = _get_nasal_variants(word)
        assert variants == set()


class TestGetStopCodaVariants:
    """Tests for _get_stop_coda_variants()."""

    def test_generates_internal_stop_coda_swap(self):
        """Generate က်↔တ် swaps for non-final coda positions."""
        variants = _get_stop_coda_variants("ဘက်ဂျက်")
        assert "ဘတ်ဂျက်" in variants


# ===========================================================================
# _is_aspirated_confusable
# ===========================================================================


class TestIsAspiratedConfusable:
    """Tests for _is_aspirated_confusable helper."""

    def test_ka_kha_swap(self):
        """က↔ခ initial swap detected."""
        assert _is_aspirated_confusable("\u1000\u103b\u1000\u103a", "\u1001\u103b\u1000\u103a")

    def test_ta_tha_swap(self):
        """တ↔ထ initial swap detected."""
        assert _is_aspirated_confusable(
            "\u1010\u1019\u1004\u103a\u1038", "\u1011\u1019\u1004\u103a\u1038"
        )

    def test_pa_pha_swap(self):
        """ပ↔ဖ initial swap detected."""
        assert _is_aspirated_confusable(
            "\u1015\u102d\u1014\u1015\u103a", "\u1016\u102d\u1014\u1015\u103a"
        )

    def test_non_aspirated_pair_returns_false(self):
        """Non-aspiration consonant change is not aspirated confusable."""
        # က→င is NOT an aspiration pair
        assert not _is_aspirated_confusable("\u1000\u102c", "\u1004\u102c")

    def test_medial_change_returns_false(self):
        """Medial insertion/swap is not aspirated confusable."""
        assert not _is_aspirated_confusable("\u1019\u102c", "\u1019\u103e\u102c")

    def test_same_word_returns_false(self):
        assert not _is_aspirated_confusable("\u1000\u102c", "\u1000\u102c")

    def test_different_length_returns_false(self):
        assert not _is_aspirated_confusable("\u1000\u102c", "\u1000\u102c\u1038")
