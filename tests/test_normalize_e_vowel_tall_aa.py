"""Unit tests for the consonant-gated ``normalize_e_vowel_tall_aa``.

Per MLC orthography + UTN #11 §3.3, the "aw" vowel after certain round-bottom
consonants takes TALL AA (U+102B); elsewhere it takes AA (U+102C). The v1.5
benchmark gold validates the subset {ပ, ခ, ဒ} — that's the whitelist this
module uses. Other consonants that classical orthography might put in the
round-bottom set (ဖ, ဘ, မ, ရ, ...) are intentionally NOT in the whitelist
because modern standard Burmese has them commonly using AA (e.g., ဖော်,
ဘော, ရော).

This test module pins the canonical behaviour and guards against a regression
to the pre-2026-04-19 flat rewrite that collapsed all ေါ → ော.
"""

from __future__ import annotations

import pytest

from myspellchecker.text.normalize import normalize_e_vowel_tall_aa

_E = "\u1031"
_AA = "\u102c"
_TALL_AA = "\u102b"


class TestCanonicalWhitelistPreserved:
    """ပေါ်, ခေါင်း, ဒေါ် must be preserved unchanged."""

    @pytest.mark.parametrize(
        "word",
        [
            "ပေါ်",  # "on, upon"
            "ပေါင်း",  # "to add up"
            "ခေါ်",  # "to call"
            "ခေါင်း",  # "head"
            "ဒေါ်",  # "Mrs."
            "ပေါ်ပေါက်",  # compound, both positions ပ
            "ခေါင်စာရင်း",  # compound with ခေါင်
        ],
    )
    def test_whitelist_tall_aa_preserved(self, word: str) -> None:
        assert normalize_e_vowel_tall_aa(word) == word


class TestComplementKeepsFlatAA:
    """Complement-set consonants must use U+102C and stay unchanged."""

    @pytest.mark.parametrize(
        "word",
        [
            "ကောင်း",  # "good"
            "ကော်",  # "sticky"
            "ကော်ဖီ",  # "coffee"
            "တော",  # "forest"
            "စော",  # "early"
            "ထော",  # component
            "အော်",  # "to shout"
            "ဆော်",  # "to knock"
            "ဟော်",  # "hall"
            "နော်",  # colloquial "right?"
            "ယော",  # interjection
            "သော",  # "the one that"
            "လော်",  # "to chase"
            # Consonants that classical MLC puts in round-bottom but which
            # modern Burmese pairs with AA — must stay AA, not get corrupted
            # to TALL AA:
            "ဖော်",  # "friend, to prop up"
            "ဘော",  # colloquial
            "မော",  # "tired"
            "ရော",  # "also, mixed"
        ],
    )
    def test_complement_flat_aa_unchanged(self, word: str) -> None:
        assert normalize_e_vowel_tall_aa(word) == word


class TestOcrFlatteningAfterWhitelist:
    """Flat `ော` after {ပ, ခ, ဒ} is repaired to `ေါ`."""

    @pytest.mark.parametrize(
        "flat,expected",
        [
            ("ပော်", "ပေါ်"),
            ("ပောင်း", "ပေါင်း"),
            ("ခော်", "ခေါ်"),
            ("ခောင်း", "ခေါင်း"),
            ("ဒော်", "ဒေါ်"),
        ],
    )
    def test_flat_to_tall_whitelist(self, flat: str, expected: str) -> None:
        assert normalize_e_vowel_tall_aa(flat) == expected

    @pytest.mark.parametrize(
        "word",
        [
            "ဖော်",  # ဖ not in whitelist — stays flat (modern Burmese uses AA here)
            "ဘော",  # ဘ not in whitelist
            "မော်",  # မ not in whitelist
            "ရော",  # ရ not in whitelist
            "ကော်",  # က not in whitelist
            "တော",  # တ not in whitelist
        ],
    )
    def test_flat_outside_whitelist_unchanged(self, word: str) -> None:
        """Consonants outside the narrow whitelist must not get repaired.
        This prevents corrupting correct modern Burmese forms."""
        assert normalize_e_vowel_tall_aa(word) == word


class TestWrongTallAfterComplement:
    """`ေါ` after a complement-set consonant is flattened to `ော`."""

    @pytest.mark.parametrize(
        "wrong,expected",
        [
            ("ကေါင်း", "ကောင်း"),
            ("တေါ", "တော"),
            ("အေါ်", "အော်"),
            ("ဆေါ်", "ဆော်"),
            ("ထေါ", "ထော"),
            ("စေါ", "စော"),
            ("သေါ", "သော"),
            # Even consonants classical MLC puts in round-bottom but that
            # modern Burmese treats with AA — a stray TALL AA is flattened:
            ("ဖေါ်", "ဖော်"),
            ("မေါ", "မော"),
            ("ရေါ", "ရော"),
        ],
    )
    def test_tall_to_flat(self, wrong: str, expected: str) -> None:
        assert normalize_e_vowel_tall_aa(wrong) == expected


class TestMidCompound:
    """Pattern must fire mid-compound, not only at word start."""

    def test_mid_compound_whitelist_preserved(self) -> None:
        word = "သန်းခေါင်စာရင်း"  # "census" — ခ mid-compound
        assert normalize_e_vowel_tall_aa(word) == word

    def test_mid_compound_flat_repaired(self) -> None:
        flat = "စုစုပေါင်း"  # Use ပ mid-compound
        assert normalize_e_vowel_tall_aa(flat) == flat  # already canonical

        flat_variant = "စုစုပောင်း"  # flat ော after mid-compound ပ → repair
        assert normalize_e_vowel_tall_aa(flat_variant) == "စုစုပေါင်း"

    def test_multiple_aw_vowels_in_one_word(self) -> None:
        flat = "ပောက်ပောက်"
        expected = "ပေါက်ပေါက်"
        assert normalize_e_vowel_tall_aa(flat) == expected


class TestPassthrough:
    """Non-targeted inputs must be untouched."""

    def test_empty(self) -> None:
        assert normalize_e_vowel_tall_aa("") == ""

    def test_non_myanmar_passthrough(self) -> None:
        assert normalize_e_vowel_tall_aa("hello world") == "hello world"

    def test_myanmar_without_e_vowel(self) -> None:
        word = "မြန်မာ"
        assert normalize_e_vowel_tall_aa(word) == word

    def test_e_vowel_without_aw(self) -> None:
        # ေ not followed by AA/TALL_AA — passthrough.
        word = "ေကာ"
        assert normalize_e_vowel_tall_aa(word) == word

    def test_isolated_aa_without_e_prefix(self) -> None:
        word = "ကာ"
        assert normalize_e_vowel_tall_aa(word) == word

    def test_mixed_latin_and_myanmar(self) -> None:
        text = "This says ပေါ်ပေါက် and continues"
        assert normalize_e_vowel_tall_aa(text) == text


class TestBenchmarkRegressions:
    """Exact benchmark gold forms from [[Tone-Zawgyi Slice 2026-04-19]]."""

    @pytest.mark.parametrize(
        "gold",
        [
            "ပေါ်",
            "ပေါင်း",
            "ဒေါ်",
            "ခေါင်း",
            "ခေါ်",
            "သန်းခေါင်စာရင်း",
            "စုစုပေါင်း",
            "ဖြစ်ပေါ်",
            "စိတ်ပေါက်",
            "အပေါ်",
        ],
    )
    def test_gold_forms_preserved(self, gold: str) -> None:
        """Pre-2026-04-19 these gold forms were corrupted by the flat
        rewrite, causing ~28 spelling-FN. The consonant-gated rule must
        preserve them byte-for-byte."""
        assert normalize_e_vowel_tall_aa(gold) == gold

    @pytest.mark.parametrize(
        "typo,gold",
        [
            # Representative rows from b3_zawgyi_vowel.jsonl
            ("သန်းခောင်စာရင်း", "သန်းခေါင်စာရင်း"),
            ("စုစုပောင်း", "စုစုပေါင်း"),
            ("ခော်", "ခေါ်"),
            ("ဖြစ်ပော်", "ဖြစ်ပေါ်"),
            ("စိတ်ပောက်", "စိတ်ပေါက်"),
            ("ပော့", "ပေါ့"),
            ("ခောင်း", "ခေါင်း"),
            ("အပော်", "အပေါ်"),
        ],
    )
    def test_typo_repaired_to_gold(self, typo: str, gold: str) -> None:
        """The B3 typos in the benchmark must be repaired to gold by the
        normalizer — that's what recovers the +28 FN."""
        assert normalize_e_vowel_tall_aa(typo) == gold
