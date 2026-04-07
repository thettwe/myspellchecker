"""Direct tests for SyllableTokenizer.tokenize().

These tests were added when the regex was replaced; they serve as the
acceptance criteria for the new implementation.
"""

import pytest

from myspellchecker.tokenizers.syllable import SyllableTokenizer


class TestSyllableTokenizerTokenize:
    @pytest.fixture
    def t(self):
        return SyllableTokenizer()

    # --- Basic correctness (must still pass after the change) ---

    def test_empty_string_returns_empty_list(self, t):
        assert t.tokenize("") == []

    def test_docstring_example(self, t):
        assert t.tokenize("မြန်မာနိုင်ငံ") == ["မြန်", "မာ", "နိုင်", "ငံ"]

    def test_myanmar_language(self, t):
        assert t.tokenize("မြန်မာဘာသာ") == ["မြန်", "မာ", "ဘာ", "သာ"]

    def test_complex_medial_syllable(self, t):
        # ကျွန်တော် (I/me, male) — two syllables with wa-swe + ya-yit medials
        assert t.tokenize("ကျွန်တော်") == ["ကျွန်", "တော်"]

    def test_kinzi_kept_intact(self, t):
        # သင်္ကြန် (Thingyan) — kinzi (င်္) must not split the syllable
        assert t.tokenize("သင်္ကြန်") == ["သင်္ကြန်"]

    # --- English / non-Myanmar grouping (the regression being fixed) ---

    def test_english_word_not_split_per_char(self, t):
        result = t.tokenize("Hello မြန်မာ")
        # Individual characters must NOT appear as tokens
        for char in ["H", "e", "l", "o"]:
            assert char not in result, f"char {char!r} should not be a standalone token"

    def test_english_word_grouped_as_single_token(self, t):
        result = t.tokenize("Hello မြန်မာ")
        assert result[0] == "Hello"  # grouped as one token, no trailing space
        assert "မြန်" in result
        assert "မာ" in result

    def test_non_myanmar_run_is_one_token(self, t):
        result = t.tokenize("(COVID-19) ကပ်ရောဂါ")
        non_myanmar = [tok for tok in result if not any("\u1000" <= c <= "\u104f" for c in tok)]
        # All non-Myanmar characters collapse into exactly one token
        assert len(non_myanmar) == 1
        assert "(COVID-19)" in non_myanmar[0]

    # --- Whitespace handling ---

    def test_space_between_myanmar_words_not_a_token(self, t):
        # Spaces between words must be dropped, not emitted as standalone tokens
        result = t.tokenize("မြန်မာ နိုင်ငံ")
        assert result == ["မြန်", "မာ", "နိုင်", "ငံ"]
        assert " " not in result

    # --- Stacking consonants ---

    def test_stacking_consonant_stays_attached(self, t):
        # virama (္) glues the stacked consonant to its base — no break inside the cluster
        assert t.tokenize("သတ္တဝါ") == ["သတ္တ", "ဝါ"]  # creature
        assert t.tokenize("ပစ္စည်း") == ["ပစ္စည်း"]  # item/equipment
        assert t.tokenize("သဒ္ဒါ") == ["သဒ္ဒါ"]  # grammar

    # --- Myanmar numeral grouping ---

    def test_myanmar_numerals_grouped(self, t):
        result = t.tokenize("မြန်မာ၂၀၂၃")
        assert "၂၀၂၃" in result  # four-digit run → one token, not four

    def test_single_myanmar_numeral(self, t):
        result = t.tokenize("အခန်း၁")
        assert "၁" in result
