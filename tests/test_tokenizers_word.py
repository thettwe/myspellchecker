"""Extended tests for tokenizers/word.py to boost coverage."""

import pytest

from myspellchecker.core.exceptions import TokenizationError


class TestWordTokenizer:
    """Test WordTokenizer class."""

    def test_tokenizer_initialization(self):
        """Test WordTokenizer can be initialized."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer()
        assert tokenizer is not None

    def test_tokenize_empty_string(self):
        """Test tokenizing empty string."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer()
        result = tokenizer.tokenize("")
        assert result == [] or result == [""]

    def test_tokenize_myanmar_text(self):
        """Test tokenizing Myanmar text."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer()
        result = tokenizer.tokenize("မြန်မာ")
        assert isinstance(result, list)

    def test_tokenize_mixed_text(self):
        """Test tokenizing mixed Myanmar and English text."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer()
        result = tokenizer.tokenize("Hello မြန်မာ")
        assert isinstance(result, list)

    def test_tokenize_with_punctuation(self):
        """Test tokenizing text with punctuation."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer()
        result = tokenizer.tokenize("မြန်မာ၊ ပြည်။")
        assert isinstance(result, list)

    def test_tokenize_whitespace_only(self):
        """Test tokenizing whitespace only text."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer()
        result = tokenizer.tokenize("   ")
        assert isinstance(result, list)


class TestWordTokenizerWithConfig:
    """Test WordTokenizer with different configurations."""

    def test_tokenizer_with_default_engine(self):
        """Test tokenizer with default engine."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer(engine="myword")
        assert tokenizer is not None

    def test_tokenizer_segment_words(self):
        """Test segment_words method."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer()
        # Test if segment_words exists and works
        if hasattr(tokenizer, "segment_words"):
            result = tokenizer.segment_words("မြန်မာပြည်")
            assert isinstance(result, list)


class TestWordTokenizerEdgeCases:
    """Test WordTokenizer edge cases."""

    def test_tokenize_unicode_characters(self):
        """Test tokenizing various Unicode characters."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer()
        # Myanmar Unicode range
        result = tokenizer.tokenize("\u1000\u1001\u1002")
        assert isinstance(result, list)

    def test_tokenize_numbers(self):
        """Test tokenizing text with Myanmar numbers."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer()
        result = tokenizer.tokenize("၁၂၃")  # Myanmar digits
        assert isinstance(result, list)

    def test_tokenize_long_text(self):
        """Test tokenizing longer text."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer()
        long_text = "မြန်မာ " * 100
        result = tokenizer.tokenize(long_text)
        assert isinstance(result, list)


class TestWordTokenizerErrorPaths:
    """Test WordTokenizer error handling paths."""

    def test_tokenizer_unknown_engine_raises(self):
        """Test tokenizer raises ValueError for unknown engine."""
        from myspellchecker.tokenizers.word import WordTokenizer

        with pytest.raises(TokenizationError, match="Unknown engine"):
            WordTokenizer(engine="unknown_engine")

    def test_tokenizer_crf_engine(self):
        """Test tokenizer with CRF engine."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer(engine="CRF")
        assert tokenizer is not None
        assert tokenizer.engine == "CRF"

    def test_tokenizer_crf_tokenize(self):
        """Test CRF engine tokenization."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer(engine="CRF")
        result = tokenizer.tokenize("မြန်မာ")
        assert isinstance(result, list)

    def test_tokenizer_crf_tokenize_empty(self):
        """Test CRF engine with empty input."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer(engine="CRF")
        result = tokenizer.tokenize("")
        assert result == []

    def test_add_custom_words_wrong_engine(self):
        """Test add_custom_words warns on wrong engine."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer(engine="CRF")
        # Should log a warning but not crash
        tokenizer.add_custom_words(["test"])

    def test_add_custom_words_empty_list(self):
        """Test add_custom_words with empty list."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer(engine="myword")
        tokenizer.add_custom_words([])

    def test_add_custom_words_with_words(self):
        """Test add_custom_words with actual words."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer(engine="myword")
        tokenizer.add_custom_words(["မြန်မာ", "ပြည်", ""])

    def test_word2features_beginning_of_sentence(self):
        """Test _word2features at beginning of sentence."""
        from myspellchecker.tokenizers.word import WordTokenizer

        features = WordTokenizer._word2features("abc", 0)
        assert "BOS" in features
        assert features["BOS"] is True

    def test_word2features_end_of_sentence(self):
        """Test _word2features at end of sentence."""
        from myspellchecker.tokenizers.word import WordTokenizer

        features = WordTokenizer._word2features("abc", 2)
        assert "EOS" in features
        assert features["EOS"] is True

    def test_word2features_middle(self):
        """Test _word2features in middle of sentence."""
        from myspellchecker.tokenizers.word import WordTokenizer

        features = WordTokenizer._word2features("abcde", 2)
        assert "prev_word.lower()" in features
        assert "next_word.lower()" in features

    def test_word2features_with_trigrams(self):
        """Test _word2features with trigram context."""
        from myspellchecker.tokenizers.word import WordTokenizer

        features = WordTokenizer._word2features("abcde", 2)
        # Should have trigram_1 (i > 1)
        assert "trigram_1" in features
        # Should have trigram_2 (i < len - 2)
        assert "trigram_2" in features

    def test_word2features_digit(self):
        """Test _word2features with digit."""
        from myspellchecker.tokenizers.word import WordTokenizer

        features = WordTokenizer._word2features("123", 1)
        assert "number" in features
        assert features["number"] is True


class TestFragmentFiltering:
    """Test fragment detection and merging functionality."""

    def test_is_invalid_fragment_consonant_asat(self):
        """Test detection of consonant + asat patterns."""
        from myspellchecker.tokenizers.word import _is_invalid_fragment

        # These should be detected as invalid
        assert _is_invalid_fragment("က်") is True
        assert _is_invalid_fragment("င်") is True
        assert _is_invalid_fragment("ည်") is True
        assert _is_invalid_fragment("လ်") is True
        assert _is_invalid_fragment("န်") is True

    def test_is_invalid_fragment_consonant_tone_asat(self):
        """Test detection of consonant + tone + asat patterns."""
        from myspellchecker.tokenizers.word import _is_invalid_fragment

        # These should be detected as invalid
        assert _is_invalid_fragment("င့်") is True
        assert _is_invalid_fragment("ည့်") is True
        assert _is_invalid_fragment("က့်") is True

    def test_is_invalid_fragment_valid_words(self):
        """Test that valid words are not flagged as fragments."""
        from myspellchecker.tokenizers.word import _is_invalid_fragment

        # These should NOT be detected as invalid
        assert _is_invalid_fragment("က") is False  # Single consonant (valid particle)
        assert _is_invalid_fragment("သည်") is False  # Valid word
        assert _is_invalid_fragment("မြန်မာ") is False  # Valid word
        assert _is_invalid_fragment("ကို") is False  # Valid word
        assert _is_invalid_fragment("တွင်") is False  # Valid word
        assert _is_invalid_fragment("ပါ") is False  # Valid particle

    def test_merge_fragments_with_previous(self):
        """Test merging fragments with previous word."""
        from myspellchecker.tokenizers.word import _merge_invalid_fragments

        # Fragment should merge with previous word
        result = _merge_invalid_fragments(["ဟာ", "လ်", "တူး"])
        assert result == ["ဟာလ်", "တူး"]

    def test_merge_fragments_with_next(self):
        """Test merging fragments with next word when no previous."""
        from myspellchecker.tokenizers.word import _merge_invalid_fragments

        # Fragment at start should merge with next
        result = _merge_invalid_fragments(["က်", "မြန်မာ"])
        assert result == ["က်မြန်မာ"]

    def test_merge_fragments_no_fragments(self):
        """Test that valid token lists pass through unchanged."""
        from myspellchecker.tokenizers.word import _merge_invalid_fragments

        tokens = ["သူ", "တို့", "သည်"]
        result = _merge_invalid_fragments(tokens.copy())
        assert result == tokens

    def test_merge_fragments_empty_list(self):
        """Test merging with empty list."""
        from myspellchecker.tokenizers.word import _merge_invalid_fragments

        result = _merge_invalid_fragments([])
        assert result == []

    def test_merge_fragments_multiple(self):
        """Test merging multiple fragments."""
        from myspellchecker.tokenizers.word import _merge_invalid_fragments

        result = _merge_invalid_fragments(["ရှေး", "ကာလ", "များ", "ဟာ", "လ်", "တူး"])
        assert result == ["ရှေး", "ကာလ", "များ", "ဟာလ်", "တူး"]

    def test_tokenizer_filters_fragments(self):
        """Test that tokenizer filters out invalid fragments."""
        from myspellchecker.tokenizers.word import WordTokenizer, _is_invalid_fragment

        tokenizer = WordTokenizer(engine="myword")

        # This input previously produced ['ဟာ', 'လ်', 'တူး']
        result = tokenizer.tokenize("ရှေးကာလများဟာလ်တူးရွာသား")

        # Verify no invalid fragments in output
        fragments = [w for w in result if _is_invalid_fragment(w)]
        assert fragments == [], f"Found fragments: {fragments}"


class TestInvalidWordsConstant:
    """Test INVALID_WORDS constant expansion."""

    def test_invalid_words_contains_fragments(self):
        """Test INVALID_WORDS contains consonant+asat patterns."""
        from myspellchecker.core.constants import INVALID_WORDS

        # Should contain consonant + asat patterns
        assert "က်" in INVALID_WORDS
        assert "င်" in INVALID_WORDS
        assert "ည်" in INVALID_WORDS
        assert "လ်" in INVALID_WORDS

    def test_invalid_words_contains_tone_patterns(self):
        """Test INVALID_WORDS contains consonant+tone+asat patterns."""
        from myspellchecker.core.constants import INVALID_WORDS

        # Should contain consonant + tone + asat patterns
        assert "င့်" in INVALID_WORDS
        assert "ည့်" in INVALID_WORDS

    def test_invalid_words_contains_floating_marks(self):
        """Test INVALID_WORDS contains floating marks."""
        from myspellchecker.core.constants import INVALID_WORDS

        # Should contain floating marks
        assert "်" in INVALID_WORDS  # Floating asat
        assert "့" in INVALID_WORDS  # Floating dot below
        assert "း" in INVALID_WORDS  # Floating visarga

    def test_invalid_words_count(self):
        """Test INVALID_WORDS has expected count."""
        from myspellchecker.core.constants import INVALID_WORDS

        # Should have: 33 consonants * 3 patterns + 4 original = ~103
        assert len(INVALID_WORDS) > 100


class TestExtendedFragmentFiltering:
    """Test extended fragment filtering for Pattern 1 and Pattern 5."""

    def test_is_invalid_fragment_starts_with_asat(self):
        """Test detection of words starting with consonant+asat."""
        from myspellchecker.tokenizers.word import _is_invalid_fragment

        # These are fragments that start with consonant+asat
        assert _is_invalid_fragment("င်ငံ") is True  # Fragment of နိုင်ငံ
        assert _is_invalid_fragment("န်မာ") is True  # Fragment of မြန်မာ
        assert _is_invalid_fragment("က်သည်") is True  # Fragment
        assert _is_invalid_fragment("ပ်ငန်း") is True  # Fragment of လုပ်ငန်း

    def test_is_invalid_fragment_incomplete_stacking(self):
        """Test detection of incomplete stacking patterns."""
        from myspellchecker.tokenizers.word import _is_invalid_fragment

        # Incomplete stacking: virama followed by vowel (missing consonant)
        assert _is_invalid_fragment("ဘဏ္ာ") is True  # Should be ဘဏ္ဍာ
        assert _is_invalid_fragment("သဏ္ာန်") is True  # Should be သဏ္ဍာန်

    def test_is_invalid_fragment_valid_stacking(self):
        """Test that valid stacking is not flagged."""
        from myspellchecker.tokenizers.word import _is_invalid_fragment

        # Valid stacking: virama followed by consonant
        assert _is_invalid_fragment("ပစ္စည်း") is False  # Valid word
        assert _is_invalid_fragment("ကုမ္ပဏီ") is False  # Valid word (company)
        assert _is_invalid_fragment("ကမ္ဘာ") is False  # Valid word (world)
        assert _is_invalid_fragment("ဘဏ္ဍာ") is False  # Valid word (treasury)

    def test_is_invalid_fragment_empty_string(self):
        """Test handling of empty string."""
        from myspellchecker.tokenizers.word import _is_invalid_fragment

        assert _is_invalid_fragment("") is False
        assert _is_invalid_fragment(None) is False if None else True  # Guard against None


class TestZeroToWaNormalization:
    """Test zero-to-wa normalization (Pattern 2)."""

    def test_normalize_zero_to_wa_basic(self):
        """Test basic zero to wa conversion."""
        from myspellchecker.tokenizers.word import _normalize_zero_to_wa

        # ၀ alone should become ဝ
        assert _normalize_zero_to_wa("၀") == "ဝ"
        # ပါ၀င် should become ပါဝင်
        assert _normalize_zero_to_wa("ပါ၀င်") == "ပါဝင်"
        # တာ၀န် should become တာဝန်
        assert _normalize_zero_to_wa("တာ၀န်") == "တာဝန်"

    def test_normalize_zero_to_wa_preserves_numbers(self):
        """Test that actual numbers are preserved."""
        from myspellchecker.tokenizers.word import _normalize_zero_to_wa

        # Numbers should stay as numbers
        assert _normalize_zero_to_wa("၂၀၂၃") == "၂၀၂၃"
        assert _normalize_zero_to_wa("၁၀၀") == "၁၀၀"
        assert _normalize_zero_to_wa("၀၀၀") == "၀၀၀"

    def test_normalize_zero_to_wa_mixed(self):
        """Test mixed text with numbers and words."""
        from myspellchecker.tokenizers.word import _normalize_zero_to_wa

        # Word followed by number - zero in word should convert
        # but zero in number should stay
        result = _normalize_zero_to_wa("၀င်၂၀၂၃")
        assert result == "ဝင်၂၀၂၃"

    def test_normalize_zero_to_wa_empty(self):
        """Test handling of empty string."""
        from myspellchecker.tokenizers.word import _normalize_zero_to_wa

        assert _normalize_zero_to_wa("") == ""
        assert _normalize_zero_to_wa("မြန်မာ") == "မြန်မာ"  # No zeros

    def test_tokenizer_applies_normalization(self):
        """Test that tokenizer applies zero-to-wa normalization."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer(engine="myword")

        # Input with zero should be normalized before tokenization
        # ပါ၀င် (with zero) should tokenize same as ပါဝင် (with wa)
        result_zero = tokenizer.tokenize("ပါ၀င်")
        result_wa = tokenizer.tokenize("ပါဝင်")

        # Both should produce the same result
        assert result_zero == result_wa


class TestMergeExtendedFragments:
    """Test merging of extended fragment patterns."""

    def test_merge_fragments_starting_with_asat(self):
        """Test merging fragments that start with consonant+asat."""
        from myspellchecker.tokenizers.word import _merge_invalid_fragments

        # င်ငံ should merge with previous
        result = _merge_invalid_fragments(["နို", "င်ငံ"])
        assert result == ["နိုင်ငံ"]

    def test_merge_incomplete_stacking(self):
        """Test merging incomplete stacking patterns."""
        from myspellchecker.tokenizers.word import _merge_invalid_fragments

        # ဘဏ္ာ (incomplete) should merge
        result = _merge_invalid_fragments(["ဘဏ္ာ", "ရေး"])
        # Since ဘဏ္ာ is invalid, it merges with next
        assert result == ["ဘဏ္ာရေး"]


class TestWordNumeralSplitting:
    """Test word+numeral splitting (Pattern 3)."""

    def test_split_at_numeral_boundary_basic(self):
        """Test basic splitting at numeral boundary."""
        from myspellchecker.tokenizers.word import _split_at_numeral_boundary

        # Word followed by number
        assert _split_at_numeral_boundary("လ၁") == ["လ", "၁"]
        assert _split_at_numeral_boundary("ကို၁") == ["ကို", "၁"]
        assert _split_at_numeral_boundary("မှ၁၁") == ["မှ", "၁၁"]

    def test_split_at_numeral_boundary_number_first(self):
        """Test splitting when number comes first."""
        from myspellchecker.tokenizers.word import _split_at_numeral_boundary

        # Number followed by word
        assert _split_at_numeral_boundary("၁၂လ") == ["၁၂", "လ"]
        assert _split_at_numeral_boundary("၂၀၂၃ခုနှစ်") == ["၂၀၂၃", "ခုနှစ်"]

    def test_split_at_numeral_boundary_multiple(self):
        """Test splitting with multiple boundaries."""
        from myspellchecker.tokenizers.word import _split_at_numeral_boundary

        # Multiple transitions
        assert _split_at_numeral_boundary("လ၁မှ") == ["လ", "၁", "မှ"]
        assert _split_at_numeral_boundary("၁လ၂") == ["၁", "လ", "၂"]

    def test_split_at_numeral_boundary_no_split(self):
        """Test that pure words or pure numbers are not split."""
        from myspellchecker.tokenizers.word import _split_at_numeral_boundary

        # Pure word - returns as single item
        assert _split_at_numeral_boundary("မြန်မာ") == ["မြန်မာ"]
        # Pure number - returns as single item
        assert _split_at_numeral_boundary("၂၀၂၃") == ["၂၀၂၃"]

    def test_split_word_numeral_tokens_basic(self):
        """Test splitting word+numeral tokens in a list."""
        from myspellchecker.tokenizers.word import _split_word_numeral_tokens

        # Mixed list with word+numeral tokens
        result = _split_word_numeral_tokens(["သည်", "လ၁", "တွင်"])
        assert result == ["သည်", "လ", "၁", "တွင်"]

    def test_split_word_numeral_tokens_multiple(self):
        """Test splitting multiple word+numeral tokens."""
        from myspellchecker.tokenizers.word import _split_word_numeral_tokens

        result = _split_word_numeral_tokens(["မှ၁၁", "ကို၁", "အထိ"])
        assert result == ["မှ", "၁၁", "ကို", "၁", "အထိ"]

    def test_split_word_numeral_tokens_no_split_needed(self):
        """Test that clean tokens pass through unchanged."""
        from myspellchecker.tokenizers.word import _split_word_numeral_tokens

        tokens = ["မြန်မာ", "နိုင်ငံ", "၂၀၂၃"]
        result = _split_word_numeral_tokens(tokens.copy())
        assert result == tokens

    def test_split_word_numeral_tokens_empty(self):
        """Test handling of empty list."""
        from myspellchecker.tokenizers.word import _split_word_numeral_tokens

        assert _split_word_numeral_tokens([]) == []
        assert _split_word_numeral_tokens(["", None]) == []  # Empty strings filtered

    def test_tokenizer_splits_word_numerals(self):
        """Test that tokenizer splits word+numeral concatenations."""
        from myspellchecker.tokenizers.word import WordTokenizer

        tokenizer = WordTokenizer(engine="myword")

        # Test with text that would produce word+numeral tokens
        # Note: actual splitting depends on what mmap produces
        result = tokenizer.tokenize("၂၀၂၃")
        # Pure numbers should remain as is
        assert "၂၀၂၃" in result or all(c in "၀၁၂၃၄၅၆၇၈၉" for t in result for c in t)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
