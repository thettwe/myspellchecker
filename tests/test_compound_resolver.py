"""Tests for the CompoundResolver."""

from unittest.mock import MagicMock

import pytest

from myspellchecker.text.compound_resolver import CompoundResolver


@pytest.fixture
def mock_segmenter():
    """Create a mock segmenter."""
    segmenter = MagicMock()
    return segmenter


@pytest.fixture
def resolver(mock_segmenter):
    """Create a CompoundResolver with default settings."""
    return CompoundResolver(
        segmenter=mock_segmenter,
        min_morpheme_frequency=10,
        max_parts=4,
        cache_size=128,
    )


def _dict_check(valid_words):
    return lambda word: word in valid_words


def _freq_check(frequencies):
    return lambda word: frequencies.get(word, 0)


def _pos_check(pos_tags):
    return lambda word: pos_tags.get(word)


class TestCompoundResolverNN:
    """Test N+N compound resolution."""

    def test_valid_nn_compound(self, resolver, mock_segmenter):
        """N+N compound with valid parts should be accepted."""
        # ကျောင်းသား = ကျောင်း (school) + သား (son/child) = student
        mock_segmenter.segment_syllables.return_value = ["ကျောင်း", "သား"]

        result = resolver.resolve(
            "ကျောင်းသား",
            dictionary_check=_dict_check({"ကျောင်း", "သား"}),
            frequency_check=_freq_check({"ကျောင်း": 200, "သား": 300}),
            pos_check=_pos_check({"ကျောင်း": "N", "သား": "N"}),
        )

        assert result is not None
        assert result.is_valid is True
        assert result.pattern == "N+N"
        assert result.parts == ["ကျောင်း", "သား"]
        assert result.confidence > 0.8

    def test_nn_compound_multisyllable_parts(self, resolver, mock_segmenter):
        """N+N compound where parts span multiple syllables."""
        # 3 syllables: first two form left part, last one is right part
        mock_segmenter.segment_syllables.return_value = ["တက္ကသို", "လ်", "ကျောင်း"]

        result = resolver.resolve(
            "တက္ကသိုလ်ကျောင်း",
            dictionary_check=_dict_check({"တက္ကသိုလ်", "ကျောင်း"}),
            frequency_check=_freq_check({"တက္ကသိုလ်": 50, "ကျောင်း": 200}),
            pos_check=_pos_check({"တက္ကသိုလ်": "N", "ကျောင်း": "N"}),
        )

        assert result is not None
        assert result.is_valid is True
        assert result.pattern == "N+N"


class TestCompoundResolverRejections:
    """Test rejection scenarios."""

    def test_vn_accepted_in_phase3(self, resolver, mock_segmenter):
        """V+N pattern should be accepted in Phase 3 (full POS patterns)."""
        mock_segmenter.segment_syllables.return_value = ["စား", "ခန်း"]

        result = resolver.resolve(
            "စားခန်း",
            dictionary_check=_dict_check({"စား", "ခန်း"}),
            frequency_check=_freq_check({"စား": 500, "ခန်း": 100}),
            pos_check=_pos_check({"စား": "V", "ခန်း": "N"}),
        )

        assert result is not None
        assert result.is_valid is True
        assert result.pattern == "V+N"

    def test_unknown_pattern_rejected(self, resolver, mock_segmenter):
        """Unknown POS pattern (e.g., ADV+P) should be rejected."""
        mock_segmenter.segment_syllables.return_value = ["ကောင်း", "ကို"]

        result = resolver.resolve(
            "ကောင်းကို",
            dictionary_check=_dict_check({"ကောင်း", "ကို"}),
            frequency_check=_freq_check({"ကောင်း": 500, "ကို": 1000}),
            pos_check=_pos_check({"ကောင်း": "ADV", "ကို": "P"}),
        )

        assert result is None

    def test_pp_rejected(self, resolver, mock_segmenter):
        """P+P pattern rejected (not in ALL_ALLOWED_PATTERNS)."""
        mock_segmenter.segment_syllables.return_value = ["ကို", "မှာ"]

        result = resolver.resolve(
            "ကိုမှာ",
            dictionary_check=_dict_check({"ကို", "မှာ"}),
            frequency_check=_freq_check({"ကို": 1000, "မှာ": 800}),
            pos_check=_pos_check({"ကို": "P", "မှာ": "P"}),
        )

        assert result is None

    def test_low_frequency_morpheme(self, resolver, mock_segmenter):
        """Morpheme below frequency floor should be rejected."""
        mock_segmenter.segment_syllables.return_value = ["ကျောင်း", "သား"]

        result = resolver.resolve(
            "ကျောင်းသား",
            dictionary_check=_dict_check({"ကျောင်း", "သား"}),
            frequency_check=_freq_check({"ကျောင်း": 200, "သား": 3}),  # Below floor
            pos_check=_pos_check({"ကျောင်း": "N", "သား": "N"}),
        )

        assert result is None

    def test_part_not_in_dict(self, resolver, mock_segmenter):
        """Parts not in dictionary should be rejected."""
        mock_segmenter.segment_syllables.return_value = ["abc", "def"]

        result = resolver.resolve(
            "abcdef",
            dictionary_check=_dict_check(set()),
            frequency_check=_freq_check({}),
            pos_check=_pos_check({}),
        )

        assert result is None

    def test_single_syllable(self, resolver, mock_segmenter):
        """Single syllable word can't be compound."""
        mock_segmenter.segment_syllables.return_value = ["ကောင်း"]

        result = resolver.resolve(
            "ကောင်း",
            dictionary_check=_dict_check({"ကောင်း"}),
            frequency_check=_freq_check({"ကောင်း": 100}),
            pos_check=_pos_check({"ကောင်း": "ADJ"}),
        )

        assert result is None


class TestCompoundResolverMultipart:
    """Test 3+ part compound splits."""

    def test_three_part_nn(self, resolver, mock_segmenter):
        """3-part N+N+N compound should be accepted."""
        mock_segmenter.segment_syllables.return_value = ["မြို့", "ပြ", "ခန်း"]

        result = resolver.resolve(
            "မြို့ပြခန်း",
            dictionary_check=_dict_check({"မြို့", "ပြ", "ခန်း"}),
            frequency_check=_freq_check({"မြို့": 100, "ပြ": 50, "ခန်း": 80}),
            pos_check=_pos_check({"မြို့": "N", "ပြ": "N", "ခန်း": "N"}),
        )

        assert result is not None
        assert result.is_valid is True
        assert len(result.parts) == 3


class TestCompoundResolverCache:
    """Test caching behavior."""

    def test_cached_result(self, resolver, mock_segmenter):
        """Same word should return cached result."""
        mock_segmenter.segment_syllables.return_value = ["ကျောင်း", "သား"]
        dict_check = _dict_check({"ကျောင်း", "သား"})
        freq_check = _freq_check({"ကျောင်း": 200, "သား": 300})
        pos_check = _pos_check({"ကျောင်း": "N", "သား": "N"})

        result1 = resolver.resolve("ကျောင်းသား", dict_check, freq_check, pos_check)
        result2 = resolver.resolve("ကျောင်းသား", dict_check, freq_check, pos_check)

        assert result1 is result2

    def test_cache_eviction(self, mock_segmenter):
        """Cache should evict when full."""
        resolver = CompoundResolver(
            segmenter=mock_segmenter,
            cache_size=2,
        )
        mock_segmenter.segment_syllables.return_value = ["x"]
        dict_check = _dict_check(set())
        freq_check = _freq_check({})
        pos_check = _pos_check({})

        resolver.resolve("w1", dict_check, freq_check, pos_check)
        resolver.resolve("w2", dict_check, freq_check, pos_check)
        resolver.resolve("w3", dict_check, freq_check, pos_check)

        assert "w1" not in resolver._cache
        assert "w2" in resolver._cache
        assert "w3" in resolver._cache


class TestCompoundResolverPhase3Patterns:
    """Test Phase 3 POS patterns (V+V, N+V, ADJ+N)."""

    def test_vv_compound(self, resolver, mock_segmenter):
        """V+V compound should be accepted."""
        mock_segmenter.segment_syllables.return_value = ["စား", "သောက်"]

        result = resolver.resolve(
            "စားသောက်",
            dictionary_check=_dict_check({"စား", "သောက်"}),
            frequency_check=_freq_check({"စား": 500, "သောက်": 300}),
            pos_check=_pos_check({"စား": "V", "သောက်": "V"}),
        )

        assert result is not None
        assert result.pattern == "V+V"
        assert result.is_valid is True

    def test_adjn_compound(self, resolver, mock_segmenter):
        """ADJ+N compound should be accepted."""
        mock_segmenter.segment_syllables.return_value = ["ကြီး", "မြို့"]

        result = resolver.resolve(
            "ကြီးမြို့",
            dictionary_check=_dict_check({"ကြီး", "မြို့"}),
            frequency_check=_freq_check({"ကြီး": 400, "မြို့": 200}),
            pos_check=_pos_check({"ကြီး": "ADJ", "မြို့": "N"}),
        )

        assert result is not None
        assert result.pattern == "ADJ+N"
        assert result.is_valid is True

    def test_nv_compound(self, resolver, mock_segmenter):
        """N+V compound should be accepted."""
        mock_segmenter.segment_syllables.return_value = ["ရေ", "ချိုး"]

        result = resolver.resolve(
            "ရေချိုး",
            dictionary_check=_dict_check({"ရေ", "ချိုး"}),
            frequency_check=_freq_check({"ရေ": 600, "ချိုး": 200}),
            pos_check=_pos_check({"ရေ": "N", "ချိုး": "V"}),
        )

        assert result is not None
        assert result.pattern == "N+V"
        assert result.is_valid is True


class TestCompoundResolverDP:
    """Test DP segmentation optimality."""

    def test_dp_finds_optimal_split(self, resolver, mock_segmenter):
        """DP should find the split with highest combined score."""
        # 4 syllables: could split as 1+3, 2+2, 3+1, or 1+1+2, etc.
        # DP should pick the split with best score
        mock_segmenter.segment_syllables.return_value = ["a", "b", "c", "d"]

        # Only "ab"+"cd" is a valid N+N split
        result = resolver.resolve(
            "abcd",
            dictionary_check=_dict_check({"ab", "cd"}),
            frequency_check=_freq_check({"ab": 100, "cd": 100}),
            pos_check=_pos_check({"ab": "N", "cd": "N"}),
        )

        assert result is not None
        assert result.parts == ["ab", "cd"]

    def test_dp_prefers_fewer_parts(self, resolver, mock_segmenter):
        """DP should prefer fewer parts when scores are similar."""
        mock_segmenter.segment_syllables.return_value = ["a", "b", "c"]

        # Both "a"+"bc" and "a"+"b"+"c" are valid, but 2-part should win
        result = resolver.resolve(
            "abc",
            dictionary_check=_dict_check({"a", "b", "c", "bc", "abc"}),
            frequency_check=_freq_check({"a": 50, "b": 50, "c": 50, "bc": 100}),
            pos_check=_pos_check({"a": "N", "b": "N", "c": "N", "bc": "N"}),
        )

        assert result is not None
        assert len(result.parts) == 2
        assert result.parts == ["a", "bc"]


class TestCompoundResolverBestSplit:
    """Test selection of best split."""

    def test_prefers_higher_frequency_split(self, resolver, mock_segmenter):
        """When multiple valid splits exist, prefer higher combined frequency."""
        # 3 syllables: split at 1 or 2
        mock_segmenter.segment_syllables.return_value = ["က", "ခ", "ဂ"]

        result = resolver.resolve(
            "ကခဂ",
            # split at 1: "က" + "ခဂ" → both N, both freq >= 10
            # split at 2: "ကခ" + "ဂ" → both N, both freq >= 10
            dictionary_check=_dict_check({"က", "ခဂ", "ကခ", "ဂ"}),
            frequency_check=_freq_check({"က": 50, "ခဂ": 200, "ကခ": 10, "ဂ": 10}),
            pos_check=_pos_check({"က": "N", "ခဂ": "N", "ကခ": "N", "ဂ": "N"}),
        )

        assert result is not None
        # Should prefer "က" + "ခဂ" (higher combined frequency)
        assert result.parts == ["က", "ခဂ"]
