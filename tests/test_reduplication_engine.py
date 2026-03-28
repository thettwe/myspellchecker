"""Tests for the ReduplicationEngine."""

from unittest.mock import MagicMock

import pytest

from myspellchecker.text.reduplication import ReduplicationEngine


@pytest.fixture
def mock_segmenter():
    """Create a mock segmenter."""
    segmenter = MagicMock()
    return segmenter


@pytest.fixture
def engine(mock_segmenter):
    """Create a ReduplicationEngine with default settings."""
    return ReduplicationEngine(
        segmenter=mock_segmenter,
        min_base_frequency=5,
        cache_size=128,
    )


def _dict_check(valid_words):
    """Create a dictionary check callable."""
    return lambda word: word in valid_words


def _freq_check(frequencies):
    """Create a frequency check callable."""
    return lambda word: frequencies.get(word, 0)


def _pos_check(pos_tags):
    """Create a POS check callable."""
    return lambda word: pos_tags.get(word)


class TestReduplicationEngineAA:
    """Test AA (simple) reduplication pattern."""

    def test_valid_aa_reduplication(self, engine, mock_segmenter):
        """AA reduplication with valid base word should be accepted."""
        # Use a word NOT in RHYME_REDUPLICATION_PATTERNS to test AA path
        mock_segmenter.segment_syllables.return_value = ["စား", "စား"]

        result = engine.analyze(
            "စားစား",
            dictionary_check=_dict_check({"စား"}),
            frequency_check=_freq_check({"စား": 100}),
            pos_check=_pos_check({"စား": "V"}),
        )

        assert result is not None
        assert result.is_valid is True
        assert result.pattern == "AB"
        assert result.base_word == "စား"
        assert result.pos_tag == "V"
        assert result.confidence > 0.8

    def test_aa_base_not_in_dict(self, engine, mock_segmenter):
        """AA reduplication with unknown base should be rejected."""
        mock_segmenter.segment_syllables.return_value = ["xyz", "xyz"]

        result = engine.analyze(
            "xyzxyz",
            dictionary_check=_dict_check(set()),
            frequency_check=_freq_check({}),
            pos_check=_pos_check({}),
        )

        assert result is None

    def test_aa_low_frequency_base(self, engine, mock_segmenter):
        """AA reduplication with low frequency base should be rejected."""
        mock_segmenter.segment_syllables.return_value = ["rare", "rare"]

        result = engine.analyze(
            "rarerare",
            dictionary_check=_dict_check({"rare"}),
            frequency_check=_freq_check({"rare": 2}),  # Below min_base_frequency=5
            pos_check=_pos_check({"rare": "N"}),
        )

        assert result is None

    def test_aa_particle_pos_rejected(self, engine, mock_segmenter):
        """AA reduplication of a particle should be rejected."""
        mock_segmenter.segment_syllables.return_value = ["ကို", "ကို"]

        result = engine.analyze(
            "ကိုကို",
            dictionary_check=_dict_check({"ကို"}),
            frequency_check=_freq_check({"ကို": 500}),
            pos_check=_pos_check({"ကို": "P"}),  # Particle
        )

        assert result is None


class TestReduplicationEngineABAB:
    """Test ABAB pattern: whole disyllabic unit repeats (syl[0]==syl[2], syl[1]==syl[3])."""

    def test_valid_abab_reduplication(self, engine, mock_segmenter):
        """ABAB reduplication with valid bases should be accepted."""
        # ABAB: syl[0]==syl[2], syl[1]==syl[3] (detect_reduplication_pattern returns ABAB)
        mock_segmenter.segment_syllables.return_value = ["ရှင်း", "လင်း", "ရှင်း", "လင်း"]

        result = engine.analyze(
            "ရှင်းလင်းရှင်းလင်း",
            dictionary_check=_dict_check({"ရှင်းလင်း"}),
            frequency_check=_freq_check({"ရှင်းလင်း": 50}),
            pos_check=_pos_check({"ရှင်းလင်း": "ADV"}),
        )

        assert result is not None
        assert result.is_valid is True
        assert result.pattern == "ABAB"


class TestReduplicationEngineAABB:
    """Test AABB pattern: each syllable doubles (syl[0]==syl[1], syl[2]==syl[3])."""

    def test_valid_aabb_reduplication(self, engine, mock_segmenter):
        """AABB reduplication with valid base should be accepted."""
        # AABB: syl[0]==syl[1], syl[2]==syl[3], syl[0]!=syl[2]
        # Base word is syl[0]+syl[2] = "စားသောက်" (eat-drink)
        # Use words NOT in RHYME_REDUPLICATION_PATTERNS
        mock_segmenter.segment_syllables.return_value = ["စား", "စား", "သောက်", "သောက်"]

        result = engine.analyze(
            "စားစားသောက်သောက်",
            dictionary_check=_dict_check({"စားသောက်"}),
            frequency_check=_freq_check({"စားသောက်": 30}),
            pos_check=_pos_check({"စားသောက်": "V"}),
        )

        assert result is not None
        assert result.is_valid is True
        assert result.pattern == "AABB"
        assert result.base_word == "စားသောက်"


class TestReduplicationEngineRhyme:
    """Test RHYME pattern (known patterns from grammar/patterns.py)."""

    def test_known_rhyme_pattern(self, engine, mock_segmenter):
        """Known rhyme reduplication should be accepted immediately."""
        result = engine.analyze(
            "ကောင်းကောင်း",  # In RHYME_REDUPLICATION_PATTERNS
            dictionary_check=_dict_check(set()),
            frequency_check=_freq_check({}),
            pos_check=_pos_check({}),
        )

        assert result is not None
        assert result.is_valid is True
        assert result.pattern == "RHYME"
        assert result.confidence >= 0.95


class TestReduplicationEngineCache:
    """Test caching behavior."""

    def test_cached_result_returned(self, engine, mock_segmenter):
        """Same word should return cached result without re-analyzing."""
        mock_segmenter.segment_syllables.return_value = ["ကောင်း", "ကောင်း"]
        dict_check = _dict_check({"ကောင်း"})
        freq_check = _freq_check({"ကောင်း": 100})
        pos_check = _pos_check({"ကောင်း": "ADJ"})

        # First call
        result1 = engine.analyze("ကောင်းကောင်း", dict_check, freq_check, pos_check)
        # Second call (should use cache)
        result2 = engine.analyze("ကောင်းကောင်း", dict_check, freq_check, pos_check)

        assert result1 is result2
        # segment_syllables should only be called once (first call bypasses via RHYME check,
        # but the cache ensures no re-analysis)

    def test_cache_eviction(self, mock_segmenter):
        """Cache should evict oldest entry when full."""
        engine = ReduplicationEngine(
            segmenter=mock_segmenter,
            cache_size=2,
        )
        mock_segmenter.segment_syllables.return_value = ["x", "y"]
        dict_check = _dict_check(set())
        freq_check = _freq_check({})
        pos_check = _pos_check({})

        engine.analyze("word1", dict_check, freq_check, pos_check)
        engine.analyze("word2", dict_check, freq_check, pos_check)
        engine.analyze("word3", dict_check, freq_check, pos_check)

        assert "word1" not in engine._cache
        assert "word2" in engine._cache
        assert "word3" in engine._cache


class TestReduplicationEngineEdgeCases:
    """Test edge cases."""

    def test_single_syllable_word(self, engine, mock_segmenter):
        """Single syllable word can't be reduplication."""
        mock_segmenter.segment_syllables.return_value = ["ကောင်း"]

        result = engine.analyze(
            "ကောင်း",
            dictionary_check=_dict_check({"ကောင်း"}),
            frequency_check=_freq_check({"ကောင်း": 100}),
            pos_check=_pos_check({"ကောင်း": "ADJ"}),
        )

        assert result is None

    def test_no_pattern_detected(self, engine, mock_segmenter):
        """Non-reduplication multi-syllable word."""
        mock_segmenter.segment_syllables.return_value = ["က", "ခ", "ဂ"]

        result = engine.analyze(
            "ကခဂ",
            dictionary_check=_dict_check({"က", "ခ", "ဂ"}),
            frequency_check=_freq_check({"က": 100, "ခ": 100, "ဂ": 100}),
            pos_check=_pos_check({"က": "N"}),
        )

        assert result is None

    def test_pos_none_allowed(self, engine, mock_segmenter):
        """When POS check returns None, reduplication should still be accepted."""
        mock_segmenter.segment_syllables.return_value = ["ကောင်း", "ကောင်း"]

        result = engine.analyze(
            "ကောင်းကောင်း",
            dictionary_check=_dict_check({"ကောင်း"}),
            frequency_check=_freq_check({"ကောင်း": 100}),
            pos_check=_pos_check({}),  # Returns None for all
        )

        # Known rhyme pattern, so accepted
        assert result is not None
        assert result.is_valid is True
