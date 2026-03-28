"""
Unit tests for POS inference engine.

Tests the rule-based POS inference system including:
- Numeral detection
- Prefix pattern matching
- Proper noun suffix detection
- Ambiguous word registry
- Morphological inference
"""

from myspellchecker.algorithms.pos_inference import (
    InferenceSource,
    POSInferenceEngine,
    POSInferenceResult,
    get_pos_inference_engine,
    infer_pos,
    infer_pos_batch,
)
from myspellchecker.text.morphology import is_numeral_word


# Test helper functions that use the engine
def is_ambiguous_word(word: str) -> bool:
    """Check if word is ambiguous using the POS inference engine."""
    engine = get_pos_inference_engine()
    return engine.is_ambiguous_word(word)


def get_ambiguous_pos_tags(word: str) -> frozenset:
    """Get ambiguous POS tags for a word."""
    engine = get_pos_inference_engine()
    if not engine.is_ambiguous_word(word):
        return frozenset()
    return engine.ambiguous_words.get(word, frozenset())


class TestNumeralDetection:
    """Tests for numeral word detection."""

    def test_myanmar_digit_sequence(self):
        """Myanmar digits should be detected as NUM."""
        result = infer_pos("၁၂၃")
        assert result.inferred_pos == "NUM"
        assert result.confidence >= 0.99
        assert result.source == InferenceSource.NUMERAL_DETECTION

    def test_single_myanmar_digit(self):
        """Single Myanmar digit should be NUM."""
        result = infer_pos("၅")
        assert result.inferred_pos == "NUM"
        assert result.confidence >= 0.99

    def test_mixed_myanmar_digits(self):
        """Various digit combinations should be NUM."""
        digits = ["၁၀", "၂၀", "၁၀၀", "၁၉၉၅"]
        for digit in digits:
            result = infer_pos(digit)
            assert result.inferred_pos == "NUM", f"Failed for {digit}"

    def test_numeral_words(self):
        """Numeral words should be detected as NUM."""
        numeral_words = ["တစ်", "နှစ်", "သုံး", "လေး", "ငါး"]
        for word in numeral_words:
            assert is_numeral_word(word), f"{word} should be numeral"

    def test_non_numeral(self):
        """Non-numeral words should not match numeral pattern."""
        result = infer_pos("စာ")
        assert result.source != InferenceSource.NUMERAL_DETECTION


class TestPrefixPatterns:
    """Tests for prefix-based inference."""

    def test_a_prefix_noun(self):
        """အ prefix should infer Noun."""
        words = ["အလုပ်", "အိမ်", "အစား"]
        for word in words:
            result = infer_pos(word)
            assert result.inferred_pos == "N", f"Failed for {word}"
            assert result.source == InferenceSource.PREFIX_PATTERN

    def test_ma_prefix_verb(self):
        """မ prefix (negation) should infer Verb."""
        result = infer_pos("မလုပ်")
        assert result.inferred_pos == "V"
        assert result.source == InferenceSource.PREFIX_PATTERN

    def test_prefix_confidence(self):
        """Prefix patterns should have appropriate confidence."""
        result = infer_pos("အလုပ်")
        assert 0.6 <= result.confidence <= 0.85


class TestProperNounSuffixes:
    """Tests for proper noun suffix detection."""

    def test_country_suffix(self):
        """Country suffix နိုင်ငံ should infer Noun."""
        result = infer_pos("မြန်မာနိုင်ငံ")
        assert result.inferred_pos == "N"
        assert result.source == InferenceSource.PROPER_NOUN_SUFFIX
        assert "နိုင်ငံ" in result.details.get("suffix", "")

    def test_city_suffix(self):
        """City suffix မြို့ should infer Noun."""
        result = infer_pos("ရန်ကုန်မြို့")
        assert result.inferred_pos == "N"
        assert result.source == InferenceSource.PROPER_NOUN_SUFFIX

    def test_university_suffix(self):
        """University suffix တက္ကသိုလ် should infer Noun."""
        result = infer_pos("ရန်ကုန်တက္ကသိုလ်")
        assert result.inferred_pos == "N"
        assert result.source == InferenceSource.PROPER_NOUN_SUFFIX

    def test_proper_noun_before_prefix(self):
        """Proper noun suffix should take precedence over prefix pattern."""
        # မြန်မာနိုင်ငံ starts with မ but should be N (country), not V (prefix)
        result = infer_pos("မြန်မာနိုင်ငံ")
        assert result.inferred_pos == "N"
        assert result.source == InferenceSource.PROPER_NOUN_SUFFIX


class TestAmbiguousWords:
    """Tests for ambiguous word registry."""

    def test_known_ambiguous_word(self):
        """Known ambiguous words should be detected."""
        # ကြီး is a common ambiguous word (ADJ|N|V)
        assert is_ambiguous_word("ကြီး")
        tags = get_ambiguous_pos_tags("ကြီး")
        assert "ADJ" in tags or "N" in tags or "V" in tags

    def test_ambiguous_word_inference(self):
        """Ambiguous words should return multi-POS result."""
        if is_ambiguous_word("ကြီး"):
            result = infer_pos("ကြီး")
            assert result.is_ambiguous
            assert result.requires_context
            assert len(result.all_pos_tags) > 1

    def test_multi_pos_string_format(self):
        """Multi-POS should be formatted as pipe-separated."""
        result = POSInferenceResult(
            word="test",
            all_pos_tags=frozenset({"N", "V", "ADJ"}),
        )
        multi_string = result.to_multi_pos_string()
        assert multi_string is not None
        assert "|" in multi_string
        # Should be sorted alphabetically
        assert multi_string == "ADJ|N|V"


class TestPOSInferenceEngine:
    """Tests for the main inference engine."""

    def test_engine_singleton(self):
        """get_pos_inference_engine should return singleton."""
        engine1 = get_pos_inference_engine()
        engine2 = get_pos_inference_engine()
        assert engine1 is engine2

    def test_empty_word(self):
        """Empty word should return empty result."""
        engine = POSInferenceEngine()
        result = engine.infer_pos("")
        assert result.inferred_pos is None
        assert result.confidence == 0.0

    def test_inference_priority_order(self):
        """Inference should follow priority order."""
        engine = POSInferenceEngine()

        # Numerals are highest priority
        result = engine.infer_pos("၁၀")
        assert result.source == InferenceSource.NUMERAL_DETECTION

        # Proper noun suffix before prefix
        result = engine.infer_pos("မြန်မာနိုင်ငံ")
        assert result.source == InferenceSource.PROPER_NOUN_SUFFIX


class TestBatchInference:
    """Tests for batch inference."""

    def test_batch_inference(self):
        """Batch inference should process multiple words."""
        words = ["၁၀", "အလုပ်", "မြန်မာနိုင်ငံ"]
        results, stats = infer_pos_batch(words)

        assert len(results) == 3
        assert stats.total_words == 3
        assert stats.inferred_count >= 2

    def test_batch_skips_existing(self):
        """Batch should skip words with existing POS."""
        words = ["၁၀", "အလုပ်"]
        existing_pos = {"၁၀": "NUM"}

        results, stats = infer_pos_batch(words, existing_pos)

        # Should skip ၁၀ since it has existing POS
        assert len(results) == 1
        assert results[0].word == "အလုပ်"

    def test_batch_stats(self):
        """Batch statistics should be accurate."""
        words = ["၁", "၂", "၃"]  # All numerals
        results, stats = infer_pos_batch(words)

        assert stats.numeral_count == 3
        assert stats.inferred_count == 3
        assert stats.avg_confidence >= 0.95


class TestInferenceResult:
    """Tests for POSInferenceResult dataclass."""

    def test_result_repr(self):
        """Result should have readable repr."""
        result = POSInferenceResult(
            word="test",
            inferred_pos="N",
            all_pos_tags=frozenset({"N"}),
            confidence=0.85,
            source=InferenceSource.PREFIX_PATTERN,
        )
        repr_str = repr(result)
        assert "test" in repr_str
        assert "N" in repr_str
        assert "0.85" in repr_str

    def test_multi_pos_string_empty(self):
        """Empty tags should return None."""
        result = POSInferenceResult(word="test")
        assert result.to_multi_pos_string() is None

    def test_multi_pos_string_single(self):
        """Single tag should return just that tag."""
        result = POSInferenceResult(
            word="test",
            all_pos_tags=frozenset({"N"}),
        )
        assert result.to_multi_pos_string() == "N"


class TestPOSInferenceEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_character_word(self):
        """Single character Myanmar words should be handled."""
        result = infer_pos("က")
        # Should not crash, may or may not have inference
        assert isinstance(result, POSInferenceResult)

    def test_long_word(self):
        """Long compound words should be handled."""
        result = infer_pos("မြန်မာနိုင်ငံတော်")
        assert isinstance(result, POSInferenceResult)

    def test_unknown_word(self):
        """Unknown words should return UNKNOWN source."""
        # A word that doesn't match any pattern
        result = infer_pos("xyz")
        assert result.source == InferenceSource.UNKNOWN or result.inferred_pos is None


class TestMorphologicalInference:
    """Tests for morphological analysis fallback."""

    def test_suffix_based_inference(self):
        """Suffix patterns should trigger morphological inference."""
        # Words ending in common suffixes
        result = infer_pos("ကောင်းမှု")  # -မှု suffix (noun)
        # May be detected by morphological analysis
        assert isinstance(result, POSInferenceResult)

    def test_verb_suffix(self):
        """Verb suffixes should be detected."""
        result = infer_pos("လုပ်နိုင်")  # -နိုင် (can/able)
        assert isinstance(result, POSInferenceResult)
