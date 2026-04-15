"""Tests for error_suppression.py — FP suppression heuristics."""

from unittest.mock import MagicMock

from myspellchecker.core.error_suppression import (
    _NER_IMMUNE,
    ErrorSuppressionMixin,
)
from myspellchecker.core.response import Error, SyllableError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_error(
    text="ကို",
    position=0,
    suggestions=None,
    error_type="invalid_syllable",
    confidence=0.9,
):
    return Error(
        text=text,
        position=position,
        suggestions=suggestions or [],
        error_type=error_type,
        confidence=confidence,
    )


def _make_syllable_error(text="မျန်", position=0, suggestions=None):
    return SyllableError(
        text=text,
        position=position,
        suggestions=suggestions or ["မြန်"],
    )


def _make_mixin(**overrides):
    """Create a minimal ErrorSuppressionMixin instance with mocked dependencies."""
    mixin = object.__new__(ErrorSuppressionMixin)
    provider = MagicMock()
    provider.is_valid_word.return_value = True
    provider.get_word_frequency.return_value = 5000
    provider.is_valid_words_bulk.return_value = {}
    mixin.provider = overrides.get("provider", provider)
    mixin.config = overrides.get("config", MagicMock())
    mixin.segmenter = overrides.get("segmenter", MagicMock())
    mixin.logger = MagicMock()
    mixin._ner_model = overrides.get("_ner_model", None)
    mixin._KEEP_ATTACHED_SUFFIXES = overrides.get("_KEEP_ATTACHED_SUFFIXES", ())
    mixin._MISSING_ASAT_PARTICLES = overrides.get("_MISSING_ASAT_PARTICLES", {})
    mixin._MISSING_VISARGA_SUFFIXES = overrides.get("_MISSING_VISARGA_SUFFIXES", {})
    return mixin


# ---------------------------------------------------------------------------
# Cascade syllable suppression
# ---------------------------------------------------------------------------


class TestSuppressCascadeSyllableErrors:
    def test_removes_secondary_syllable_errors_without_space_gap(self):
        e1 = _make_syllable_error(text="က္ေ", position=0)
        e2 = _make_syllable_error(text="ယျ", position=3)
        errors = [e1, e2]
        ErrorSuppressionMixin._suppress_cascade_syllable_errors(errors, "က္ေယျ")
        assert len(errors) == 1
        assert errors[0] is e1

    def test_keeps_both_errors_when_space_between_them(self):
        e1 = _make_syllable_error(text="ကို", position=0)
        e2 = _make_syllable_error(text="မြန်", position=7)
        errors = [e1, e2]
        ErrorSuppressionMixin._suppress_cascade_syllable_errors(errors, "ကို   မြန်")
        assert len(errors) == 2

    def test_no_crash_on_single_error(self):
        errors = [_make_syllable_error()]
        ErrorSuppressionMixin._suppress_cascade_syllable_errors(errors)
        assert len(errors) == 1

    def test_no_crash_on_empty_list(self):
        errors = []
        ErrorSuppressionMixin._suppress_cascade_syllable_errors(errors)
        assert errors == []


# ---------------------------------------------------------------------------
# Pali stacking suppression
# ---------------------------------------------------------------------------


class TestSuppressPaliStackingErrors:
    def test_suppresses_fragment_inside_valid_stacking_word(self):
        # ပစ္စည်း contains valid virama stacking (စ္စ)
        text = "ပစ္စည်း"
        e = _make_syllable_error(text="ပ", position=0)
        errors = [e]
        ErrorSuppressionMixin._suppress_pali_stacking_errors(errors, text)
        assert len(errors) == 0

    def test_keeps_errors_outside_stacking_words(self):
        text = "မြန်မာ"
        e = _make_syllable_error(text="မြ", position=0)
        errors = [e]
        ErrorSuppressionMixin._suppress_pali_stacking_errors(errors, text)
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# Bare consonant suppression near text-level errors
# ---------------------------------------------------------------------------


class TestSuppressBareConsonantNearTextErrors:
    def test_suppresses_bare_consonant_adjacent_to_broken_stacking(self):
        text_err = _make_error(text="ဏ္ာ", position=1, error_type="broken_stacking")
        bare = _make_syllable_error(text="ဘ", position=0)
        errors = [text_err, bare]
        ErrorSuppressionMixin._suppress_bare_consonant_near_text_errors(errors)
        assert len(errors) == 1
        assert errors[0] is text_err

    def test_keeps_bare_consonant_far_from_text_errors(self):
        text_err = _make_error(text="ဏ္ာ", position=20, error_type="broken_stacking")
        bare = _make_syllable_error(text="ဘ", position=0)
        errors = [text_err, bare]
        ErrorSuppressionMixin._suppress_bare_consonant_near_text_errors(errors)
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# Dedup by position
# ---------------------------------------------------------------------------


class TestDedupByPosition:
    def test_text_detector_beats_context_probability_at_same_position(self):
        mixin = _make_mixin()
        detector_err = _make_error(
            text="မှာ",
            position=5,
            suggestions=["မာ"],
            error_type="ha_htoe_confusion",
        )
        ctx_err = _make_error(
            text="မှာ",
            position=5,
            suggestions=["မာ"],
            error_type="context_probability",
        )
        errors = [ctx_err, detector_err]
        mixin._dedup_errors_by_position(errors)
        assert len(errors) == 1
        assert errors[0].error_type == "ha_htoe_confusion"

    def test_higher_confidence_wins_at_same_position_same_length(self):
        mixin = _make_mixin()
        low = _make_error(text="ကို", position=0, confidence=0.6, error_type="context_probability")
        high = _make_error(text="ကို", position=0, confidence=0.9, error_type="context_probability")
        errors = [low, high]
        mixin._dedup_errors_by_position(errors)
        assert len(errors) == 1
        assert errors[0].confidence == 0.9

    def test_suggestions_carried_over_from_displaced_error(self):
        mixin = _make_mixin()
        short = _make_error(
            text="က",
            position=0,
            suggestions=["ခ"],
            error_type="context_probability",
        )
        long = _make_error(
            text="ကို",
            position=0,
            suggestions=["ခို"],
            error_type="context_probability",
        )
        errors = [short, long]
        mixin._dedup_errors_by_position(errors)
        assert len(errors) == 1
        assert "ခ" in errors[0].suggestions


# ---------------------------------------------------------------------------
# Dedup by span
# ---------------------------------------------------------------------------


class TestDedupBySpan:
    def test_removes_error_fully_contained_in_wider_span(self):
        mixin = _make_mixin()
        wide = _make_error(text="ကိုယ်တိုင်", position=0, error_type="invalid_word")
        narrow = _make_error(text="ကို", position=0, error_type="invalid_syllable")
        errors = [wide, narrow]
        mixin._dedup_errors_by_span(errors)
        assert len(errors) == 1
        assert errors[0] is wide

    def test_narrow_root_cause_replaces_wider_generic(self):
        mixin = _make_mixin()
        wide = _make_error(
            text="ကြောင်း",
            position=0,
            error_type="pos_sequence_error",
        )
        narrow = _make_error(
            text="ကြော",
            position=0,
            suggestions=["ကျော"],
            error_type="medial_confusion",
        )
        errors = [wide, narrow]
        mixin._dedup_errors_by_span(errors)
        assert len(errors) == 1
        assert errors[0].error_type == "medial_confusion"


# ---------------------------------------------------------------------------
# NER immunity constants
# ---------------------------------------------------------------------------


class TestNerImmunity:
    def test_ha_htoe_confusion_is_ner_immune(self):
        assert "ha_htoe_confusion" in _NER_IMMUNE

    def test_confusable_error_is_ner_immune(self):
        assert "confusable_error" in _NER_IMMUNE

    def test_context_probability_is_not_ner_immune(self):
        assert "context_probability" not in _NER_IMMUNE


# ---------------------------------------------------------------------------
# Low-value context probability suppression
# ---------------------------------------------------------------------------


class TestSuppressLowValueContextProbability:
    def test_suppresses_high_freq_valid_word_with_no_suggestions(self):
        mixin = _make_mixin()
        e = _make_error(
            text="သည်",
            position=0,
            suggestions=[],
            error_type="context_probability",
        )
        errors = [e]
        mixin._suppress_low_value_context_probability(errors)
        assert len(errors) == 0

    def test_keeps_low_freq_word_with_no_suggestions(self):
        provider = MagicMock()
        provider.is_valid_word.return_value = True
        provider.get_word_frequency.return_value = 50
        mixin = _make_mixin(provider=provider)
        e = _make_error(
            text="သည်",
            position=0,
            suggestions=[],
            error_type="context_probability",
        )
        errors = [e]
        mixin._suppress_low_value_context_probability(errors)
        assert len(errors) == 1

    def test_no_crash_without_provider(self):
        mixin = _make_mixin(provider=None)
        errors = [_make_error(error_type="context_probability")]
        mixin._suppress_low_value_context_probability(errors)
        assert len(errors) == 1


class TestMLMWordSuppression:
    """Tests for _suppress_invalid_word_via_mlm morpheme boundary filtering."""

    def test_does_not_suppress_when_prefix_is_morpheme_boundary(self):
        """MLM predicting 'အတိုင်းအတာ' should NOT suppress 'အတိုင်' (missing visarga)."""
        semantic = MagicMock()
        # MLM predicts a compound starting with the error word + visarga
        semantic.predict_mask.return_value = [("အတိုင်းအတာ", 9.5)]

        mixin = _make_mixin()
        mixin._semantic_checker = semantic
        mixin.config.validation.mlm_plausibility_threshold = 5.0
        mixin.config.validation.mlm_plausibility_top_k = 10

        errors = [_make_error(text="အတိုင်", error_type="invalid_word")]
        mixin._suppress_invalid_word_via_mlm(errors, "test text")
        assert len(errors) == 1, "Should keep error — visarga boundary means missing-visarga"

    def test_does_not_suppress_dot_below_boundary(self):
        """MLM predicting 'ခွင့်ပြု' should NOT suppress 'ခွင်' (missing dot-below)."""
        semantic = MagicMock()
        semantic.predict_mask.return_value = [("ခွင့်ပြု", 9.0)]

        mixin = _make_mixin()
        mixin._semantic_checker = semantic
        mixin.config.validation.mlm_plausibility_threshold = 5.0
        mixin.config.validation.mlm_plausibility_top_k = 10

        errors = [_make_error(text="ခွင်", error_type="invalid_word")]
        mixin._suppress_invalid_word_via_mlm(errors, "test text")
        assert len(errors) == 1, "Should keep error — dot-below boundary"

    def test_suppresses_legitimate_compound_prefix(self):
        """MLM predicting 'ကျွန်တော်' should suppress 'ကျွန်' (valid prefix)."""
        semantic = MagicMock()
        # Consonant suffix — legitimate compound
        semantic.predict_mask.return_value = [("ကျွန်တော်", 9.0)]

        mixin = _make_mixin()
        mixin._semantic_checker = semantic
        mixin.config.validation.mlm_plausibility_threshold = 5.0
        mixin.config.validation.mlm_plausibility_top_k = 10

        errors = [_make_error(text="ကျွန်", error_type="invalid_word")]
        mixin._suppress_invalid_word_via_mlm(errors, "test text")
        assert len(errors) == 0, "Should suppress — consonant suffix is legitimate prefix"

    def test_suppresses_exact_word_match(self):
        """MLM predicting the exact word should always suppress."""
        semantic = MagicMock()
        semantic.predict_mask.return_value = [("ကျောင်း", 9.0)]

        mixin = _make_mixin()
        mixin._semantic_checker = semantic
        mixin.config.validation.mlm_plausibility_threshold = 5.0
        mixin.config.validation.mlm_plausibility_top_k = 10

        errors = [_make_error(text="ကျောင်း", error_type="invalid_word")]
        mixin._suppress_invalid_word_via_mlm(errors, "test text")
        assert len(errors) == 0, "Exact match should always suppress"
