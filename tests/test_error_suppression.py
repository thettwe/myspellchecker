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


# ---------------------------------------------------------------------------
# Compound-split-valid-words suppression + skip-rule confidence gate
# (ssr-implement-01 — seg-skip-rule-refactor workstream)
# ---------------------------------------------------------------------------


class TestSuppressCompoundSplitValidWords:
    """Regression tests for the 4+syllable-all-valid suppression path."""

    def _build_mixin(
        self,
        *,
        max_ed: int = 2,
        min_freq: int = 1000,
        syllables: list[str],
        valid_syllables: set[str] | None = None,
        symspell_top1: tuple[str, float, int] | None = None,
    ):
        """Create a mixin with just enough wiring to exercise the suppressor.

        The greedy reassembly in _suppress_compound_split_valid_words first
        tries the whole word as one valid part, then shorter prefixes. For
        the suppressor's "all_valid + len(parts)>=2" predicate to fire, the
        whole word must NOT be a dict word while the individual syllables
        MUST be. We model that by defaulting ``valid_syllables`` to the
        syllable set and treating anything else as invalid.
        """
        mixin = _make_mixin()
        if valid_syllables is None:
            valid_syllables = set(syllables)
        mixin.provider.is_valid_word = lambda w: w in valid_syllables
        mixin.segmenter.segment_syllables.return_value = syllables

        # Wire the skip-rule gate params.
        mixin.config.validation.skip_rule_gate_max_ed = max_ed
        mixin.config.validation.skip_rule_gate_min_freq = min_freq

        # SymSpell mock.
        if symspell_top1 is None:
            mixin.symspell = MagicMock()
            mixin.symspell.lookup.return_value = []
        else:
            term, ed, freq = symspell_top1
            cand = MagicMock()
            cand.term = term
            cand.edit_distance = ed
            cand.frequency = freq
            mixin.symspell = MagicMock()
            mixin.symspell.lookup.return_value = [cand]
        return mixin

    def test_suppresses_4_syllable_all_valid_without_candidate(self):
        # No SymSpell candidate → skip rule must fire and suppress the error.
        mixin = self._build_mixin(
            syllables=["စွမ်း", "ဆောင်", "ရ", "ည"],
            symspell_top1=None,
        )
        errors = [_make_error(text="စွမ်းဆောင်ရည", error_type="invalid_word")]
        mixin._suppress_compound_split_valid_words(errors)
        assert errors == []

    def test_keeps_error_when_symspell_has_confident_candidate(self):
        # SymSpell top-1 inside the gate → confidence says this is a real typo.
        mixin = self._build_mixin(
            syllables=["စွမ်း", "ဆောင်", "ရ", "ည"],
            symspell_top1=("စွမ်းဆောင်ရည်", 1.0, 48_971),
        )
        err = _make_error(text="စွမ်းဆောင်ရည", error_type="invalid_word")
        errors = [err]
        mixin._suppress_compound_split_valid_words(errors)
        assert errors == [err]

    def test_suppresses_when_candidate_below_freq_gate(self):
        mixin = self._build_mixin(
            syllables=["စွမ်း", "ဆောင်", "ရ", "ည"],
            symspell_top1=("စွမ်းဆောင်ရည်", 1.0, 500),  # below min_freq=1000
        )
        errors = [_make_error(text="စွမ်းဆောင်ရည", error_type="invalid_word")]
        mixin._suppress_compound_split_valid_words(errors)
        assert errors == []

    def test_suppresses_when_candidate_above_ed_gate(self):
        mixin = self._build_mixin(
            syllables=["စွမ်း", "ဆောင်", "ရ", "ည"],
            symspell_top1=("some_far_word", 3.0, 50_000),  # ed=3 > max_ed=2
        )
        errors = [_make_error(text="စွမ်းဆောင်ရည", error_type="invalid_word")]
        mixin._suppress_compound_split_valid_words(errors)
        assert errors == []

    def test_skip_rule_helper_returns_false_when_candidate_matches_input(self):
        # SymSpell echoing the input back is not a useful suggestion.
        mixin = self._build_mixin(
            syllables=["a", "b", "c", "d"],
            symspell_top1=("WORD", 0.0, 9999),
        )
        assert mixin._skip_rule_has_confident_candidate("WORD") is False

    def test_skip_rule_helper_tolerates_missing_symspell(self):
        mixin = self._build_mixin(
            syllables=["a", "b", "c", "d"],
            symspell_top1=None,
        )
        mixin.symspell = None
        assert mixin._skip_rule_has_confident_candidate("anything") is False

    def test_does_not_suppress_short_word_even_without_candidate(self):
        # len(word) < 4 is guarded earlier; should not be touched.
        mixin = self._build_mixin(syllables=["a"], symspell_top1=None)
        err = _make_error(text="abc", error_type="invalid_word")
        errors = [err]
        mixin._suppress_compound_split_valid_words(errors)
        assert errors == [err]


# ---------------------------------------------------------------------------
# Compound-split confusable boost (ccb-implement-01)
# ---------------------------------------------------------------------------


class TestBoostInnerConfusableForCompoundSplits:
    """Regression tests for the combined-signal boost helper.

    The helper runs BEFORE _CONFIDENCE_THRESHOLDS filter and boosts inner
    confusable_error confidence when both (a) compound-split-predicate fires
    on an outer long token AND (b) an inner confusable at a sub-span is
    below the ceiling.
    """

    def _build_mixin(
        self,
        *,
        enabled: bool = True,
        boost: float = 0.20,
        ceiling: float = 0.75,
        min_syllables: int = 4,
        outer_syllables: list[str] | None = None,
        outer_valid_syllables: set[str] | None = None,
    ):
        mixin = _make_mixin()
        mixin.config.validation.compound_split_confusable_boost_enabled = enabled
        mixin.config.validation.compound_split_confusable_boost = boost
        mixin.config.validation.compound_split_confusable_boost_inner_conf_ceiling = ceiling
        mixin.config.validation.compound_split_confusable_boost_min_syllables = min_syllables

        if outer_syllables is not None:
            mixin.segmenter.segment_syllables.return_value = outer_syllables
            if outer_valid_syllables is None:
                outer_valid_syllables = set(outer_syllables)
            mixin.provider.is_valid_word = lambda w: w in outer_valid_syllables
        return mixin

    def test_boost_fires_when_conditions_met(self):
        # Outer ET_WORD at [0, 24) containing inner confusable at [12, 18)
        mixin = self._build_mixin(
            outer_syllables=["စွမ်း", "ဆောင်", "ရ", "ည"],
            outer_valid_syllables={"စွမ်း", "ဆောင်", "ရ", "ည"},
        )
        outer = _make_error(text="အသုံးပျုသူအား", position=0, error_type="invalid_word")
        inner = _make_error(
            text="ပျု",
            position=5,
            suggestions=["ပြု"],
            error_type="confusable_error",
            confidence=0.56,
        )
        errors = [outer, inner]
        mixin._boost_inner_confusable_for_compound_splits(errors)
        # inner conf should have been boosted by 0.20 → 0.76
        assert abs(inner.confidence - 0.76) < 1e-9

    def test_no_boost_when_disabled(self):
        mixin = self._build_mixin(
            enabled=False,
            outer_syllables=["a", "b", "c", "d"],
        )
        outer = _make_error(text="abcdefgh", position=0, error_type="invalid_word")
        inner = _make_error(
            text="bc",
            position=1,
            suggestions=["bd"],
            error_type="confusable_error",
            confidence=0.56,
        )
        errors = [outer, inner]
        mixin._boost_inner_confusable_for_compound_splits(errors)
        assert inner.confidence == 0.56

    def test_no_boost_when_no_outer_compound_split_span(self):
        # No invalid_word error at all → no outer span to trigger boost
        mixin = self._build_mixin(
            outer_syllables=["a", "b", "c", "d"],
        )
        inner = _make_error(
            text="bc",
            position=1,
            suggestions=["bd"],
            error_type="confusable_error",
            confidence=0.56,
        )
        errors = [inner]
        mixin._boost_inner_confusable_for_compound_splits(errors)
        assert inner.confidence == 0.56

    def test_no_boost_when_inner_outside_outer_span(self):
        mixin = self._build_mixin(
            outer_syllables=["a", "b", "c", "d"],
        )
        outer = _make_error(text="abcdefgh", position=0, error_type="invalid_word")
        far_inner = _make_error(
            text="xy",
            position=100,  # far outside outer span
            suggestions=["xz"],
            error_type="confusable_error",
            confidence=0.56,
        )
        errors = [outer, far_inner]
        mixin._boost_inner_confusable_for_compound_splits(errors)
        assert far_inner.confidence == 0.56

    def test_no_boost_when_inner_already_above_ceiling(self):
        mixin = self._build_mixin(
            outer_syllables=["a", "b", "c", "d"],
        )
        outer = _make_error(text="abcdefgh", position=0, error_type="invalid_word")
        high_conf_inner = _make_error(
            text="bc",
            position=1,
            suggestions=["bd"],
            error_type="confusable_error",
            confidence=0.90,
        )
        errors = [outer, high_conf_inner]
        mixin._boost_inner_confusable_for_compound_splits(errors)
        # Above ceiling (0.75 default) → no boost needed
        assert high_conf_inner.confidence == 0.90

    def test_boost_bounded_at_1(self):
        mixin = self._build_mixin(
            outer_syllables=["a", "b", "c", "d"],
            boost=0.50,
            ceiling=0.95,
        )
        outer = _make_error(text="abcdefgh", position=0, error_type="invalid_word")
        inner = _make_error(
            text="bc",
            position=1,
            suggestions=["bd"],
            error_type="confusable_error",
            confidence=0.70,
        )
        errors = [outer, inner]
        mixin._boost_inner_confusable_for_compound_splits(errors)
        # 0.70 + 0.50 = 1.20 → clipped to 1.0
        assert inner.confidence == 1.0

    def test_no_boost_when_outer_syllables_below_min(self):
        # Only 3 syllables — doesn't meet the 4+ predicate
        mixin = self._build_mixin(
            outer_syllables=["a", "b", "c"],
        )
        outer = _make_error(text="abcdef", position=0, error_type="invalid_word")
        inner = _make_error(
            text="bc",
            position=1,
            suggestions=["bd"],
            error_type="confusable_error",
            confidence=0.56,
        )
        errors = [outer, inner]
        mixin._boost_inner_confusable_for_compound_splits(errors)
        assert inner.confidence == 0.56

    def test_no_boost_when_outer_syllable_invalid(self):
        # Compound-split predicate requires ALL syllables valid.
        mixin = self._build_mixin(
            outer_syllables=["a", "b", "c", "d"],
            outer_valid_syllables={"a", "b", "c"},  # 'd' missing → predicate fails
        )
        outer = _make_error(text="abcdefgh", position=0, error_type="invalid_word")
        inner = _make_error(
            text="bc",
            position=1,
            suggestions=["bd"],
            error_type="confusable_error",
            confidence=0.56,
        )
        errors = [outer, inner]
        mixin._boost_inner_confusable_for_compound_splits(errors)
        assert inner.confidence == 0.56

    def test_tolerates_missing_segmenter_or_provider(self):
        mixin = self._build_mixin(outer_syllables=["a", "b", "c", "d"])
        mixin.segmenter = None
        outer = _make_error(text="abcdefgh", position=0, error_type="invalid_word")
        inner = _make_error(
            text="bc",
            position=1,
            suggestions=["bd"],
            error_type="confusable_error",
            confidence=0.56,
        )
        errors = [outer, inner]
        # Should not crash; no boost applied
        mixin._boost_inner_confusable_for_compound_splits(errors)
        assert inner.confidence == 0.56
