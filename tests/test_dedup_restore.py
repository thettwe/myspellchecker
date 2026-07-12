"""Tests for the Option R dedup graveyard + restore-through-tail mechanism.

Covers: graveyard capture at both dedup sites, per-check reset, the measured
slice blocklist, and the restore orchestration (empty-slot admission,
occupied-slot skip, confidence gates). Integrated behavior (tail filters,
suggestion processing, benchmark deltas) is verified by the full benchmark;
these tests pin the orchestration logic.
"""

import threading
from unittest.mock import MagicMock

from myspellchecker import SpellChecker
from myspellchecker.core.config.main import SpellCheckerConfig
from myspellchecker.core.response import Error


def _make_error(
    text="စမ်း",
    position=0,
    error_type="invalid_word",
    confidence=0.9,
    suggestions=None,
    source_strategy="",
):
    e = Error(
        text=text,
        position=position,
        error_type=error_type,
        suggestions=suggestions if suggestions is not None else ["စမ်"],
        confidence=confidence,
    )
    e.source_strategy = source_strategy
    return e


def _checker(restore_enabled=True):
    c = SpellChecker.__new__(SpellChecker)
    c.provider = MagicMock()
    c.provider.get_word_frequency.return_value = 0
    c.provider.is_valid_word.return_value = False
    c.config = SpellCheckerConfig()
    c.config.validation.dedup_restore_displaced = restore_enabled
    # narrow the restore gauntlet for unit scope: no MLM / compound-split /
    # word suppressor (integration is benchmark-verified)
    c.config.validation.bypass_word_heuristic_suppression = True
    c._dedup_graveyard_tls = threading.local()
    c._semantic_checker = None
    c._meta_classifier = None
    c._ner_model = None
    c._rerank_telemetry_lock = threading.Lock()
    c._CONFIDENCE_THRESHOLDS = c.config.validation.output_confidence_thresholds
    c._SECONDARY_CONFIDENCE_THRESHOLDS = {}
    # suggestion stages are integration-level; neutralize here
    c._extend_suggestions_with_sentence_context = MagicMock()
    c._append_morpheme_subwords = MagicMock()
    c._rerank_detector_suggestions_by_distance = MagicMock()
    return c


class TestGraveyardCapture:
    def test_position_dedup_captures_loser(self):
        c = _checker()
        wide = _make_error(text="စမ်းသပ်မှု", position=5)
        narrow = _make_error(text="စမ်း", position=5, error_type="confusable_error")
        errors = [narrow, wide]
        c._dedup_errors_by_position(errors)
        assert len(errors) == 1
        graveyard = c._dedup_graveyard_items()
        assert len(graveyard) == 1
        assert graveyard[0] in (narrow, wide)
        assert graveyard[0] not in errors

    def test_span_dedup_captures_contained_loser(self):
        c = _checker()
        wide = _make_error(text="စမ်းသပ်မှု", position=0)
        inner = _make_error(text="သပ်", position=2, error_type="invalid_word")
        errors = [wide, inner]
        c._dedup_errors_by_span(errors)
        assert inner not in errors
        assert inner in c._dedup_graveyard_items()

    def test_flag_off_captures_nothing(self):
        c = _checker(restore_enabled=False)
        wide = _make_error(text="စမ်းသပ်မှု", position=5)
        narrow = _make_error(text="စမ်း", position=5)
        c._dedup_errors_by_position([narrow, wide])
        assert c._dedup_graveyard_items() == []

    def test_reset_clears_graveyard(self):
        c = _checker()
        c._dedup_graveyard_items().append(_make_error())
        c._reset_dedup_graveyard()
        assert c._dedup_graveyard_items() == []

    def test_bare_mixin_without_tls_is_graceful(self):
        c = SpellChecker.__new__(SpellChecker)
        assert c._dedup_graveyard_items() == []
        c._reset_dedup_graveyard()  # must not raise


class TestSliceBlocklist:
    def test_broken_compound_family_blocked(self):
        for src in ("", "CrossWhitespaceProbeStrategy"):
            e = _make_error(error_type="broken_compound", source_strategy=src)
            assert SpellChecker._restore_slice_blocked(e) == "broken_compound_family"

    def test_broken_compound_curated_sources_pass(self):
        e = _make_error(error_type="broken_compound", source_strategy="BrokenCompoundStrategy")
        assert SpellChecker._restore_slice_blocked(e) is None

    def test_ngram_context_blocked(self):
        e = _make_error(
            error_type="context_probability",
            source_strategy="NgramContextValidationStrategy",
        )
        assert SpellChecker._restore_slice_blocked(e) == "ngram_context"

    def test_low_conf_detector_invalid_word_blocked_and_high_passes(self):
        low = _make_error(error_type="invalid_word", confidence=0.80)
        high = _make_error(error_type="invalid_word", confidence=0.85)
        sourced = _make_error(
            error_type="invalid_word",
            confidence=0.6,
            source_strategy="CompoundMergeProbeStrategy",
        )
        assert SpellChecker._restore_slice_blocked(low) == "low_conf_detector_invalid_word"
        assert SpellChecker._restore_slice_blocked(high) is None
        assert SpellChecker._restore_slice_blocked(sourced) is None


class TestRestoreOrchestration:
    def test_restores_viable_loser_into_empty_slot(self):
        c = _checker()
        loser = _make_error(text="ထာနမှူး", position=10, error_type="invalid_word", confidence=0.9)
        c._dedup_graveyard_items().append(loser)
        restored = c._restore_displaced_errors([], "ထာနမှူးက ပြောသည်")
        assert restored == [loser]
        # post-gauntlet suggestion processing ran on the restored error
        c._rerank_detector_suggestions_by_distance.assert_called_once()

    def test_occupied_slot_is_skipped(self):
        c = _checker()
        loser = _make_error(text="ထာနမှူး", position=10, confidence=0.9)
        survivor = _make_error(text="ထာနမှူးက", position=10, confidence=0.9)
        c._dedup_graveyard_items().append(loser)
        assert c._restore_displaced_errors([survivor], "text") == []

    def test_blocked_slice_is_skipped(self):
        c = _checker()
        loser = _make_error(
            text="မြန်မာ နိုင်ငံ", position=0, error_type="broken_compound", confidence=0.9
        )
        c._dedup_graveyard_items().append(loser)
        assert c._restore_displaced_errors([], "text") == []

    def test_two_losers_same_slot_only_first_restores(self):
        c = _checker()
        a = _make_error(text="ထာနမှူး", position=10, confidence=0.9)
        b = _make_error(text="ထာန", position=10, confidence=0.9)
        c._dedup_graveyard_items().extend([a, b])
        restored = c._restore_displaced_errors([], "text")
        assert restored == [a]

    def test_meta_scores_restored_candidate(self):
        c = _checker()
        meta = MagicMock()
        meta.filter_errors.return_value = []  # meta rejects
        c._meta_classifier = meta
        loser = _make_error(text="ထာနမှူး", position=10, confidence=0.9)
        c._dedup_graveyard_items().append(loser)
        assert c._restore_displaced_errors([], "text") == []
        meta.filter_errors.assert_called_once()

    def test_empty_graveyard_returns_fast(self):
        c = _checker()
        assert c._restore_displaced_errors([], "text") == []
        c._rerank_detector_suggestions_by_distance.assert_not_called()

    def test_inlayer_suppressors_run_unless_source_is_immune(self):
        """M1 fix: restored losers pass the post-dedup in-layer suppressors;
        suppression-immune sources skip them (parity with the immune
        extraction survivors get)."""
        c = _checker()
        c.config.validation.suppression_immune_strategies = frozenset({"VisargaStrategy"})

        def _kill_all(probe, **kwargs):
            probe.clear()

        c._suppress_low_value_syllable_errors = MagicMock(side_effect=_kill_all)
        immune_loser = _make_error(
            text="ထာန", position=0, confidence=0.9, source_strategy="VisargaStrategy"
        )
        plain_loser = _make_error(text="မှူး", position=30, confidence=0.9)
        c._dedup_graveyard_items().extend([immune_loser, plain_loser])
        restored = c._restore_displaced_errors([], "text")
        assert immune_loser in restored
        assert plain_loser not in restored
