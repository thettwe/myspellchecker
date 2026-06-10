"""Unit tests for the pre-normalization aw-vowel un-mask detector.

The pre-lookup normalizer silently repairs aw-vowel typos (flat ော → tall
ေါ after {ပ,ခ,ဒ}; stray ေါ → ော after other bases) BEFORE the validator
judges the token, masking genuine spelling errors like ခော်.
``_detect_aw_vowel_unmask_errors`` runs on the raw text and emits the
silently-applied repair as an explicit confusable error with the canonical
form as the single suggestion, at up to three nested span granularities
(violated syllable, segmenter word, all-content compound chunk),
narrowest first.
"""

from __future__ import annotations

import logging

import pytest

from myspellchecker.core.constants import ET_CONFUSABLE_ERROR
from myspellchecker.core.detectors.pre_normalization import PreNormalizationDetectorsMixin


class _StubProvider:
    """Exact-key dictionary stub mirroring SQLiteProvider.is_valid_word."""

    def __init__(self, words: set[str]) -> None:
        self._words = set(words)

    def is_valid_word(self, word: str) -> bool:
        return word in self._words


class _StubSegmenter:
    """Returns canned segmentations; falls back to the whole text."""

    def __init__(
        self,
        words: dict[str, list[str]] | None = None,
        syllables: dict[str, list[str]] | None = None,
    ) -> None:
        self._words = words or {}
        self._syllables = syllables or {}

    def segment_words(self, text: str) -> list[str]:
        return self._words.get(text, [text])

    def segment_syllables(self, text: str) -> list[str]:
        return self._syllables.get(text, [text])


class _Harness(PreNormalizationDetectorsMixin):
    """Minimal host providing the mixin's attribute stubs."""

    def __init__(
        self,
        words: set[str],
        segments: dict[str, list[str]] | None = None,
        syllables: dict[str, list[str]] | None = None,
    ) -> None:
        self.provider = _StubProvider(words)
        self.segmenter = _StubSegmenter(segments, syllables)
        self.logger = logging.getLogger("test_aw_vowel_unmask")


# Canonical dictionary forms used across tests (all benchmark golds).
_DICT = {
    "ခေါ်",
    "ခေါင်း",
    "ခေါင်းကွဲ",
    "ပေါ်",
    "ပေါက်",
    "ဒေါ်",
    "စိတ်ပေါက်",
    "စုစုပေါင်း",
    "သန်းခေါင်စာရင်း",
    "ကျောင်း",
    "ကောင်း",
    "ဆောင်",
    "သည်",
    "စိတ်",
    "ဝောလ်",  # loanword "Wall" — keyed FLAT post flat-AA migration
    "ဂေါ",  # loanword — raw tall-AA form IS the in-dict key here
}


class TestDirectionFlatToTall:
    """Flat ော after round-bottom {ပ,ခ,ဒ} — the 28-row direction."""

    def test_single_token_fires(self) -> None:
        det = _Harness(_DICT)
        errors = det._detect_aw_vowel_unmask_errors("ခော်")
        assert len(errors) == 1
        err = errors[0]
        assert err.text == "ခော်"
        assert err.position == 0
        assert err.error_type == ET_CONFUSABLE_ERROR
        assert err.suggestions[0] == "ခေါ်"
        assert err._structural_early_exit is True
        assert err.confidence >= 0.60  # must not downgrade to INFORM

    def test_token_inside_sentence_position(self) -> None:
        det = _Harness(_DICT)
        text = "ဒီနေ့ ခော် သည်"
        errors = det._detect_aw_vowel_unmask_errors(text)
        assert len(errors) == 1
        assert errors[0].position == text.index("ခော်")
        assert errors[0].suggestions[0] == "ခေါ်"

    def test_multi_syllable_compound_whole_word(self) -> None:
        det = _Harness(
            _DICT,
            syllables={"သန်းခေါင်စာရင်း": ["သန်း", "ခေါင်", "စာ", "ရင်း"]},
        )
        errors = det._detect_aw_vowel_unmask_errors("သန်းခောင်စာရင်း")
        # Violated syllable ခေါင် is not in the stub dict, so only the
        # whole compound (segmenter keeps it as one word) is emitted.
        assert len(errors) == 1
        assert errors[0].text == "သန်းခောင်စာရင်း"
        assert errors[0].suggestions[0] == "သန်းခေါင်စာရင်း"

    def test_deterministic_gold_for_da_row(self) -> None:
        """ဒော် → ဒေါ် — the row generic SymSpell ranking missed (top-1 တော်).
        The deterministic canonical construction must recover it."""
        det = _Harness(_DICT)
        errors = det._detect_aw_vowel_unmask_errors("ဒော်")
        assert len(errors) == 1
        assert errors[0].suggestions[0] == "ဒေါ်"

    def test_particle_glued_chunk_emits_word_only(self) -> None:
        """Particle-glued chunk → emission at the inner word, no whole-chunk
        emission even when the glued form is a (noisy) dictionary entry."""
        det = _Harness(
            _DICT | {"ခေါ်သည်"},  # noisy glued dict entry must NOT widen the span
            segments={"ခေါ်သည်": ["ခေါ်", "သည်"]},
            syllables={"ခေါ်သည်": ["ခေါ်", "သည်"]},
        )
        errors = det._detect_aw_vowel_unmask_errors("ခော်သည်")
        assert len(errors) == 1
        assert errors[0].text == "ခော်"
        assert errors[0].position == 0
        assert errors[0].suggestions[0] == "ခေါ်"

    def test_compound_token_emits_whole_word(self) -> None:
        """A violated syllable inside a lexical compound token emits at the
        token span (the full word, the in-text typo unit)."""
        det = _Harness(
            _DICT,
            segments={"ခေါင်းကွဲ": ["ခေါင်းကွဲ"]},
            syllables={"ခေါင်းကွဲ": ["ခေါင်း", "ကွဲ"]},
        )
        errors = det._detect_aw_vowel_unmask_errors("ခောင်းကွဲ")
        assert [e.text for e in errors] == ["ခောင်းကွဲ"]
        assert [str(e.suggestions[0]) for e in errors] == ["ခေါင်းကွဲ"]

    def test_all_content_compound_chunk_emitted(self) -> None:
        """A multi-word all-content chunk (lexical compound) emits the
        whole-chunk span — the natural typo unit for compounds like
        စိတ်ပေါက် that the segmenter splits."""
        det = _Harness(
            _DICT,
            segments={"စိတ်ပေါက်": ["စိတ်", "ပေါက်"]},
            syllables={"စိတ်ပေါက်": ["စိတ်", "ပေါက်"]},
        )
        errors = det._detect_aw_vowel_unmask_errors("စိတ်ပောက်")
        assert [e.text for e in errors] == ["စိတ်ပောက်"]
        assert [str(e.suggestions[0]) for e in errors] == ["စိတ်ပေါက်"]

    def test_postposition_glued_chunk_emits_inner_word(self) -> None:
        """A noun + locative postposition phrase is NOT a lexical compound:
        the emission is the postposition word carrying the typo."""
        det = _Harness(
            _DICT | {"ပြဿနာ", "အပေါ်", "ပြဿနာအပေါ်"},
            segments={"ပြဿနာအပေါ်": ["ပြဿနာ", "အပေါ်"]},
            syllables={"ပြဿနာအပေါ်": ["ပြ", "ဿ", "နာ", "အ", "ပေါ်"]},
        )
        errors = det._detect_aw_vowel_unmask_errors("ပြဿနာအပော်")
        assert [e.text for e in errors] == ["အပော်"]
        assert [str(e.suggestions[0]) for e in errors] == ["အပေါ်"]

    def test_syllable_fallback_when_token_gate_fails(self) -> None:
        """When the covering token's canonical form is dictionary-OOV
        (noisy segmentation), the violated syllable is emitted instead."""
        det = _Harness(
            _DICT,  # ဒေါ် in dict; the glued name token is NOT
            segments={"ဒေါ်အောင်ဆန်းမူ": ["ဒေါ်အောင်ဆန်းမူ"]},
            syllables={"ဒေါ်အောင်ဆန်းမူ": ["ဒေါ်", "အောင်", "ဆန်း", "မူ"]},
        )
        errors = det._detect_aw_vowel_unmask_errors("ဒော်အောင်ဆန်းမူ")
        assert [e.text for e in errors] == ["ဒော်"]
        assert [str(e.suggestions[0]) for e in errors] == ["ဒေါ်"]


class TestDirectionTallToFlat:
    """Stray tall ေါ after complement bases — the 2-row reverse direction."""

    def test_after_complement_consonant(self) -> None:
        det = _Harness(_DICT)
        errors = det._detect_aw_vowel_unmask_errors("ဆေါင်")
        assert len(errors) == 1
        assert errors[0].suggestions[0] == "ဆောင်"

    def test_after_medial_ya(self) -> None:
        """ကျေါင်း → ကျောင်း: the medial sign occupies the base slot and
        medial clusters take flat AA (see TestMedialClusterContract in
        test_normalize_e_vowel_tall_aa.py)."""
        det = _Harness(_DICT)
        errors = det._detect_aw_vowel_unmask_errors("ကျေါင်း")
        assert len(errors) == 1
        assert errors[0].suggestions[0] == "ကျောင်း"

    def test_vowel_reorder_map_forms_deferred(self) -> None:
        """Chunks containing a _VOWEL_REORDER_ERRORS key are owned by the
        dedicated detector: ကေါင်း can be a ခ→က consonant typo (gold
        ခေါင်းလောင်း), so the single-suggestion unmask emission must not
        displace that detector's multi-candidate list."""
        det = _Harness(_DICT | {"ကောင်းလောင်း"})
        assert det._detect_aw_vowel_unmask_errors("ကေါင်းလောင်း") == []
        assert det._detect_aw_vowel_unmask_errors("ကေါင်း") == []

    def test_independent_aw_typo_glued_to_reorder_key_still_fires(self) -> None:
        """A recoverable aw-typo (ခော်) glued to a reorder-key form (ကေါင်း) in
        the same chunk still fires for the typo; only the reorder-key span is
        deferred. The old whole-chunk substring defer dropped both."""
        det = _Harness(
            _DICT,
            segments={"ခေါ်ကောင်း": ["ခေါ်", "ကောင်း"]},
            syllables={"ခေါ်ကောင်း": ["ခေါ်", "ကောင်း"]},
        )
        errors = det._detect_aw_vowel_unmask_errors("ခော်ကေါင်း")
        assert [e.text for e in errors] == ["ခော်"]
        assert [str(e.suggestions[0]) for e in errors] == ["ခေါ်"]


class TestLoanwordGuard:
    """The {ဂ,င,ဝ} bases — classical round-bottom consonants excluded from
    the narrow whitelist — must never fire in the tall→flat direction.
    These are exactly the 2 clean-text flips from the unmask-probe-01
    kill-gate (ဝေါလ်, ဂေါ)."""

    def test_wa_loanword_not_flagged(self) -> None:
        # ဝေါလ် is OOV as typed (dict keys the flat form ဝောလ်) and the
        # canonical form IS in-dict — only the base guard blocks the fire.
        det = _Harness(_DICT)
        assert det._detect_aw_vowel_unmask_errors("ဝေါလ်") == []

    def test_ga_loanword_not_flagged(self) -> None:
        det = _Harness(_DICT)
        assert det._detect_aw_vowel_unmask_errors("ဂေါ") == []


class TestNoFireGates:
    """Each precision gate must independently block the emission."""

    def test_canonical_text_no_fire(self) -> None:
        det = _Harness(_DICT)
        assert det._detect_aw_vowel_unmask_errors("ခေါ် သည် ကျောင်း") == []

    def test_complement_flat_aa_no_fire(self) -> None:
        det = _Harness(_DICT)
        assert det._detect_aw_vowel_unmask_errors("ကောင်း တော သော") == []

    def test_canonical_not_in_dict_no_fire(self) -> None:
        det = _Harness(set())
        assert det._detect_aw_vowel_unmask_errors("ခော်") == []

    def test_raw_form_in_dict_no_fire(self) -> None:
        det = _Harness(_DICT | {"ခော်"})
        assert det._detect_aw_vowel_unmask_errors("ခော်") == []

    def test_non_aw_diff_in_chunk_no_fire(self) -> None:
        """A chunk whose canonicalization changes anything besides the
        aw-vowel is out of scope (other detectors own those repairs)."""
        det = _Harness(_DICT | {"ဆောင်း"})
        # ဆေါငး် has BOTH a stray tall AA and a visarga-asat reorder:
        # normalize() fixes both, so the diff is not aw-only → skip.
        assert det._detect_aw_vowel_unmask_errors("ဆေါငး်") == []

    def test_empty_and_non_myanmar(self) -> None:
        det = _Harness(_DICT)
        assert det._detect_aw_vowel_unmask_errors("") == []
        assert det._detect_aw_vowel_unmask_errors("hello world") == []


class TestDiffGuardUnit:
    """Direct unit coverage of _aw_vowel_diffs_guarded."""

    @pytest.fixture()
    def det(self) -> _Harness:
        return _Harness(_DICT)

    def test_flat_to_tall_round_bottom_ok(self, det: _Harness) -> None:
        assert det._aw_vowel_diffs_guarded("ခော်", "ခေါ်") is True

    def test_flat_to_tall_outside_whitelist_rejected(self, det: _Harness) -> None:
        # Normalizer never repairs flat→tall outside {ပ,ခ,ဒ}; reject defensively.
        assert det._aw_vowel_diffs_guarded("ကော်", "ကေါ်") is False

    def test_tall_to_flat_ambiguous_base_rejected(self, det: _Harness) -> None:
        assert det._aw_vowel_diffs_guarded("ဝေါလ်", "ဝောလ်") is False

    def test_tall_to_flat_medial_base_ok(self, det: _Harness) -> None:
        assert det._aw_vowel_diffs_guarded("ကျေါင်း", "ကျောင်း") is True

    def test_identical_strings_rejected(self, det: _Harness) -> None:
        assert det._aw_vowel_diffs_guarded("ခေါ်", "ခေါ်") is False

    def test_non_aw_diff_rejected(self, det: _Harness) -> None:
        assert det._aw_vowel_diffs_guarded("ခမ်", "ခန်") is False

    def test_multiple_aw_diffs_all_guarded(self, det: _Harness) -> None:
        assert det._aw_vowel_diffs_guarded("ပော်ပောက်", "ပေါ်ပေါက်") is True

    def test_no_base_before_aw_rejected(self, det: _Harness) -> None:
        # Diff at index 1 — no base two positions back (the i < 2 guard).
        assert det._aw_vowel_diffs_guarded("ေါ", "ော") is False

    def test_non_myanmar_base_rejected(self, det: _Harness) -> None:
        # tall→flat requires a Myanmar consonant / medial base; Latin rejected.
        assert det._aw_vowel_diffs_guarded("Aေါ", "Aော") is False

    def test_mixed_valid_and_invalid_diffs_rejected(self, det: _Harness) -> None:
        # One guarded aw-swap (ပ: ော→ေါ) plus a non-aw diff (မ→န): the
        # all-diffs-must-pass contract must reject the whole token.
        assert det._aw_vowel_diffs_guarded("ပော်ခမ်", "ပေါ်ခန်") is False


class TestSpanFallbacksAndPositions:
    """Defensive span-selection branches + duplicate-chunk position tracking."""

    def test_segmenter_empty_falls_back_to_whole_chunk(self) -> None:
        # segment_words returns [] for the canonical chunk → whole-chunk gate
        # is the fallback emission (non-default / unavailable segmenter).
        det = _Harness(_DICT, segments={"ခေါ်": []})
        errors = det._detect_aw_vowel_unmask_errors("ခော်")
        assert [e.text for e in errors] == ["ခော်"]
        assert [str(e.suggestions[0]) for e in errors] == ["ခေါ်"]

    def test_walk_skips_untraceable_segmenter_part(self) -> None:
        # A segmenter part that is not a substring of the canonical chunk
        # (non-tiling backend) is skipped without derailing the emission.
        det = _Harness(_DICT, segments={"ခေါ်": ["ခေါ်", "ZZZ"]})
        errors = det._detect_aw_vowel_unmask_errors("ခော်")
        assert [e.text for e in errors] == ["ခော်"]

    def test_duplicate_typo_chunk_distinct_positions(self) -> None:
        # The same typo chunk repeated must anchor to start-advanced offsets,
        # not collapse both onto the first occurrence.
        text = "ခော် ခော်"
        det = _Harness(_DICT)
        errors = det._detect_aw_vowel_unmask_errors(text)
        assert len(errors) == 2
        assert [e.position for e in errors] == [
            text.index("ခော်"),
            text.rindex("ခော်"),
        ]
