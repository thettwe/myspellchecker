"""Unit tests for the context-aukmyit suppression-immunity + whole-word
emission (avt-02 B2).

The dot-below (့) context detectors ``_detect_aukmyit_confusion`` (ထည်→ထည့်)
and ``_detect_extra_aukmyit_confusion`` (ပြော့→ပြော) only fire when a
disambiguating right-context trigger follows, so their emissions are
high-precision. Two behaviours are covered:

1. When ``aukmyit_context_suppression_immune`` is on, the emitted error
   carries ``_structural_early_exit`` so the generic low-value-confusable
   suppressor (which drops every ့-only difference) cannot kill it.
2. The error spans the whole enclosing whitespace-delimited word and
   suggests the whole corrected word (ပြော့သည်→ပြောသည်, not ပြော့→ပြော) so
   it matches the gold granularity and earns top-1 credit.
"""

from __future__ import annotations

from types import SimpleNamespace

from myspellchecker.core.constants import ET_CONFUSABLE_ERROR
from myspellchecker.core.detectors.post_normalization import (
    PostNormalizationDetectorsMixin,
)


class _Harness(PostNormalizationDetectorsMixin):
    """Minimal host exposing only the config flag the detector reads."""

    def __init__(self, immune: bool = True) -> None:
        self.config = SimpleNamespace(
            validation=SimpleNamespace(aukmyit_context_suppression_immune=immune)
        )


def _run_extra(text: str, immune: bool = True):
    harness = _Harness(immune=immune)
    errors: list = []
    harness._detect_extra_aukmyit_confusion(text, errors)
    return errors


def test_extra_aukmyit_fires_on_glued_trigger() -> None:
    """ပြော့ + glued declarative သည် → emits a confusable error."""
    errors = _run_extra("သူ ပြော့သည်။")
    assert len(errors) == 1
    assert errors[0].error_type == ET_CONFUSABLE_ERROR


def test_extra_aukmyit_emits_whole_word_suggestion() -> None:
    """Span + suggestion cover the whole word, not just the ပြော့ syllable."""
    errors = _run_extra("သူ ပြော့သည်။")
    err = errors[0]
    assert err.text == "ပြော့သည်။"
    assert [str(s) for s in err.suggestions] == ["ပြောသည်။"]
    # position points at the start of the whole word (offset of "ပြော့သည်။").
    assert err.position == "သူ ".__len__()


def test_extra_aukmyit_survival_flag_when_immune() -> None:
    """Immune flag on → emission is marked structural-early-exit (survives)."""
    err = _run_extra("သူ ပြော့သည်။", immune=True)[0]
    assert getattr(err, "_structural_early_exit", False) is True


def test_extra_aukmyit_no_survival_flag_when_not_immune() -> None:
    """Immune flag off → emission still fires but is not suppression-immune."""
    err = _run_extra("သူ ပြော့သည်။", immune=False)[0]
    assert getattr(err, "_structural_early_exit", False) is False


def test_extra_aukmyit_requires_trigger() -> None:
    """Bare ပြော့ at clause end (no disambiguating right context) is silent."""
    assert _run_extra("အရမ်း ပြော့။") == []
