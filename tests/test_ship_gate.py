"""WS-RG rg-01: the v1.9 ship-gate caps are real code gates, not policy.

Covers the cap enforcer (`benchmarks/ship_gate.py`) on synthetic reports
(CI-runnable — no production DB) plus a config snapshot that pins the probe
sweet-spot thresholds and the df-02 default-on detection flags so a later edit
cannot silently regress them (STOP #4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from myspellchecker.core.config import SpellCheckerConfig

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))

from ship_gate import (  # noqa: E402
    CLEAN_FP_MAX,
    check_report,
    gate_passed,
)


def _report(composite=0.7821, clean_fp=84, fpr=0.1078, p95=404.1):
    """Synthetic benchmark report shaped like run_benchmark's overall_metrics."""
    return {
        "overall_metrics": {
            "composite_score": composite,
            "false_positive_rate": {"rate": fpr, "clean_sentences_with_fp": clean_fp},
            "latency_ms": {"p95": p95},
        }
    }


# ── Cap enforcer ────────────────────────────────────────────────────────────


def test_passes_on_frozen_baseline() -> None:
    """fph02-verify2 reference (0.7821 / 84 / 10.78% / 404ms) clears every cap."""
    assert gate_passed(check_report(_report()))


def test_fails_on_clean_fp_breach() -> None:
    assert not gate_passed(check_report(_report(clean_fp=89)))


def test_clean_fp_at_cap_passes() -> None:
    assert gate_passed(check_report(_report(clean_fp=CLEAN_FP_MAX)))


def test_fails_on_composite_below_floor() -> None:
    assert not gate_passed(check_report(_report(composite=0.7780)))


def test_fails_on_fpr_regress() -> None:
    assert not gate_passed(check_report(_report(fpr=0.1150)))


def test_fails_on_p95_hard_breach() -> None:
    assert not gate_passed(check_report(_report(p95=520.0)))


def test_p95_policy_is_warn_not_fail() -> None:
    """p95 above the ~400ms policy but under the 500ms hard gate -> WARN only."""
    results = check_report(_report(p95=450.0))
    assert gate_passed(results)
    assert any(r.severity == "WARN" and not r.ok for r in results)


def test_partial_report_only_checks_present_metrics() -> None:
    report = {"overall_metrics": {"composite_score": 0.79}}
    assert gate_passed(check_report(report))


# ── Probe sweet-spot + default-on snapshot ──────────────────────────────────


def test_probe_sweet_spot_config_snapshot() -> None:
    """Pin the v1.7.1 probe sweet-spot thresholds against accidental edits."""
    v = SpellCheckerConfig().validation
    assert v.probe_corrector_threshold == pytest.approx(0.75)
    assert v.probe_compound_threshold == pytest.approx(0.70)
    assert v.probe_compound_min_freq == 50
    assert v.probe_rescue_threshold == pytest.approx(0.75)
    assert v.probe_rescue_min_freq == 2000
    assert v.probe_max_existing_errors == 100


def test_v19_detection_defaults_on() -> None:
    """df-02 flipped these on for v1.9; pin so they can't silently flip off."""
    v = SpellCheckerConfig().validation
    assert v.use_probe_corrector is True
    assert v.use_probe_compound is True
    assert v.use_probe_segmenter_rescue is True
    assert v.detect_aw_vowel_unmask is True
