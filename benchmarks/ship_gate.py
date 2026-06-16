"""v1.9 ship-gate cap enforcer (WS-RG rg-01).

Makes the binding ship-gate caps a real code gate instead of policy-only.
Today ``run_benchmark.py`` only enforces ``latency_pass`` (p95 <= 500ms); the
clean-FP / FPR / composite caps were "auto-reject" in name only. This module
asserts every binding cap against a benchmark result JSON and exits non-zero on
any breach, so CI / a pre-merge hook can block a regression.

Usage::

    python benchmarks/ship_gate.py benchmarks/results/<run>/<result>.json

The cap values are the ratified v1.9 ship gate (re-anchored to the
``1.5.0-v19-granularity-normalization`` scale; clean-FP re-ratified <= 88 after
WS-FPH fph-02; baseline reference ``fph02-verify2`` composite 0.7821 / clean-FP
84 / FPR 10.78% / p95 404ms). See ``00_Index`` -> ``v1.9.0-ship-gate``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── Ratified v1.9 ship-gate caps (single source of truth) ──────────────────
COMPOSITE_FLOOR = 0.7791  # FLOOR; ship only when composite holds or improves
CLEAN_FP_MAX = 88  # binding cap, re-ratified after fph-02 (measured 84, 4 headroom)
FPR_MAX = 0.1078  # non-regress vs the post-fph-02 reference (10.78%)
FPR_TOLERANCE = 0.0010  # absorb run-to-run noise in the FPR term
P95_HARD_MS = 500.0  # hard latency gate (matches run_benchmark latency_pass)
P95_POLICY_MS = 420.0  # policy target ~400ms + a noise margin (baseline ~404ms)


@dataclass(frozen=True)
class GateResult:
    name: str
    value: float
    threshold: float
    ok: bool
    severity: str  # "FAIL" | "WARN"

    def line(self) -> str:
        mark = "PASS" if self.ok else self.severity
        return f"  [{mark}] {self.name}: {self.value} (cap {self.threshold})"


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    om = report.get("overall_metrics") or report.get("metrics") or {}
    fpr = om.get("false_positive_rate") or {}
    lat = om.get("latency_ms") or {}
    return {
        "composite": om.get("composite_score"),
        "clean_fp": fpr.get("clean_sentences_with_fp"),
        "fpr": fpr.get("rate"),
        "p95": lat.get("p95") if isinstance(lat, dict) else None,
    }


def check_report(report: dict[str, Any]) -> list[GateResult]:
    """Return the per-cap gate results for a benchmark report dict."""
    m = _metrics(report)
    results: list[GateResult] = []

    if m["composite"] is not None:
        v = float(m["composite"])
        results.append(
            GateResult(
                "composite >= FLOOR", round(v, 4), COMPOSITE_FLOOR, v >= COMPOSITE_FLOOR, "FAIL"
            )
        )
    if m["clean_fp"] is not None:
        v = int(m["clean_fp"])
        results.append(GateResult("clean-FP <= cap", v, CLEAN_FP_MAX, v <= CLEAN_FP_MAX, "FAIL"))
    if m["fpr"] is not None:
        v = float(m["fpr"])
        results.append(
            GateResult(
                "FPR non-regress",
                round(v, 4),
                FPR_MAX + FPR_TOLERANCE,
                v <= FPR_MAX + FPR_TOLERANCE,
                "FAIL",
            )
        )
    if m["p95"] is not None:
        v = float(m["p95"])
        results.append(
            GateResult("p95 <= hard", round(v, 1), P95_HARD_MS, v <= P95_HARD_MS, "FAIL")
        )
        # Policy target is advisory (latency is run-to-run noisy) -> WARN only.
        results.append(
            GateResult("p95 <= policy", round(v, 1), P95_POLICY_MS, v <= P95_POLICY_MS, "WARN")
        )
    return results


def gate_passed(results: list[GateResult]) -> bool:
    """True iff no FAIL-severity cap is breached (WARN does not block)."""
    return all(r.ok for r in results if r.severity == "FAIL")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python benchmarks/ship_gate.py <result.json>", file=sys.stderr)
        return 2
    report = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    results = check_report(report)
    if not results:
        print("[ship-gate] no recognised metrics in report", file=sys.stderr)
        return 2
    print("=== v1.9 ship-gate caps ===")
    for r in results:
        print(r.line())
    passed = gate_passed(results)
    warns = [r for r in results if r.severity == "WARN" and not r.ok]
    print(
        f"\n=== SHIP-GATE: {'PASS' if passed else 'FAIL'} ==="
        + (f"  ({len(warns)} warning)" if warns else "")
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
