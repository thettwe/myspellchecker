"""Select and mark the frozen holdout subset in myspellchecker_benchmark.yaml.

The holdout is a stratified ~12% sample of the benchmark whose rows are
annotation-frozen: they may not be re-annotated, re-labeled, or re-spanned by
the same process that optimizes against the benchmark, and from bp-02 onward
tuning runs exclude them (run_benchmark.py --holdout exclude). Ship gates
score them separately (--holdout only) to detect drift between the mutable
main set and the frozen subset.

Selection is deterministic and seed-free: within each stratum, rows are
ranked by sha256("v19-holdout:" + sentence id) and the lowest hashes are
taken. Re-running --select on the same yaml always yields the same set.

Strata:
  - clean rows: by `domain` (12% of each)
  - error rows: by primary error_subtype (first expected error; 12% of each)
Global targets use largest-remainder rounding to hit exactly 12% overall.

Usage:
  python benchmarks/select_holdout.py --select          # print selection summary
  python benchmarks/select_holdout.py --apply           # insert markers + bump version
  python benchmarks/select_holdout.py --verify          # check applied state
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import yaml

BENCHMARK_PATH = Path(__file__).parent / "myspellchecker_benchmark.yaml"
HOLDOUT_FRACTION = 0.12
HASH_PREFIX = "v19-holdout:"
NEW_VERSION = "1.5.0-v19h-holdout-freeze"


def _rank(sentence_id: str) -> str:
    return hashlib.sha256((HASH_PREFIX + sentence_id).encode("utf-8")).hexdigest()


def _stratum(sentence: dict) -> str:
    if sentence.get("is_clean"):
        return f"clean/{sentence.get('domain', 'unknown')}"
    errors = sentence.get("expected_errors") or []
    subtype = errors[0].get("error_subtype", "unknown") if errors else "no-errors"
    return f"error/{subtype}"


def _largest_remainder_quotas(strata: dict[str, list[dict]], target_total: int) -> dict[str, int]:
    raw = {name: len(rows) * HOLDOUT_FRACTION for name, rows in strata.items()}
    quotas = {name: int(v) for name, v in raw.items()}
    remainder = target_total - sum(quotas.values())
    by_fraction = sorted(raw, key=lambda n: (raw[n] - quotas[n], n), reverse=True)
    for name in by_fraction[:remainder]:
        quotas[name] += 1
    return quotas


def select_holdout(benchmark: dict) -> list[str]:
    sentences = benchmark["sentences"]
    strata: dict[str, list[dict]] = defaultdict(list)
    for s in sentences:
        strata[_stratum(s)].append(s)

    target_total = round(len(sentences) * HOLDOUT_FRACTION)
    quotas = _largest_remainder_quotas(strata, target_total)

    selected: list[str] = []
    for name, rows in strata.items():
        ranked = sorted(rows, key=lambda s: _rank(s["id"]))
        selected.extend(s["id"] for s in ranked[: quotas[name]])
    return sorted(selected)


def apply_markers(selected: set[str]) -> None:
    lines = BENCHMARK_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    current_id: str | None = None
    inserted = 0
    for line in lines:
        m = re.match(r"^- id: (\S+)\s*$", line)
        if m:
            current_id = m.group(1)
        out.append(line)
        if current_id in selected and re.match(r"^  is_clean: (true|false)\s*$", line):
            out.append("  holdout: true\n")
            inserted += 1
            current_id = None
    if inserted != len(selected):
        raise SystemExit(f"marker insertion mismatch: {inserted} != {len(selected)}")

    text = "".join(out)
    old_version_line = re.search(r"^version: (\S+)$", text, re.MULTILINE)
    if not old_version_line:
        raise SystemExit("version line not found")
    text = text.replace(f"version: {old_version_line.group(1)}", f"version: {NEW_VERSION}", 1)
    BENCHMARK_PATH.write_text(text, encoding="utf-8")
    print(f"Applied {inserted} holdout markers; version -> {NEW_VERSION}")


def verify() -> None:
    benchmark = yaml.safe_load(BENCHMARK_PATH.read_text(encoding="utf-8"))
    marked = [s["id"] for s in benchmark["sentences"] if s.get("holdout")]
    expected = select_holdout(benchmark)
    clean = sum(1 for s in benchmark["sentences"] if s.get("holdout") and s.get("is_clean"))
    print(f"version: {benchmark.get('version')}")
    print(f"marked: {len(marked)} (clean {clean} / error {len(marked) - clean})")
    if set(marked) != set(expected):
        raise SystemExit(
            f"holdout set does not match deterministic selection "
            f"(+{len(set(marked) - set(expected))} / -{len(set(expected) - set(marked))})"
        )
    print("verify OK: marked set == deterministic selection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--select", action="store_true", help="print selection summary")
    group.add_argument("--apply", action="store_true", help="insert holdout markers + bump version")
    group.add_argument("--verify", action="store_true", help="verify applied markers")
    args = parser.parse_args()

    if args.verify:
        verify()
        return

    benchmark = yaml.safe_load(BENCHMARK_PATH.read_text(encoding="utf-8"))
    if args.apply and any(s.get("holdout") for s in benchmark["sentences"]):
        raise SystemExit("holdout markers already present — refusing to re-apply")

    selected = select_holdout(benchmark)
    strata: dict[str, int] = defaultdict(int)
    by_id = {s["id"]: s for s in benchmark["sentences"]}
    for sid in selected:
        strata[_stratum(by_id[sid])] += 1

    clean_n = sum(n for name, n in strata.items() if name.startswith("clean/"))
    print(
        f"selected {len(selected)} of {len(benchmark['sentences'])} "
        f"({clean_n} clean / {len(selected) - clean_n} error)"
    )
    for name in sorted(strata):
        print(f"  {name}: {strata[name]}")

    if args.apply:
        apply_markers(set(selected))
        verify()


if __name__ == "__main__":
    main()
