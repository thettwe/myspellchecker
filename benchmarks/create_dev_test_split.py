#!/usr/bin/env python3
"""
Create Dev/Test Split — v1.6.0 Sprint 1.3

Splits the benchmark into 80% dev / 20% test, stratified by:
- domain (academic, conversational, general, literary, news, religious, social_media, technical)
- difficulty_tier (1, 2, 3, None for clean)
- is_clean (True/False)

The test set should NEVER be used for threshold tuning. All tuning uses dev only.
Test set is for final go/no-go evaluation of releases.

Usage:
    python benchmarks/create_dev_test_split.py
    python benchmarks/create_dev_test_split.py --test-ratio 0.25
    python benchmarks/create_dev_test_split.py --seed 123
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


def stratified_split(
    sentences: list[dict],
    test_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split sentences with stratification by domain + difficulty + is_clean."""
    import numpy as np

    rng = np.random.default_rng(seed)

    # Build strata
    strata: dict[str, list[dict]] = defaultdict(list)
    for s in sentences:
        domain = s.get("domain", "unknown")
        tier = s.get("difficulty_tier", "clean" if s.get("is_clean") else "none")
        key = f"{domain}_{tier}"
        strata[key].append(s)

    dev_sentences: list[dict] = []
    test_sentences: list[dict] = []

    for _stratum_key, items in sorted(strata.items()):
        n = len(items)
        n_test = max(1, round(n * test_ratio))  # At least 1 per stratum

        # Shuffle within stratum
        indices = list(range(n))
        rng.shuffle(indices)

        for i, idx in enumerate(indices):
            if i < n_test:
                test_sentences.append(items[idx])
            else:
                dev_sentences.append(items[idx])

    return dev_sentences, test_sentences


def compute_split_stats(sentences: list[dict]) -> dict:
    """Compute distribution stats for a split."""
    domains = defaultdict(int)
    tiers = defaultdict(int)
    clean_count = 0
    error_count = 0

    for s in sentences:
        domains[s.get("domain", "unknown")] += 1
        if s.get("is_clean"):
            clean_count += 1
            tiers["clean"] += 1
        else:
            error_count += 1
            tier = s.get("difficulty_tier", "none")
            tiers[f"tier_{tier}"] += 1

    return {
        "total": len(sentences),
        "clean": clean_count,
        "error": error_count,
        "domains": dict(sorted(domains.items())),
        "tiers": dict(sorted(tiers.items())),
    }


def build_split_yaml(
    original_metadata: dict,
    sentences: list[dict],
    split_name: str,
    split_stats: dict,
    original_hash: str,
) -> dict:
    """Build a YAML document for one split."""
    return {
        "version": original_metadata.get("version", "1.0.0"),
        "category": "benchmark",
        "description": (
            f"{split_name.upper()} split of Myanmar spell checker benchmark. "
            f"Stratified by domain + difficulty tier. "
            + (
                "Use for threshold tuning and development."
                if split_name == "dev"
                else "NEVER use for tuning. Final evaluation only."
            )
        ),
        "metadata": {
            "split": split_name,
            "created_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "source_benchmark": "benchmarks/myspellchecker_benchmark.yaml",
            "source_hash": original_hash,
            "total_sentences": split_stats["total"],
            "clean_count": split_stats["clean"],
            "error_count": split_stats["error"],
            "domain_distribution": split_stats["domains"],
            "tier_distribution": split_stats["tiers"],
        },
        "sentences": sentences,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified dev/test benchmark split.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/myspellchecker_benchmark.yaml"),
        help="Source benchmark YAML.",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.20, help="Test set ratio (default 0.20)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Output directory for split files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.benchmark.exists():
        print(f"Error: Benchmark not found: {args.benchmark}")
        return 1

    print(f"Loading benchmark: {args.benchmark}")
    raw_content = args.benchmark.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(raw_content.encode()).hexdigest()[:12]
    benchmark = yaml.safe_load(raw_content)

    sentences = benchmark["sentences"]
    metadata = benchmark.get("metadata", {})
    print(f"  Total sentences: {len(sentences)}")

    # Split
    dev_sentences, test_sentences = stratified_split(
        sentences, test_ratio=args.test_ratio, seed=args.seed
    )

    dev_stats = compute_split_stats(dev_sentences)
    test_stats = compute_split_stats(test_sentences)

    print(
        f"\n  Dev split: {dev_stats['total']} sentences "
        f"({dev_stats['clean']} clean, {dev_stats['error']} error)"
    )
    print(
        f"  Test split: {test_stats['total']} sentences "
        f"({test_stats['clean']} clean, {test_stats['error']} error)"
    )

    # Verify stratification quality
    print("\n  Domain distribution comparison:")
    all_domains = sorted(
        set(list(dev_stats["domains"].keys()) + list(test_stats["domains"].keys()))
    )
    for domain in all_domains:
        dev_n = dev_stats["domains"].get(domain, 0)
        test_n = test_stats["domains"].get(domain, 0)
        total = dev_n + test_n
        test_pct = test_n / total * 100 if total > 0 else 0
        print(f"    {domain:15s}: dev={dev_n:4d}  test={test_n:3d}  (test={test_pct:.0f}%)")

    # Build YAML documents
    dev_yaml = build_split_yaml(metadata, dev_sentences, "dev", dev_stats, source_hash)
    test_yaml = build_split_yaml(metadata, test_sentences, "test", test_stats, source_hash)

    # Write
    dev_path = args.output_dir / "benchmark_dev.yaml"
    test_path = args.output_dir / "benchmark_test.yaml"

    dev_path.write_text(
        yaml.dump(dev_yaml, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    test_path.write_text(
        yaml.dump(test_yaml, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    print(f"\n  Saved: {dev_path}")
    print(f"  Saved: {test_path}")
    print(f"\n  Source hash: {source_hash}")
    print("  IMPORTANT: Never tune thresholds on the test split!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
