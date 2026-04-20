"""ByT5 training data preparation for Myanmar spell correction.

Extracts and formats training data from three sources:
1. Benchmark gold (error_sentence, corrected_sentence) pairs
2. Existing reranker training data (sentence → corrupted pairs)
3. SyntheticErrorGenerator on clean corpus text

All pairs are formatted as T5-style seq2seq:
    Input:  "correct: <error_text>"
    Output: "<corrected_text>"

Usage:
    python -m myspellchecker.training.byt5_data \
        --benchmark benchmarks/myspellchecker_benchmark.yaml \
        --reranker-data data/reranker_training_combined_120k.jsonl \
        --output data/byt5_training \
        --max-reranker 50000 \
        --identity-ratio 0.20

    # Or just extract from benchmark gold:
    python -m myspellchecker.training.byt5_data \
        --benchmark benchmarks/myspellchecker_benchmark.yaml \
        --output data/byt5_training \
        --benchmark-only
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

# T5-style prefix for the correction task.
TASK_PREFIX = "correct: "


def _myanmar_char_ratio(text: str) -> float:
    """Return the fraction of non-whitespace characters that are Myanmar script.

    Myanmar primary block: U+1000-U+109F (consonants, vowels, digits, punctuation).
    """
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    myanmar_count = sum(1 for c in non_ws if "\u1000" <= c <= "\u109f")
    return myanmar_count / len(non_ws)


def _normalize_spaces(text: str) -> str:
    """Strip inter-word spaces from Myanmar text.

    Myanmar doesn't use spaces for word boundaries in standard orthography.
    Keeps spaces around non-Myanmar content (numbers, Latin text).
    """
    return text.replace(" ", "")


def extract_benchmark_gold(benchmark_path: Path) -> list[dict]:
    """Extract (error_sentence, corrected_sentence) pairs from benchmark YAML.

    For each sentence with errors, applies gold corrections to produce
    the corrected version. Returns both the error pair and (for clean
    sentences) identity pairs.
    """
    with open(benchmark_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    pairs = []
    for item in data.get("sentences", []):
        input_text = item["input"]
        expected = item.get("expected_errors", [])
        is_clean = item.get("is_clean", False)

        if is_clean:
            # Identity pair: clean sentence maps to itself
            pairs.append(
                {
                    "input": TASK_PREFIX + input_text,
                    "output": input_text,
                    "source": "benchmark_clean",
                    "error_type": "identity",
                }
            )
            continue

        if not expected:
            continue

        # Apply gold corrections to produce corrected sentence
        # Sort by span start (reverse) to avoid position shifts
        corrections = []
        for err in expected:
            span = err.get("span", {})
            gold = err.get("gold_correction")
            if not gold or not span:
                continue
            corrections.append((span["start"], span["end"], gold))

        if not corrections:
            continue

        # Validate no overlapping spans (skip if found)
        sorted_corrections = sorted(corrections, key=lambda x: x[0])
        has_overlap = False
        for i in range(len(sorted_corrections) - 1):
            if sorted_corrections[i][1] > sorted_corrections[i + 1][0]:
                # Check if one is a no-op
                s1, e1, g1 = sorted_corrections[i]
                s2, e2, g2 = sorted_corrections[i + 1]
                if input_text[s1:e1] != g1 and input_text[s2:e2] != g2:
                    has_overlap = True
                    break
        if has_overlap:
            continue

        # Apply corrections from right to left
        corrected = input_text
        for start, end, gold in sorted(corrections, key=lambda x: -x[0]):
            corrected = corrected[:start] + gold + corrected[end:]

        pairs.append(
            {
                "input": TASK_PREFIX + input_text,
                "output": corrected,
                "source": "benchmark_gold",
                "error_type": "gold_correction",
            }
        )

    return pairs


def extract_reranker_pairs(
    reranker_path: Path,
    max_pairs: int = 50_000,
    min_myanmar_ratio: float = 0.8,
    min_length_ratio: float = 0.7,
    max_byte_length: int = 512,
    seed: int = 42,
) -> list[dict]:
    """Extract (corrupted_sentence, clean_sentence) pairs from reranker JSONL.

    The reranker data has 'sentence' (clean) and 'corrupted_sentence' (error)
    fields, plus 'error_type' labels. We reverse the direction for correction:
    input = corrupted, output = clean.

    Filters applied:
    - Reject pairs where target is <min_myanmar_ratio Myanmar characters (B1)
    - Normalize spaces in both input and target (B2)
    - Reject pairs where corrupted/clean length ratio < min_length_ratio (W3)
    - Reject pairs exceeding max_byte_length (W4)
    """
    all_entries = []
    seen_inputs = set()
    filtered_stats = {"non_myanmar": 0, "length_ratio": 0, "too_long": 0, "space_only": 0}

    with open(reranker_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            clean = entry.get("sentence", "").strip()
            corrupted = entry.get("corrupted_sentence", "").strip()
            error_type = entry.get("error_type", "unknown")

            if not clean or not corrupted or clean == corrupted:
                continue

            # B1: Filter non-Myanmar content in targets
            if _myanmar_char_ratio(clean) < min_myanmar_ratio:
                filtered_stats["non_myanmar"] += 1
                continue

            # B2: Normalize spaces — strip inter-word spaces from both sides
            clean_norm = _normalize_spaces(clean)
            corrupted_norm = _normalize_spaces(corrupted)

            # Skip if the only difference was spaces
            if clean_norm == corrupted_norm:
                filtered_stats["space_only"] += 1
                continue

            # W3: Filter severely truncated pairs (corruption removed too much)
            if len(corrupted_norm) / max(len(clean_norm), 1) < min_length_ratio:
                filtered_stats["length_ratio"] += 1
                continue

            # W4: Filter pairs that exceed max byte length (avoid truncation)
            input_text = TASK_PREFIX + corrupted_norm
            if (
                len(input_text.encode("utf-8")) > max_byte_length
                or len(clean_norm.encode("utf-8")) > max_byte_length
            ):
                filtered_stats["too_long"] += 1
                continue

            # Dedup by normalized corrupted input
            if corrupted_norm in seen_inputs:
                continue
            seen_inputs.add(corrupted_norm)

            all_entries.append(
                {
                    "input": input_text,
                    "output": clean_norm,
                    "source": "reranker",
                    "error_type": error_type,
                }
            )

    print(f"  Reranker filtering: {filtered_stats}")

    # Sample if we have more than requested
    rng = random.Random(seed)
    if len(all_entries) > max_pairs:
        all_entries = rng.sample(all_entries, max_pairs)

    return all_entries


def generate_identity_pairs(
    reranker_path: Path,
    count: int = 15_000,
    min_myanmar_ratio: float = 0.8,
    max_byte_length: int = 512,
    seed: int = 42,
) -> list[dict]:
    """Generate identity pairs (clean → clean) from reranker clean sentences.

    These teach the model that most text is correct and should not be modified.
    Target ~20-25% of total dataset to prevent over-correction.
    """
    clean_sents: set[str] = set()

    with open(reranker_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            clean = entry.get("sentence", "").strip()
            if not clean or len(clean) < 10:
                continue
            # Apply same filters as reranker pairs
            if _myanmar_char_ratio(clean) < min_myanmar_ratio:
                continue
            norm = _normalize_spaces(clean)
            input_text = TASK_PREFIX + norm
            if len(input_text.encode("utf-8")) > max_byte_length:
                continue
            clean_sents.add(norm)

    rng = random.Random(seed)
    sampled = rng.sample(sorted(clean_sents), min(count, len(clean_sents)))

    return [
        {
            "input": TASK_PREFIX + s,
            "output": s,
            "source": "identity",
            "error_type": "identity",
        }
        for s in sampled
    ]


def split_data(
    pairs: list[dict],
    train_ratio: float = 0.9,
    dev_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split pairs into train/dev/test sets, stratified by source.

    Benchmark gold/clean pairs go entirely into train (W6: avoid leakage).
    Reranker and identity pairs are split randomly.
    """
    # Separate benchmark pairs (always in train)
    benchmark_pairs = [p for p in pairs if p["source"] in ("benchmark_gold", "benchmark_clean")]
    other_pairs = [p for p in pairs if p["source"] not in ("benchmark_gold", "benchmark_clean")]

    rng = random.Random(seed)
    rng.shuffle(other_pairs)

    n = len(other_pairs)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    train = benchmark_pairs + other_pairs[:train_end]
    dev = other_pairs[train_end:dev_end]
    test = other_pairs[dev_end:]

    # Shuffle train so benchmark pairs aren't all at the start
    rng.shuffle(train)

    return train, dev, test


def write_jsonl(pairs: list[dict], path: Path) -> None:
    """Write pairs to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def print_stats(pairs: list[dict], label: str) -> None:
    """Print distribution stats for a set of pairs."""
    sources = {}
    error_types = {}
    for p in pairs:
        sources[p["source"]] = sources.get(p["source"], 0) + 1
        error_types[p["error_type"]] = error_types.get(p["error_type"], 0) + 1

    print(f"\n  {label}: {len(pairs)} pairs")
    print(f"  Sources: {dict(sorted(sources.items(), key=lambda x: -x[1]))}")
    top_types = sorted(error_types.items(), key=lambda x: -x[1])[:10]
    print(f"  Top error types: {dict(top_types)}")

    # Length stats
    input_lens = [len(p["input"].encode("utf-8")) for p in pairs]
    if input_lens:
        print(
            f"  Input byte lengths: min={min(input_lens)}, "
            f"mean={sum(input_lens) // len(input_lens)}, "
            f"max={max(input_lens)}"
        )

    # Identity ratio
    identity_count = sum(1 for p in pairs if p["error_type"] == "identity")
    if pairs:
        print(
            f"  Identity ratio: {identity_count}/{len(pairs)} "
            f"({identity_count / len(pairs) * 100:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ByT5 training data for Myanmar spell correction"
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/myspellchecker_benchmark.yaml"),
        help="Path to benchmark YAML",
    )
    parser.add_argument(
        "--reranker-data",
        type=Path,
        default=Path("data/reranker_training_combined_120k.jsonl"),
        help="Path to reranker training JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/byt5_training"),
        help="Output directory for train/dev/test JSONL",
    )
    parser.add_argument(
        "--max-reranker",
        type=int,
        default=50_000,
        help="Max pairs to extract from reranker data",
    )
    parser.add_argument(
        "--identity-count",
        type=int,
        default=15_000,
        help="Number of identity (clean→clean) pairs (~20-25%% of dataset)",
    )
    parser.add_argument(
        "--min-myanmar-ratio",
        type=float,
        default=0.8,
        help="Minimum Myanmar character ratio in targets (filters web artifacts)",
    )
    parser.add_argument(
        "--max-byte-length",
        type=int,
        default=512,
        help="Max byte length for input/output (pairs exceeding this are dropped)",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only extract from benchmark (no reranker data)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("  ByT5 Training Data Preparation")
    print("=" * 60)

    all_pairs: list[dict] = []

    # 1. Benchmark gold pairs
    if args.benchmark.exists():
        gold_pairs = extract_benchmark_gold(args.benchmark)
        print_stats(gold_pairs, "Benchmark gold")
        all_pairs.extend(gold_pairs)
    else:
        print(f"  WARNING: Benchmark not found: {args.benchmark}")

    if not args.benchmark_only:
        # 2. Reranker correction pairs
        if args.reranker_data.exists():
            reranker_pairs = extract_reranker_pairs(
                args.reranker_data,
                max_pairs=args.max_reranker,
                min_myanmar_ratio=args.min_myanmar_ratio,
                max_byte_length=args.max_byte_length,
                seed=args.seed,
            )
            print_stats(reranker_pairs, "Reranker pairs")
            all_pairs.extend(reranker_pairs)

            # 3. Identity pairs from reranker clean sentences
            identity_pairs = generate_identity_pairs(
                args.reranker_data,
                count=args.identity_count,
                min_myanmar_ratio=args.min_myanmar_ratio,
                max_byte_length=args.max_byte_length,
                seed=args.seed,
            )
            print_stats(identity_pairs, "Identity pairs")
            all_pairs.extend(identity_pairs)
        else:
            print(f"  WARNING: Reranker data not found: {args.reranker_data}")

    # 4. Split into train/dev/test (stratified: benchmark in train only)
    train, dev, test = split_data(all_pairs, seed=args.seed)

    print_stats(train, "TRAIN")
    print_stats(dev, "DEV")
    print_stats(test, "TEST")

    # 5. Write output
    write_jsonl(train, args.output / "train.jsonl")
    write_jsonl(dev, args.output / "dev.jsonl")
    write_jsonl(test, args.output / "test.jsonl")

    # Also write a config summary
    config = {
        "task_prefix": TASK_PREFIX,
        "total_pairs": len(all_pairs),
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "sources": {
            "benchmark_gold": sum(1 for p in all_pairs if p["source"] == "benchmark_gold"),
            "benchmark_clean": sum(1 for p in all_pairs if p["source"] == "benchmark_clean"),
            "reranker": sum(1 for p in all_pairs if p["source"] == "reranker"),
            "identity": sum(1 for p in all_pairs if p["source"] == "identity"),
        },
        "filters": {
            "min_myanmar_ratio": args.min_myanmar_ratio,
            "max_byte_length": args.max_byte_length,
        },
        "seed": args.seed,
    }
    with open(args.output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Output written to: {args.output}/")
    print(
        f"  Files: train.jsonl ({len(train)}), dev.jsonl ({len(dev)}), "
        f"test.jsonl ({len(test)}), config.json"
    )


if __name__ == "__main__":
    main()
