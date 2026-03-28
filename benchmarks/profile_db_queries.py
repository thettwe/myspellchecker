#!/usr/bin/env python3
"""
DB Query Profiler for Validation Pipeline.

Instruments SQLiteProvider to count and time every database call per sentence
and per strategy. Identifies redundant queries, cache misses, and N+1 patterns.

Usage:
    source venv/bin/activate
    python benchmarks/profile_db_queries.py \
        --db "/Volumes/External 8TB/myspellchecker/comparison/mySpellChecker_production.db"
"""

import argparse
import functools
import json
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class QueryRecord:
    """Single DB query record."""

    method: str
    args: tuple
    duration_ms: float
    cache_hit: bool
    caller: str  # Shortened caller stack (strategy name)


@dataclass
class SentenceProfile:
    """Profile data for one benchmark sentence."""

    sentence_id: str
    input_text: str
    total_ms: float = 0.0
    queries: list[QueryRecord] = field(default_factory=list)
    strategy_timings: dict[str, float] = field(default_factory=dict)

    @property
    def query_count(self) -> int:
        return len(self.queries)

    @property
    def db_time_ms(self) -> float:
        return sum(q.duration_ms for q in self.queries)

    @property
    def cache_hits(self) -> int:
        return sum(1 for q in self.queries if q.cache_hit)


# ── Provider wrapper ─────────────────────────────────────────────────────────

# Methods to instrument on SQLiteProvider
TRACKED_METHODS = [
    "is_valid_syllable",
    "is_valid_word",
    "get_word_id",
    "get_word_frequency",
    "get_syllable_frequency",
    "get_bigram_probability",
    "get_trigram_probability",
    "get_word_pos",
    "is_valid_vocabulary",
    "get_top_continuations",
    "get_pos_unigram_probabilities",
    "get_pos_bigram_probabilities",
    "get_pos_trigram_probabilities",
    "get_neighbors",
    "get_words_by_edit_distance",
    "is_valid_syllables_bulk",
    "is_valid_words_bulk",
    "get_syllable_frequencies_bulk",
    "get_word_frequencies_bulk",
]


class QueryProfiler:
    """Wraps a SQLiteProvider to record all query calls."""

    def __init__(self, provider):
        self.provider = provider
        self.recording = False
        self.current_queries: list[QueryRecord] = []
        self._originals: dict[str, Any] = {}
        self._install_hooks()

    def _install_hooks(self):
        """Monkey-patch tracked methods on the provider instance."""
        for method_name in TRACKED_METHODS:
            original = getattr(self.provider, method_name, None)
            if original is None:
                continue
            self._originals[method_name] = original
            wrapped = self._make_wrapper(method_name, original)
            setattr(self.provider, method_name, wrapped)

    def _make_wrapper(self, method_name, original):
        profiler = self

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            if not profiler.recording:
                return original(*args, **kwargs)

            # Detect caller (walk stack to find strategy class)
            caller = _get_caller_strategy()

            t0 = time.perf_counter()
            result = original(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Heuristic: if call took <0.01ms, it was likely a cache hit
            cache_hit = elapsed_ms < 0.01

            record = QueryRecord(
                method=method_name,
                args=args,
                duration_ms=elapsed_ms,
                cache_hit=cache_hit,
                caller=caller,
            )
            profiler.current_queries.append(record)
            return result

        return wrapper

    def start_recording(self):
        self.recording = True
        self.current_queries = []

    def stop_recording(self) -> list[QueryRecord]:
        self.recording = False
        queries = self.current_queries
        self.current_queries = []
        return queries

    def restore(self):
        """Remove monkey-patches."""
        for method_name, original in self._originals.items():
            setattr(self.provider, method_name, original)


def _get_caller_strategy() -> str:
    """Walk the call stack to find the most specific calling module.

    We walk from the innermost frame outward but collect ALL matches,
    then return the most specific one. This avoids the problem where
    spellchecker.py (always on the stack) masks the actual caller.
    """
    # Priority order: most specific first
    best = "unknown"
    best_priority = 999

    # Map filename patterns to (label, priority) — lower priority = more specific
    patterns = [
        ("compound_resolver", "compound_resolver", 1),
        ("reduplication", "reduplication", 1),
        ("morphology", "morphology", 2),
        ("symspell", "symspell", 2),
        ("validation_strategies/", None, 3),  # Special: use filename stem
        ("algorithms/", None, 4),  # Special: use filename stem
        ("validators.py", "validators", 5),
        ("spellchecker.py", "spellchecker", 10),
    ]

    for frame_info in traceback.extract_stack():
        filename = frame_info.filename
        for pattern, label, priority in patterns:
            if pattern in filename and priority < best_priority:
                if label is None:
                    # Use filename stem with prefix
                    stem = Path(filename).stem
                    prefix = "strategy" if "validation_strategies" in pattern else "algo"
                    best = f"{prefix}:{stem}"
                else:
                    best = label
                best_priority = priority

    return best


# ── Analysis ─────────────────────────────────────────────────────────────────


def analyze_profiles(profiles: list[SentenceProfile]) -> dict:
    """Analyze all sentence profiles for redundancy patterns."""

    # Global stats
    total_queries = sum(p.query_count for p in profiles)
    total_db_time = sum(p.db_time_ms for p in profiles)
    total_cache_hits = sum(p.cache_hits for p in profiles)

    # Per-method stats
    method_counts: dict[str, int] = defaultdict(int)
    method_times: dict[str, float] = defaultdict(float)
    method_cache_hits: dict[str, int] = defaultdict(int)

    # Per-caller stats
    caller_counts: dict[str, int] = defaultdict(int)
    caller_times: dict[str, float] = defaultdict(float)

    # Redundancy detection: same (method, args) called multiple times per sentence
    per_sentence_redundancy: list[dict] = []

    # Cross-strategy duplication: same query made by different callers
    cross_strategy_dupes: dict[str, set] = defaultdict(set)

    for profile in profiles:
        # Track (method, args) -> [callers] within this sentence
        call_map: dict[tuple, list[str]] = defaultdict(list)

        for q in profile.queries:
            method_counts[q.method] += 1
            method_times[q.method] += q.duration_ms
            if q.cache_hit:
                method_cache_hits[q.method] += 1

            caller_counts[q.caller] += 1
            caller_times[q.caller] += q.duration_ms

            # Build call signature (use repr for guaranteed hashability)
            sig = (q.method, repr(q.args))
            call_map[sig].append(q.caller)

        # Find redundant calls within this sentence
        redundant = {}
        for sig, callers in call_map.items():
            if len(callers) > 1:
                method, args = sig
                key = f"{method}({_truncate_args(args)})"
                redundant[key] = {
                    "count": len(callers),
                    "callers": callers,
                }
                # Track cross-strategy duplication
                unique_callers = set(callers)
                if len(unique_callers) > 1:
                    cross_strategy_dupes[method].update(unique_callers)

        if redundant:
            per_sentence_redundancy.append(
                {
                    "sentence_id": profile.sentence_id,
                    "redundant_calls": redundant,
                    "redundant_count": sum(v["count"] - 1 for v in redundant.values()),
                }
            )

    # Sort methods by total time (descending)
    method_summary = []
    for method in sorted(method_times, key=method_times.get, reverse=True):
        method_summary.append(
            {
                "method": method,
                "calls": method_counts[method],
                "total_ms": round(method_times[method], 2),
                "avg_ms": round(method_times[method] / method_counts[method], 4),
                "cache_hits": method_cache_hits[method],
                "cache_rate": round(method_cache_hits[method] / method_counts[method] * 100, 1)
                if method_counts[method] > 0
                else 0,
            }
        )

    # Sort callers by total time
    caller_summary = []
    for caller in sorted(caller_times, key=caller_times.get, reverse=True):
        caller_summary.append(
            {
                "caller": caller,
                "calls": caller_counts[caller],
                "total_ms": round(caller_times[caller], 2),
                "avg_ms": round(caller_times[caller] / caller_counts[caller], 4),
            }
        )

    # Aggregate strategy timings from ContextValidator
    strategy_timing_agg: dict[str, float] = defaultdict(float)
    for p in profiles:
        for name, t in p.strategy_timings.items():
            strategy_timing_agg[name] += t

    # Top redundant sentences
    top_redundant = sorted(
        per_sentence_redundancy,
        key=lambda x: x["redundant_count"],
        reverse=True,
    )[:10]

    return {
        "summary": {
            "sentences_profiled": len(profiles),
            "total_queries": total_queries,
            "total_db_time_ms": round(total_db_time, 2),
            "total_cache_hits": total_cache_hits,
            "cache_hit_rate": round(total_cache_hits / total_queries * 100, 1)
            if total_queries > 0
            else 0,
            "avg_queries_per_sentence": round(total_queries / len(profiles), 1) if profiles else 0,
            "avg_db_time_per_sentence_ms": round(total_db_time / len(profiles), 2)
            if profiles
            else 0,
        },
        "by_method": method_summary,
        "by_caller": caller_summary,
        "strategy_timings_ms": {
            k: round(v * 1000, 2)
            for k, v in sorted(strategy_timing_agg.items(), key=lambda x: x[1], reverse=True)
        },
        "cross_strategy_duplication": {
            method: sorted(callers) for method, callers in cross_strategy_dupes.items()
        },
        "top_redundant_sentences": top_redundant,
    }


def _truncate_args(args: tuple) -> str:
    """Truncate args for readable display."""
    parts = []
    for a in args:
        s = repr(a)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(s)
    return ", ".join(parts)


# ── Printing ─────────────────────────────────────────────────────────────────


def print_report(analysis: dict):
    """Print human-readable profiling report."""
    s = analysis["summary"]
    print("\n" + "=" * 70)
    print("  DB QUERY PROFILE REPORT")
    print("=" * 70)

    print(f"\n  Sentences profiled:       {s['sentences_profiled']}")
    print(f"  Total DB queries:         {s['total_queries']}")
    print(f"  Total DB time:            {s['total_db_time_ms']:.1f} ms")
    print(f"  Cache hit rate:           {s['cache_hit_rate']:.1f}%")
    print(f"  Avg queries/sentence:     {s['avg_queries_per_sentence']:.1f}")
    print(f"  Avg DB time/sentence:     {s['avg_db_time_per_sentence_ms']:.2f} ms")

    print("\n── By Method (sorted by total time) ──────────────────────────────")
    print(f"  {'Method':<35} {'Calls':>7} {'Total ms':>10} {'Avg ms':>8} {'Cache%':>7}")
    print(f"  {'─' * 35} {'─' * 7} {'─' * 10} {'─' * 8} {'─' * 7}")
    for m in analysis["by_method"]:
        print(
            f"  {m['method']:<35} {m['calls']:>7} {m['total_ms']:>10.2f} "
            f"{m['avg_ms']:>8.4f} {m['cache_rate']:>6.1f}%"
        )

    print("\n── By Caller (sorted by total time) ──────────────────────────────")
    print(f"  {'Caller':<35} {'Calls':>7} {'Total ms':>10} {'Avg ms':>8}")
    print(f"  {'─' * 35} {'─' * 7} {'─' * 10} {'─' * 8}")
    for c in analysis["by_caller"]:
        print(f"  {c['caller']:<35} {c['calls']:>7} {c['total_ms']:>10.2f} {c['avg_ms']:>8.4f}")

    if analysis["strategy_timings_ms"]:
        print("\n── Strategy Timings (wall clock, ms) ─────────────────────────────")
        for name, ms in analysis["strategy_timings_ms"].items():
            print(f"  {name:<45} {ms:>10.2f} ms")

    if analysis["cross_strategy_duplication"]:
        print("\n── Cross-Strategy Duplication ─────────────────────────────────────")
        print("  Same method called by DIFFERENT strategies on same data:")
        for method, callers in analysis["cross_strategy_duplication"].items():
            print(f"  {method}: {', '.join(callers)}")

    if analysis["top_redundant_sentences"]:
        print("\n── Top Redundant Sentences ────────────────────────────────────────")
        for entry in analysis["top_redundant_sentences"][:5]:
            print(f"\n  {entry['sentence_id']} (+{entry['redundant_count']} redundant calls)")
            for call_key, info in list(entry["redundant_calls"].items())[:5]:
                print(f"    {call_key}: {info['count']}x by {info['callers']}")

    print("\n" + "=" * 70)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Profile DB queries in validation pipeline")
    parser.add_argument("--db", required=True, help="Path to spell checker database")
    parser.add_argument(
        "--benchmark",
        default="benchmarks/myspellchecker_benchmark.yaml",
        help="Path to benchmark YAML",
    )
    parser.add_argument("--output", help="Save JSON report to file")
    parser.add_argument("--limit", type=int, help="Only profile first N sentences")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs")
    args = parser.parse_args()

    # Imports
    from myspellchecker import SpellChecker
    from myspellchecker.core.config.main import SpellCheckerConfig
    from myspellchecker.core.constants import ValidationLevel
    from myspellchecker.providers.sqlite import SQLiteProvider

    # Load benchmark
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"Error: Benchmark file not found: {benchmark_path}", file=sys.stderr)
        sys.exit(1)

    with open(benchmark_path) as f:
        benchmark = yaml.safe_load(f)
    sentences = benchmark["sentences"]

    if args.limit:
        sentences = sentences[: args.limit]

    # Initialize
    db_path = Path(args.db)
    print(f"Database: {db_path}")
    print(f"Sentences: {len(sentences)}")

    config = SpellCheckerConfig()
    # Enable strategy timing
    config.validation.enable_strategy_timing = True

    provider = SQLiteProvider(database_path=str(db_path))
    profiler = QueryProfiler(provider)

    checker = SpellChecker(config=config, provider=provider)
    val_level = ValidationLevel.WORD

    # Warmup
    print(f"Warming up ({args.warmup} runs)...")
    warmup_text = "ကျွန်တော် ကျန်းမာပါတယ်"
    for _ in range(args.warmup):
        checker.check(warmup_text, level=val_level)

    # Clear caches after warmup to get clean profile
    provider.clear_caches()

    # Profile each sentence
    profiles: list[SentenceProfile] = []
    print(f"Profiling {len(sentences)} sentences...")

    for i, sentence_data in enumerate(sentences):
        sid = sentence_data["id"]
        input_text = sentence_data["input"]

        # Reset strategy timings
        if hasattr(checker, "context_validator") and checker.context_validator:
            checker.context_validator.strategy_timings.clear()

        profiler.start_recording()
        t0 = time.perf_counter()
        checker.check(input_text, level=val_level)
        total_ms = (time.perf_counter() - t0) * 1000
        queries = profiler.stop_recording()

        # Collect strategy timings
        strategy_timings = {}
        if hasattr(checker, "context_validator") and checker.context_validator:
            strategy_timings = dict(checker.context_validator.strategy_timings)

        profile = SentenceProfile(
            sentence_id=sid,
            input_text=input_text,
            total_ms=total_ms,
            queries=queries,
            strategy_timings=strategy_timings,
        )
        profiles.append(profile)

        if (i + 1) % 10 == 0:
            print(
                f"  [{i + 1}/{len(sentences)}] {sid}: {profile.query_count} queries, "
                f"{total_ms:.1f}ms total, {profile.db_time_ms:.1f}ms DB"
            )

    # Analyze
    analysis = analyze_profiles(profiles)

    # Print cache stats from provider
    print("\n── Provider Cache Stats ───────────────────────────────────────────")
    cache_stats = provider.get_cache_stats()
    for cache_name, stats in cache_stats.items():
        print(f"  {cache_name}: {stats}")

    # Print report
    print_report(analysis)

    # Save JSON
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\nJSON report saved to: {output_path}")

    profiler.restore()
    return analysis


if __name__ == "__main__":
    main()
