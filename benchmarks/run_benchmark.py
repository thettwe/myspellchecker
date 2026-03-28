#!/usr/bin/env python3
"""
Myanmar Spell Checker Benchmark Runner v1.0

Runs the benchmark suite against the spell checker and produces accuracy metrics.
This is a reference benchmark — not a CI gate — to document detection and suggestion
accuracy rates with a specific database version.

Usage:
    python benchmarks/run_benchmark.py --db data/myspellchecker.db
    python benchmarks/run_benchmark.py --db data/myspellchecker_large.db --level word
    python benchmarks/run_benchmark.py --db data/myspellchecker.db --output benchmarks/results/
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class SpanMatch:
    """Result of matching a system detection to a gold error span."""

    gold_error_id: str
    gold_span: tuple[int, int]
    gold_correction: Optional[str]
    system_text: str
    system_position: int
    system_suggestions: list[str]
    system_error_type: str
    matched: bool  # Did system detect this error?
    top1_correct: bool  # Is suggestions[0] the gold correction?
    top3_correct: bool  # Is gold correction in suggestions[:3]?
    top5_correct: bool  # Is gold correction in suggestions[:5]?
    rank_of_correct: Optional[int]  # 1-indexed rank, None if not found


@dataclass
class SentenceResult:
    """Result of running the benchmark on one sentence."""

    sentence_id: str
    input_text: str
    is_clean: bool
    difficulty_tier: Optional[int]
    domain: str
    latency_ms: float
    # Detection
    expected_error_count: int
    detected_error_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    # Suggestion quality
    span_matches: list[SpanMatch] = field(default_factory=list)
    # Raw system output
    system_errors: list[dict] = field(default_factory=list)
    has_system_errors: bool = False


@dataclass
class TierMetrics:
    """Aggregated metrics for a difficulty tier."""

    tier: str
    sentence_count: int = 0
    total_errors: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    top1_correct: int = 0
    top3_correct: int = 0
    top5_correct: int = 0
    mrr_sum: float = 0.0
    detected_with_suggestion: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def top1_accuracy(self) -> float:
        return (
            self.top1_correct / self.detected_with_suggestion
            if self.detected_with_suggestion > 0
            else 0.0
        )

    @property
    def top3_accuracy(self) -> float:
        return (
            self.top3_correct / self.detected_with_suggestion
            if self.detected_with_suggestion > 0
            else 0.0
        )

    @property
    def mrr(self) -> float:
        return (
            self.mrr_sum / self.detected_with_suggestion
            if self.detected_with_suggestion > 0
            else 0.0
        )


def load_benchmark(path: Path) -> dict:
    """Load benchmark YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_db_hash(db_path: Path) -> str:
    """Compute SHA256 hash of the database file."""
    h = hashlib.sha256()
    with open(db_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _norm_text(s: str | None) -> str | None:
    """Normalize Myanmar text for comparison (tall AA → regular AA, etc.)."""
    if not s:
        return s
    # Tall AA (U+102B) → regular AA (U+102C) — the library normalizes to regular
    s = s.replace("\u102b", "\u102c")
    # Canonical ordering: aukmyit+asat (U+1037 U+103A) → asat+aukmyit (U+103A U+1037)
    # The library normalizes to asat-before-aukmyit order
    s = s.replace("\u1037\u103a", "\u103a\u1037")
    return s


def match_errors(
    gold_errors: list[dict],
    system_errors: list[dict],
    overlap_threshold: float = 0.3,
) -> tuple[list[SpanMatch], int, int, int]:
    """
    Match system detections to gold error spans.

    Uses position-based matching: a system error matches a gold error if
    the system error position falls within the gold span.

    Returns:
        (span_matches, true_positives, false_positives, false_negatives)
    """
    span_matches = []
    matched_system = set()
    matched_gold = set()

    for gi, gold in enumerate(gold_errors):
        gold_start = gold["span"]["start"]
        gold_end = gold["span"]["end"]
        gold_correction = gold.get("gold_correction")
        best_match = None
        best_overlap = 0

        for si, sys_err in enumerate(system_errors):
            if si in matched_system:
                continue

            sys_pos = sys_err.get("position", -1)
            sys_text = sys_err.get("text", "")
            sys_end = sys_pos + len(sys_text)

            # Calculate overlap
            overlap_start = max(gold_start, sys_pos)
            overlap_end = min(gold_end, sys_end)
            overlap = max(0, overlap_end - overlap_start)
            gold_len = gold_end - gold_start
            sys_len = sys_end - sys_pos

            # Bidirectional overlap: match if overlap covers enough of
            # EITHER the gold span OR the system span. This handles both:
            # - System wider than gold (e.g., compound word containing error)
            # - System narrower than gold (e.g., detecting core error only)
            gold_ratio = overlap / gold_len if gold_len > 0 else 0
            sys_ratio = overlap / sys_len if sys_len > 0 else 0

            if gold_ratio >= overlap_threshold or sys_ratio >= overlap_threshold:
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = si

        if best_match is not None:
            sys_err = system_errors[best_match]
            suggestions = sys_err.get("suggestions", [])
            matched_system.add(best_match)
            matched_gold.add(gi)

            # Check suggestion quality (normalize both sides for comparison)
            norm_gold = _norm_text(gold_correction)
            norm_sugs = [_norm_text(s) for s in suggestions]
            top1 = len(norm_sugs) > 0 and norm_sugs[0] == norm_gold
            top3 = norm_gold in norm_sugs[:3] if norm_gold else False
            top5 = norm_gold in norm_sugs[:5] if norm_gold else False
            rank = None
            if norm_gold and norm_gold in norm_sugs:
                rank = norm_sugs.index(norm_gold) + 1

            span_matches.append(
                SpanMatch(
                    gold_error_id=gold["error_id"],
                    gold_span=(gold_start, gold_end),
                    gold_correction=gold_correction,
                    system_text=sys_err.get("text", ""),
                    system_position=sys_err.get("position", -1),
                    system_suggestions=suggestions[:5],
                    system_error_type=sys_err.get("error_type", ""),
                    matched=True,
                    top1_correct=top1,
                    top3_correct=top3,
                    top5_correct=top5,
                    rank_of_correct=rank,
                )
            )
        else:
            span_matches.append(
                SpanMatch(
                    gold_error_id=gold["error_id"],
                    gold_span=(gold_start, gold_end),
                    gold_correction=gold.get("gold_correction"),
                    system_text="",
                    system_position=-1,
                    system_suggestions=[],
                    system_error_type="",
                    matched=False,
                    top1_correct=False,
                    top3_correct=False,
                    top5_correct=False,
                    rank_of_correct=None,
                )
            )

    tp = len(matched_gold)

    # Second pass: absorb redundant detections.
    # System errors that overlap an already-matched gold span are redundant
    # (same underlying error detected by multiple strategies), not true FPs.
    redundant = set()
    for si, sys_err in enumerate(system_errors):
        if si in matched_system:
            continue
        sys_pos = sys_err.get("position", -1)
        sys_text = sys_err.get("text", "")
        sys_end = sys_pos + len(sys_text)

        for gi in matched_gold:
            gold = gold_errors[gi]
            gs = gold["span"]["start"]
            ge = gold["span"]["end"]
            overlap_start = max(gs, sys_pos)
            overlap_end = min(ge, sys_end)
            overlap = max(0, overlap_end - overlap_start)
            gold_len = ge - gs
            sys_len = sys_end - sys_pos
            gold_ratio = overlap / gold_len if gold_len > 0 else 0
            sys_ratio = overlap / sys_len if sys_len > 0 else 0
            # If this unmatched error overlaps the gold span at all, it is
            # a redundant detection of the same error, not a true FP.
            if gold_ratio >= 0.1 or sys_ratio >= 0.1:
                redundant.add(si)
                break

    # Exclude unmatched informational notes (colloquial_info) from FP count —
    # these are advisory, not error detections
    _INFO_TYPES = {"colloquial_info"}
    unmatched_info = sum(
        1
        for si, sys_err in enumerate(system_errors)
        if si not in matched_system
        and si not in redundant
        and sys_err.get("error_type", "") in _INFO_TYPES
    )
    fp = len(system_errors) - len(matched_system) - len(redundant) - unmatched_info
    fn = len(gold_errors) - len(matched_gold)

    return span_matches, tp, fp, fn


def _count_occurrences_before(text: str, token: str, end_pos: int) -> int:
    """Count token occurrences before end_pos for stable MLM masking occurrence index."""
    if not text or not token or end_pos <= 0:
        return 0
    count = 0
    start = 0
    while True:
        idx = text.find(token, start)
        if idx == -1 or idx >= end_pos:
            break
        count += 1
        start = idx + len(token)
    return count


def _is_token_boundary_hidden(target_text: str, words: list[str]) -> bool:
    """Return True when target span appears only as a substring inside merged tokens."""
    if not target_text:
        return False
    if target_text in words:
        return False
    for word in words:
        if not word or len(word) <= len(target_text):
            continue
        if target_text in word:
            return True
    return False


def classify_false_negative_reason(
    *,
    checker: Any,
    input_text: str,
    gold_error: dict[str, Any],
    segmented_words: list[str],
) -> str:
    """Classify likely cause for one false negative."""
    span = gold_error.get("span", {})
    start = int(span.get("start", -1))
    end = int(span.get("end", -1))
    if start < 0 or end <= start or end > len(input_text):
        return "candidate_not_generated"

    target_text = input_text[start:end]
    if _is_token_boundary_hidden(target_text, segmented_words):
        return "token_boundary_hidden"

    semantic_checker = getattr(checker, "semantic_checker", None)
    if semantic_checker is None:
        return "candidate_not_generated"

    occurrence = _count_occurrences_before(input_text, target_text, start)
    try:
        predictions = semantic_checker.predict_mask(
            input_text,
            target_text,
            top_k=15,
            occurrence=occurrence,
        )
    except Exception:
        return "candidate_not_generated"

    if not predictions:
        return "candidate_not_generated"

    pred_map = {word: score for word, score in predictions}
    top5_words = [word for word, _score in predictions[:5]]

    skip_due_to_prefix = False
    if hasattr(semantic_checker, "_should_skip_due_to_prefix_evidence"):
        try:
            skip_due_to_prefix = bool(
                semantic_checker._should_skip_due_to_prefix_evidence(
                    target_text,
                    predictions,
                    top_n=min(5, len(predictions)),
                )
            )
        except Exception:
            skip_due_to_prefix = False
    if skip_due_to_prefix and target_text not in top5_words:
        return "semantic_prefix_suppressed"

    current_score = float(pred_map.get(target_text, min(pred_map.values())))
    gold_correction = gold_error.get("gold_correction")
    if isinstance(gold_correction, str) and gold_correction:
        candidate_score = pred_map.get(gold_correction)
        if candidate_score is None:
            return "candidate_not_generated"
        if float(candidate_score) - current_score < 1.2:
            return "candidate_generated_below_margin"

    try:
        top_score = float(predictions[0][1])
        logit_scale = float(semantic_checker._get_model_logit_scale())
        confidence = float(semantic_checker._calibrate_confidence(top_score, logit_scale))
        proactive_threshold = float(
            getattr(
                getattr(checker.config, "semantic", object()),
                "proactive_confidence_threshold",
                0.85,
            )
        )
        if confidence < proactive_threshold:
            return "semantic_low_confidence"
    except Exception:
        return "candidate_not_generated"

    return "candidate_not_generated"


def merge_strategy_debug_telemetry(
    aggregate: dict[str, Any],
    check_telemetry: dict[str, Any],
    sentence_id: str,
    input_text: str,
    sample_limit: int = 40,
) -> None:
    """Merge one check's strategy debug telemetry into benchmark-level aggregate."""
    if not isinstance(check_telemetry, dict):
        return

    check_strategies = check_telemetry.get("strategies", {})
    if not isinstance(check_strategies, dict):
        return

    aggregate["enabled"] = True
    aggregate_strategies = aggregate.setdefault("strategies", {})
    if not isinstance(aggregate_strategies, dict):
        return

    numeric_fields = (
        "calls",
        "emitted",
        "new_positions",
        "overlap_emitted",
        "shadow_potential_positions",
        "overlap_blocked_positions",
    )

    for strategy_name, stats in check_strategies.items():
        if not isinstance(strategy_name, str) or not isinstance(stats, dict):
            continue

        entry = aggregate_strategies.setdefault(
            strategy_name,
            {
                "calls": 0,
                "emitted": 0,
                "new_positions": 0,
                "overlap_emitted": 0,
                "shadow_potential_positions": 0,
                "overlap_blocked_positions": 0,
                "overlap_blocked_by_type": {},
                "overlap_blocked_examples": [],
            },
        )
        if not isinstance(entry, dict):
            continue

        for metric_name in numeric_fields:
            entry[metric_name] = int(entry.get(metric_name, 0)) + int(stats.get(metric_name, 0))

        agg_by_type = entry.setdefault("overlap_blocked_by_type", {})
        cur_by_type = stats.get("overlap_blocked_by_type", {})
        if isinstance(agg_by_type, dict) and isinstance(cur_by_type, dict):
            for error_type, count in cur_by_type.items():
                if not isinstance(error_type, str):
                    continue
                agg_by_type[error_type] = int(agg_by_type.get(error_type, 0)) + int(count)

        agg_examples = entry.setdefault("overlap_blocked_examples", [])
        cur_examples = stats.get("overlap_blocked_examples", [])
        if isinstance(agg_examples, list) and isinstance(cur_examples, list):
            for example in cur_examples:
                if len(agg_examples) >= sample_limit:
                    break
                if not isinstance(example, dict):
                    continue
                enriched = dict(example)
                enriched["sentence_id"] = sentence_id
                enriched["input_text"] = input_text
                agg_examples.append(enriched)


def run_benchmark(
    benchmark_path: Path,
    db_path: Path,
    level: str = "word",
    warmup: int = 3,
    semantic_path: Optional[Path] = None,
    enable_ner: bool = False,
    targeted_rerank_hints: bool = True,
    targeted_candidate_injections: bool = True,
    targeted_grammar_completion_templates: bool = True,
    proactive_threshold: float = 0.85,
    enable_proactive: bool = False,
    no_confusable_semantic: bool = False,
    confusable_preset: str = "relaxed",
    enable_strategy_debug: bool = False,
    reranker_path: Optional[Path] = None,
) -> dict:
    """
    Run the full benchmark suite.

    Args:
        benchmark_path: Path to benchmark YAML file.
        db_path: Path to spell checker database.
        level: Validation level ("syllable" or "word").
        warmup: Number of warmup runs before timing.

        enable_ner: Enable NER-based FP suppression (PER+LOC) using heuristic
            NER with place-name dictionary.
        targeted_rerank_hints: Enable targeted rerank hint rules.
        targeted_candidate_injections: Enable targeted candidate injections.
        targeted_grammar_completion_templates: Enable targeted grammar completion templates.
        enable_strategy_debug: Enable per-strategy gate-debug telemetry collection.

    Returns:
        Complete benchmark results as a dictionary.
    """
    # Import spell checker
    from myspellchecker import SpellChecker
    from myspellchecker.core.config.algorithm_configs import (
        NeuralRerankerConfig,
        RankerConfig,
        SemanticConfig,
    )
    from myspellchecker.core.config.main import SpellCheckerConfig
    from myspellchecker.core.constants import ValidationLevel
    from myspellchecker.providers.sqlite import SQLiteProvider

    # Load benchmark data
    benchmark = load_benchmark(benchmark_path)
    sentences = benchmark["sentences"]

    # Build config — wire in detector and/or semantic model when requested
    config_kwargs = {}

    if semantic_path is not None:
        sem_model_file = (
            semantic_path if semantic_path.suffix == ".onnx" else semantic_path / "model.onnx"
        )
        sem_tokenizer_dir = sem_model_file.parent
        if not sem_model_file.exists():
            print(f"Error: Semantic ONNX model not found: {sem_model_file}", file=sys.stderr)
            sys.exit(1)
        print(f"  Semantic model: {sem_model_file}")
        print(f"  Semantic tokenizer: {sem_tokenizer_dir}")
        config_kwargs["semantic"] = SemanticConfig(
            model_path=str(sem_model_file),
            tokenizer_path=str(sem_tokenizer_dir),
            use_proactive_scanning=enable_proactive,
            proactive_confidence_threshold=proactive_threshold,
        )
        print(f"  Proactive scanning: {enable_proactive} (threshold: {proactive_threshold})")
        confusable_label = "disabled" if no_confusable_semantic else "enabled"
        print(f"  Confusable semantic: {confusable_label} (MLM-enhanced)")

    if enable_ner:
        from myspellchecker.text.ner_model import NERConfig

        print("  NER: heuristic (PER+LOC with place-name dictionary)")
        config_kwargs["ner"] = NERConfig(
            enabled=True,
            model_type="heuristic",
            ner_entity_types=["PER", "LOC"],
            loc_confidence_threshold=0.85,
        )

    config_kwargs["ranker"] = RankerConfig(
        enable_targeted_rerank_hints=targeted_rerank_hints,
        enable_targeted_candidate_injections=targeted_candidate_injections,
        enable_targeted_grammar_completion_templates=targeted_grammar_completion_templates,
    )

    if reranker_path is not None:
        model_file = reranker_path / "reranker.onnx"
        stats_file = reranker_path / "reranker.onnx.stats.json"
        if not model_file.exists():
            print(f"Error: Reranker model not found: {model_file}", file=sys.stderr)
            sys.exit(1)
        print(f"  Neural reranker: {model_file}")
        config_kwargs["neural_reranker"] = NeuralRerankerConfig(
            enabled=True,
            model_path=str(model_file),
            stats_path=str(stats_file) if stats_file.exists() else None,
        )

    config = SpellCheckerConfig(**config_kwargs)
    config.validation.enable_strategy_debug = enable_strategy_debug
    if enable_strategy_debug:
        print("  Strategy gate debug: enabled")

    # Confusable semantic detection: only enable when explicitly NOT disabled.
    # Default matches production config (use_confusable_semantic=False).
    # Use --no-confusable-semantic to explicitly disable.
    if semantic_path is not None and not no_confusable_semantic:
        config.validation.use_confusable_semantic = True
        # Apply confusable preset overrides
        if confusable_preset == "conservative":
            config.validation.confusable_semantic_freq_ratio_penalty_high = 3.0
            config.validation.confusable_semantic_freq_ratio_penalty_mid = 1.5
            config.validation.confusable_semantic_sentence_final_penalty = 1.0
            config.validation.confusable_semantic_max_threshold = 0.0
            config.validation.confusable_semantic_reverse_ratio_min_freq = 0
            config.validation.confusable_semantic_visarga_high_freq_hard_block = True
            print("  Confusable preset: conservative (original strict guards)")
        else:
            print("  Confusable preset: relaxed (loosened guards)")

    # Initialize checker with specified database
    provider = SQLiteProvider(database_path=str(db_path))
    checker = SpellChecker(config=config, provider=provider)
    val_level = ValidationLevel.WORD if level == "word" else ValidationLevel.SYLLABLE

    # Warmup runs
    warmup_text = "ကျွန်တော် ကျန်းမာပါတယ်"
    for _ in range(warmup):
        checker.check(warmup_text, level=val_level)

    # Run benchmark
    results: list[SentenceResult] = []
    rerank_rule_telemetry: dict[str, dict[str, int]] = {}
    strategy_debug_telemetry: dict[str, Any] = {"enabled": False, "strategies": {}}
    fn_reason_telemetry: dict[str, Any] = {"histogram": {}, "examples": []}
    total_start = time.perf_counter()

    for sentence in sentences:
        input_text = sentence["input"]
        is_clean = sentence["is_clean"]
        gold_errors = sentence.get("expected_errors", [])

        # Time the check
        start = time.perf_counter()
        response = checker.check(input_text, level=val_level)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Aggregate per-check rerank telemetry when available.
        check_telemetry = response.metadata.get("rerank_rule_telemetry", {})
        if isinstance(check_telemetry, dict):
            for rule_id, stats in check_telemetry.items():
                if not isinstance(rule_id, str) or not isinstance(stats, dict):
                    continue
                agg = rerank_rule_telemetry.setdefault(rule_id, {"fires": 0, "top1_changes": 0})
                agg["fires"] += int(stats.get("fires", 0))
                agg["top1_changes"] += int(stats.get("top1_changes", 0))
        if enable_strategy_debug:
            check_strategy_telemetry = response.metadata.get("strategy_debug_telemetry", {})
            if isinstance(check_strategy_telemetry, dict):
                merge_strategy_debug_telemetry(
                    strategy_debug_telemetry,
                    check_strategy_telemetry,
                    sentence_id=sentence["id"],
                    input_text=input_text,
                )

        # Convert system errors to dicts
        system_errors = [err.to_dict() for err in response.errors]

        if is_clean:
            # Clean sentence — any detection is a false positive
            # (exclude informational notes like colloquial_info)
            fp = sum(1 for e in system_errors if e.get("error_type", "") not in {"colloquial_info"})
            result = SentenceResult(
                sentence_id=sentence["id"],
                input_text=input_text,
                is_clean=True,
                difficulty_tier=sentence.get("difficulty_tier"),
                domain=sentence.get("domain", ""),
                latency_ms=elapsed_ms,
                expected_error_count=0,
                detected_error_count=len(system_errors),
                true_positives=0,
                false_positives=fp,
                false_negatives=0,
                span_matches=[],
                system_errors=system_errors,
                has_system_errors=response.has_errors,
            )
        else:
            # Error sentence — match detections to gold
            span_matches, tp, fp, fn = match_errors(gold_errors, system_errors)

            if fn > 0:
                segmented_words = checker.segmenter.segment_words(input_text)
                gold_by_id = {
                    err.get("error_id"): err for err in gold_errors if isinstance(err, dict)
                }
                for match in span_matches:
                    if match.matched:
                        continue
                    gold = gold_by_id.get(match.gold_error_id)
                    if not isinstance(gold, dict):
                        continue
                    reason = classify_false_negative_reason(
                        checker=checker,
                        input_text=input_text,
                        gold_error=gold,
                        segmented_words=segmented_words,
                    )
                    hist = fn_reason_telemetry.setdefault("histogram", {})
                    if isinstance(hist, dict):
                        hist[reason] = int(hist.get(reason, 0)) + 1
                    examples = fn_reason_telemetry.setdefault("examples", [])
                    if isinstance(examples, list) and len(examples) < 120:
                        span = gold.get("span", {})
                        start = int(span.get("start", -1))
                        end = int(span.get("end", -1))
                        target_text = (
                            input_text[start:end]
                            if start >= 0 and end > start and end <= len(input_text)
                            else ""
                        )
                        examples.append(
                            {
                                "sentence_id": sentence["id"],
                                "error_id": match.gold_error_id,
                                "reason": reason,
                                "target_text": target_text,
                                "gold_correction": gold.get("gold_correction"),
                            }
                        )

            result = SentenceResult(
                sentence_id=sentence["id"],
                input_text=input_text,
                is_clean=False,
                difficulty_tier=sentence.get("difficulty_tier"),
                domain=sentence.get("domain", ""),
                latency_ms=elapsed_ms,
                expected_error_count=len(gold_errors),
                detected_error_count=len(system_errors),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                span_matches=span_matches,
                system_errors=system_errors,
                has_system_errors=response.has_errors,
            )

        results.append(result)

    total_elapsed = (time.perf_counter() - total_start) * 1000
    checker.close()

    # Compute metrics
    report = compute_report(
        results,
        benchmark,
        db_path,
        level,
        total_elapsed,
        rerank_rule_telemetry=rerank_rule_telemetry,
        strategy_debug_telemetry=strategy_debug_telemetry if enable_strategy_debug else None,
        fn_reason_telemetry=fn_reason_telemetry,
    )
    report["config"]["targeted_rerank_hints"] = targeted_rerank_hints
    report["config"]["targeted_candidate_injections"] = targeted_candidate_injections
    report["config"]["targeted_grammar_completion_templates"] = (
        targeted_grammar_completion_templates
    )
    report["config"]["strategy_debug"] = enable_strategy_debug
    return report


def compute_report(
    results: list[SentenceResult],
    benchmark: dict,
    db_path: Path,
    level: str,
    total_elapsed_ms: float,
    rerank_rule_telemetry: dict[str, dict[str, int]] | None = None,
    strategy_debug_telemetry: dict[str, Any] | None = None,
    fn_reason_telemetry: dict[str, Any] | None = None,
) -> dict:
    """Compute all metrics and produce the final report."""

    # --- Overall detection metrics ---
    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)
    total_expected = sum(r.expected_error_count for r in results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- False positive rate on clean sentences ---
    clean_results = [r for r in results if r.is_clean]
    clean_fp_total = sum(r.false_positives for r in clean_results)
    clean_sentences_with_fp = sum(1 for r in clean_results if r.false_positives > 0)
    clean_total = len(clean_results)
    # FPR = fraction of clean sentences that were incorrectly flagged
    fpr = clean_sentences_with_fp / clean_total if clean_total > 0 else 0.0

    # --- Suggestion metrics (across all matched errors) ---
    all_matches = []
    for r in results:
        all_matches.extend(r.span_matches)

    detected_matches = [m for m in all_matches if m.matched]
    # Only count suggestions for errors that have a gold_correction
    suggestion_matches = [m for m in detected_matches if m.gold_correction is not None]

    top1_count = sum(1 for m in suggestion_matches if m.top1_correct)
    top3_count = sum(1 for m in suggestion_matches if m.top3_correct)
    top5_count = sum(1 for m in suggestion_matches if m.top5_correct)
    mrr_sum = sum(1.0 / m.rank_of_correct for m in suggestion_matches if m.rank_of_correct)
    suggestion_total = len(suggestion_matches)

    top1_acc = top1_count / suggestion_total if suggestion_total > 0 else 0.0
    top3_acc = top3_count / suggestion_total if suggestion_total > 0 else 0.0
    top5_acc = top5_count / suggestion_total if suggestion_total > 0 else 0.0
    mrr = mrr_sum / suggestion_total if suggestion_total > 0 else 0.0

    # --- Per-tier metrics ---
    tier_metrics = {}
    for tier_label in [
        "tier_1",
        "tier_2",
        "tier_3",
    ]:
        tier_metrics[tier_label] = TierMetrics(tier=tier_label)

    for r in results:
        if r.is_clean:
            continue
        tier = r.difficulty_tier

        # Use difficulty_tier field directly
        if tier is not None:
            tier_key = f"tier_{tier}"
        else:
            continue

        # Auto-create tier bucket if not pre-defined (handles tier_4+ gracefully)
        if tier_key not in tier_metrics:
            tier_metrics[tier_key] = TierMetrics(tier=tier_key)

        tm = tier_metrics[tier_key]
        tm.sentence_count += 1
        tm.total_errors += r.expected_error_count
        tm.true_positives += r.true_positives
        tm.false_positives += r.false_positives
        tm.false_negatives += r.false_negatives

        for m in r.span_matches:
            if m.matched and m.gold_correction is not None:
                tm.detected_with_suggestion += 1
                if m.top1_correct:
                    tm.top1_correct += 1
                if m.top3_correct:
                    tm.top3_correct += 1
                if m.top5_correct:
                    tm.top5_correct += 1
                if m.rank_of_correct:
                    tm.mrr_sum += 1.0 / m.rank_of_correct

    # --- Latency metrics ---
    latencies = [r.latency_ms for r in results]
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = latencies_sorted[n // 2] if n > 0 else 0.0
    p95 = latencies_sorted[int(n * 0.95)] if n > 0 else 0.0
    p99 = latencies_sorted[int(n * 0.99)] if n > 0 else 0.0

    # --- Composite score ---
    latency_normalized = min(p95 / 500.0, 1.0)
    composite = (
        0.30 * f1
        + 0.25 * mrr
        + 0.20 * (1.0 - fpr)
        + 0.15 * top1_acc
        + 0.10 * (1.0 - latency_normalized)
    )

    # --- Build report ---
    report = {
        "benchmark_version": benchmark.get("version", "1.0.0"),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "system_info": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "database": {
            "path": str(db_path),
            "hash": compute_db_hash(db_path),
            "size_mb": round(db_path.stat().st_size / (1024 * 1024), 1),
        },
        "config": {
            "validation_level": level,
            "total_sentences": len(results),
            "clean_sentences": clean_total,
            "error_sentences": len(results) - clean_total,
        },
        "overall_metrics": {
            "detection": {
                "total_expected_errors": total_expected,
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            },
            "false_positive_rate": {
                "clean_sentences_tested": clean_total,
                "clean_sentences_with_fp": clean_sentences_with_fp,
                "total_fp_on_clean": clean_fp_total,
                "rate": round(fpr, 4),
            },
            "suggestions": {
                "errors_with_gold_correction": suggestion_total,
                "top1_accuracy": round(top1_acc, 4),
                "top3_accuracy": round(top3_acc, 4),
                "top5_accuracy": round(top5_acc, 4),
                "mrr": round(mrr, 4),
            },
            "latency_ms": {
                "total": round(total_elapsed_ms, 1),
                "mean": round(sum(latencies) / n, 1) if n > 0 else 0.0,
                "p50": round(p50, 1),
                "p95": round(p95, 1),
                "p99": round(p99, 1),
            },
            "composite_score": round(composite, 4),
        },
        "per_tier_metrics": {},
        "per_sentence_results": [],
        "rerank_rule_telemetry": {},
        "strategy_debug_telemetry": {},
        "fn_reason_telemetry": {"histogram": {}, "examples": []},
    }

    if rerank_rule_telemetry:
        telemetry_sorted = sorted(
            rerank_rule_telemetry.items(),
            key=lambda item: (-int(item[1].get("fires", 0)), item[0]),
        )
        for rule_id, stats in telemetry_sorted:
            fires = int(stats.get("fires", 0))
            top1_changes = int(stats.get("top1_changes", 0))
            report["rerank_rule_telemetry"][rule_id] = {
                "fires": fires,
                "top1_changes": top1_changes,
                "top1_change_rate": round((top1_changes / fires), 4) if fires > 0 else 0.0,
            }

    if strategy_debug_telemetry and isinstance(strategy_debug_telemetry, dict):
        report["strategy_debug_telemetry"]["enabled"] = bool(
            strategy_debug_telemetry.get("enabled", False)
        )
        strategies = strategy_debug_telemetry.get("strategies", {})
        if isinstance(strategies, dict):
            sorted_strategies = sorted(
                strategies.items(),
                key=lambda item: (
                    -int(
                        item[1].get("overlap_blocked_positions", 0)
                        if isinstance(item[1], dict)
                        else 0
                    ),
                    -int(item[1].get("emitted", 0) if isinstance(item[1], dict) else 0),
                    item[0],
                ),
            )
            report["strategy_debug_telemetry"]["strategies"] = {}
            for strategy_name, stats in sorted_strategies:
                if not isinstance(strategy_name, str) or not isinstance(stats, dict):
                    continue
                shadow_potential = int(stats.get("shadow_potential_positions", 0))
                blocked = int(stats.get("overlap_blocked_positions", 0))
                blocked_by_type_raw = stats.get("overlap_blocked_by_type", {})
                blocked_by_type_sorted: dict[str, int] = {}
                if isinstance(blocked_by_type_raw, dict):
                    blocked_by_type_sorted = {
                        error_type: int(count)
                        for error_type, count in sorted(
                            blocked_by_type_raw.items(),
                            key=lambda item: (-int(item[1]), item[0]),
                        )
                        if isinstance(error_type, str)
                    }

                entry = {
                    "calls": int(stats.get("calls", 0)),
                    "emitted": int(stats.get("emitted", 0)),
                    "new_positions": int(stats.get("new_positions", 0)),
                    "overlap_emitted": int(stats.get("overlap_emitted", 0)),
                    "shadow_potential_positions": shadow_potential,
                    "overlap_blocked_positions": blocked,
                    "overlap_blocked_ratio": (
                        round(blocked / shadow_potential, 4) if shadow_potential > 0 else 0.0
                    ),
                    "overlap_blocked_by_type": blocked_by_type_sorted,
                    "overlap_blocked_examples": stats.get("overlap_blocked_examples", []),
                }
                report["strategy_debug_telemetry"]["strategies"][strategy_name] = entry

    if fn_reason_telemetry and isinstance(fn_reason_telemetry, dict):
        raw_hist = fn_reason_telemetry.get("histogram", {})
        if isinstance(raw_hist, dict):
            report["fn_reason_telemetry"]["histogram"] = {
                reason: int(count)
                for reason, count in sorted(
                    raw_hist.items(), key=lambda item: (-int(item[1]), item[0])
                )
                if isinstance(reason, str)
            }
        examples = fn_reason_telemetry.get("examples", [])
        if isinstance(examples, list):
            report["fn_reason_telemetry"]["examples"] = examples

    # Per-tier
    for key, tm in tier_metrics.items():
        if tm.sentence_count == 0:
            continue
        report["per_tier_metrics"][key] = {
            "sentences": tm.sentence_count,
            "total_errors": tm.total_errors,
            "true_positives": tm.true_positives,
            "false_positives": tm.false_positives,
            "false_negatives": tm.false_negatives,
            "precision": round(tm.precision, 4),
            "recall": round(tm.recall, 4),
            "f1": round(tm.f1, 4),
            "top1_accuracy": round(tm.top1_accuracy, 4),
            "top3_accuracy": round(tm.top3_accuracy, 4),
            "mrr": round(tm.mrr, 4),
        }

    # Per-sentence detail
    for r in results:
        entry: dict[str, Any] = {
            "id": r.sentence_id,
            "is_clean": r.is_clean,
            "tier": r.difficulty_tier,
            "latency_ms": round(r.latency_ms, 2),
            "expected_errors": r.expected_error_count,
            "detected_errors": r.detected_error_count,
            "tp": r.true_positives,
            "fp": r.false_positives,
            "fn": r.false_negatives,
        }
        if r.span_matches:
            entry["matches"] = []
            for m in r.span_matches:
                match_entry: dict[str, Any] = {
                    "gold_id": m.gold_error_id,
                    "detected": m.matched,
                }
                if m.matched:
                    match_entry["system_type"] = m.system_error_type
                    match_entry["suggestions"] = m.system_suggestions
                    if m.gold_correction:
                        match_entry["top1_correct"] = m.top1_correct
                        match_entry["rank"] = m.rank_of_correct
                entry["matches"].append(match_entry)
        if r.is_clean and r.false_positives > 0:
            entry["false_positive_details"] = [
                {
                    "text": e.get("text", ""),
                    "type": e.get("error_type", ""),
                    "pos": e.get("position", -1),
                }
                for e in r.system_errors
            ]
        report["per_sentence_results"].append(entry)

    return report


def print_summary(report: dict) -> None:
    """Print a human-readable summary to stdout."""
    det = report["overall_metrics"]["detection"]
    sug = report["overall_metrics"]["suggestions"]
    fpr_data = report["overall_metrics"]["false_positive_rate"]
    lat = report["overall_metrics"]["latency_ms"]

    print("\n" + "=" * 70)
    print("  Myanmar Spell Checker Benchmark Results")
    print("=" * 70)

    db = report["database"]
    print(f"\n  Database: {Path(db['path']).name} ({db['size_mb']} MB, hash: {db['hash']})")
    print(f"  Level: {report['config']['validation_level']}")
    cfg = report["config"]
    total = cfg["total_sentences"]
    clean = cfg["clean_sentences"]
    errs = cfg["error_sentences"]
    print(f"  Sentences: {total} ({clean} clean, {errs} with errors)")
    print(f"  Run: {report['run_timestamp']}")

    print(f"\n{'─' * 70}")
    print("  OVERALL DETECTION")
    print(f"{'─' * 70}")
    print(f"  Expected errors:  {det['total_expected_errors']}")
    print(f"  True positives:   {det['true_positives']}")
    print(f"  False positives:  {det['false_positives']}")
    print(f"  False negatives:  {det['false_negatives']}")
    print(f"  Precision:        {det['precision']:.1%}")
    print(f"  Recall:           {det['recall']:.1%}")
    print(f"  F1 Score:         {det['f1']:.1%}")

    print(f"\n{'─' * 70}")
    print("  FALSE POSITIVE RATE")
    print(f"{'─' * 70}")
    print(f"  Clean sentences:  {fpr_data['clean_sentences_tested']}")
    fp_count = fpr_data["clean_sentences_with_fp"]
    fp_total = fpr_data["clean_sentences_tested"]
    print(f"  Sentences with FP: {fp_count}/{fp_total}")
    print(f"  Total FP count:   {fpr_data['total_fp_on_clean']}")
    print(f"  FP Rate:          {fpr_data['rate']:.1%} of clean sentences flagged")

    print(f"\n{'─' * 70}")
    print("  SUGGESTION QUALITY")
    print(f"{'─' * 70}")
    print(f"  Top-1 Accuracy:   {sug['top1_accuracy']:.1%}")
    print(f"  Top-3 Accuracy:   {sug['top3_accuracy']:.1%}")
    print(f"  Top-5 Accuracy:   {sug['top5_accuracy']:.1%}")
    print(f"  MRR:              {sug['mrr']:.4f}")

    print(f"\n{'─' * 70}")
    print("  PER-TIER BREAKDOWN")
    print(f"{'─' * 70}")
    hdr = (
        f"  {'Tier':<16} {'Errors':>6} {'TP':>4} {'FP':>4}"
        f" {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}"
        f" {'Top1':>7} {'MRR':>7}"
    )
    sep = (
        f"  {'─' * 14:<16} {'─' * 6:>6} {'─' * 4:>4} {'─' * 4:>4}"
        f" {'─' * 4:>4} {'─' * 7:>7} {'─' * 7:>7} {'─' * 7:>7}"
        f" {'─' * 7:>7} {'─' * 7:>7}"
    )
    print(hdr)
    print(sep)

    tier_order = [
        "tier_1",
        "tier_2",
        "tier_3",
    ]
    tier_labels = {
        "tier_1": "Tier 1 (Easy)",
        "tier_2": "Tier 2 (Medium)",
        "tier_3": "Tier 3 (Hard)",
    }

    # Include any extra tiers found in data (tier_4, etc.)
    for key in sorted(report["per_tier_metrics"].keys()):
        if key not in tier_order:
            tier_order.append(key)
            tier_labels.setdefault(key, key.replace("_", " ").title())

    for key in tier_order:
        if key not in report["per_tier_metrics"]:
            continue
        tm = report["per_tier_metrics"][key]
        label = tier_labels.get(key, key)
        print(
            f"  {label:<16} {tm['total_errors']:>6} {tm['true_positives']:>4} "
            f"{tm['false_positives']:>4} {tm['false_negatives']:>4} "
            f"{tm['precision']:>6.1%} {tm['recall']:>6.1%} {tm['f1']:>6.1%} "
            f"{tm['top1_accuracy']:>6.1%} {tm['mrr']:>7.4f}"
        )

    print(f"\n{'─' * 70}")
    print("  LATENCY")
    print(f"{'─' * 70}")
    print(f"  Total:    {lat['total']:.0f} ms")
    print(f"  Mean:     {lat['mean']:.1f} ms/sentence")
    print(f"  P50:      {lat['p50']:.1f} ms")
    print(f"  P95:      {lat['p95']:.1f} ms")

    print(f"\n{'─' * 70}")
    print(f"  COMPOSITE SCORE: {report['overall_metrics']['composite_score']:.4f}")
    print(f"{'─' * 70}")

    rerank_telemetry = report.get("rerank_rule_telemetry", {})
    if rerank_telemetry:
        print(f"\n{'─' * 70}")
        print("  RERANK RULE TELEMETRY (TOP 10 BY FIRES)")
        print(f"{'─' * 70}")
        print(f"  {'Rule ID':<52} {'Fires':>7} {'Top1Δ':>7} {'Rate':>7}")
        print(f"  {'─' * 52:<52} {'─' * 7:>7} {'─' * 7:>7} {'─' * 7:>7}")
        for idx, (rule_id, stats) in enumerate(rerank_telemetry.items()):
            if idx >= 10:
                break
            print(
                f"  {rule_id[:52]:<52} {stats['fires']:>7} "
                f"{stats['top1_changes']:>7} {stats['top1_change_rate']:>7.1%}"
            )

    strategy_debug = report.get("strategy_debug_telemetry", {})
    strategy_debug_items = strategy_debug.get("strategies", {})
    if isinstance(strategy_debug_items, dict) and strategy_debug_items:
        print(f"\n{'─' * 70}")
        print("  STRATEGY GATE DEBUG TELEMETRY")
        print(f"{'─' * 70}")
        print(
            f"  {'Strategy':<34} {'Calls':>6} {'Emit':>6} {'New':>6} {'Blocked':>8} {'Block%':>8}"
        )
        print(f"  {'─' * 34:<34} {'─' * 6:>6} {'─' * 6:>6} {'─' * 6:>6} {'─' * 8:>8} {'─' * 8:>8}")
        shown = 0
        for strategy_name, stats in strategy_debug_items.items():
            if shown >= 10:
                break
            shown += 1
            print(
                f"  {strategy_name[:34]:<34} {int(stats.get('calls', 0)):>6} "
                f"{int(stats.get('emitted', 0)):>6} {int(stats.get('new_positions', 0)):>6} "
                f"{int(stats.get('overlap_blocked_positions', 0)):>8} "
                f"{float(stats.get('overlap_blocked_ratio', 0.0)):>7.1%}"
            )

        semantic_stats = strategy_debug_items.get("SemanticValidationStrategy")
        if isinstance(semantic_stats, dict):
            blocked_by_type = semantic_stats.get("overlap_blocked_by_type", {})
            if isinstance(blocked_by_type, dict) and blocked_by_type:
                top_blockers = list(blocked_by_type.items())[:5]
                print("  Semantic overlap blockers (top 5):")
                for blocker, count in top_blockers:
                    print(f"    {blocker}: {count}")

    fn_reason = report.get("fn_reason_telemetry", {})
    fn_histogram = fn_reason.get("histogram", {})
    if isinstance(fn_histogram, dict) and fn_histogram:
        print(f"\n{'─' * 70}")
        print("  FALSE-NEGATIVE REASON TELEMETRY")
        print(f"{'─' * 70}")
        for reason, count in fn_histogram.items():
            print(f"  {reason:<32} {int(count):>5}")

    # Print missed detections
    missed = []
    for sr in report["per_sentence_results"]:
        if sr.get("matches"):
            for m in sr["matches"]:
                if not m["detected"]:
                    missed.append((sr["id"], m["gold_id"]))
    if missed:
        print(f"\n  MISSED DETECTIONS ({len(missed)}):")
        for sid, eid in missed:
            print(f"    {sid} / {eid}")

    # Print false positives on clean
    fp_clean = []
    for sr in report["per_sentence_results"]:
        if sr["is_clean"] and sr.get("false_positive_details"):
            for fpd in sr["false_positive_details"]:
                fp_clean.append((sr["id"], fpd["text"], fpd["type"]))
    if fp_clean:
        print(f"\n  FALSE POSITIVES ON CLEAN SENTENCES ({len(fp_clean)}):")
        for sid, text, etype in fp_clean:
            print(f"    {sid}: '{text}' flagged as {etype}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run Myanmar Spell Checker Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to spell checker database (.db file)",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(__file__).parent / "myspellchecker_benchmark.yaml",
        help="Path to benchmark YAML file",
    )
    parser.add_argument(
        "--level",
        choices=["syllable", "word"],
        default="word",
        help="Validation level (default: word)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for results JSON (default: benchmarks/results/)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output only JSON, no human-readable summary",
    )
    parser.add_argument(
        "--semantic",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to ONNX model file or directory containing model.onnx for semantic checking. "
            "Tokenizer is expected in the same directory."
        ),
    )
    parser.add_argument(
        "--ner",
        action="store_true",
        default=False,
        help="Enable NER-based FP suppression (PER+LOC) via heuristic NER with place-name dict.",
    )
    parser.add_argument(
        "--proactive-threshold",
        type=float,
        default=0.85,
        help="Proactive scanning confidence threshold (default: 0.85). Lower = more aggressive.",
    )
    parser.add_argument(
        "--proactive",
        action="store_true",
        default=False,
        help="Enable proactive semantic scanning (off by default due to FP rate).",
    )
    parser.add_argument(
        "--no-confusable-semantic",
        action="store_true",
        default=False,
        help="Disable MLM-enhanced confusable detection (ConfusableSemanticStrategy).",
    )
    parser.add_argument(
        "--confusable-preset",
        choices=["relaxed", "conservative"],
        default="relaxed",
        help=(
            "Confusable semantic threshold preset. "
            "'relaxed' (default): loosened guards, model sees more. "
            "'conservative': original strict guards for A/B comparison."
        ),
    )
    parser.add_argument(
        "--disable-targeted-rerank-hints",
        action="store_true",
        default=False,
        help="Disable targeted rerank hint rules.",
    )
    parser.add_argument(
        "--disable-targeted-candidate-injections",
        action="store_true",
        default=False,
        help="Disable targeted candidate injection rules.",
    )
    parser.add_argument(
        "--disable-targeted-grammar-completion-templates",
        action="store_true",
        default=False,
        help="Disable targeted grammar completion templates.",
    )
    parser.add_argument(
        "--debug-strategy-gates",
        action="store_true",
        default=False,
        help=(
            "Enable strategy gate-debug telemetry. Adds per-strategy emitted/new/overlap "
            "stats and semantic overlap suppression analysis."
        ),
    )
    parser.add_argument(
        "--reranker",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to directory containing reranker.onnx and reranker.onnx.stats.json "
            "for neural MLP suggestion reranking."
        ),
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    if not args.benchmark.exists():
        print(f"Error: Benchmark file not found: {args.benchmark}", file=sys.stderr)
        sys.exit(1)

    # Run benchmark
    report = run_benchmark(
        benchmark_path=args.benchmark,
        db_path=args.db,
        level=args.level,
        warmup=args.warmup,
        semantic_path=args.semantic,
        enable_ner=args.ner,
        targeted_rerank_hints=not args.disable_targeted_rerank_hints,
        targeted_candidate_injections=not args.disable_targeted_candidate_injections,
        targeted_grammar_completion_templates=(
            not args.disable_targeted_grammar_completion_templates
        ),
        proactive_threshold=args.proactive_threshold,
        enable_proactive=args.proactive,
        no_confusable_semantic=args.no_confusable_semantic,
        confusable_preset=args.confusable_preset,
        enable_strategy_debug=args.debug_strategy_gates,
        reranker_path=args.reranker,
    )

    # Print summary
    if not args.json_only:
        print_summary(report)

    # Save results
    output_dir = args.output or Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    db_name = args.db.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    semantic_tag = "_semantic" if args.semantic else ""
    ner_tag = "_ner" if args.ner else ""
    targeted_tags = ""
    if args.disable_targeted_rerank_hints:
        targeted_tags += "_no_hint"
    if args.disable_targeted_candidate_injections:
        targeted_tags += "_no_inject"
    if args.disable_targeted_grammar_completion_templates:
        targeted_tags += "_no_grammar_tpl"
    debug_tag = "_debug_gates" if args.debug_strategy_gates else ""
    output_file = output_dir / (
        f"benchmark_{db_name}_{args.level}{semantic_tag}{ner_tag}"
        f"{targeted_tags}{debug_tag}_{timestamp}.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if not args.json_only:
        print(f"  Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
