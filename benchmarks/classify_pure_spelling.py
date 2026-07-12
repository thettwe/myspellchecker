"""Classify benchmark spelling golds into PURE-SPELLING vs CONTEXT-SPELLING.

The v2.0 north-star metric is clearance on the PURE bucket: errors whose
erroneous surface form is itself invalid (non-word, malformed segmentation,
encoding artifact) and therefore detectable without sentence context. Errors
whose surface form is a valid word (real-word confusions, homophones,
synonyms) form the CONTEXT bucket, deferred past v2.0.

Classification is scorer-side only — the benchmark yaml is never relabeled
(the v19h holdout freeze and the annotation-independence policy forbid it).
Rules, ratified 2026-07-02 ("v2.0 Scope - Pure-Spelling Reframe"):

  1. synonym_substitution            -> CONTEXT (dict absence there is a
     coverage gap on valid rare words, not non-wordness)
  2. loan_word_misspelling           -> PURE (wrong loan forms are corpus-
     frequent but the fix is lexicon curation, not context)
  3. encoding subtypes               -> PURE
  4. structural/segmentation subtypes-> PURE (the span as written is
     malformed regardless of meaning)
  5. whitespace inside the span      -> PURE
  6. otherwise, dictionary lookup:      absent or frequency 0 -> PURE
                                        present with frequency -> CONTEXT

Usage:
  python benchmarks/classify_pure_spelling.py --classify
      print classification summary (and optionally --json <path> the map)
  python benchmarks/classify_pure_spelling.py --report <result.json>
      join against a benchmark result JSON and report per-bucket clearance
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import yaml

BENCHMARK_PATH = Path(__file__).parent / "myspellchecker_benchmark.yaml"
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "mySpellChecker_production.db"

STRUCTURAL_SUBTYPES = frozenset({"spacing", "broken_compound", "word_boundary", "merged_word"})
ENCODING_SUBTYPES = frozenset({"zawgyi_conversion_error", "zawgyi_encoding", "zero_width_chars"})
PURE_OVERRIDE_SUBTYPES = frozenset({"loan_word_misspelling"})
CONTEXT_OVERRIDE_SUBTYPES = frozenset({"synonym_substitution"})

ZERO_WIDTH = "​‌‍﻿"

# Spelling-vs-grammar categorization mirrors scripts/fn_audit/spelling_only_metrics.py
# (kept in sync manually; scripts/ is local-only and cannot be imported here).
GRAMMAR_KEYWORDS = (
    "particle",
    "tense",
    "aspect",
    "register",
    "classifier",
    "collocation",
    "ngram",
    "word_order",
    "missing_word",
    "missing_words",
    "missing_information",
    "redundancy",
    "repetition",
    "incomplete",
    "contextual",
    "illogical",
    "logical_contradiction",
    "wrong_word_in_context",
    "punctuation",
    "extraneous",
    "omission",
    "negation",
    "semantic_error",
    "ending_mark_error",
    "colloquial_in_formal",
)

EXPLICIT_GRAMMAR = frozenset(
    {
        "register_mismatch",
        "verb_tense_agreement",
        "tense_mismatch",
        "particle_misuse",
        "particle_error",
        "wrong_particle",
        "particle_spelling",
        "particle_confusion",
        "invalid_particle_combination",
        "informal_particle_in_formal_context",
        "classifier_error",
        "aspect_error",
        "negation_error",
        "word_order",
        "word_order_error",
        "collocation_error",
        "ngram_unlikely",
        "semantic_error",
        "missing_word",
        "missing_words",
        "missing_information",
        "redundancy",
        "repetition",
        "incomplete_sentence",
        "contextual_error",
        "illogical_usage",
        "logical_contradiction",
        "wrong_word_in_context",
        "punctuation_error",
        "extraneous_word",
        "omission",
        "ending_mark_error",
        "colloquial_in_formal",
        "missing_conjunction",
    }
)


def categorize_domain(subtype: str) -> str:
    """Return 'spelling' or 'grammar' for an error_subtype."""
    if not subtype:
        return "spelling"
    s = subtype.lower()
    if s in EXPLICIT_GRAMMAR:
        return "grammar"
    for kw in GRAMMAR_KEYWORDS:
        if kw in s:
            return "grammar"
    return "spelling"


class DictLookup:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    def frequency(self, form: str) -> int | None:
        """Corpus frequency of the exact form, or None if absent."""
        for candidate in (form, form.translate(str.maketrans("", "", ZERO_WIDTH + " "))):
            row = self._conn.execute(
                "SELECT frequency FROM words WHERE word = ?", (candidate,)
            ).fetchone()
            if row is not None:
                return int(row[0])
        return None


def classify_instance(subtype: str, erroneous_text: str, lookup: DictLookup) -> tuple[str, str]:
    """Return (bucket, rule) for one gold error instance."""
    if subtype in CONTEXT_OVERRIDE_SUBTYPES:
        return "context", "synonym-override"
    if subtype in PURE_OVERRIDE_SUBTYPES:
        return "pure", "loan-override"
    if subtype in ENCODING_SUBTYPES:
        return "pure", "encoding"
    if subtype in STRUCTURAL_SUBTYPES:
        return "pure", "structural"
    if any(ch.isspace() for ch in erroneous_text):
        return "pure", "whitespace-span"
    freq = lookup.frequency(erroneous_text)
    if freq is None or freq == 0:
        return "pure", "nonword"
    return "context", "realword"


def build_classification(db_path: Path) -> dict[str, dict]:
    """Map '<sentence_id>::<error_id>' -> {bucket, rule, subtype} for spelling golds."""
    benchmark = yaml.safe_load(BENCHMARK_PATH.read_text(encoding="utf-8"))
    lookup = DictLookup(db_path)
    out: dict[str, dict] = {}
    for s in benchmark["sentences"]:
        if s.get("is_clean"):
            continue
        for e in s.get("expected_errors") or []:
            subtype = e.get("error_subtype", "")
            if categorize_domain(subtype) != "spelling":
                continue
            bucket, rule = classify_instance(subtype, e.get("erroneous_text", "") or "", lookup)
            out[f"{s['id']}::{e.get('error_id')}"] = {
                "bucket": bucket,
                "rule": rule,
                "subtype": subtype,
            }
    return out


def print_summary(classification: dict[str, dict]) -> None:
    pure = sum(1 for v in classification.values() if v["bucket"] == "pure")
    print(f"spelling golds classified: {len(classification)}")
    print(f"  PURE:    {pure}")
    print(f"  CONTEXT: {len(classification) - pure}")
    by_rule: dict[str, int] = {}
    for v in classification.values():
        by_rule[v["rule"]] = by_rule.get(v["rule"], 0) + 1
    for rule in sorted(by_rule):
        print(f"    {rule}: {by_rule[rule]}")


def report(result_path: Path, classification: dict[str, dict]) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    stats = {
        "pure": {"tp": 0, "fn": 0, "top1": 0},
        "context": {"tp": 0, "fn": 0, "top1": 0},
    }
    joined = 0
    for s in result["per_sentence_results"]:
        for m in s.get("matches", []):
            key = f"{s['id']}::{m.get('gold_id')}"
            entry = classification.get(key)
            if entry is None:
                continue
            joined += 1
            bucket = stats[entry["bucket"]]
            if m.get("detected"):
                bucket["tp"] += 1
                if m.get("top1_correct"):
                    bucket["top1"] += 1
            else:
                bucket["fn"] += 1

    print(f"result: {result_path}")
    print(f"joined spelling golds: {joined} of {len(classification)} classified")
    total_tp = total = 0
    for name in ("pure", "context"):
        b = stats[name]
        n = b["tp"] + b["fn"]
        total_tp += b["tp"]
        total += n
        clearance = b["tp"] / n if n else 0.0
        top1 = b["top1"] / n if n else 0.0
        bar = "  <- PRIMARY KPI (target >= 0.80)" if name == "pure" else ""
        print(
            f"  {name.upper():8} n={n:5} TP={b['tp']:4} FN={b['fn']:4} "
            f"clearance={clearance:.4f} top1={top1:.4f}{bar}"
        )
    if total:
        print(f"  UNSPLIT  n={total:5} clearance={total_tp / total:.4f} (cross-check)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--classify", action="store_true", help="print classification summary")
    group.add_argument("--report", type=Path, help="benchmark result JSON to join and report on")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="dictionary DB path")
    parser.add_argument("--json", type=Path, help="also write the per-gold map to this path")
    args = parser.parse_args()

    classification = build_classification(args.db)
    if args.json:
        args.json.write_text(json.dumps(classification, ensure_ascii=False, indent=1))
        print(f"map written: {args.json}")
    if args.classify:
        print_summary(classification)
    else:
        report(args.report, classification)


if __name__ == "__main__":
    main()
