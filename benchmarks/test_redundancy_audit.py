"""Redundancy audit: test which overlap-tagged compound entries are truly redundant.

Temporarily removes overlap-tagged entries from the compound confusion dict,
runs the benchmark, and reports which benchmark cases regress.

Usage:
    source venv/bin/activate
    python benchmarks/test_redundancy_audit.py \
        --db data/mySpellChecker_production.db \
        --semantic /Users/thettwe/Works/myspellchecker-training/models/semantic-v2.3-final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# The 19 overlap-tagged patterns (keys in _CONSONANT_CONFUSION_COMPOUNDS)
# These are the POST-NORMALIZATION forms (must match what _norm_dict_tuple produces)
OVERLAP_TAGGED_KEYS_RAW = [
    # missing_visarga (7)
    "တင်သွင်",
    "ဆုံဖြတ်",
    "စွမ်ဆောင်",
    "တိုတက်",
    "သင်တန်",
    "သင်ကြာ",
    "အင်အာ",
    # medial_confusion (5)
    "ကြွေးစား",
    "ပြော်ရွင်",
    "မနက်ဖန်",
    "နေပီ",
    "ဆင်ချင်",
    # vowel_confusion (5)
    "ဘူရား",
    "ပြူပြင်",
    "ကူသ",
    "ရှာဖွဲ",
    "ဝင်ငွဲ",
    # missing_anusvara (2)
    "ဘောလုး",
    "ထောက်ပန့်",
]


def main():
    parser = argparse.ArgumentParser(description="Redundancy audit for compound confusion entries")
    parser.add_argument("--db", required=True, help="Path to production DB")
    parser.add_argument("--semantic", default=None, help="Path to semantic model directory")
    args = parser.parse_args()

    from myspellchecker.text.normalize import normalize

    # Normalize the keys to match what the dict uses
    overlap_keys = {normalize(k) for k in OVERLAP_TAGGED_KEYS_RAW}

    # Import and patch the compound detection module
    from myspellchecker.core.detectors.post_norm_mixins import compound_detection_mixin as cdm

    original_dict = dict(cdm._LOADED_CONSONANT_CONFUSION_COMPOUNDS)
    print(f"Original dict size: {len(original_dict)}")

    # Find which keys to remove
    keys_to_remove = set()
    for key in original_dict:
        if key in overlap_keys:
            keys_to_remove.add(key)

    print(f"Overlap-tagged keys found in dict: {len(keys_to_remove)}")
    if len(keys_to_remove) < len(overlap_keys):
        missing = overlap_keys - keys_to_remove
        print(f"  WARNING: {len(missing)} keys not found in dict (normalization mismatch?)")
        for m in sorted(missing):
            print(f"    - {repr(m)}")

    # Run benchmark with FULL dict (baseline)
    print("\n=== BASELINE (all entries enabled) ===")
    baseline = run_benchmark(args.db, args.semantic)

    # Patch: remove overlap entries
    patched_dict = {k: v for k, v in original_dict.items() if k not in keys_to_remove}
    cdm._LOADED_CONSONANT_CONFUSION_COMPOUNDS = patched_dict
    cdm.CompoundDetectionMixin._CONSONANT_CONFUSION_COMPOUNDS = patched_dict
    print(
        f"\n=== PATCHED (removed {len(keys_to_remove)} entries, {len(patched_dict)} remaining) ==="
    )
    patched = run_benchmark(args.db, args.semantic)

    # Compare
    print("\n=== COMPARISON ===")
    print(f"  Baseline:  TP={baseline['tp']}, FP={baseline['fp']}, FN={baseline['fn']}")
    print(f"  Patched:   TP={patched['tp']}, FP={patched['fp']}, FN={patched['fn']}")
    print(f"  TP delta:  {patched['tp'] - baseline['tp']}")
    print(f"  FP delta:  {patched['fp'] - baseline['fp']}")
    print(f"  FN delta:  {patched['fn'] - baseline['fn']}")

    # Show specific regressions (new FNs)
    baseline_fn_ids = set(baseline.get("fn_ids", []))
    patched_fn_ids = set(patched.get("fn_ids", []))
    new_fns = patched_fn_ids - baseline_fn_ids
    fixed_fps = set(baseline.get("fp_ids", [])) - set(patched.get("fp_ids", []))

    if new_fns:
        print(f"\n  NEW FALSE NEGATIVES ({len(new_fns)} cases):")
        for fn_id in sorted(new_fns):
            print(f"    - {fn_id}")
        print("  → These entries are NEEDED (keep in dict)")
    else:
        print("\n  No new false negatives! All 19 entries are REDUNDANT.")

    if fixed_fps:
        print(f"\n  FIXED FALSE POSITIVES ({len(fixed_fps)} cases):")
        for fp_id in sorted(fixed_fps):
            print(f"    - {fp_id}")

    # Restore original
    cdm._LOADED_CONSONANT_CONFUSION_COMPOUNDS = original_dict
    cdm.CompoundDetectionMixin._CONSONANT_CONFUSION_COMPOUNDS = original_dict


def run_benchmark(db_path: str, semantic_path: str | None) -> dict:
    """Run the benchmark and return TP/FP/FN counts + case IDs."""
    import yaml

    from myspellchecker.core.config.algorithm_configs import SemanticConfig
    from myspellchecker.core.config.main import SpellCheckerConfig
    from myspellchecker.core.spellchecker import SpellChecker
    from myspellchecker.providers.sqlite import SQLiteProvider

    bench_file = Path(__file__).parent / "myspellchecker_benchmark.yaml"
    with open(bench_file, encoding="utf-8") as f:
        bench_data = yaml.safe_load(f)

    sentences = bench_data.get("sentences", [])

    config_kwargs: dict = {}
    if semantic_path:
        config_kwargs["semantic"] = SemanticConfig(
            use_semantic_refinement=True,
            model_path=semantic_path,
            tokenizer_path=str(Path(semantic_path) / "tokenizer.json"),
        )
    config = SpellCheckerConfig(**config_kwargs)
    if semantic_path:
        config.validation.use_confusable_semantic = True
    provider = SQLiteProvider(database_path=str(db_path))
    checker = SpellChecker(config=config, provider=provider)

    tp, fp, fn = 0, 0, 0
    fn_ids, fp_ids = [], []

    for sent in sentences:
        sent_id = sent.get("id", "?")
        text = sent.get("input", sent.get("text", ""))
        expected_errors = sent.get("expected_errors", sent.get("errors", []))
        is_clean = sent.get("is_clean", len(expected_errors) == 0)

        result = checker.check(text)
        detected = result.errors

        if is_clean:
            if detected:
                fp += len(detected)
                fp_ids.append(sent_id)
        else:
            # Match detected errors to expected
            gold_spans = set()
            for err in expected_errors:
                span = err.get("span", {})
                start = span.get("start", 0)
                end = span.get("end", start + len(err.get("erroneous_text", "")))
                gold_spans.add((start, end))

            detected_spans = set()
            for err in detected:
                start = err.position
                end = start + len(err.text)
                detected_spans.add((start, end))

            # Overlap-based matching (30% threshold)
            matched_gold = set()
            matched_detected = set()
            for gs, ge in gold_spans:
                gold_len = ge - gs
                if gold_len <= 0:
                    continue
                for ds, de in detected_spans:
                    overlap = max(0, min(ge, de) - max(gs, ds))
                    if overlap >= 0.3 * gold_len:
                        matched_gold.add((gs, ge))
                        matched_detected.add((ds, de))

            tp += len(matched_gold)
            fn += len(gold_spans) - len(matched_gold)
            fp += len(detected_spans) - len(matched_detected)

            if len(gold_spans) > len(matched_gold):
                fn_ids.append(sent_id)
            if len(detected_spans) > len(matched_detected):
                fp_ids.append(sent_id)

    print(f"  TP={tp}, FP={fp}, FN={fn}")
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"  Precision={precision:.4f}")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"  Recall={recall:.4f}")

    return {"tp": tp, "fp": fp, "fn": fn, "fn_ids": fn_ids, "fp_ids": fp_ids}


if __name__ == "__main__":
    main()
