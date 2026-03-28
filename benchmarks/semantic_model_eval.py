"""
Semantic Model Evaluation — Confusable Discrimination & Logit Analysis

Test 1: Confusable Discrimination
  For known confusable pairs, create sentences where one word is correct.
  Mask the target word, run predict_mask(). Measure whether the model
  ranks the correct word higher than the confusable variant.

Test 2: Raw Logit Diff Comparison
  Run all models on benchmark sentences, capture logit_diff values at
  each position. Compare separation quality across models.

Test 3: Perplexity
  Compute pseudo-perplexity on held-out clean Myanmar text.

Usage:
  python benchmarks/semantic_model_eval.py \
    --models v2-final=/path/to/v2-final v2.2=/path/to/v2.2 v2.3=/path/to/v2.3 \
    --db data/mySpellChecker_production.db
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from myspellchecker.algorithms.semantic_checker import SemanticChecker

# ── Test 1: Confusable Discrimination ─────────────────────────────────────────

# Sentence pairs: (sentence_with_correct_word, correct_word, confusable_word, meaning)
CONFUSABLE_TEST_PAIRS = [
    # Ya-pin vs Ya-yit (ျ↔ြ)
    ("ကျောင်းသားများ စာကြိုးစားနေသည်", "ကျောင်း", "ကြောင်း", "school vs reason"),
    ("ကြောင်းကြောင့် အဲဒီလို ဖြစ်တာ", "ကြောင်း", "ကျောင်း", "reason vs school"),
    ("ကြောက်စရာ ကောင်းတယ်", "ကြောက်", "ကျောက်", "fear vs stone"),
    ("ကျောက်တုံးကြီး ကျသွားတယ်", "ကျောက်", "ကြောက်", "stone vs fear"),
    ("ကြီးမားတဲ့ အိမ်ကြီးတစ်လုံး", "ကြီး", "ကျီး", "big vs crow"),
    ("ကျီးကန်းတွေ ပျံနေတယ်", "ကျီး", "ကြီး", "crow vs big"),
    ("ပြည်ထောင်စု မြန်မာနိုင်ငံ", "ပြည်", "ပျည်", "country vs pus"),
    ("ကြက်သားကြော် ချက်မယ်", "ကြက်", "ကျက်", "chicken vs memorize"),
    # Ha-htoe pairs
    ("နှစ်ယောက်အတူ သွားတယ်", "နှစ်", "နစ်", "two/year vs sink"),
    ("ရေထဲ နစ်မြုပ်သွားတယ်", "နစ်", "နှစ်", "sink vs two"),
    ("ရှာဖွေနေပါတယ်", "ရှာ", "ရာ", "search vs hundred"),
    ("တစ်ရာ ပေးပါ", "ရာ", "ရှာ", "hundred vs search"),
    ("လုပ်ငန်းခွင် ဝင်မယ်", "လုပ်", "လှုပ်", "work vs shake"),
    ("လှုပ်ရှားမှု ပြုလုပ်နေတယ်", "လှုပ်", "လုပ်", "shake vs work"),
    ("နှာခေါင်း ပိတ်နေတယ်", "နှာ", "နာ", "nose vs hurt"),
    ("ခေါင်းနာ ဖြစ်နေတယ်", "နာ", "နှာ", "hurt vs nose"),
    # Aspirated / Unaspirated
    ("ခေါ်ယူလိုက်ပါ", "ခေါ်", "ကော်", "call vs glue"),
    ("တင်သွင်းလိုက်မယ်", "တင်", "ထင်", "submit vs think"),
    ("ထင်မြင်ချက် ပေးပါ", "ထင်", "တင်", "think vs submit"),
    # Tone / final confusions
    ("စာမေးပွဲ ကျရှုံးသွားတယ်", "ကျ", "ကြ", "fail vs do"),
    ("မှာယူလိုက်ပါ", "မှာ", "မာ", "order vs hard"),
    ("သံချပ် မာတယ်", "မာ", "မှာ", "hard vs order"),
    # Particles
    ("စားပွဲပေါ်မှာ ထားလိုက်", "မှာ", "မာ", "at/on vs hard"),
    ("ကျောင်းကို သွားမယ်", "ကို", "ကိုယ်", "to(particle) vs self"),
]


def run_discrimination_test(models: dict[str, SemanticChecker]) -> dict:
    """Test 1: For each confusable pair, check if model ranks correct > confusable."""
    results = {}

    for name, checker in models.items():
        correct_ranks = []
        confusable_ranks = []
        wins = 0
        ties = 0
        losses = 0
        details = []

        for sentence, correct, confusable, meaning in CONFUSABLE_TEST_PAIRS:
            try:
                predictions = checker.predict_mask(sentence, correct, top_k=50)
                pred_map = {w: score for w, score in predictions}

                correct_score = pred_map.get(correct)
                confusable_score = pred_map.get(confusable)

                # Find ranks (1-indexed, None if not in top-50)
                ranked_words = [w for w, _ in predictions]
                correct_rank = (
                    (ranked_words.index(correct) + 1) if correct in ranked_words else None
                )
                confusable_rank = (
                    (ranked_words.index(confusable) + 1) if confusable in ranked_words else None
                )

                if correct_score is not None and confusable_score is not None:
                    diff = correct_score - confusable_score
                    if diff > 0.1:
                        wins += 1
                        outcome = "WIN"
                    elif diff < -0.1:
                        losses += 1
                        outcome = "LOSS"
                    else:
                        ties += 1
                        outcome = "TIE"
                elif correct_score is not None and confusable_score is None:
                    wins += 1
                    outcome = "WIN (confusable not in top-50)"
                    diff = None
                elif correct_score is None and confusable_score is not None:
                    losses += 1
                    outcome = "LOSS (correct not in top-50)"
                    diff = None
                else:
                    ties += 1
                    outcome = "TIE (neither in top-50)"
                    diff = None

                if correct_rank:
                    correct_ranks.append(correct_rank)
                if confusable_rank:
                    confusable_ranks.append(confusable_rank)

                details.append(
                    {
                        "sentence": sentence,
                        "correct": correct,
                        "confusable": confusable,
                        "meaning": meaning,
                        "correct_score": round(correct_score, 4) if correct_score else None,
                        "confusable_score": (
                            round(confusable_score, 4) if confusable_score else None
                        ),
                        "correct_rank": correct_rank,
                        "confusable_rank": confusable_rank,
                        "score_diff": round(diff, 4) if diff is not None else None,
                        "outcome": outcome,
                    }
                )
            except Exception as e:
                details.append(
                    {
                        "sentence": sentence,
                        "correct": correct,
                        "confusable": confusable,
                        "error": str(e),
                        "outcome": "ERROR",
                    }
                )

        total = wins + ties + losses
        results[name] = {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "total": total,
            "win_rate": round(wins / total, 4) if total else 0,
            "avg_correct_rank": round(sum(correct_ranks) / len(correct_ranks), 2)
            if correct_ranks
            else None,
            "avg_confusable_rank": round(sum(confusable_ranks) / len(confusable_ranks), 2)
            if confusable_ranks
            else None,
            "details": details,
        }

    return results


# ── Test 2: Raw Logit Diff on Benchmark Sentences ────────────────────────────


def run_logit_diff_comparison(
    models: dict[str, SemanticChecker],
    benchmark_path: str,
    db_path: str,
) -> dict:
    """Test 2: Compare logit_diff distributions across models on benchmark error sentences."""

    with open(benchmark_path) as f:
        benchmark = yaml.safe_load(f)

    sentences = benchmark.get("sentences", [])
    # Pick error sentences with confusable-type errors (where MLM should help)
    confusable_subtypes = {
        "homophone_confusion",
        "medial_ya_ra_confusion",
        "missing_ha_htoe",
        "aspirated_unaspirated_confusion",
        "wrong_final_consonant",
        "wrong_medial_type",
        "missing_medial_ya_yit",
        "vowel_u_uu_confusion",
        "vowel_length_confusion",
        "extra_medial_wa",
        "medial_and_vowel_confusion",
        "real_word_context_confusion",
        "confusable_word",
        "medial_confusion",
        "wrong_vowel",
        "wrong_vowel_tone",
        "zero_wa_confusion",
    }

    test_sentences = []
    for s in sentences:
        errors = s.get("expected_errors", [])
        for e in errors:
            esubtype = e.get("error_subtype", "")
            if esubtype in confusable_subtypes:
                erroneous = e.get("erroneous_text", "")
                gold = e.get("gold_correction", "")
                if erroneous and gold:
                    test_sentences.append(
                        {
                            "id": s["id"],
                            "text": s["input"],
                            "error_word": erroneous,
                            "correct_word": gold,
                            "error_type": e.get("error_type", ""),
                            "error_subtype": esubtype,
                        }
                    )

    results = {}
    for name, checker in models.items():
        diffs = []
        for item in test_sentences:
            if not item["error_word"] or not item["correct_word"]:
                continue
            try:
                # Mask the error word position, see what model predicts
                predictions = checker.predict_mask(item["text"], item["error_word"], top_k=50)
                pred_map = {w: score for w, score in predictions}

                error_score = pred_map.get(item["error_word"])
                correct_score = pred_map.get(item["correct_word"])

                diffs.append(
                    {
                        "id": item["id"],
                        "error_word": item["error_word"],
                        "correct_word": item["correct_word"],
                        "error_type": item["error_type"],
                        "error_score": round(error_score, 4) if error_score else None,
                        "correct_score": round(correct_score, 4) if correct_score else None,
                        "logit_diff": (
                            round(correct_score - error_score, 4)
                            if correct_score is not None and error_score is not None
                            else None
                        ),
                        "correct_rank": next(
                            (
                                i + 1
                                for i, (w, _) in enumerate(predictions)
                                if w == item["correct_word"]
                            ),
                            None,
                        ),
                        "error_rank": next(
                            (
                                i + 1
                                for i, (w, _) in enumerate(predictions)
                                if w == item["error_word"]
                            ),
                            None,
                        ),
                    }
                )
            except Exception as e:
                diffs.append({"id": item["id"], "error": str(e)})

        # Compute summary stats
        valid_diffs = [d["logit_diff"] for d in diffs if d.get("logit_diff") is not None]
        correct_higher = sum(1 for d in valid_diffs if d > 0.1)
        error_higher = sum(1 for d in valid_diffs if d < -0.1)

        results[name] = {
            "total_tested": len(diffs),
            "valid_pairs": len(valid_diffs),
            "correct_ranked_higher": correct_higher,
            "error_ranked_higher": error_higher,
            "correct_rate": round(correct_higher / len(valid_diffs), 4) if valid_diffs else 0,
            "mean_logit_diff": round(sum(valid_diffs) / len(valid_diffs), 4)
            if valid_diffs
            else None,
            "median_logit_diff": round(sorted(valid_diffs)[len(valid_diffs) // 2], 4)
            if valid_diffs
            else None,
            "details": diffs,
        }

    return results


# ── Test 3: Pseudo-Perplexity ─────────────────────────────────────────────────


def compute_pseudo_perplexity(
    checker: SemanticChecker,
    sentences: list[str],
    segmenter=None,
) -> dict:
    """
    Compute pseudo-perplexity via masked language modeling.
    For each word in each sentence, mask it and check if model predicts it.
    PPL = exp(-1/N * sum(log P(word_i | context)))
    """
    total_log_prob = 0.0
    total_words = 0
    word_hits = 0  # Words found in top-K

    for sentence in sentences:
        if segmenter:
            words = segmenter.segment(sentence)
        else:
            # For Myanmar, use syllable tokenizer as fallback
            try:
                from myspellchecker.tokenizers.syllable import segment_syllables

                words = segment_syllables(sentence)
            except ImportError:
                words = sentence.split()

        for word in words:
            if len(word.strip()) == 0:
                continue
            try:
                predictions = checker.predict_mask(sentence, word, top_k=50)
                pred_map = {w: score for w, score in predictions}

                if word in pred_map:
                    # Use softmax-normalized score as probability proxy
                    scores = list(pred_map.values())
                    max_score = max(scores)
                    exp_scores = [math.exp(s - max_score) for s in scores]
                    total_exp = sum(exp_scores)
                    word_prob = math.exp(pred_map[word] - max_score) / total_exp
                    total_log_prob += math.log(word_prob)
                    word_hits += 1
                else:
                    # Word not in top-K — assign minimum probability
                    total_log_prob += math.log(1e-6)

                total_words += 1
            except Exception:
                continue

    if total_words == 0:
        return {"perplexity": float("inf"), "total_words": 0}

    avg_log_prob = total_log_prob / total_words
    perplexity = math.exp(-avg_log_prob)

    return {
        "perplexity": round(perplexity, 2),
        "total_words": total_words,
        "word_hits": word_hits,
        "hit_rate": round(word_hits / total_words, 4) if total_words else 0,
        "avg_log_prob": round(avg_log_prob, 4),
    }


# Clean Myanmar sentences for perplexity test
PERPLEXITY_SENTENCES = [
    "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်",
    "ကလေးများ ကျောင်းသို့ သွားကြသည်",
    "မိုးရွာနေသောကြောင့် ထီးဆောင်းရမည်",
    "ဆရာမ စာသင်နေသည်",
    "ငါးဖမ်းသမားများ ပင်လယ်သို့ ထွက်သွားကြသည်",
    "ဈေးဝယ်ထွက်ဖို့ ပြင်ဆင်နေတယ်",
    "နိုင်ငံတော်သမ္မတက မိန့်ခွန်းပြောကြားသည်",
    "ကျန်းမာရေးအတွက် အားကစားလုပ်သင့်သည်",
    "စာအုပ်ဖတ်ခြင်းသည် အသိပညာ တိုးပွားစေသည်",
    "ရာသီဥတုပြောင်းလဲမှု ပြဿနာကြီး ဖြစ်လာနေသည်",
    "မိသားစုနှင့်အတူ ညစာစားပွဲ ပြင်ဆင်သည်",
    "တက္ကသိုလ်တွင် ပညာသင်ကြားနေသည်",
    "လယ်သမားများ စပါးစိုက်ပျိုးနေကြသည်",
    "ဆေးရုံတွင် လူနာများ တန်းစီနေကြသည်",
    "မြို့ပြစီမံကိန်းအသစ် စတင်အကောင်အထည်ဖော်မည်",
    "ကွန်ပြူတာသိပ္ပံ ဘာသာရပ်ကို လေ့လာနေသည်",
    "ငွေကြေးစီမံခန့်ခွဲမှု အရေးကြီးသည်",
    "ခရီးသွားလုပ်ငန်း ဖွံ့ဖြိုးတိုးတက်လာသည်",
    "သဘာဝပတ်ဝန်းကျင် ထိန်းသိမ်းကာကွယ်ရမည်",
    "ပညာရေးစနစ် ပြုပြင်ပြောင်းလဲရန် လိုအပ်သည်",
]


# ── Main ──────────────────────────────────────────────────────────────────────


def load_models(model_specs: list[str]) -> dict[str, SemanticChecker]:
    """Parse 'name=/path/to/model' specs and load SemanticChecker instances."""
    models = {}
    for spec in model_specs:
        name, path = spec.split("=", 1)
        model_path = Path(path)
        onnx_file = model_path / "model.onnx" if model_path.is_dir() else model_path
        tokenizer_path = model_path if model_path.is_dir() else model_path.parent

        print(f"Loading {name} from {onnx_file}...")
        checker = SemanticChecker(
            model_path=str(onnx_file),
            tokenizer_path=str(tokenizer_path),
            predict_top_k=50,
            check_top_k=50,
        )
        models[name] = checker
        print(
            f"  Loaded: vocab="
            f"{checker.tokenizer.vocab_size if hasattr(checker.tokenizer, 'vocab_size') else '?'}"
        )

    return models


def print_discrimination_summary(results: dict):
    print("\n" + "=" * 70)
    print("  TEST 1: CONFUSABLE DISCRIMINATION")
    print("=" * 70)

    # Header
    print(
        f"\n  {'Model':<12} {'Wins':>6} {'Ties':>6} {'Loss':>6} "
        f"{'Total':>6} {'Win%':>8} {'AvgCorrectRk':>14} {'AvgConfRk':>12}"
    )
    print("  " + "─" * 68)

    for name, r in results.items():
        print(
            f"  {name:<12} {r['wins']:>6} {r['ties']:>6} {r['losses']:>6} "
            f"{r['total']:>6} {r['win_rate'] * 100:>7.1f}% "
            f"{r['avg_correct_rank'] or 'N/A':>14} {r['avg_confusable_rank'] or 'N/A':>12}"
        )

    # Per-pair comparison
    print("\n  Per-pair breakdown:")
    print(f"  {'Pair':<30} ", end="")
    for name in results:
        print(f"{'  ' + name:>16}", end="")
    print()
    print("  " + "─" * (30 + 16 * len(results)))

    first_model = list(results.values())[0]
    for i, detail in enumerate(first_model["details"]):
        pair_label = f"{detail['correct']}↔{detail['confusable']}"
        print(f"  {pair_label:<30} ", end="")
        for _name, r in results.items():
            d = r["details"][i]
            diff = d.get("score_diff")
            outcome = d.get("outcome", "?")
            if diff is not None:
                marker = "+" if "WIN" in outcome else ("-" if "LOSS" in outcome else "=")
                print(
                    f"{marker}{abs(diff):>7.2f} "
                    f"({d.get('correct_rank', '?'):>2}v"
                    f"{d.get('confusable_rank', '?'):<2})",
                    end="",
                )
            else:
                print(f"{'  ' + outcome[:12]:>16}", end="")
        print()


def print_logit_diff_summary(results: dict):
    print("\n" + "=" * 70)
    print("  TEST 2: RAW LOGIT DIFF ON BENCHMARK ERRORS")
    print("=" * 70)

    print(
        f"\n  {'Model':<12} {'Tested':>8} {'Valid':>7} {'Correct↑':>10} "
        f"{'Error↑':>8} {'Rate':>8} {'MeanDiff':>10} {'MedianDiff':>12}"
    )
    print("  " + "─" * 73)

    for name, r in results.items():
        print(
            f"  {name:<12} {r['total_tested']:>8} {r['valid_pairs']:>7} "
            f"{r['correct_ranked_higher']:>10} {r['error_ranked_higher']:>8} "
            f"{r['correct_rate'] * 100:>7.1f}% "
            f"{r['mean_logit_diff'] or 0:>10.2f} {r['median_logit_diff'] or 0:>12.2f}"
        )


def print_perplexity_summary(results: dict):
    print("\n" + "=" * 70)
    print("  TEST 3: PSEUDO-PERPLEXITY (lower is better)")
    print("=" * 70)

    print(f"\n  {'Model':<12} {'PPL':>10} {'Words':>8} {'Hits':>7} {'HitRate':>9} {'AvgLogP':>10}")
    print("  " + "─" * 54)

    for name, r in results.items():
        print(
            f"  {name:<12} {r['perplexity']:>10.2f} {r['total_words']:>8} "
            f"{r['word_hits']:>7} {r['hit_rate'] * 100:>8.1f}% {r['avg_log_prob']:>10.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Semantic Model Evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model specs as name=/path/to/model (e.g., v2.3=/path/to/v2.3-final)",
    )
    parser.add_argument("--db", required=True, help="Path to spell checker database")
    parser.add_argument(
        "--benchmark",
        default="benchmarks/myspellchecker_benchmark.yaml",
        help="Path to benchmark YAML",
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["1", "2", "3"],
        choices=["1", "2", "3"],
        help="Which tests to run (default: all)",
    )
    args = parser.parse_args()

    models = load_models(args.models)
    all_results = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if "1" in args.tests:
        print("\nRunning Test 1: Confusable Discrimination...")
        disc_results = run_discrimination_test(models)
        all_results["discrimination"] = disc_results
        print_discrimination_summary(disc_results)

    if "2" in args.tests:
        print("\nRunning Test 2: Raw Logit Diff on Benchmark Errors...")
        logit_results = run_logit_diff_comparison(models, args.benchmark, args.db)
        all_results["logit_diff"] = logit_results
        print_logit_diff_summary(logit_results)

    if "3" in args.tests:
        print("\nRunning Test 3: Pseudo-Perplexity...")
        ppl_results = {}
        for name, checker in models.items():
            ppl_results[name] = compute_pseudo_perplexity(checker, PERPLEXITY_SENTENCES)
        all_results["perplexity"] = ppl_results
        print_perplexity_summary(ppl_results)

    # Save results
    output_path = (
        args.output or f"benchmarks/results/semantic_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
