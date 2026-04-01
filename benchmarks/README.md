# mySpellChecker Benchmark

Benchmark suite for evaluating the myspellchecker error detection and suggestion pipeline.
489 hand-annotated Myanmar sentences covering syllable errors, word errors, grammar errors,
register violations, and adversarial near-misses.

## Benchmark Suite

- **Version**: 1.0.0
- **Total sentences**: 489 (60 clean, 429 with errors, 470 total error spans)
- **Definition**: [`myspellchecker_benchmark.yaml`](myspellchecker_benchmark.yaml)
- **Runner**: [`run_benchmark.py`](run_benchmark.py)

### Sentence Distribution

| Tier | Errors | What It Tests |
|------|-------:|---------------|
| Tier 1 (Easy) | 160 | Invalid syllable structure |
| Tier 2 (Medium) | 164 | Valid syllable, wrong word |
| Tier 3 (Hard) | 146 | Valid word, wrong in context |
| Clean | 0 | False positive resistance (60 sentences) |

### Composite Score Formula

```
composite = 0.30 * F1
          + 0.25 * MRR
          + 0.20 * (1 - FPR)
          + 0.15 * Top1_Accuracy
          + 0.10 * (1 - latency_normalized)
```

Where `latency_normalized = min(p95 / 500ms, 1.0)`.

## Current Results

### Run Configuration

- **Database**: `mySpellChecker_production.db` (565 MB, 601K words, full POS + enrichment tables)
- **Validation level**: word
- **Platform**: macOS (Apple Silicon)

> **Note:** The dictionary database and semantic model (v2.3) used in these benchmarks are **not included** in the library. They were built from our own proprietary corpus using the [data pipeline](https://docs.myspellchecker.com/data-pipeline/index) and [training pipeline](https://docs.myspellchecker.com/guides/training) respectively. Your results will vary depending on the dictionary database you build and the semantic model you train.

### Overall Metrics (no semantic)

| Metric | Value |
|--------|------:|
| **F1** | 96.2% |
| **Precision** | 97.8% |
| **Recall** | 94.7% |
| True Positives | 445 |
| False Positives | 10 |
| False Negatives | 25 |
| FPR (clean sentences) | 0.0% |
| Top-1 Suggestion Acc | 85.2% |
| MRR | 0.8731 |

### Overall Metrics (with semantic v2.3)

| Metric | Value |
|--------|------:|
| **F1** | 98.3% |
| **Precision** | 97.1% |
| **Recall** | 99.6% |
| True Positives | 468 |
| False Positives | 14 |
| False Negatives | 2 |
| FPR (clean sentences) | 0.0% |
| Top-1 Suggestion Acc | 81.2% |
| MRR | 0.8395 |

### Per-Tier Breakdown (no semantic)

| Tier | Errors | TP | FP | FN | Prec | Rec | F1 | Top-1 | MRR |
|------|-------:|---:|---:|---:|-----:|----:|---:|------:|----:|
| Tier 1 (Easy) | 160 | 152 | 3 | 8 | 98.1% | 95.0% | 96.5% | 85.5% | 0.876 |
| Tier 2 (Medium) | 164 | 157 | 1 | 7 | 99.4% | 95.7% | 97.5% | 86.0% | 0.883 |
| Tier 3 (Hard) | 146 | 136 | 6 | 10 | 95.8% | 93.2% | 94.4% | 83.8% | 0.859 |

## How to Run

```bash
# Activate the library venv
source venv/bin/activate

# Run with production DB (no semantic)
python benchmarks/run_benchmark.py \
  --db data/mySpellChecker_production.db

# Run with semantic model
python benchmarks/run_benchmark.py \
  --db data/mySpellChecker_production.db \
  --semantic /path/to/semantic-model/

# Run with neural reranker
python benchmarks/run_benchmark.py \
  --db data/mySpellChecker_production.db \
  --reranker /path/to/reranker-dir/

# Enable NER-based false positive suppression
python benchmarks/run_benchmark.py \
  --db data/mySpellChecker_production.db \
  --ner

# JSON-only output (for automation)
python benchmarks/run_benchmark.py \
  --db data/mySpellChecker_production.db \
  --json-only

# Gate/layer debug telemetry (per-strategy overlap + semantic blocking)
python benchmarks/run_benchmark.py \
  --db data/mySpellChecker_production.db \
  --semantic /path/to/semantic-model/ \
  --debug-strategy-gates \
  --json-only

# Ablation toggles (disable targeted rule groups without code edits)
python benchmarks/run_benchmark.py \
  --db data/mySpellChecker_production.db \
  --disable-targeted-rerank-hints \
  --disable-targeted-candidate-injections \
  --disable-targeted-grammar-completion-templates \
  --json-only
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--db` | Path to spell checker database (required) |
| `--benchmark` | Benchmark YAML path (default: `myspellchecker_benchmark.yaml`) |
| `--level` | Validation level: `syllable` or `word` (default: `word`) |
| `--semantic` | Path to ONNX semantic model directory |
| `--reranker` | Path to neural MLP reranker directory |
| `--ner` | Enable NER-based FP suppression |
| `--json-only` | Output JSON only, no human-readable summary |
| `--output` | Custom output directory for results JSON |
| `--warmup` | Number of warmup runs (default: 3) |
| `--debug-strategy-gates` | Enable per-strategy gate telemetry |
| `--confusable-preset` | `relaxed` (default) or `conservative` |
| `--no-confusable-semantic` | Disable MLM-enhanced confusable detection |

## Utilities

### Compare Runs

Compare two benchmark result JSON files to track regressions:

```bash
python benchmarks/compare_runs.py \
  --baseline run_a.json \
  --current run_b.json \
  --output-json comparison.json \
  --output-md comparison.md
```

### Audit Targeted Rules

Aggregate rerank rule telemetry and rank rules by risk/opportunity:

```bash
python benchmarks/audit_targeted_rules.py \
  --reports run.json \
  --output-json audit.json \
  --output-md audit.md
```

### Run Ablation Matrix

Run full ablation study (default + each targeted group off + all off):

```bash
python benchmarks/run_ablation.py \
  --db data/mySpellChecker_production.db \
  --level word \
  --semantic /path/to/semantic-model/ \
  --output-dir ablation_results/
```

### Evaluate Semantic Models

Head-to-head model comparison (confusable discrimination, logit analysis, perplexity):

```bash
python benchmarks/semantic_model_eval.py \
  --models v2.3=/path/to/v2.3-final \
  --db data/mySpellChecker_production.db
```

### Profile DB Queries

Instrument SQLiteProvider to count and time every database call per sentence:

```bash
python benchmarks/profile_db_queries.py \
  --db data/mySpellChecker_production.db \
  --output profile_report.json
```

## Known Limitations

1. **10 residual FPs** — false positives on edge-case constructions, documented and accepted.
2. **25 FNs without semantic** — context-dependent errors requiring MLM; semantic model rescues 23 of 25.
3. **Suggestion quality plateau** — remaining rank>1 cases are inherent morpheme/compound ambiguity where the same error pattern has conflicting gold corrections.
