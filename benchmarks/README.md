# mySpellChecker Benchmark

Benchmark suite for evaluating the myspellchecker error detection and suggestion pipeline.
1,146 hand-annotated Myanmar sentences covering syllable errors, word errors, grammar errors,
confusable pairs, register violations, and adversarial near-misses across 7 domains.

## Benchmark Suite

- **Total sentences**: 1,146 (425 clean, 721 with errors, 743 error spans)
- **Definition**: [`myspellchecker_benchmark.yaml`](myspellchecker_benchmark.yaml)
- **Runner**: [`run_benchmark.py`](run_benchmark.py)

### Sentence Distribution

| Tier | Sentences | What It Tests |
|------|----------:|---------------|
| Tier 1 (Easy) | 185 | Invalid syllable structure |
| Tier 2 (Medium) | 381 | Valid syllable, wrong word |
| Tier 3 (Hard) | 136 | Valid word, wrong in context |
| Clean | 425 | False positive resistance |

### Domain Coverage

| Domain | Sentences |
|--------|----------:|
| Conversational | 303 |
| News | 218 |
| Technical | 215 |
| Academic | 198 |
| Religious | 77 |
| Literary | 72 |
| General | 63 |

### Composite Score Formula

```
composite = 0.30 * F1
          + 0.25 * MRR
          + 0.20 * (1 - FPR)
          + 0.15 * Top1_Accuracy
          + 0.10 * (1 - latency_normalized)
```

Where `latency_normalized = min(p95 / 500ms, 1.0)`.

## Current Results (v1.6.0)

### Run Configuration

- **Database**: `mySpellChecker_production.db` (495 MB, flat-AA migrated, full POS + enrichment tables)
- **Semantic model**: v2.4-final (ONNX, MLM-based)
- **Validation level**: word
- **Platform**: macOS (Apple Silicon)
- **Benchmark suite**: 2,084 sentences (570 clean, 1,514 with errors, 1,716 expected error spans) — `myspellchecker_benchmark.yaml@v1.5.0`

> **Note:** The dictionary database and semantic model used in these benchmarks are **not included** in the library. They were built from a proprietary corpus using the [data pipeline](https://docs.myspellchecker.com/data-pipeline/index) and [training pipeline](https://docs.myspellchecker.com/guides/training) respectively. Your results will vary depending on the dictionary database you build and the semantic model you train.

### Overall Metrics — full benchmark (spelling + grammar)

| Metric | Value |
|--------|------:|
| **F1** | 62.2% |
| **Precision** | 83.7% |
| **Recall** | 49.5% |
| FPR (clean sentences) | 11.1% |
| Top-1 Suggestion Acc | 44.4% |
| MRR | 0.5481 |
| p95 latency | 298 ms |
| Composite score | 0.6267 |

### Overall Metrics — spelling domain only (`--domain spelling`)

Spelling-first is the v1.6.0 product priority. On the spelling-only subset (1,415 spelling error spans; grammar and ambiguous spans filtered out at evaluation time):

| Metric | Value |
|--------|------:|
| **F1** | 64.6% |
| **Precision** | 83.1% |
| **Recall** | 52.9% |
| FPR (clean sentences) | 9.8% |
| Top-1 Suggestion Acc | 43.6% |
| MRR | 0.5413 |
| p95 latency | 292 ms |
| Composite score | 0.6345 |

Previous release rows (v1.5.0 at composite 0.7227 on the earlier spelling-scoped 1,304-sentence suite; v1.4.0 at 0.660) used a narrower, earlier benchmark — direct row-to-row composite comparison across releases is not apples-to-apples because the benchmark itself grew from 1,304 → 2,084 sentences with an added `domain` dimension. See `50_Metrics/Benchmark History.md` in the internal knowledge base for per-commit composite trajectories.

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

# Enable candidate fusion (calibrated Noisy-OR voting)
python benchmarks/run_benchmark.py \
  --db data/mySpellChecker_production.db \
  --fusion --fusion-threshold 0.5

# JSON-only output (for automation)
python benchmarks/run_benchmark.py \
  --db data/mySpellChecker_production.db \
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
| `--fusion` | Enable calibrated Noisy-OR candidate fusion |
| `--fusion-threshold` | Fusion confidence threshold (default: 0.5) |
| `--calibration` | Path to calibration YAML (used with `--fusion`) |
| `--scope` | Comma-separated scopes to evaluate (default: `spelling`) |
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

### Audit Compound Redundancy

Test which overlap-tagged compound confusion entries are truly redundant:

```bash
python benchmarks/test_redundancy_audit.py \
  --db data/mySpellChecker_production.db
```
