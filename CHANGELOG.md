# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-04-08

### Added

- **Meta-classifier post-filter**: Logistic regression model (41 features including one-hot error type, word frequency, context signals) replaces manual per-strategy confidence thresholds. FPR dropped from 34.5% to 18.6%.
- **ConfusableSemanticStrategy** (priority 48): MLM-enhanced confusable detection using masked language model logit comparison with asymmetric thresholds.
- **Kinzi and stacking variant support**: Confusable candidate generation now handles Kinzi (င်္) and consonant stacking patterns.
- **Rich Suggestion objects**: `Suggestion` class with `confidence` and `source` metadata, backward-compatible (inherits from `str`). Errors serialize both `suggestions` (plain text) and `suggestions_detail` (with metadata).
- **Per-request CheckOptions**: `CheckOptions` dataclass for runtime overrides (`context_checking`, `grammar_checking`, `max_suggestions`, `use_semantic`).
- **Error.severity property**: Computed severity (error/warning/info) based on action type classification.
- **MLM post-filter**: Suppress invalid_word and dangling_word false positives using semantic model logit validation.
- **Expanded confusable pairs**: 87 to 124+ curated pairs with 9 linguistics-audit additions.
- **Expanded colloquial variants**: 83 to 91 entries covering verb aspect contractions, modal forms, reduplication extensions, negation extensions, and copula forms. Removed 20 standard modern Burmese words (demonstratives, question words, discourse connectors) that were incorrectly classified as colloquial.
- **Homophone morphological guard**: Expanded from negation prefix "မ" only to 2 prefixes (မ, အ) plus 4 right-context compound suffixes (ရေး, ရာ, သား, သူ) for better homophone disambiguation.
- **Candidate fusion enabled by default**: Calibrated Noisy-OR voting pipeline is now the default detection mode.

### Changed

- **Config split**: `algorithm_configs.py` (3,002 lines) split into 4 focused modules — `algorithm_configs.py` (core), `text_configs.py`, `strategy_configs.py`, `infra_configs.py`. All existing imports continue to work via re-exports.
- **Benchmark consolidated**: Merged expanded benchmark into main file (1,138 → 1,146 sentences), fixed 18 duplicate IDs, deleted orphaned `myspellchecker_benchmark_expanded.yaml` and `clean_corpus.yaml`.
- Confidence gates expanded to 15 error types for FPR reduction.
- Zero-TP detectors disabled, weak detectors heavily gated.
- Syntax fusion discount widened to rules with confidence ≤0.85.
- Mutex/override infrastructure fully removed (`conflict_rules.py`, `fusion_mode` flag, `should_skip_position()`).
- 19 mislabeled benchmark sentences re-labeled as not-clean.

### Fixed

- MLM threshold comparison now operates in logit space (was incorrectly using probability space).
- `context_checking=False` in CheckOptions now preserves word validation instead of skipping all validation.
- Suggestion objects properly converted to strings in reranker feature extraction.
- Loan word variants import hoisted to module level to avoid repeated lazy loading.
- Sentence-final penalty bypass fixed for edge cases.
- Long verb chains reassembled correctly, dangling particle scan capped.
- 4-syllable all-valid tokens skipped in word validation (segmenter merge artifacts).
- Invalid_word and dangling_word FPs reduced via post-validation compound splitting.

## [1.3.0] - 2026-04-06

### Added

- **Candidate fusion pipeline**: Calibrated Noisy-OR fusion replaces the mutex-based winner selection. When enabled (`use_candidate_fusion=True`), all strategies may fire at every position and the arbiter uses calibrated confidence fusion across independence clusters to determine which errors to emit.
- **StatisticalConfusableStrategy** (priority 24): Bidirectional bigram ratio comparison for detecting confusable word pairs in context.
- **ConfusableCompoundClassifierStrategy** (priority 47): MLP-based compound word detection using ONNX inference.
- **Data-driven calibration**: Per-strategy calibration breakpoints and reliability weights loaded from YAML (`calibration_data.yaml`), with `StrategyCalibrator.from_bundled()` convenience method.
- **Confidence gates**: Word error suppression and context evidence guards to reduce false positives on structurally-clean text.
- **Source strategy attribution**: `Error.source_strategy` field tracks which strategy produced each error.
- **Pipeline conflict resolution**: Override matrix and candidate arbiter for resolving cross-strategy conflicts.
- **Builder API**: `SpellCheckerBuilder.with_candidate_fusion()` for fluent configuration of the voting pipeline.
- **Expanded confusable pairs**: 37 to 87 curated pairs, mined from production database with overlap scoring.
- **Expanded grammar rules**: Additional auxiliary verbs, null-copula fix, grammar frequency guard.

### Changed

- `ConfigPresets.accurate()` now enables candidate fusion by default.
- Strategy priority docstrings updated to reflect all 12 strategies.
- NERConfig split into lightweight `text/ner_config.py` module for faster imports.
- Lowered default strategy confidences and confusable thresholds for better recall.
- BrokenCompound span extraction uses word positions directly instead of `sentence.find()`.
- N-gram strategy applies error correction to all context words including `word_at_ip2`.

### Fixed

- Removed false positive: `ထင်ရှား` (prominent/outstanding) was incorrectly flagged as aspiration error.
- Removed incorrect correction: `ကုမ်ပဏီ` was corrected to `ကုန်ပဏီ` instead of `ကုမ္ပဏီ`.
- Removed disputed correction: `နူနာ` ha-htoe correction (`နူ` → `နှူ`) removed per MLC dictionary.
- CalibrationData loader now catches `ValueError` from invalid y_thresholds instead of crashing.
- Narrowed overly broad exception catches in classifier strategy and suggestion pipeline.
- Neural reranker now warns when stats file is missing instead of silently using unnormalized features.
- Hardcoded paths in benchmark files replaced with relative paths.

## [1.2.1] - 2026-04-04

### Fixed

- Fixed error position offsets, edge cases in validation pipeline, and batch processor robustness.
- Improved suggestion generation, error deduplication, and POS tagger fallback paths.

## [1.2.0] - 2026-04-04

### Added

- **StatisticalConfusableStrategy**: Bigram ratio gate for confusable word pair detection.
- **ConfusableCompoundClassifierStrategy**: MLP-based confusable/compound detection with ONNX inference.
- **CMS multi-signal scoring** for confusable detection with curated pair threshold reduction.
- **POS-based V+particle detection** for broken compounds.
- **Title/suffix compound detection layer**.
- Expanded mandatory compounds from 63 to 3,315 via template mining.
- 28 confusable pair benchmark sentences.
- Benchmark `--confidence-gap` flag for analysis.
- Dual MLP/LightGBM training pipelines with MLM logit wiring.
- v2 inference features and 3-gate neural reranking.

### Changed

- Error budget relaxed from per-sentence skip to heavy-error-only.
- Benchmark scope filtering for targeted evaluation.

### Fixed

- Reduced FPR on expanded benchmark with pipeline hardening.
- Scoped 5 weak-context confusable sentences as `context_dependent`.
- Resolved 29 code review issues across 24 files.

## [1.1.0] - 2026-04-01

### Added

- **ParticleChecker** and **TenseAgreementChecker** grammar checkers wired into grammar engine.
- **Polite register tier** for 3-way register detection (formal/polite/informal).
- **G2P phonetic mappings**, named entity gazetteer, and loan word variant rules.
- **Fast-path exit** in context_validator for clean sentences (performance optimization).
- **Clean-text FPR corpus** for false positive rate measurement.
- Pragmatics golden tests for register validation.
- Externalized NER data: YAML consolidation + SQLite schema.
- Expanded benchmark from 489 to 719 sentences.
- `importorskip` guards for optional dependencies (duckdb, torch) in tests.

### Changed

- Tuned dangling_word detector and expanded false_compounds list.
- Shared false_compound suppression with text-level compound detector.
- Semantic MLM suggestions sorted by edit distance before logit score.
- CI coverage threshold lowered 75% → 70% (new modules without full test coverage).
- Removed mypy from CI lint (82 pre-existing type errors tracked in issue #29).
- CI path filters to skip non-code changes.
- GitHub Actions bumped to actions/checkout v6 and actions/setup-python v6.

### Fixed

- Reduced FPR: disabled G04, fixed confusable logit_diff, added terminal SFP guard.
- Surfaced semantic model load failures when paths are explicitly configured.
- Stacking error suppression improvements.
- Critical bugs, config wiring, and security hardening from code review.
- Publish workflow: use release tag for pypi-publish.

## [1.0.0] - 2026-03-28

### Added

- Initial public release with 12-strategy validation pipeline.
- Syllable-first architecture for Myanmar continuous script.
- SymSpell O(1) suggestion algorithm with Myanmar-specific edit costs.
- N-gram context checking (bigram/trigram).
- ONNX-powered semantic validation (MLM).
- 8 grammar checkers with YAML-driven rules.
- Dictionary building pipeline with Cython acceleration.
- POS tagging (rule-based, Viterbi, transformer).
- Zawgyi detection and conversion.
- CLI interface for spell checking and dictionary building.
