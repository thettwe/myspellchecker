# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [1.2.1] - 2026-03-23

### Fixed

- Fixed error position offsets, edge cases in validation pipeline, and batch processor robustness.
- Improved suggestion generation, error deduplication, and POS tagger fallback paths.

## [1.2.0] - 2026-03-20

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
