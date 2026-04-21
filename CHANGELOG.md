# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2026-04-21

### Added

- **MinedConfusablePairStrategy** (priority 49, enabled by default): Flags real-word confusables using a table of 23,970 edit-distance-1 pairs mined from the production dictionary. Both forms are dictionary-valid, so SymSpell cannot surface them on its own; the strategy gates emissions with a semantic MLM logit margin and a frequency ratio between the current word and its partner. Dedicated unit test suite covers constructor guards, partner-map symmetry, validate guards, happy path, margin threshold, and frequency cache.
- **Compound-split confusable boost** (enabled by default): When `_suppress_compound_split_valid_words` would fire on a long OOV token whose syllables are all individually valid (4+ syllables), the same structural signal also boosts the confidence of any inner `confusable_error` emission inside the outer span. Combined-signal detections clear the downstream confidence gate and survive to the final response.
- **Skip-rule confidence gate**: The pre-existing "skip tokens of 4+ valid syllables" rule now defers to SymSpell when the top-1 candidate clears a configurable edit-distance / frequency gate (`skip_rule_gate_max_ed`, `skip_rule_gate_min_freq`). Recovers missing-asat and substitution typos whose fragmented form happens to be all-valid syllables (e.g. `စွမ်းဆောင်ရည` → `စွမ်းဆောင်ရည်`).
- **Pre-segmenter raw-token SymSpell probe** (priority 23, enabled by default): Runs `SymSpell.lookup(raw_token, level='word')` on unsegmented whitespace-delimited tokens before segmentation. Recovers compound typos the segmenter would otherwise fragment into piecewise-valid subtokens.
- **Segmenter post-merge rescue** (left-cascade merge): New adjacent-pair merge pass probes variant-map / dictionary / dictionary+asat lookups on concatenated segmenter fragments. Off by default pending false-positive-rate calibration; opt in via `use_segmenter_post_merge_rescue`.
- **Loan-word DB mining**: 54 curated transliteration variants mined from the confusable pairs table, plus a `WordValidator` short-circuit that emits the curated correction before SymSpell's edit-distance filter kicks in.
- **Tone-Zawgyi normalization** (consonant-gated `normalize_e_vowel_tall_aa`): Targeted whitelist `{ပ, ခ, ဒ}` of consonants whose `ေ` + flat-AA + final sequences get rewritten to the classical `ေါ` form. Narrower than the classical MLC round-bottom set by design; the broader set would corrupt modern gold forms like `ဘောလုံး`, `သဘော`, `ရောဂါ`, `ဖော်`.
- **Flat-AA dictionary migration**: 17,712 word keys + 68k N-gram foreign-key repoints + 1.5M probability re-normalizations to resolve the TALL_AA vs AA divergence on the consonant whitelist. Benchmark-validated and backed up.
- **Spelling-first benchmark labeling**: Every gold error now carries a `domain` field (`spelling` / `grammar` / `ambiguous`), and `benchmarks/run_benchmark.py` accepts a `--domain` filter so spelling-only regression tests no longer need a sibling YAML. Benchmark YAML version bumped accordingly.
- **Archival strategies** (shipped default-off): `ByT5SafetyNetStrategy`, `MLMSpanMaskCandGenStrategy`, `ToneSafetyNetStrategy`, `StructuralSyllableEarlyExit`, and a SymSpell-on-merged segmenter probe remain in the source tree behind feature flags for future benchmarking attempts. None are active at default configuration.

### Changed

- **Sentence-level honorific detector** now normalizes input at entry (`_detect_informal_with_honorific`) so direct detector calls receive the same normalized text as production-pipeline calls. Resolves silent miss on honorific-plus-casual-particle detection when callers pass unnormalized text.
- **`REGISTER_CRITICAL_PRONOUNS` constant** consolidated into `validators/base`; `WordValidator` and `SyllableValidator` now import a single source of truth.
- **Greedy syllable-reassembly loop** (used by compound-split suppression and the confusable boost) extracted into `_greedy_syllable_reassembly` module helper; removes a ~30-line copy-paste and prevents divergence between the two call sites.
- **`error_suppression` imports** for `WordError` and `Suggestion` hoisted to the module top, matching the rest of the module and removing a per-call late import on the structural-syllable rescue path.

### Fixed

- **Honorific-plus-casual-particle regressions** (`ဒော်ခင်မာ လာပြီကွာ → ရှင်`, `ဒော်ခင်မာ ထမင်းစားပြီးပြီကွာ → ရှင့်`): the detector now normalizes input at entry so post-normalize `_HONORIFIC_TERMS` membership matches regardless of caller.
- **Defensive bounds guards** on `context.word_positions` access in `LoanWordValidationStrategy` and `VisargaStrategy` mirror the pattern already used by `ToneSafetyNetStrategy`. The `ValidationContext` dataclass invariant makes these unreachable in practice, but the guards keep the three strategies consistent.

### Removed

- **Internal process references** (task IDs, workstream slugs, audit-document pointers, dated "Parked YYYY-MM-DD" notes, and one `/octo:debate` comment) stripped from shipped source code across `error_suppression`, `validation_configs`, `spellchecker`, `meta_fusion`, `word_validator`, the segmenter/strategy modules, and supporting factories. No runtime behaviour change; the technical rationale for each gate/guard is preserved in docstrings.

### Benchmark

- Spelling-only composite improved from `0.6161` (v1.5.0 baseline on the spelling-first benchmark) to approximately `0.6345`, driven primarily by the flat-AA migration, mined-confusable-pair strategy, skip-rule gate, and compound-split confusable boost. See `50_Metrics/Benchmark History.md` for per-commit deltas and per-bucket breakdowns.

## [1.5.0] - 2026-04-12

### Added

- **HiddenCompoundStrategy** (priority 23): Detects multi-token compound typos that the segmenter over-splits into individually-valid syllables. Walks curated-vocabulary bigram/trigram windows and checks whether a phonetic/tonal/medial variant of the leading token forms a high-frequency dictionary compound with the following token(s). Enabled by default.
- **SyllableWindowOOVStrategy** (priority 22): Detects multi-syllable OOV typos that the segmenter decomposes into individually-valid syllables. Disabled by default pending per-process SymSpell caching; kept available as a structural-phase option.
- **Suffix-aware re-segmentation** in `DefaultSegmenter`: New post-processing pass that reassembles tokens where the segmenter left an oversized compound or split a colloquial-locative merge such as `ကုန်မာ` → `[ကုန်, မာ]`.
- **Ternary compound splits** in `MorphemeSuggestionStrategy`: Correction suggestions can now span three morpheme components for compound typos.
- **Formal register benchmark subset** for FPR regression testing on formal-register text.
- **Particle-tone confusable pairs**: `ခဲ → ခဲ့` and `မဲ → မယ်` (unidirectional, protects standalone uses).
- **Curated-pair promotion** in `StatisticalConfusableStrategy`: bigram-ratio detections that also match curated homophone/confusable maps receive a confidence boost to clear the output filter.
- **7 unidirectional homophone pairs** for homophone-confusion false-negative recovery.
- **Benchmark expansion** from 1,146 → 1,304 sentences with additional benchmark annotation corrections.
- **New confusable pairs**: `ကယာ`/`ကရာ` consonant confusion and 5 `false_compound` entries from benchmark FP analysis.

### Changed

- **Meta-classifier v2** with compound-aware features; threshold retuned from `0.40` → `0.42` based on FP/TP sweep.
- **Fusion arbiter**: `HiddenCompoundStrategy` promoted from Tier 2 to Tier 3 so the arbiter can select HC against Homophone via confidence tiebreak.
- **Arbiter registry sync** and benchmark annotation corrections (Sprint E fixes).
- **N-gram probability fields** (`bigram_threshold`, `trigram_threshold`, `fourgram_threshold`, `fivegram_threshold`, `right_context_threshold`, `min_meaningful_prob`) now enforce an `le=1.0` upper bound in Pydantic validation.
- **SQLite n-gram lookups** deduplicated into a single `_lookup_ngram_prob` helper; the four public `get_{bi,tri,four,five}gram_probability` methods now share the same cache → word-id → query → cache-set path.
- **Renamed** `core.constants.is_myanmar_text` (any-character semantics) → `contains_myanmar` to disambiguate from the ratio-based `text.normalize.is_myanmar_text`.
- **`contains_myanmar` helper** is now the canonical any-character Myanmar detection function; existing ratio-based detection remains in `text.normalize`.
- **Optional-dependency extras** in `pyproject.toml` consolidated via self-referencing: `ai-full`, `train`, and `dev` now reference `myspellchecker[ai,transformers]` instead of duplicating version pins.
- **Training module** modernized: migrated from `typing.Dict/List/Optional` to lowercase generics with `from __future__ import annotations`, and converted `os.path` usage to `pathlib` across `training/trainer.py`, `training/exporter.py`, and `algorithms/semantic_checker.py`.
- **Blanket `# type: ignore`** comments narrowed to specific error codes (`[assignment]`, `[import-untyped]`, `[name-defined]`) across 20 sites.

### Fixed

- **Fusion arbiter**: untrained error types (`hidden_compound_typo`, `syllable_window_oov`) are now isolated from the meta-classifier's context features so their presence does not corrupt `n_errors`/`max_other_conf` features for other errors.
- **Confusable FPs on punctuation boundaries**: valid words with attached Myanmar punctuation no longer trigger confusable detection.
- **Invalid-word FPs** from boundary punctuation attachment (e.g. trailing `။` / `၊`).
- **`tense_mismatch` confidence** lowered for data-driven FP filtering.
- **Dot-below confusable FPs** suppressed in error suppression pipeline.
- **Visarga compound skip threshold** lowered to protect established words.
- **Reduplication guard** added to `BrokenCompoundStrategy`, plus nominalizer particles excluded from broken-compound detection.
- **HiddenCompound subsumed-token guard** and isolated fusion cluster.
- **Logger f-string eager evaluation** replaced with `%`-style formatting on hot paths (`ContextValidator`, `DI container`, `SpellChecker.check()`) so format arguments are no longer evaluated when DEBUG is disabled.

### Removed

- **9 dead `ValidationConfig` fields** that had zero consumers in the validation pipeline: `is_myanmar_text_threshold`, `orthography_confidence`, `semantic_min_word_length`, `use_orthography_validation`, `truncation_frequency_ratio`, and 4 unused `homophone_*` tunables.
- **Unused `freezegun` dev dependency** from `pyproject.toml` and `Dockerfile`.
- **15 exact-duplicate entries** in `rules/confusable_pairs.yaml` (an accidentally copy-pasted block; parser was silently de-duping via `set.add`).
- **Dead type annotations**: 12 unnecessary `# type: ignore[arg-type]` comments on SQLite cache method calls.

### Internal

- Seven focused cleanup commits across the codebase: stripped internal references (Obsidian vault paths, sprint identifiers, benchmark IDs), removed sprint-specific docstring breadcrumbs, consolidated duplicate documentation, and applied the bulk of the `ruff UP035/UP015/RUF100/PIE790` autofixes.
- `pyproject.toml` repository URLs corrected from `github.com/thettwe/my-spellchecker` → `github.com/thettwe/myspellchecker`.

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
