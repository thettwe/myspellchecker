"""
Validation and Provider Configuration Classes.

This module contains configuration classes for validation and provider settings:
- ValidationConfig: Error validation and confidence scoring
- ProviderConfig: Dictionary provider caching and connection pooling
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ValidationConfig(BaseModel):
    """
    Configuration for error validation and confidence scoring.

    Controls error detection thresholds and confidence scores.

    Attributes:
        syllable_error_confidence: Confidence for syllable errors (default: 1.0).
        word_error_confidence: Confidence for word errors (default: 0.8).
        context_error_confidence_high: High confidence for context errors (default: 0.9).
        context_error_confidence_low: Low confidence for context errors (default: 0.6).
        max_syllable_length: Maximum valid syllable length (default: 12).
        syllable_corruption_threshold: Threshold for syllable corruption (default: 3).
        use_zawgyi_detection: Enable Zawgyi encoding detection (default: True).
        use_zawgyi_conversion: Enable automatic Zawgyi to Unicode conversion (default: True).
        zawgyi_confidence_threshold: Confidence threshold for Zawgyi detection (default: 0.95).
        medial_confusion_confidence: Confidence for medial confusion corrections (default: 0.85).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    syllable_error_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for syllable-level errors (higher = more certain)",
    )
    word_error_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score for word-level errors",
    )
    context_error_confidence_high: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="High confidence score for context-level errors",
    )
    context_error_confidence_low: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Low confidence score for context-level errors",
    )
    max_syllable_length: int = Field(
        default=12,
        ge=1,
        description="Maximum allowed syllable length in characters",
    )
    syllable_corruption_threshold: int = Field(
        default=3,
        ge=1,
        description="Threshold for detecting corrupted syllables",
    )
    use_zawgyi_detection: bool = Field(
        default=True,
        description="Enable detection of legacy Zawgyi encoding",
    )
    use_zawgyi_conversion: bool = Field(
        default=True,
        description="Enable automatic conversion from Zawgyi to Unicode",
    )
    zawgyi_confidence_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for Zawgyi detection",
    )
    strict_validation: bool = Field(
        default=True,
        description="Enable strict validation rules",
    )
    medial_confusion_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence score for medial confusion corrections (ျ vs ြ)",
    )
    raise_on_strategy_error: bool = Field(
        default=False,
        description=(
            "If True, re-raise exceptions from validation strategies instead of catching them. "
            "Useful for debugging when a strategy fails unexpectedly. "
            "In production, keep False to allow graceful degradation when a strategy fails."
        ),
    )
    # Colloquial variant handling
    colloquial_strictness: Literal["strict", "lenient", "off"] = Field(
        default="lenient",
        description=(
            "Strictness level for colloquial spelling variants. "
            "Options: 'strict' (flag all colloquial variants as errors), "
            "'lenient' (accept colloquial variants with info note), "
            "'off' (no special handling for colloquial variants). "
            "Colloquial variants include informal spellings like 'ကျနော်' (colloquial) "
            "vs 'ကျွန်တော်' (standard)."
        ),
    )
    colloquial_info_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for colloquial variant informational notes. "
            "Lower values indicate 'less certain' errors (more like suggestions). "
            "Only used when colloquial_strictness='lenient'."
        ),
    )
    # Extended Myanmar character handling
    allow_extended_myanmar: bool = Field(
        default=False,
        description=(
            "Allow Extended Myanmar characters for non-Burmese Myanmar-script languages. "
            "When True, enables: "
            "(1) Extended Core Block (U+1050-U+109F) for Shan/Mon/Karen in main block; "
            "(2) Extended-A (U+AA60-U+AA7F) for Shan, Khamti Shan, Aiton, Phake, Pa'O Karen; "
            "(3) Extended-B (U+A9E0-U+A9FF) for Shan, Pa'O; "
            "(4) Non-standard core chars (U+1022, U+1028, U+1033-U+1035) for Mon/Shan. "
            "False (default) enforces strict Burmese-only scope (U+1000-U+104F minus non-standard)."
        ),
    )
    stacking_pairs_path: str | None = Field(
        default=None,
        description=(
            "Path to a YAML file defining valid virama stacking pairs. "
            "If None, uses the built-in stacking_pairs.yaml from the rules directory. "
            "The YAML file organizes pairs into categories (gemination, cross_aspirated, "
            "cross_row, etc.), each with an 'enabled' flag for category-level toggling."
        ),
    )
    # Strategy-specific confidence thresholds
    tone_validation_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for tone validation strategy",
    )
    syntactic_validation_confidence: float = Field(
        default=0.78,
        ge=0.0,
        le=1.0,
        description="Confidence for syntactic (rule-based) validation strategy",
    )
    pos_sequence_confidence: float = Field(
        default=0.68,
        ge=0.0,
        le=1.0,
        description="Confidence for POS sequence validation strategy",
    )
    question_structure_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence for question structure validation strategy",
    )
    homophone_confidence: float = Field(
        default=0.68,
        ge=0.0,
        le=1.0,
        description="Confidence for homophone validation strategy",
    )
    use_homophone_detection: bool = Field(
        default=True,
        description="Enable homophone detection in validation pipeline",
    )
    # Loan Word Detection (priority 22)
    use_loan_word_detection: bool = Field(
        default=True,
        description=(
            "Enable loan word transliteration error detection at priority 22. "
            "Detects valid-in-DB loan word variants and suggests standard forms."
        ),
    )
    loan_word_detection_confidence: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence for loan-word variant short-circuit in WordValidator "
            "(Prong-3 propagation fix). Fires when an OOV word is a known "
            "variant in loan_words.yaml or loan_words_mined.yaml, bypassing "
            "SymSpell's max_edit_distance gate. High default since the "
            "variant lookup is rule-based and curated/gated by linguist review."
        ),
    )
    # Segmenter post-merge rescue (default off until FPR calibration lands).
    use_segmenter_post_merge_rescue: bool = Field(
        default=False,
        description=(
            "Enable post-segmentation probe-and-merge rescue in WordValidator. "
            "For each adjacent fragment pair (a, b) in the segmenter output, "
            "probe `a+b` against the loan-word variant map, dict, dict+asat, "
            "and bigram store. If any probe hits, replace (a, b) with the "
            "merged token for validation. Off by default until FPR calibration "
            "confirms the thresholds below hold."
        ),
    )
    segmenter_merge_bigram_threshold: float = Field(
        default=-1.0,
        ge=-1.0,
        description=(
            "Minimum bigram probability for the weakest merge-probe (bigram "
            "association). Negative = probe 4 disabled (the default). Setting "
            "to 0.0 accepts any positive probability; higher = stricter. "
            "Probe 4 is hard to calibrate cleanly — prior sweeps showed large "
            "FPR regressions even with fragment-rarity guards. Kept in code "
            "for future calibration but disabled by default."
        ),
    )
    # Probe-5: SymSpell on merged string. Off by default — opens the
    # "merged near-match" path targeting segmenter over-split tokens where
    # the merged token differs from the gold by a small edit distance.
    use_segmenter_merge_symspell_probe: bool = Field(
        default=False,
        description=(
            "Enable Probe-5 in the segmenter post-merge rescue. After probes "
            "1-4 fail, run SymSpell on the merged string; if a candidate with "
            "frequency >= segmenter_merge_symspell_min_freq exists at "
            "edit_distance <= segmenter_merge_symspell_max_ed, accept the "
            "merge and let WordValidator emit the correction. Requires "
            "use_segmenter_post_merge_rescue=True."
        ),
    )
    segmenter_merge_symspell_max_ed: int = Field(
        default=2,
        ge=1,
        le=3,
        description=(
            "Max edit distance for the Probe-5 SymSpell lookup on merged strings. Default 2."
        ),
    )
    segmenter_merge_symspell_min_freq: int = Field(
        default=100,
        ge=0,
        description=(
            "Minimum corpus frequency for the Probe-5 SymSpell top-1 "
            "candidate. Filters low-quality dict neighbours. Default 100."
        ),
    )
    segmenter_merge_symspell_min_merged_len: int = Field(
        default=4,
        ge=2,
        description=(
            "Minimum character length of the merged string to run Probe-5. "
            "Short merges are too prone to spurious ed<=2 matches. Default 4."
        ),
    )
    # Statistical Confusable Gate (priority 24)
    use_statistical_confusable_gate: bool = Field(
        default=True,
        description=(
            "Enable bigram-ratio confusable detection at priority 24. "
            "Runs within the structural phase to avoid fast-path skip."
        ),
    )
    statistical_confusable_threshold: float = Field(
        default=5.0,
        ge=1.0,
        description=(
            "Bigram ratio threshold for the statistical confusable gate. "
            "Higher = fewer detections, more precision."
        ),
    )

    # MLP Confusable/Compound Classifier (priority 47)
    confusable_compound_classifier_path: str | None = Field(
        default=None,
        description=(
            "Path to ONNX MLP classifier for confusable/compound detection. "
            "When set, enables the ConfusableCompoundClassifierStrategy "
            "at priority 47. Independent of MLM and error budget."
        ),
    )
    confusable_compound_classifier_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Sigmoid threshold for the MLP classifier.",
    )

    # Confusable Semantic Detection (MLM-enhanced, priority 48)
    use_confusable_semantic: bool = Field(
        default=True,
        description=(
            "Enable MLM-enhanced confusable detection (priority 48). "
            "Uses predict_mask() to detect valid-word confusables missed by n-gram. "
            "Requires semantic model to be loaded."
        ),
    )
    confusable_semantic_confidence: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Confidence score for confusable semantic errors.",
    )
    confusable_semantic_top_k: int = Field(
        default=50,
        ge=5,
        description="Number of top predictions to request from predict_mask.",
    )
    confusable_semantic_logit_diff: float = Field(
        default=2.5,
        ge=0.0,
        description=(
            "Default logit diff threshold (~12x probability ratio). "
            "Variant must score this much higher than current word. "
            "Lowered from 3.0 to improve confusable recall."
        ),
    )
    confusable_semantic_logit_diff_medial: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Logit diff threshold for medial ျ↔ြ swaps (~7x ratio). "
            "Lower because these are highest-signal confusables."
        ),
    )
    confusable_semantic_logit_diff_current_in_topk: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Logit diff threshold when current word IS in top-K (~148x ratio). "
            "Need stronger evidence when model already considers current word."
        ),
    )
    confusable_semantic_high_freq_threshold: int = Field(
        default=50000,
        ge=0,
        description="Word frequency above which the high-freq logit diff applies.",
    )
    confusable_semantic_high_freq_logit_diff: float = Field(
        default=3.5,
        ge=0.0,
        description=(
            "Logit diff threshold for high-frequency words (~33x ratio). "
            "Protects common particles/words against false positives. "
            "Lowered from 6.0 to enable detection of Kinzi and stacking confusables."
        ),
    )
    confusable_semantic_freq_ratio_penalty_high: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Extra threshold penalty when variant_freq/word_freq > 5.0. "
            "Counters MLM frequency bias for much-more-common confusables. "
            "(Conservative: 3.0)"
        ),
    )
    confusable_semantic_freq_ratio_penalty_mid: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Extra threshold penalty when variant_freq/word_freq > 2.0. "
            "Moderate guard against frequency-biased MLM predictions. "
            "(Conservative: 1.5)"
        ),
    )
    confusable_semantic_visarga_penalty: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Extra threshold penalty when word and variant differ only by visarga (း). "
            "Visarga pairs are almost always different morphemes in Myanmar."
        ),
    )
    confusable_semantic_sentence_final_penalty: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Extra threshold penalty for sentence-final words. "
            "MLM has no right context at sentence boundaries, reducing reliability. "
            "(Conservative: 1.0)"
        ),
    )
    confusable_semantic_max_threshold: float = Field(
        default=8.0,
        ge=0.0,
        description=(
            "Maximum allowed stacked threshold. Caps total threshold after "
            "all penalties. Prevents impossible thresholds that silence the model. "
            "Set to 0 to disable capping. (Conservative: 0 / no cap)"
        ),
    )
    confusable_semantic_reverse_ratio_min_freq: int = Field(
        default=50000,
        ge=0,
        description=(
            "Only apply reverse_ratio penalty when word_freq >= this value. "
            "Low-freq misspellings with high reverse ratio are real errors. "
            "(Conservative: 0 / always apply)"
        ),
    )
    confusable_semantic_visarga_high_freq_hard_block: bool = Field(
        default=True,
        description=(
            "When True, hard-block visarga pairs where both are high-frequency. "
            "When False, use a high but reachable threshold instead. "
            "Keep True — ပြီ/ပြီး FP proves the block is needed."
        ),
    )

    # Output confidence filter thresholds
    # Calibration: confusable_error FPs cluster at 0.72, TPs at 0.88+.
    # colloquial_info notes use confidence 0.3 (informational, not errors)
    # and are suppressed by default; lower the threshold to surface them.
    output_confidence_thresholds: dict[str, float] = Field(
        default={},
        description=(
            "Per-error-type minimum confidence for the output filter. "
            "Errors whose confidence is below the threshold for their type "
            "are suppressed. Empty by default — the meta-classifier handles "
            "FP suppression with learned per-type boundaries. Set explicit "
            "thresholds here only if use_meta_classifier=False."
        ),
    )
    secondary_confidence_thresholds: dict[str, float] = Field(
        default={},
        description=(
            "Cascade guard: suppress error types when the sentence already "
            "has a higher-confidence error of a different type. Empty by "
            "default — the meta-classifier's context features handle this. "
            "Set explicit thresholds here only if use_meta_classifier=False."
        ),
    )

    # -- Candidate fusion (voting architecture) --
    use_candidate_fusion: bool = Field(
        default=True,
        description=(
            "Enable calibrated Noisy-OR candidate fusion instead of mutex-based "
            "winner selection. When True, all strategies may fire at every position "
            "(mutex bypass), fast-path exit is automatically disabled, and the "
            "arbiter uses calibrated confidence fusion "
            "across independence clusters to determine which errors to emit. "
            "When False, falls back to mutex-based winner selection with shadow mode."
        ),
    )
    fusion_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fused confidence (after calibrated Noisy-OR) to emit an error "
            "in candidate-fusion mode. Lower values increase recall but risk FPR."
        ),
    )
    calibration_path: str | None = Field(
        default=None,
        description=(
            "Path to a YAML file with per-strategy calibration breakpoints "
            "and reliability weights, as produced by train_calibrators.py. "
            "When set and use_candidate_fusion=True, the fusion pipeline "
            "uses data-driven calibration instead of bootstrap weights."
        ),
    )

    use_meta_classifier: bool = Field(
        default=True,
        description=(
            "Use a learned meta-classifier as post-validation error filter. "
            "When True, a logistic regression model scores every error and "
            "suppresses likely false positives. Replaces hand-tuned "
            "output_confidence_thresholds with learned per-type boundaries. "
            "Falls back to manual thresholds if model is unavailable."
        ),
    )
    meta_classifier_path: str | None = Field(
        default=None,
        description=(
            "Path to a YAML file with meta-classifier coefficients, "
            "as produced by train_meta_classifier.py. When None and "
            "use_meta_classifier=True, attempts to load from bundled "
            "rules/meta_classifier.yaml."
        ),
    )

    suppression_immune_strategies: frozenset[str] = Field(
        default=frozenset(),
        description=(
            "Set of strategy class names whose errors are immune to post-context "
            "suppression rules. Errors from these strategies pass through suppression "
            "untouched. Use for strategies empirically shown to have high precision "
            "but whose TPs are incorrectly suppressed."
        ),
    )

    use_pos_sequence: bool = Field(
        default=True,
        description="Enable POSSequenceValidationStrategy in context validation.",
    )
    use_ngram_context: bool = Field(
        default=True,
        description="Enable NgramContextValidationStrategy in context validation.",
    )

    enable_fast_path: bool = Field(
        default=True,
        description=(
            "Enable fast-path exit in context validation. When True, if structural "
            "strategies (Tone, Orthography, Syntactic, BrokenCompound) find no errors, "
            "contextual strategies are skipped. Reduces FPR on clean text but may miss "
            "context-only errors. Set to False for maximum recall. "
            "Note: automatically disabled when use_candidate_fusion=True."
        ),
    )

    enable_strategy_timing: bool = Field(
        default=False,
        description=(
            "Enable per-strategy timing instrumentation. "
            "When True, each strategy's execution time is logged at DEBUG level "
            "and accumulated in ContextValidator.strategy_timings."
        ),
    )
    enable_strategy_debug: bool = Field(
        default=False,
        description=(
            "Enable per-strategy gate-debug telemetry in ContextValidator. "
            "Captures emitted/new positions per strategy and semantic overlap "
            "suppression (shadow potential vs actual). Increases latency."
        ),
    )
    debug_blocked_example_limit: int = Field(
        default=12,
        ge=0,
        description=(
            "Maximum number of blocked-example entries to collect per strategy "
            "in strategy-debug telemetry. Controls memory usage of debug output."
        ),
    )

    # -- Segment-skip guard tuning (WordValidator) --
    segment_skip_min_freq: int = Field(
        default=80,
        ge=0,
        description=(
            "Minimum corpus frequency for a SymSpell candidate to be considered "
            "a strong whole-word correction when deciding whether to skip "
            "segmenter-merge validation of multi-syllable OOV tokens."
        ),
    )
    segment_skip_min_ratio: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Minimum frequency ratio (candidate_freq / current_freq) for a "
            "SymSpell candidate to be considered a strong whole-word correction."
        ),
    )
    segment_skip_max_edit_distance: int = Field(
        default=2,
        ge=0,
        description=(
            "Maximum edit distance for a SymSpell candidate to qualify as a "
            "strong whole-word correction for segment-skip decisions."
        ),
    )
    segment_skip_max_length: int = Field(
        default=24,
        ge=1,
        description=(
            "Maximum word length (in characters) for segment-skip candidate "
            "checking. Longer words are skipped to avoid expensive lookups."
        ),
    )

    # -- Bare-consonant merge guard (SyllableValidator) --
    bare_consonant_merge_min_freq: int = Field(
        default=1000,
        ge=0,
        description=(
            "Minimum frequency for the merged word when suppressing "
            "bare-consonant segmenter artifacts. Higher values filter out "
            "noise while catching common compound boundaries."
        ),
    )

    # Morphological synthesis: reduplication validation
    use_reduplication_validation: bool = Field(
        default=True,
        description=(
            "Enable productive reduplication validation for OOV words. "
            "When True, words that are valid reduplications of known dictionary "
            "words (e.g., ကောင်းကောင်း from ကောင်း) are accepted without errors."
        ),
    )
    reduplication_min_base_frequency: int = Field(
        default=5,
        ge=0,
        description=(
            "Minimum corpus frequency for the base word in reduplication validation. "
            "Prevents accepting reduplications of rare or noisy dictionary entries."
        ),
    )
    reduplication_cache_size: int = Field(
        default=1024,
        ge=0,
        description="Cache size for reduplication analysis results (0=disabled).",
    )

    # Morphological synthesis: compound resolution
    use_compound_synthesis: bool = Field(
        default=True,
        description=(
            "Enable compound word synthesis validation for OOV words. "
            "When True, words that split into valid dictionary morphemes "
            "(e.g., ကျောင်းသား = ကျောင်း + သား) are accepted without errors."
        ),
    )
    compound_min_morpheme_frequency: int = Field(
        default=50,
        ge=0,
        description=(
            "Minimum corpus frequency for each morpheme in compound validation. "
            "Higher than reduplication floor since compound splits are more speculative."
        ),
    )
    compound_max_parts: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Maximum number of parts allowed in compound word splitting.",
    )
    compound_cache_size: int = Field(
        default=1024,
        ge=0,
        description="Cache size for compound resolution results (0=disabled).",
    )

    # Word validator compound bigram thresholds
    min_syllable_word_freq: int = Field(
        default=50,
        ge=0,
        description="Minimum frequency for syllables in compound bigram validation.",
    )
    asat_freq_guard: int = Field(
        default=1000,
        ge=0,
        description="Frequency threshold for prefix-asat and compound synthesis guards.",
    )

    # Structural syllable early-exit.
    # When `syllable_validator` emits `invalid_syllable` AND the
    # `syllable_rule_validator.validate()` returns False (categorical
    # Myanmar language violation: two consecutive vowels, broken stacking,
    # medial-vowel reorder, etc.), AND the enclosing segmenter token is
    # OOV, AND SymSpell returns a top-1 hit at ed≤max_ed freq≥min_freq,
    # emit an authoritative `invalid_word` on the enclosing token and
    # bypass downstream suppressors.
    #
    # Disabled by default: when benchmarked, baseline strategies already
    # covered the same FN subset on their own, so the rescue added
    # negligible TPs. Kept archived behind a flag for future revival.
    structural_syllable_early_exit_enabled: bool = Field(
        default=False,
        description=(
            "Enable structural-syllable early-exit rescue. When the "
            "syllable rule validator rejects a syllable AND the enclosing "
            "segmenter token has a confident SymSpell correction, emit an "
            "authoritative word-level error bypassing downstream filters. "
            "Disabled by default (archived); see CHANGELOG for rationale."
        ),
    )
    structural_syllable_early_exit_max_ed: int = Field(
        default=1,
        ge=0,
        le=3,
        description=(
            "Max SymSpell edit distance for the structural-syllable "
            "early-exit gate. Default 1 for high precision; raising to 2 "
            "widens recall at a precision cost."
        ),
    )
    structural_syllable_early_exit_min_freq: int = Field(
        default=500,
        ge=0,
        description=(
            "Min SymSpell top-1 frequency for the structural-syllable "
            "early-exit gate. 500 chosen for precision; lowering adds "
            "TPs at a precision cost."
        ),
    )

    # Compound-split confusable boost.
    # When `_suppress_compound_split_valid_words` would fire on a long OOV
    # token whose syllables are all individually valid (4+ syllables),
    # the same structural signal that marks it as a "benign merge" ALSO
    # indicates an inner confusable_error is more likely a real typo than
    # a clean-text FP. Boosting inner confusable confidence pushes
    # eligible emissions past the _CONFIDENCE_THRESHOLDS['confusable_error']
    # gate at 100% clean-text precision in prior benchmarking.
    compound_split_confusable_boost_enabled: bool = Field(
        default=True,
        description=(
            "Enable confidence boost for inner confusable_error emissions "
            "inside compound-split-suppressible spans. Structural AND-gate: "
            "boost only fires when BOTH (a) a long token would be killed "
            "by _suppress_compound_split_valid_words AND (b) a confusable "
            "emission exists at a sub-span below the ceiling."
        ),
    )
    compound_split_confusable_boost: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence increment applied to eligible inner confusable "
            "emissions. Default 0.20 pushes conf ≥ 0.55 past the "
            "_CONFIDENCE_THRESHOLDS['confusable_error']=0.75 gate. "
            "Final confidence is clipped at 1.0."
        ),
    )
    compound_split_confusable_boost_inner_conf_ceiling: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Only boost inner confusable emissions below this ceiling. "
            "Default matches _CONFIDENCE_THRESHOLDS['confusable_error']; "
            "emissions at or above already pass the downstream gate so "
            "boosting them is a no-op."
        ),
    )
    compound_split_confusable_boost_min_syllables: int = Field(
        default=4,
        ge=2,
        description=(
            "Minimum syllables in the outer token for the structural "
            "signal to fire. Matches _suppress_compound_split_valid_words "
            "'4+ syllables' predicate; keep in sync if that rule changes."
        ),
    )

    # Skip-rule confidence gate.
    # At word_validator.py the 4+syllable-all-valid-parts unconditional skip
    # was added to suppress FPs on genuine compound/verb-chain merges. That
    # rule also hid true-positive missing-asat / substitution typos whose
    # fragmented form happens to produce all-valid syllables (canonical
    # example: "စွမ်းဆောင်ရည" → ["စွမ်း", "ဆောင်", "ရ", "ည"], all dict-valid,
    # but SymSpell finds "စွမ်းဆောင်ရည်" at ed=1 freq 48k). The confidence
    # gate lets validation proceed when SymSpell has a strong top-1 candidate.
    skip_rule_gate_max_ed: int = Field(
        default=2,
        ge=0,
        le=3,
        description=(
            "Max SymSpell edit distance for the skip-rule confidence gate. "
            "When the 4+syllable-all-valid skip predicate fires, validation "
            "is allowed to proceed if SymSpell's top-1 candidate is within "
            "this edit distance and clears skip_rule_gate_min_freq."
        ),
    )
    skip_rule_gate_min_freq: int = Field(
        default=1000,
        ge=0,
        description=(
            "Min SymSpell top-1 candidate frequency for the skip-rule "
            "confidence gate. The default threshold balances precision "
            "against recall recovery; raising it tightens precision."
        ),
    )

    # Broken compound detection (wrongly split compound words)
    use_broken_compound_detection: bool = Field(
        default=True,
        description=(
            "Enable detection of wrongly split compound words. "
            "When True, adjacent word pairs that form a known compound "
            "(e.g., 'မနက် ဖြန်' → 'မနက်ဖြန်') are flagged."
        ),
    )
    broken_compound_rare_threshold: int = Field(
        default=500,
        ge=0,
        description=(
            "Maximum word frequency for a word to be considered 'rare'. "
            "At least one word in the pair must be below this threshold."
        ),
    )
    broken_compound_min_frequency: int = Field(
        default=10000,
        ge=0,
        description="Minimum compound frequency required to flag a broken compound.",
    )
    broken_compound_ratio: float = Field(
        default=25.0,
        ge=1.0,
        description="Minimum ratio of compound_freq / rare_word_freq to flag.",
    )
    broken_compound_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score for broken compound errors.",
    )

    # Hidden compound typo detection (priority 23, structural phase)
    use_hidden_compound_detection: bool = Field(
        default=True,
        description=(
            "Enable detection of hidden compound typos. When True, "
            "HiddenCompoundStrategy runs at priority 23 (before "
            "StatisticalConfusable and BrokenCompound). It examines multi-token "
            "windows where each individual token is valid but a confusable "
            "variant of w_i would form a high-frequency compound with the "
            "following token(s)."
        ),
    )
    hidden_compound_max_token_syllables: int = Field(
        default=3,
        ge=1,
        le=6,
        description=(
            "Maximum syllable count for a token to be considered as a typo "
            "source. Longer tokens are skipped for performance."
        ),
    )
    hidden_compound_max_variants_per_token: int = Field(
        default=20,
        ge=1,
        le=100,
        description=(
            "Maximum confusable variants to generate per candidate token. "
            "Caps cost of dictionary lookups per sentence."
        ),
    )
    hidden_compound_min_frequency: int = Field(
        default=100,
        ge=0,
        description=(
            "Minimum compound frequency required to flag a hidden compound "
            "typo. Mirrors broken_compound_min_frequency but lower because "
            "we also permit freq=0 subsumed compounds via trigram lookahead."
        ),
    )
    hidden_compound_confidence_floor: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum confidence to emit a hidden compound typo error. "
            "Higher floor suppresses marginal corrections."
        ),
    )
    hidden_compound_enable_trigram_lookahead: bool = Field(
        default=True,
        description=(
            "When the bigram variant lookup returns a freq=0 valid dictionary "
            "entry (subsumed compound case), try the trigram variant (v + "
            "w_next + w_next2) as well."
        ),
    )
    hidden_compound_variant_cache_size: int = Field(
        default=8192,
        ge=64,
        description=(
            "LRU cache size for generate_confusable_variants() results per "
            "process. Larger cache reduces CPU cost at the expense of memory."
        ),
    )
    hidden_compound_require_typo_prone_chars: bool = Field(
        default=True,
        description=(
            "Only process tokens containing at least one typo-prone character "
            "(e.g., ခ/က, ပ/ဖ, ြ/ျ). Pure-vowel or pure-suffix tokens are "
            "never sources of confusable typos and can be skipped."
        ),
    )
    hidden_compound_curated_only: bool = Field(
        default=True,
        description=(
            "Require both w_i and w_next to satisfy is_valid_vocabulary() "
            "(curated=1). Rejects segmenter artifacts with boundary drift."
        ),
    )

    # Pre-segmenter raw-token SymSpell probe (priority 23, structural phase).
    # Runs SymSpell.lookup(raw_token, level='word') on whitespace/punctuation-
    # delimited tokens BEFORE they reach the segmenter. Recovers compound typos
    # the segmenter would otherwise fragment into piecewise-valid subtokens
    # (e.g. "စွမ်းဆောင်ရည" → ["စွမ်းဆောင်", "ရ", "ည"] hides the asat drop).
    use_pre_segmenter_raw_probe: bool = Field(
        default=True,
        description=(
            "Run SymSpell.lookup(raw_token, level='word') on unsegmented "
            "whitespace-delimited tokens before segmentation. Catches typo'd "
            "compounds whose segmenter fragmentation makes them invisible to "
            "piecewise strategies."
        ),
    )
    pre_segmenter_raw_probe_max_ed: int = Field(
        default=2,
        ge=1,
        le=3,
        description=(
            "Maximum SymSpell edit distance accepted for a raw-token probe hit. "
            "Raising above 2 introduces a larger FP hit (real-word confusion) "
            "than candidate-gen gain; keep at 2 unless a dedicated precision "
            "effort has lowered that ceiling."
        ),
    )
    pre_segmenter_raw_probe_min_freq: int = Field(
        default=100,
        ge=0,
        description=(
            "Minimum dictionary frequency for a SymSpell suggestion to be "
            "emitted by the raw-token probe. Guards against swapping a rare "
            "OOV compound with a homophone-like dict neighbour."
        ),
    )
    pre_segmenter_raw_probe_max_length: int = Field(
        default=15,
        ge=1,
        description=(
            "Maximum character length for a raw token to be probed. Matches "
            "SymSpell's default max_word_length; tokens beyond this are "
            "excluded from SymSpell's index and cannot be recovered."
        ),
    )
    pre_segmenter_raw_probe_max_length_diff: int = Field(
        default=2,
        ge=0,
        description=(
            "Maximum absolute length difference between the raw token and the "
            "suggested candidate. Larger differences usually indicate a wrong "
            "compound join or phrase substitution rather than a typo."
        ),
    )
    pre_segmenter_raw_probe_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score emitted for raw-token probe errors. Calibrated "
            "to survive the meta-classifier's learnt boundary for word-level "
            "typos without displacing higher-priority structural emissions."
        ),
    )

    # Cross-whitespace compound probe (priority 21, early structural phase).
    # Concatenates adjacent whitespace-delimited Myanmar spans and checks
    # whether the result is a known dictionary compound. Recovers compound
    # words that users split with spaces (e.g., "လူ သွား လမ်း" → "လူသွားလမ်း").
    use_cross_whitespace_probe: bool = Field(
        default=True,
        description=(
            "Concatenate adjacent whitespace-delimited Myanmar spans and "
            "check the dictionary for a compound word. Detects space-insertion "
            "errors invisible to segmentation-dependent strategies."
        ),
    )
    cross_whitespace_probe_min_freq: int = Field(
        default=7000,
        ge=0,
        description=(
            "Minimum dictionary frequency for the concatenated compound to "
            "be emitted as a correction. Set high to avoid corpus-artifact "
            "merges where both parts are valid standalone words."
        ),
    )
    cross_whitespace_probe_max_part_length: int = Field(
        default=30,
        ge=1,
        description=(
            "Maximum codepoint length per individual whitespace-delimited "
            "part. Prevents probing excessively long chunks."
        ),
    )
    cross_whitespace_probe_max_concat_length: int = Field(
        default=25,
        ge=1,
        description=("Maximum codepoint length for the concatenated compound form."),
    )
    cross_whitespace_probe_confidence: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for cross-whitespace compound corrections. "
            "Set high (0.90) because space-insertion errors have strong "
            "signal when both parts are valid words and concat is in dict."
        ),
    )

    # Compound merge probe (priority 46, late detection phase).
    # Slides a token-level window across segmented words, concatenates
    # adjacent tokens, and probes SymSpell for compound corrections.
    # Recovers over-split compounds invisible to per-word strategies.
    use_compound_merge_probe: bool = Field(
        default=True,
        description=(
            "Slide a window of 2–N adjacent segmented tokens, concatenate, "
            "and probe SymSpell for compound corrections. Catches typo'd "
            "compounds the segmenter fragments into valid subtokens."
        ),
    )
    compound_merge_probe_max_window: int = Field(
        default=3,
        ge=2,
        le=6,
        description="Maximum number of adjacent tokens to merge in a single probe.",
    )
    compound_merge_probe_max_span_length: int = Field(
        default=20,
        ge=5,
        description="Skip merged spans longer than this (chars).",
    )
    compound_merge_probe_max_ed: float = Field(
        default=0.4,
        ge=0.0,
        le=3.0,
        description="Maximum weighted edit distance accepted for a merge probe hit.",
    )
    compound_merge_probe_min_freq: int = Field(
        default=5000,
        ge=0,
        description="Minimum dictionary frequency for a SymSpell candidate.",
    )
    compound_merge_probe_fragment_freq_floor: int = Field(
        default=50_000,
        ge=0,
        description=(
            "At least one token in the window must have frequency below this "
            "threshold (or be OOV) to trigger the probe."
        ),
    )
    compound_merge_probe_max_length_diff: int = Field(
        default=1,
        ge=0,
        description="Maximum |len(candidate) - len(span_text)| allowed.",
    )
    compound_merge_probe_confidence: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Confidence score emitted for compound merge probe errors.",
    )
    compound_merge_probe_affinity_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Compound affinity score above which a token is treated as a "
            "likely compound fragment (bypasses the frequency-floor heuristic). "
            "Only effective when compound_affinity.json is available."
        ),
    )

    # MLM-as-candidate-generator (priority 46).
    # Wraps semantic-v2.4 RoBERTa span-masking as a production candidate
    # generator — the existing MLM is already deployed for scoring at
    # priorities 48 (ConfusableSemantic) and 70 (Semantic); this strategy
    # adds a *generation* path. Disabled by default pending a benchmark
    # gate that measures composite + FPR impact.
    use_mlm_span_mask_candgen: bool = Field(
        default=False,
        description=(
            "Enable MLMSpanMaskCandGenStrategy: use semantic-v2.4 span-masking "
            "as a production candidate generator for real-word confusions. "
            "Fires per Myanmar token, filters predictions by ED ≤ 2 + dict "
            "membership, gates on score(candidate) − score(typo) ≥ margin. "
            "Default-off pending benchmark gate."
        ),
    )
    mlm_candgen_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "Number of MLM top predictions to decode per masked token. "
            "Probe sweet spot: 10. Raising to 50 adds only ~3pp recall but "
            "doubles the FP surface because more unrelated words survive "
            "the ED filter."
        ),
    )
    mlm_candgen_margin: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Required ``score(candidate) − score(typo)`` margin. Higher = "
            "fewer emissions, higher precision."
        ),
    )
    mlm_candgen_max_ed: int = Field(
        default=2,
        ge=1,
        le=3,
        description=(
            "Maximum edit distance accepted between the typo and the MLM "
            "candidate. ED = 2 keeps the strategy on-task for typo repair; "
            "widening to 3 admits whole-word semantic replacements outside "
            "the intended scope of this strategy."
        ),
    )
    mlm_candgen_skip_above_freq: int = Field(
        default=50_000,
        ge=0,
        description=(
            "Skip tokens whose own frequency already exceeds this value. "
            "Prevents the MLM from second-guessing common words where "
            "user intent is overwhelmingly the written form."
        ),
    )
    mlm_candgen_min_token_length: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum token length probed by the MLM. Single-char tokens "
            "produce very noisy MLM predictions; skipping them trades a "
            "tiny amount of recall for a large FPR reduction."
        ),
    )
    mlm_candgen_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence stamped on MLM-candgen errors. Calibrated below "
            "the raw-token probe's 0.85 because real-word confusion carries "
            "more context ambiguity than OOV recovery."
        ),
    )

    # Tone-safety-net candidate generator (priority 22, structural phase).
    # Targets real-word confusions where the gold form differs from the
    # typo by a single trailing tone character {း, ့, ံ, ်}. Disabled by
    # default pending a benchmark gate that measures composite + FPR
    # impact.
    use_tone_safety_net: bool = Field(
        default=False,
        description=(
            "Enable ToneSafetyNetStrategy: probe trailing tone insert / delete "
            "candidates for real-word confusions where both typo and gold are "
            "valid dict words. Runs at priority 22 (before StatisticalConfusable "
            "and BrokenCompound). Default-off pending benchmark gate."
        ),
    )
    tone_safety_net_min_frequency: int = Field(
        default=1000,
        ge=0,
        description=(
            "Minimum dictionary frequency a tone-variant candidate must clear. "
            "Calibrated higher than the raw-token probe (100) because real-word "
            "confusion has an asymmetric FP cost — common tokens must stay "
            "un-flagged unless the alternative is demonstrably more frequent."
        ),
    )
    tone_safety_net_freq_ratio: float = Field(
        default=10.0,
        gt=0.0,
        description=(
            "Minimum ``freq(candidate) / freq(token)`` ratio required to emit "
            "a correction. A value of 10 means the tone-variant must be at "
            "least 10x more common than the token in the dictionary corpus. "
            "Lowering this will widen coverage at the cost of FPR."
        ),
    )
    tone_safety_net_skip_above_freq: int = Field(
        default=50000,
        ge=0,
        description=(
            "Do not probe tokens whose own frequency already exceeds this "
            "value. Guards very common words (particles, pronouns, frequent "
            "verbs) from being second-guessed by the tone-variant lookup."
        ),
    )
    tone_safety_net_confidence: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score emitted for tone-safety-net errors. Calibrated "
            "below the raw-token probe's 0.85 because real-word confusion "
            "carries more context ambiguity than OOV token recovery."
        ),
    )

    # Syllable-window OOV detection (priority 22, structural phase).
    # Disabled by default: requires per-process SymSpell caching to amortise
    # the per-window lookup cost before it is viable in production.
    use_syllable_window_oov: bool = Field(
        default=False,
        description=(
            "Enable SyllableWindowOOVStrategy: detect multi-syllable OOV typos "
            "that the segmenter decomposes into individually-valid syllables. "
            "Runs at priority 22 (before HiddenCompound, StatisticalConfusable, "
            "BrokenCompound)."
        ),
    )
    syllable_window_sizes: tuple[int, ...] = Field(
        default=(3, 4),
        description=(
            "Window sizes to enumerate (in syllables). 2-syllable windows are "
            "excluded by default; on real text they are dominated by "
            "particle-deletion false positives."
        ),
    )
    syllable_window_min_frequency: int = Field(
        default=500,
        ge=0,
        description="Minimum SymSpell suggestion frequency to emit a window error.",
    )
    syllable_window_confidence_floor: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required to emit a syllable-window OOV error.",
    )
    syllable_window_require_typo_prone: bool = Field(
        default=True,
        description=(
            "Only consider windows containing at least one typo-prone character "
            "(aspirated/unaspirated pairs, medials, nasals, etc.)."
        ),
    )
    syllable_window_skip_names: bool = Field(
        default=True,
        description=(
            "Skip windows that span any word flagged as a proper name (``context.is_name_mask``)."
        ),
    )
    syllable_window_require_valid_source_words: bool = Field(
        default=True,
        description=(
            "Only emit when every word contributing syllables to the window "
            "is individually a valid dictionary entry."
        ),
    )
    syllable_window_max_edit_distance: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Maximum SymSpell edit distance accepted for the candidate.",
    )

    # Mined confusable pair detection (priority 49, post-semantic).
    # Enabled by default in v1.7.0+: mines 4K+ confusable pairs from the DB and
    # uses semantic MLM margin to flag real-word confusable errors that SymSpell
    # cannot surface (both forms are in-dictionary). Added to suppression
    # immunity by default since pipeline suppression was observed to remove
    # valid emissions.
    use_mined_confusable_pair: bool = Field(
        default=True,
        description=(
            "Enable MinedConfusablePairStrategy: flag real-word confusables using "
            "mined ed-1 pair list with semantic MLM margin gate. Requires "
            "semantic_checker to be configured. Runs at priority 49."
        ),
    )
    mined_pair_yaml_path: str | None = Field(
        default=None,
        description=(
            "Optional path to mined pairs YAML. If None, the strategy loads "
            "the bundled `rules/mined_confusable_pairs.yaml`."
        ),
    )
    mined_pair_low_freq_min: int = Field(
        default=100,
        ge=0,
        description=(
            "Minimum frequency for the lower-frequency member of a pair to enter the partner map."
        ),
    )
    mined_pair_freq_ratio: float = Field(
        default=2.0,
        ge=1.0,
        description=(
            "Minimum partner-freq / current-freq ratio for the strategy to consider "
            "swapping. Prevents low-freq partners from dominating high-freq tokens."
        ),
    )
    mined_pair_mlm_margin: float = Field(
        default=2.5,
        ge=0.0,
        description=(
            "Minimum MLM(partner) - MLM(current) margin at target position required "
            "to emit a confusable error. Higher is more conservative."
        ),
    )
    mined_pair_backend: str = Field(
        default="mlm",
        description=(
            "Scoring backend for the strategy. 'mlm' uses the existing SemanticChecker "
            "(conservative baseline). 'classifier' uses a dedicated fine-tuned "
            "classifier (higher recall at the same FPR). Set 'classifier' and "
            "provide `mined_pair_classifier_path` to load the model."
        ),
    )
    mined_pair_classifier_path: str | None = Field(
        default=None,
        description=(
            "Path to a fine-tuned classifier checkpoint (HuggingFace-format) or an "
            "ONNX model. Required when `mined_pair_backend == 'classifier'`."
        ),
    )

    # ByT5 safety net (priority 80, after every other strategy)
    # Runs only on sentences where the rest of the pipeline flagged nothing.
    # Uses a fine-tuned ByT5-small seq2seq model to propose whole-sentence
    # corrections; edits are gated by dict membership + MLM plausibility margin.
    use_byt5_safety_net: bool = Field(
        default=False,
        description=(
            "Enable ByT5SafetyNetStrategy (priority 80). Runs a fine-tuned "
            "ByT5-small seq2seq model on sentences with zero pipeline "
            "emissions, gated by dict membership + MLM margin. Requires "
            "`byt5_safety_net_model_path` to point at an ONNX bundle or HF dir."
        ),
    )
    byt5_safety_net_model_path: str | None = Field(
        default=None,
        description=(
            "Path to ByT5 model directory: either ONNX bundle "
            "(encoder.onnx + decoder.onnx + onnx_meta.json) or a HuggingFace "
            "checkpoint directory (for the PyTorch fallback)."
        ),
    )
    byt5_safety_net_mlm_gate_margin: float = Field(
        default=2.0,
        ge=-20.0,
        le=20.0,
        description=(
            "Minimum MLM(replacement) - MLM(original) margin at the masked "
            "position required to emit a ByT5 safety-net edit."
        ),
    )
    byt5_safety_net_min_typo_prone_chars: int = Field(
        default=2,
        ge=0,
        description=(
            "Minimum typo-prone character count in the sentence before the "
            "safety net fires. Filters sentences with no plausible typo source."
        ),
    )
    byt5_safety_net_max_sentence_chars: int = Field(
        default=400,
        ge=1,
        description="Skip sentences longer than this (byte-level inference cost).",
    )
    byt5_safety_net_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence score for ByT5 safety-net errors.",
    )

    # MLM post-filter for invalid_word / dangling_word FP suppression
    mlm_plausibility_threshold: float = Field(
        default=3.0,
        ge=-20.0,
        le=20.0,
        description=(
            "Minimum MLM logit score for the original word to suppress an "
            "invalid_word/dangling_word error. predict_mask returns raw logits "
            "(typical range 5-15 for top predictions). Higher values are more "
            "conservative (fewer FPs suppressed). A value of 3.0 suppresses "
            "only words the model is reasonably confident about."
        ),
    )
    mlm_plausibility_top_k: int = Field(
        default=10,
        ge=1,
        description=(
            "Number of top MLM predictions to check when evaluating "
            "contextual plausibility for invalid_word/dangling_word errors."
        ),
    )

    bypass_word_heuristic_suppression: bool = Field(
        default=False,
        description=(
            "Bypass heuristic word-level suppression (dict-check, MLM plausibility, "
            "compound-split). When True, all word-level errors flow directly to "
            "the meta-classifier. Used for meta-classifier retraining."
        ),
    )


class ProviderConfig(BaseModel):
    """
    Configuration for dictionary provider.

    Controls provider-level caching, query limits, connection pooling,
    and SQLite runtime tuning.

    Attributes:
        database_path: Path to SQLite database file (default: None, must be set explicitly).
        cache_size: LRU cache size for provider queries (default: 1024).
        pool_min_size: Minimum connections in pool (default: 1).
        pool_max_size: Maximum connections in pool (default: 5, smaller is better for SQLite).
        pool_timeout: Connection checkout timeout in seconds (default: 5.0).
        pool_max_connection_age: Max connection age before recreation in seconds (default: 3600.0).
        sqlite_max_batch_size: Max parameters per SQL query batch (default: 900).
            Safety limit for SQLITE_MAX_VARIABLE_NUMBER (999).
        iterator_fetch_size: Row fetch batch size for iterator methods (default: 10000).
        default_cache_size: Default LRU cache size for SQLite frequency lookups
            (default: 8192).
        pragma_cache_size: SQLite PRAGMA cache_size for runtime (default: -524288).
            Negative value = KiB; -524288 = 512 MB page cache.
        pragma_mmap_size: SQLite PRAGMA mmap_size for runtime (default: 2147483648).
            2 GB memory-mapped I/O.
        schema_check_timeout: Timeout in seconds for one-off schema version check
            connections (default: 5.0).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    database_path: str | None = Field(
        default=None,
        description="Path to SQLite database file (required, no bundled default)",
    )
    cache_size: int = Field(
        default=1024,
        ge=0,
        description="LRU cache size for provider queries (0=disabled)",
    )
    pool_min_size: int = Field(
        default=1,
        ge=0,
        le=100,
        description="Minimum number of connections to maintain in pool",
    )
    pool_max_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description=(
            "Maximum number of connections allowed in pool (smaller is better for SQLite)"
        ),
    )
    pool_timeout: float = Field(
        default=5.0,
        gt=0.0,
        le=60.0,
        description="Maximum time to wait for connection checkout in seconds",
    )
    pool_max_connection_age: float = Field(
        default=3600.0,
        gt=0.0,
        description=("Maximum age of a connection before recreation in seconds (default: 1 hour)"),
    )
    curated_min_frequency: int = Field(
        default=50,
        ge=0,
        description=(
            "Minimum effective frequency for curated vocabulary words "
            "(is_curated=1). Words with raw frequency below this floor "
            "are boosted to this value, ensuring they pass SymSpell "
            "count_threshold and participate in suggestions. "
            "Set to 0 to disable curated frequency boosting."
        ),
    )

    # --- SQLite-specific tuning ---

    sqlite_max_batch_size: int = Field(
        default=900,
        ge=1,
        le=999,
        description=(
            "Maximum number of parameters per SQL query batch. "
            "Safety limit for SQLITE_MAX_VARIABLE_NUMBER (999). "
            "Lower values add safety margin for different SQLite builds."
        ),
    )
    iterator_fetch_size: int = Field(
        default=10_000,
        ge=100,
        description=(
            "Row fetch batch size for iterator methods. "
            "Limits how many rows are loaded into memory at once "
            "to prevent memory exhaustion on large tables."
        ),
    )
    default_cache_size: int = Field(
        default=8192,
        ge=0,
        description=(
            "Default LRU cache size for SQLite frequency lookups. "
            "Matches AlgorithmCacheConfig.frequency_cache_size."
        ),
    )
    pragma_cache_size: int = Field(
        default=-524288,
        description=(
            "SQLite PRAGMA cache_size for runtime connections. "
            "Negative value = KiB; -524288 = 512 MB page cache."
        ),
    )
    pragma_mmap_size: int = Field(
        default=2_147_483_648,
        ge=0,
        description=(
            "SQLite PRAGMA mmap_size for runtime connections. 2147483648 = 2 GB memory-mapped I/O."
        ),
    )
    schema_check_timeout: float = Field(
        default=5.0,
        gt=0.0,
        le=60.0,
        description=("Timeout in seconds for one-off schema version check connections."),
    )

    @model_validator(mode="after")
    def validate_pool_sizes(self) -> "ProviderConfig":
        """Ensure pool_max_size >= pool_min_size."""
        if self.pool_max_size < self.pool_min_size:
            raise ValueError(
                f"pool_max_size ({self.pool_max_size}) must be >= "
                f"pool_min_size ({self.pool_min_size})"
            )
        return self


class ConnectionPoolConfig(BaseModel):
    """
    Configuration for SQLite connection pool.

    Controls connection pool sizing, timeouts, and connection lifecycle.

    Attributes:
        min_size: Minimum number of connections to maintain in pool (default: 2).
        max_size: Maximum number of connections allowed in pool (default: 10).
        timeout: Maximum time to wait for connection checkout in seconds (default: 5.0).
        max_connection_age: Maximum age of a connection before recreation (default: 3600.0).
        check_same_thread: SQLite check_same_thread parameter (default: False).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    min_size: int = Field(
        default=2,
        ge=0,
        le=100,
        description="Minimum number of connections to maintain in pool",
    )
    max_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of connections allowed in pool",
    )
    timeout: float = Field(
        default=5.0,
        gt=0.0,
        le=60.0,
        description="Maximum time to wait for connection checkout in seconds",
    )
    max_connection_age: float = Field(
        default=3600.0,
        gt=0.0,
        description="Maximum age of a connection before recreation in seconds (1 hour)",
    )
    check_same_thread: bool = Field(
        default=False,
        description="SQLite check_same_thread parameter (False for thread-safe pooling)",
    )
    sqlite_timeout: float = Field(
        default=30.0,
        gt=0.0,
        le=300.0,
        description="SQLite busy timeout in seconds (how long to wait when database is locked)",
    )
    skip_health_check: bool = Field(
        default=False,
        description="Skip connection health checks on checkout (safe for read-only databases)",
    )

    @model_validator(mode="after")
    def validate_pool_sizes(self) -> "ConnectionPoolConfig":
        """Ensure max_size >= min_size."""
        if self.max_size < self.min_size:
            raise ValueError(f"max_size ({self.max_size}) must be >= min_size ({self.min_size})")
        return self
