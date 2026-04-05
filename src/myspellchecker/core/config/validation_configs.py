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
        is_myanmar_text_threshold: Threshold for Myanmar text detection (default: 0.5).
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
    is_myanmar_text_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of Myanmar characters to classify text as Myanmar",
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
    orthography_confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence score for orthography validation errors",
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
    homophone_improvement_ratio: float = Field(
        default=5.0,
        ge=1.0,
        description="Minimum probability improvement ratio for homophone suggestions",
    )
    homophone_min_probability: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum n-gram probability threshold for homophone suggestions. "
            "Prevents false positives from infrequent n-gram occurrences."
        ),
    )
    homophone_high_freq_threshold: int = Field(
        default=1000,
        ge=0,
        description=(
            "Word frequency above which a stricter improvement ratio is required. "
            "Prevents false positives on common, correct words."
        ),
    )
    homophone_high_freq_improvement_ratio: float = Field(
        default=50.0,
        ge=1.0,
        description=(
            "Improvement ratio required for high-frequency words. "
            "Much stricter than the default ratio to avoid flagging common words."
        ),
    )
    semantic_min_word_length: int = Field(
        default=2,
        ge=1,
        description="Minimum word length for semantic validation",
    )
    use_homophone_detection: bool = Field(
        default=True,
        description="Enable homophone detection in validation pipeline",
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
        default=False,
        description=(
            "Enable MLM-enhanced confusable detection (priority 48). "
            "Uses predict_mask() to detect valid-word confusables missed by n-gram. "
            "Requires semantic model to be loaded. Opt-in due to inference cost."
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
        default=3.0,
        ge=0.0,
        description=(
            "Default logit diff threshold (~20x probability ratio). "
            "Variant must score this much higher than current word."
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
        default=6.0,
        ge=0.0,
        description=(
            "Logit diff threshold for high-frequency words (~403x ratio). "
            "Protects common particles/words against false positives."
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

    use_orthography_validation: bool = Field(
        default=True,
        description="Enable orthography validation in validation pipeline",
    )
    # Output confidence filter thresholds
    # Calibration: confusable_error FPs cluster at 0.72, TPs at 0.88+.
    # colloquial_info notes use confidence 0.3 (informational, not errors)
    # and are suppressed by default; lower the threshold to surface them.
    output_confidence_thresholds: dict[str, float] = Field(
        default={
            "confusable_error": 0.75,
            "colloquial_info": 0.35,
            "homophone_error": 0.72,
            "question_structure": 0.68,
        },
        description=(
            "Per-error-type minimum confidence for the output filter. "
            "Errors whose confidence is below the threshold for their type "
            "are suppressed."
        ),
    )
    secondary_confidence_thresholds: dict[str, float] = Field(
        default={
            "semantic_error": 0.85,
            "pos_sequence_error": 0.70,
            "syntax_error": 0.75,
        },
        description=(
            "Cascade guard: suppress error types when the sentence already "
            "has a higher-confidence error of a different type. Prevents "
            "cascade false positives from context strategies."
        ),
    )

    # -- Candidate fusion (voting architecture) --
    use_candidate_fusion: bool = Field(
        default=False,
        description=(
            "Enable calibrated Noisy-OR candidate fusion instead of mutex-based "
            "winner selection. When True, all strategies may fire at every position "
            "(mutex bypass), and the arbiter uses calibrated confidence fusion "
            "across independence clusters to determine which errors to emit. "
            "When False (default), the v1.2 mutex-and-shadow-mode behaviour is used."
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

    enable_fast_path: bool = Field(
        default=True,
        description=(
            "Enable fast-path exit in context validation. When True, if structural "
            "strategies (Tone, Orthography, Syntactic, BrokenCompound) find no errors, "
            "contextual strategies are skipped. Reduces FPR on clean text but may miss "
            "context-only errors. Set to False for maximum recall."
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

    truncation_frequency_ratio: int = Field(
        default=100,
        ge=1,
        description=(
            "Minimum frequency ratio of complete_freq / truncated_freq "
            "for truncation detection. Higher values = stricter detection."
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
