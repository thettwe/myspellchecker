"""Infrastructure configuration classes.

Configuration for caching, frequency guards, neural reranking,
and token boundary refinement.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AlgorithmCacheConfig",
    "FrequencyGuardConfig",
    "NeuralRerankerConfig",
    "TokenRefinementConfig",
]


class AlgorithmCacheConfig(BaseModel):
    """
    Unified cache configuration for algorithm caching layers.

    Consolidates cache sizes from:
    - algorithms/cache.py: CachedDictionaryLookup, CachedBigramSource, etc.
    - algorithms/factory.py: Component factory cache configurations
    - algorithms/semantic_checker.py: Encoding and alignment caches
    - algorithms/joint_segment_tagger.py: Word score, transition, emission caches

    This class provides a single source of truth for cache configuration.

    Attributes:
        syllable_cache_size: LRU cache size for syllable lookups (default: 4096).
        word_cache_size: LRU cache size for word lookups (default: 8192).
        frequency_cache_size: LRU cache size for frequency lookups (default: 8192).
        bigram_cache_size: LRU cache size for bigram probability lookups (default: 16384).
        trigram_cache_size: LRU cache size for trigram probability lookups (default: 16384).
        semantic_encoding_cache_size: Semantic checker encoding cache (default: 512).
        semantic_alignment_cache_size: Semantic checker alignment cache (default: 256).
        joint_word_score_cache_size: Joint tagger word scores (default: 8192).
        joint_transition_cache_size: Joint tagger POS transitions (default: 2048).
        joint_emission_cache_size: Joint tagger emissions (default: 4096).
        joint_valid_tags_cache_size: Joint tagger valid tags (default: 4096).

    Memory Usage (approximate):
        - ~1KB per cached entry (average)
        - Default configuration: ~50MB total cache memory

    Example:
        >>> from myspellchecker.core.config import SpellCheckerConfig, AlgorithmCacheConfig
        >>>
        >>> # Reduce cache sizes for memory-constrained environments
        >>> config = SpellCheckerConfig(
        ...     cache=AlgorithmCacheConfig(
        ...         syllable_cache_size=1024,
        ...         word_cache_size=2048,
        ...         bigram_cache_size=4096,
        ...     )
        ... )
        >>>
        >>> # Increase cache sizes for high-throughput applications
        >>> config = SpellCheckerConfig(
        ...     cache=AlgorithmCacheConfig(
        ...         syllable_cache_size=16384,
        ...         word_cache_size=32768,
        ...         bigram_cache_size=65536,
        ...     )
        ... )
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    syllable_cache_size: int = Field(
        default=4096,
        ge=0,
        description="LRU cache size for syllable validation and frequency lookups",
    )
    word_cache_size: int = Field(
        default=8192,
        ge=0,
        description="LRU cache size for word validation and frequency lookups",
    )
    frequency_cache_size: int = Field(
        default=8192,
        ge=0,
        description="LRU cache size for frequency source lookups",
    )
    bigram_cache_size: int = Field(
        default=16384,
        ge=0,
        description="LRU cache size for bigram probability lookups",
    )
    trigram_cache_size: int = Field(
        default=16384,
        ge=0,
        description="LRU cache size for trigram probability lookups",
    )
    # SemanticChecker cache sizes
    semantic_encoding_cache_size: int = Field(
        default=512,
        ge=0,
        description="LRU cache size for semantic checker tokenization encoding results",
    )
    semantic_alignment_cache_size: int = Field(
        default=256,
        ge=0,
        description="LRU cache size for semantic checker word-token alignment results",
    )

    # JointSegmentTagger cache sizes
    joint_word_score_cache_size: int = Field(
        default=8192,
        ge=0,
        description="LRU cache size for joint tagger word bigram scores",
    )
    joint_transition_cache_size: int = Field(
        default=2048,
        ge=0,
        description="LRU cache size for joint tagger POS trigram transitions",
    )
    joint_emission_cache_size: int = Field(
        default=4096,
        ge=0,
        description="LRU cache size for joint tagger word-tag emission scores",
    )
    joint_valid_tags_cache_size: int = Field(
        default=4096,
        ge=0,
        description="LRU cache size for joint tagger valid tags per word",
    )


class FrequencyGuardConfig(BaseModel):
    """
    Centralized frequency thresholds used across validators and strategies.

    These guards prevent false positives on common words by requiring
    stronger evidence before flagging high-frequency entries as errors.
    Each threshold was previously hardcoded in the consumer file noted
    in its description.

    Attributes:
        colloquial_high_freq_suppression: Frequency above which colloquial
            variant informational notes are suppressed in lenient mode.
            Used by SyllableValidator and WordValidator.
        homophone_high_freq: Frequency above which a stricter improvement
            ratio is required for homophone detection.
        homophone_high_freq_ratio: Improvement ratio required for words
            above ``homophone_high_freq``.
        ngram_high_freq_guard: Frequency above which n-gram false positives
            are suppressed when no better suggestion exists.
        semantic_high_freq_protection: Frequency above which the high-freq
            logit diff threshold applies in confusable semantic detection.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # SyllableValidator + WordValidator: suppress colloquial info for
    # very high-frequency words in lenient mode
    colloquial_high_freq_suppression: int = Field(
        default=100_000,
        ge=0,
        description=(
            "Frequency above which colloquial variant informational "
            "notes are suppressed in lenient mode. "
            "Used by SyllableValidator and WordValidator."
        ),
    )
    # HomophoneValidationStrategy: frequency guard
    homophone_high_freq: int = Field(
        default=1_000,
        ge=0,
        description=(
            "Word frequency above which a stricter improvement ratio "
            "is required for homophone detection."
        ),
    )
    homophone_high_freq_ratio: float = Field(
        default=50.0,
        ge=1.0,
        description=(
            "Improvement ratio required for words above "
            "homophone_high_freq. Much stricter than the default "
            "5x to avoid flagging common words."
        ),
    )
    # NgramContextValidationStrategy: suppress FP on common words
    ngram_high_freq_guard: int = Field(
        default=5_000,
        ge=0,
        description=(
            "Minimum dictionary frequency to suppress n-gram FP on "
            "common words when no better suggestion exists."
        ),
    )
    # ConfusableSemanticStrategy: protect common words from MLM FP
    semantic_high_freq_protection: int = Field(
        default=50_000,
        ge=0,
        description=(
            "Word frequency above which the high-freq logit diff "
            "threshold applies in confusable semantic detection."
        ),
    )


class NeuralRerankerConfig(BaseModel):
    """
    Configuration for the neural MLP suggestion reranker.

    Controls the ONNX-based MLP that re-scores spell checker suggestions
    using 16 extracted features (edit distance, frequency, phonetic
    similarity, n-gram context, etc.).

    The neural reranker runs AFTER both n-gram and semantic reranking,
    giving it the final say on suggestion ordering.

    Attributes:
        enabled: Enable neural reranking (default: False).
        model_path: Path to ONNX model file.
        stats_path: Path to normalization stats JSON (feature_means/stds).
        confidence_gap_threshold: Skip reranking if score gap > this.
        max_candidates: Max candidates to score per error.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    enabled: bool = Field(
        default=False,
        description="Enable neural MLP suggestion reranking.",
    )
    model_path: str | None = Field(
        default=None,
        description="Path to the ONNX model file for neural reranking.",
    )
    stats_path: str | None = Field(
        default=None,
        description=(
            "Path to normalization stats JSON file containing "
            "feature_means and feature_stds arrays."
        ),
    )
    confidence_gap_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Skip reranking when the gap between top-1 and top-2 "
            "neural scores exceeds this threshold. A large gap "
            "indicates the current ranking is already confident."
        ),
    )
    max_candidates: int = Field(
        default=20,
        ge=1,
        description="Maximum number of candidates to score per error.",
    )


class TokenRefinementConfig(BaseModel):
    """Configuration for token boundary refinement scoring.

    Controls scoring parameters used during validation-time token-lattice
    refinement. The refinement pass operates on already-segmented tokens to
    expose hidden error spans in merged tokens (e.g., particle attachment,
    negation attachment).

    Attributes:
        suffix_score_boost: Score boost when suffix matches a known form.
        known_part_score: Score for known dictionary parts.
        unknown_long_part_penalty: Penalty for unknown long parts.
        split_complexity_penalty: Penalty for complex multi-part splits.
        bigram_scale: Scaling factor for bigram probability contribution.
        min_token_len: Minimum token length for refinement candidates.
        keep_if_freq_at_least: Keep token if frequency is at least this value.
        min_score_gain: Minimum score improvement to accept a split.
        lattice_max_paths: Maximum lattice paths to consider.
        syllable_split_min_token_len: Minimum token length for syllable-level splitting.
        syllable_split_max_syllables: Maximum syllables for syllable-level splitting.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    suffix_score_boost: float = Field(
        default=0.85,
        description="Score boost when suffix matches a known form.",
    )
    known_part_score: float = Field(
        default=1.35,
        description="Score for known dictionary parts.",
    )
    unknown_long_part_penalty: float = Field(
        default=0.45,
        description="Penalty for unknown long parts.",
    )
    split_complexity_penalty: float = Field(
        default=0.30,
        description="Penalty for complex multi-part splits.",
    )
    bigram_scale: float = Field(
        default=120_000.0,
        description="Scaling factor for bigram probability contribution.",
    )
    min_token_len: int = Field(
        default=3,
        ge=1,
        description="Minimum token length for refinement candidates.",
    )
    keep_if_freq_at_least: int = Field(
        default=2_000,
        ge=0,
        description="Keep token if frequency is at least this value.",
    )
    min_score_gain: float = Field(
        default=0.55,
        description="Minimum score improvement to accept a split.",
    )
    lattice_max_paths: int = Field(
        default=2,
        ge=1,
        description="Maximum lattice paths to consider.",
    )
    syllable_split_min_token_len: int = Field(
        default=4,
        ge=1,
        description="Minimum token length for syllable-level splitting.",
    )
    syllable_split_max_syllables: int = Field(
        default=6,
        ge=1,
        description="Maximum syllables for syllable-level splitting.",
    )
