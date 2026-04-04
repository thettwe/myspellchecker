"""
POS Tagger Configuration Classes.

This module contains configuration classes for Part-of-Speech tagging:
- POSTaggerConfig: Pluggable POS tagger system (rule-based/transformer/viterbi/custom)
- JointConfig: Joint segmentation and POS tagging
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from myspellchecker.core.constants.detector_thresholds import DEFAULT_ALGORITHM_THRESHOLDS


class POSTaggerConfig(BaseModel):
    """
    Configuration for pluggable POS tagger system.

    Supports multiple tagger backends: rule-based (default), transformer,
    viterbi, or custom. Enables flexible POS tagging for both build-time
    (dictionary building) and runtime (spell checking).

    Tagger Types:
        - rule_based: Fast suffix-based tagging (default, no dependencies)
        - transformer: Neural models from HuggingFace (high accuracy, requires transformers)
        - viterbi: HMM-based sequence tagging (context-aware, no dependencies)
        - custom: User-provided tagger class

    Attributes:
        tagger_type: Type of POS tagger to use (default: "rule_based").
                    Options: "rule_based", "transformer", "viterbi", "custom"

        # Transformer settings (for tagger_type="transformer")
        model_name: HuggingFace model ID or local path (default: None).
                   Only used when tagger_type="transformer".
        device: Device for transformer inference. -1 for CPU, 0+ for GPU index (default: -1).
        batch_size: Batch size for transformer sequence tagging (default: 32).
        cache_dir: Directory for caching downloaded transformer models (optional).

        # Rule-based/Viterbi settings
        use_morphology_fallback: Use MorphologyAnalyzer for OOV words (default: True).
        cache_size: LRU cache size for rule-based tagger (default: 10000).

        # Viterbi-specific settings
        beam_width: Beam width for Viterbi decoding (default: 10).
        emission_weight: Weight for emission probabilities (default: 1.2).
        min_prob: Minimum probability threshold (default: 1e-10).

        # Common settings
        unknown_tag: Tag to return for completely unknown words (default: "UNK").

    Example:
        >>> # Default rule-based tagger
        >>> config = POSTaggerConfig()
        >>>
        >>> # Transformer tagger on GPU
        >>> config = POSTaggerConfig(
        ...     tagger_type="transformer",
        ...     model_name="chuuhtetnaing/myanmar-pos-model",
        ...     device=0,
        ...     batch_size=64
        ... )
        >>>
        >>> # Viterbi tagger with custom settings
        >>> config = POSTaggerConfig(
        ...     tagger_type="viterbi",
        ...     beam_width=15,
        ...     emission_weight=1.5
        ... )
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # Tagger selection
    tagger_type: Literal["rule_based", "transformer", "viterbi", "custom"] = Field(
        default="rule_based",
        description="Type of POS tagger backend to use",
    )

    # Transformer settings
    model_name: str | None = Field(
        default=None,
        description="HuggingFace model ID or local path (for transformer tagger)",
    )
    device: int = Field(
        default=-1,
        description="Device for transformer inference (-1=CPU, 0+=GPU index)",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for transformer sequence tagging",
    )
    cache_dir: str | None = Field(
        default=None,
        description="Directory for caching downloaded transformer models",
    )

    # Rule-based/Viterbi settings
    use_morphology_fallback: bool = Field(
        default=True,
        description="Use morphology analyzer for out-of-vocabulary words",
    )
    cache_size: int = Field(
        default=10000,
        ge=0,
        description="LRU cache size for rule-based tagger",
    )

    # Viterbi-specific settings
    # DEPRECATED: These top-level fields duplicate the viterbi_* prefixed fields
    # below (viterbi_beam_width, viterbi_emission_weight, viterbi_min_prob).
    # They are kept for backward compatibility with component_factory and
    # data_pipeline, which read pos_config.beam_width etc.
    # Prefer the viterbi_* prefixed fields for new code. These will be removed
    # in a future version once all consumers are migrated.
    beam_width: int = Field(
        default=DEFAULT_ALGORITHM_THRESHOLDS.beam_width_minimal,
        ge=1,
        description="Beam width for Viterbi decoding (deprecated: use viterbi_beam_width)",
    )
    emission_weight: float = Field(
        default=1.2,
        gt=0.0,
        description=(
            "Weight for emission probabilities in HMM (deprecated: use viterbi_emission_weight)"
        ),
    )
    min_prob: float = Field(
        default=1e-10,
        gt=0.0,
        description="Minimum probability threshold (deprecated: use viterbi_min_prob)",
    )
    hmm_params_path: str | None = Field(
        default=None,
        description="Path to precomputed HMM params JSON (for viterbi tagger bootstrap). "
        "If None, uses bundled myPOS data from package data directory.",
    )

    # Transformer performance settings
    use_fp16: bool = Field(
        default=True,
        description="Use float16 on GPU for ~2x throughput (transformer tagger only)",
    )
    use_torch_compile: bool = Field(
        default=False,
        description="Use torch.compile() JIT optimization (transformer tagger only)",
    )

    # Pipeline build settings
    pos_checkpoint_interval: int = Field(
        default=250_000,
        ge=1_000,
        description="Save POS checkpoint every N sentences during build",
    )
    pos_batch_buffer_size: int = Field(
        default=2000,
        ge=1,
        description="Sentences buffered before POS tagger dispatch during build",
    )

    # Viterbi cache settings
    viterbi_emission_cache_size: int = Field(
        default=4096,
        ge=0,
        description="LRU cache size for emission score lookups.",
    )
    viterbi_transition_cache_size: int = Field(
        default=2048,
        ge=0,
        description="LRU cache size for transition probability lookups.",
    )

    # Viterbi algorithm parameters (canonical — prefer these over the
    # deprecated top-level beam_width / emission_weight / min_prob fields)
    viterbi_min_prob: float = Field(
        default=1e-10,
        gt=0.0,
        description="Minimum probability floor for Viterbi algorithm.",
    )
    viterbi_beam_width: int = Field(
        default=10,
        ge=1,
        description="Default beam search width for pruning.",
    )
    viterbi_emission_weight: float = Field(
        default=1.2,
        gt=0.0,
        description="Weight for emission probabilities in scoring.",
    )
    viterbi_lambda_unigram: float = Field(
        default=0.1,
        ge=0.0,
        description="Deleted interpolation weight for unigram.",
    )
    viterbi_lambda_bigram: float = Field(
        default=0.3,
        ge=0.0,
        description="Deleted interpolation weight for bigram.",
    )
    viterbi_lambda_trigram: float = Field(
        default=0.6,
        ge=0.0,
        description="Deleted interpolation weight for trigram.",
    )

    # Viterbi adaptive beam settings
    viterbi_min_beam_width: int = Field(
        default=5,
        ge=1,
        description="Minimum beam width for adaptive beam mode.",
    )
    viterbi_max_beam_width: int = Field(
        default=20,
        ge=1,
        description="Maximum beam width for adaptive beam mode.",
    )
    viterbi_short_sequence_threshold: int = Field(
        default=5,
        ge=1,
        description="Sequences at or below this length use max beam.",
    )
    viterbi_long_sequence_threshold: int = Field(
        default=20,
        ge=1,
        description="Sequences above this length use min beam.",
    )

    # POS disambiguation confidence values
    disambiguation_confidence_resolved: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence when disambiguation succeeds.",
    )
    disambiguation_confidence_unresolved: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence when no rule applies and no resolution found.",
    )
    disambiguation_confidence_fallback: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence when no specific rule matches.",
    )
    disambiguation_r1_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="R1: Noun after verb confidence.",
    )
    disambiguation_r2_confidence: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="R2: Adjective before noun confidence.",
    )
    disambiguation_r3_confidence: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="R3: Verb before particle confidence.",
    )
    disambiguation_r4_confidence: float = Field(
        default=0.88,
        ge=0.0,
        le=1.0,
        description="R4: Noun after determiner confidence.",
    )
    disambiguation_r5_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="R5: Verb after adverb confidence.",
    )

    # Common settings
    unknown_tag: str = Field(
        default="UNK",
        description="Tag to return for completely unknown words",
    )

    @field_validator("tagger_type", mode="before")
    @classmethod
    def normalize_tagger_type(cls, v: str) -> str:
        """Normalize tagger_type to lowercase and strip whitespace."""
        if isinstance(v, str):
            return v.lower().strip()
        return v

    @model_validator(mode="after")
    def validate_transformer_dependency(self) -> "POSTaggerConfig":
        """Check transformers library availability for transformer tagger."""
        if self.tagger_type == "transformer":
            try:
                import transformers  # noqa: F401
            except ImportError as err:
                raise ValueError(
                    "tagger_type='transformer' requires the 'transformers' library.\n\n"
                    "Install with: pip install myspellchecker[transformers]\n\n"
                    "This will install:\n"
                    "  - transformers>=4.30.0\n"
                    "  - torch>=2.0.0\n\n"
                    "Alternatively, use the default rule-based tagger:\n"
                    "  config.pos_tagger.tagger_type = 'rule_based'\n"
                ) from err
        return self


class JointConfig(BaseModel):
    """
    Configuration for joint segmentation and POS tagging.

    Controls the unified Viterbi decoder that simultaneously optimizes
    word boundaries and POS tags in a single pass.

    Why Disabled by Default:
        Joint mode is disabled by default for production stability:
        - **Complexity**: State space is O(positions × word_lengths × tags²)
          vs O(words × tags²) for sequential mode
        - **Memory**: Joint beam search requires more memory (~2x sequential)
        - **Stability**: Sequential pipeline has more extensive production testing
        - **Accuracy**: For most use cases, sequential achieves comparable results

    When to Enable:
        Consider enabling joint mode (enabled=True) for:
        - Ambiguous segmentation where POS context helps word boundaries
        - OOV-heavy text where joint optimization handles unknowns better
        - Research and experimentation comparing segmentation approaches

    Example:
        >>> from myspellchecker.core.config import SpellCheckerConfig, JointConfig
        >>> config = SpellCheckerConfig(joint=JointConfig(enabled=True))
        >>> checker = SpellChecker(config=config)
        >>> words, tags = checker.segment_and_tag("မြန်မာနိုင်ငံ")

    See Also:
        - https://docs.myspellchecker.com/features/pos-tagging
          #joint-segmentation-and-tagging for full documentation
        - JointSegmentTagger class for implementation details

    Attributes:
        enabled: Enable joint segmentation-tagging mode (default: False).
            When False, uses sequential segmentation then tagging.
        beam_width: Beam width for joint decoding (default: 15).
            Larger than ViterbiConfig because state space is larger.
        max_word_length: Maximum word length in characters (default: 20).
        emission_weight: Weight for emission probabilities (default: 1.2).
        word_score_weight: Weight for word n-gram scores (default: 1.0).
        min_prob: Minimum probability threshold (default: 1e-10).
        use_morphology_fallback: Use morphology for OOV word tagging (default: True).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    enabled: bool = Field(
        default=False,
        description="Enable joint segmentation-tagging mode (vs sequential)",
    )
    beam_width: int = Field(
        default=DEFAULT_ALGORITHM_THRESHOLDS.beam_width_pos_tagger,
        ge=1,
        description="Beam width for joint Viterbi decoding (larger state space than solo)",
    )
    max_word_length: int = Field(
        default=20,
        ge=1,
        description="Maximum word length in characters for segmentation",
    )
    emission_weight: float = Field(
        default=1.2,
        gt=0.0,
        description="Weight for emission probabilities in joint model",
    )
    word_score_weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight for word n-gram scores in joint scoring",
    )
    min_prob: float = Field(
        default=1e-10,
        gt=0.0,
        description="Minimum probability threshold to prevent underflow",
    )
    use_morphology_fallback: bool = Field(
        default=True,
        description="Use morphology analyzer for out-of-vocabulary word tagging",
    )
