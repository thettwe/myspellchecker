"""Text processing configuration classes.

Configuration for morphological analysis, compound resolution, reduplication
detection, resource loading, and transformer-based segmentation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CompoundResolverConfig",
    "MorphologyConfig",
    "ReduplicationConfig",
    "ResourceConfig",
    "StemmerConfig",
    "ToneConfig",
    "TransformerSegmenterConfig",
    "ZawgyiConfig",
]


class ZawgyiConfig(BaseModel):
    """
    Configuration for Zawgyi detection and conversion.

    Controls thresholds for detecting and converting legacy Zawgyi-1
    encoded text to standard Unicode Myanmar.

    Attributes:
        detection_threshold: Confidence threshold for Zawgyi detection (default: 0.5).
            Values below this are assumed to be Unicode.
        warning_threshold: Threshold for emitting Zawgyi warnings (default: 0.7).
            Higher confidence triggers user-facing warnings.
        conversion_threshold: Threshold for automatic conversion (default: 0.9).
            Text above this threshold is converted to Unicode.
            High threshold (0.9) prevents corrupting valid Unicode text.
        myanmar_text_threshold: Minimum Myanmar character ratio (default: 0.5).
            Used by is_myanmar_text() to classify text.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    detection_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for Zawgyi detection (0.0-1.0)",
    )
    warning_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for emitting Zawgyi warnings (0.0-1.0)",
    )
    conversion_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Threshold for automatic Zawgyi to Unicode conversion (0.0-1.0)",
    )
    myanmar_text_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum Myanmar character ratio for is_myanmar_text (0.0-1.0)",
    )
    unicode_determination_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Threshold below which low Zawgyi confidence indicates Unicode text (0.0-1.0)",
    )
    space_seg_min_spaces: int = Field(
        default=2,
        ge=1,
        description="Minimum Myanmar space segments for space-segmentation detection.",
    )
    space_seg_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum space/character ratio for space-segmentation detection.",
    )
    min_myanmar_chars_for_zawgyi: int = Field(
        default=3,
        ge=1,
        description="Minimum Myanmar characters needed for Zawgyi detection.",
    )
    min_line_length_for_zawgyi: int = Field(
        default=5,
        ge=1,
        description="Minimum line length for Zawgyi analysis.",
    )


class ToneConfig(BaseModel):
    """
    Configuration for tone disambiguation.

    Controls context-based tone disambiguation for Myanmar text,
    which helps resolve tone-ambiguous words.

    Attributes:
        context_window: Number of words to consider on each side (default: 3).
            Larger windows provide more context but are slower.
        min_confidence: Minimum confidence for disambiguation (default: 0.2).
            Suggestions below this threshold are not returned.
        tone_ambiguous_map: Optional mapping of tone-ambiguous words to context patterns.
            If provided, overrides the default TONE_AMBIGUOUS dict.
            Loaded from tone_rules.yaml via GrammarRuleConfig.
        tone_errors_map: Optional mapping of common tone mark errors.
            If provided, overrides the default TONE_MARK_ERRORS dict.
            Loaded from tone_rules.yaml via GrammarRuleConfig.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    context_window: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of words to consider on each side for context",
    )
    min_confidence: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for disambiguation suggestions",
    )
    tone_ambiguous_map: dict | None = Field(
        default=None,
        description="Optional tone-ambiguous words map from tone_rules.yaml",
    )
    tone_errors_map: dict | None = Field(
        default=None,
        description="Optional tone errors map from tone_rules.yaml",
    )


class StemmerConfig(BaseModel):
    """
    Configuration for Myanmar stemmer/lemmatizer.

    Controls rule-based stemming to strip common suffixes from words.

    Attributes:
        cache_size: LRU cache size for stemmed words (default: 4096).
            Higher values use more memory but improve performance for repeated words.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    cache_size: int = Field(
        default=4096,
        ge=0,
        description="LRU cache size for stemmed words (0 = no caching)",
    )


class MorphologyConfig(BaseModel):
    """
    Configuration for morphological analysis and POS guessing.

    Controls confidence values used in OOV (Out-of-Vocabulary) recovery
    when analyzing word morphology (root extraction and suffix stripping),
    as well as suffix-based POS guessing confidence multipliers.

    Attributes:
        particle_confidence_boost: Multiplier for particle suffix confidence
            (default: 1.2). Particles are usually reliable indicators.
        particle_confidence_cap: Maximum confidence after particle boost
            (default: 1.0).
        verb_suffix_weight: Weight for verb suffix confidence scoring
            (default: 0.9). Slightly lower than particles.
        noun_suffix_weight: Weight for noun suffix confidence scoring
            (default: 0.85).
        adverb_suffix_weight: Weight for adverb suffix confidence scoring
            (default: 0.8).
        oov_base_confidence: Base confidence for OOV suffix analysis (default: 0.3).
        oov_scale_factor: Scale factor for suffix ratio contribution (default: 0.7).
        oov_cap: Maximum confidence from suffix analysis alone (default: 0.95).
        dictionary_boost: Confidence boost when dictionary confirms root (default: 0.2).
        dictionary_cap: Maximum confidence after dictionary boost (default: 0.98).
        fallback_with_dict: Confidence when no suffixes found but root in dict (default: 0.5).
        fallback_without_dict: Confidence when no suffixes found and root is unknown
            (default: 0.2).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # -- POS guessing confidence multipliers --

    particle_confidence_boost: float = Field(
        default=1.2,
        ge=0.0,
        le=5.0,
        description="Multiplier for particle suffix confidence",
    )
    particle_confidence_cap: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after particle boost",
    )
    verb_suffix_weight: float = Field(
        default=0.9,
        ge=0.0,
        le=2.0,
        description="Weight for verb suffix confidence scoring",
    )
    noun_suffix_weight: float = Field(
        default=0.85,
        ge=0.0,
        le=2.0,
        description="Weight for noun suffix confidence scoring",
    )
    adverb_suffix_weight: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Weight for adverb suffix confidence scoring",
    )

    # -- OOV recovery confidence --

    oov_base_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Base confidence for OOV suffix analysis",
    )
    oov_scale_factor: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Scale factor for suffix ratio contribution to confidence",
    )
    oov_cap: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Maximum confidence from suffix analysis alone",
    )
    dictionary_boost: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Confidence boost when dictionary confirms the extracted root",
    )
    dictionary_cap: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after dictionary boost",
    )
    fallback_with_dict: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence when no suffixes found but root is in dictionary",
    )
    fallback_without_dict: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Confidence when no suffixes found and root is unknown",
    )
    max_suffix_strip_iterations: int = Field(
        default=5,
        ge=1,
        description=(
            "Maximum iterations for suffix stripping in analyze_word(). "
            "Prevents infinite loops on pathological inputs."
        ),
    )


class CompoundResolverConfig(BaseModel):
    """Configuration for compound word resolution.

    Controls the dynamic-programming compound splitter that validates OOV words
    by segmenting them into known dictionary morphemes with valid POS patterns.

    Attributes:
        min_morpheme_frequency: Minimum frequency for each morpheme (default: 10).
        max_parts: Maximum number of parts in a compound (default: 4).
        cache_size: Maximum cache entries (default: 1024).
        parts_penalty_multiplier: Penalty multiplier per extra split beyond
            the first in DP scoring (default: 2.0).
        base_confidence: Base confidence score for compound splits (default: 0.85).
        high_freq_boost: Confidence boost when min morpheme freq >= high_freq_threshold
            (default: 0.05).
        high_freq_threshold: Frequency threshold for high_freq_boost (default: 100).
        medium_freq_boost: Confidence boost when min morpheme freq >= medium_freq_threshold
            (default: 0.03).
        medium_freq_threshold: Frequency threshold for medium_freq_boost (default: 50).
        extra_parts_penalty: Confidence penalty per extra part beyond 2 (default: 0.05).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    min_morpheme_frequency: int = Field(
        default=10,
        ge=0,
        description="Minimum frequency for each morpheme to be considered valid",
    )
    max_parts: int = Field(
        default=4,
        ge=2,
        description="Maximum number of parts allowed in compound word splitting",
    )
    cache_size: int = Field(
        default=1024,
        ge=0,
        description="Maximum cache entries for compound resolution results",
    )
    parts_penalty_multiplier: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Penalty multiplier per extra split beyond the first in DP scoring. "
            "Higher values prefer fewer parts."
        ),
    )
    base_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Base confidence score for compound splits",
    )
    high_freq_boost: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Confidence boost when min morpheme freq >= high_freq_threshold",
    )
    high_freq_threshold: int = Field(
        default=100,
        ge=0,
        description="Frequency threshold for high_freq_boost",
    )
    medium_freq_boost: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Confidence boost when min morpheme freq >= medium_freq_threshold",
    )
    medium_freq_threshold: int = Field(
        default=50,
        ge=0,
        description="Frequency threshold for medium_freq_boost",
    )
    extra_parts_penalty: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Confidence penalty per extra part beyond 2",
    )
    confidence_cap: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after high-frequency boost.",
    )
    confidence_cap_mid: float = Field(
        default=0.93,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after medium-frequency boost.",
    )


class ReduplicationConfig(BaseModel):
    """Configuration for reduplication detection.

    Controls the reduplication engine that validates OOV words formed by
    reduplicating known dictionary words (AA, AABB, ABAB, RHYME patterns).

    Attributes:
        min_base_frequency: Minimum frequency for the base word (default: 5).
        cache_size: Maximum cache entries (default: 1024).
        pattern_confidence_ab: Base confidence for AB (simple doubling) pattern
            (default: 0.90).
        pattern_confidence_aabb: Base confidence for AABB (syllable doubling) pattern
            (default: 0.85).
        pattern_confidence_abab: Base confidence for ABAB (word repeating) pattern
            (default: 0.85).
        pattern_confidence_rhyme: Base confidence for RHYME pattern (default: 0.95).
        pattern_confidence_default: Default base confidence for unknown patterns
            (default: 0.80).
        high_freq_boost: Confidence boost when base freq >= high_freq_threshold
            (default: 0.05).
        high_freq_threshold: Frequency threshold for high_freq_boost (default: 100).
        high_freq_cap: Maximum confidence after high_freq_boost (default: 0.98).
        medium_freq_boost: Confidence boost when base freq >= medium_freq_threshold
            (default: 0.03).
        medium_freq_threshold: Frequency threshold for medium_freq_boost (default: 50).
        medium_freq_cap: Maximum confidence after medium_freq_boost (default: 0.95).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    min_base_frequency: int = Field(
        default=5,
        ge=0,
        description="Minimum frequency for the base word to be considered valid",
    )
    cache_size: int = Field(
        default=1024,
        ge=0,
        description="Maximum cache entries for reduplication analysis results",
    )
    pattern_confidence_ab: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Base confidence for AB (simple doubling) pattern",
    )
    pattern_confidence_aabb: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Base confidence for AABB (syllable doubling) pattern",
    )
    pattern_confidence_abab: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Base confidence for ABAB (word repeating) pattern",
    )
    pattern_confidence_rhyme: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Base confidence for RHYME pattern",
    )
    pattern_confidence_default: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Default base confidence for unknown patterns",
    )
    high_freq_boost: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Confidence boost when base freq >= high_freq_threshold",
    )
    high_freq_threshold: int = Field(
        default=100,
        ge=0,
        description="Frequency threshold for high_freq_boost",
    )
    high_freq_cap: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after high_freq_boost",
    )
    medium_freq_boost: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Confidence boost when base freq >= medium_freq_threshold",
    )
    medium_freq_threshold: int = Field(
        default=50,
        ge=0,
        description="Frequency threshold for medium_freq_boost",
    )
    medium_freq_cap: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after medium_freq_boost",
    )


class ResourceConfig(BaseModel):
    """
    Configuration for HuggingFace resource loading.

    Controls the download and caching of tokenization resources
    (segmentation model, CRF model, curated lexicon) from HuggingFace.

    Attributes:
        resource_version: Resource version tag on HuggingFace (default: "main").
            Bump with releases for reproducibility.
        hf_repo_base: Base URL for the HuggingFace dataset repository.
        cache_dir: Local cache directory for downloaded resources.
            Defaults to ~/.cache/myspellchecker/resources.
            Can be overridden with MYSPELL_CACHE_DIR environment variable.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    resource_version: str = Field(
        default="main",
        description=(
            "Resource version tag on HuggingFace. Use 'main' until a versioned tag is created."
        ),
    )
    hf_repo_base: str = Field(
        default=("https://huggingface.co/datasets/thettwe/myspellchecker-resources/resolve"),
        description=("Base URL for the HuggingFace dataset repository (without version suffix)."),
    )
    cache_dir: str | None = Field(
        default=None,
        description=(
            "Local cache directory for downloaded resources. "
            "Defaults to ~/.cache/myspellchecker/resources. "
            "Can be overridden with MYSPELL_CACHE_DIR env var."
        ),
    )

    @property
    def hf_repo_url(self) -> str:
        """Full HuggingFace repository URL with version."""
        return f"{self.hf_repo_base}/{self.resource_version}"


class TransformerSegmenterConfig(BaseModel):
    """
    Configuration for transformer-based word segmenter.

    Controls the HuggingFace token classification model used for
    Myanmar word boundary detection via B/I labeling.

    Attributes:
        model_name: HuggingFace model ID or local path.
        device: Device for inference (-1=CPU, 0+=GPU index).
        batch_size: Batch size for batch segmentation.
        max_length: Maximum sequence length for the model.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    model_name: str = Field(
        default="chuuhtetnaing/myanmar-text-segmentation-model",
        description=("HuggingFace model ID or local path for word segmentation."),
    )
    device: int = Field(
        default=-1,
        ge=-1,
        description=("Device for inference. -1 for CPU, 0+ for GPU index."),
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for batch segmentation.",
    )
    max_length: int = Field(
        default=512,
        ge=1,
        description="Maximum sequence length for the model.",
    )
