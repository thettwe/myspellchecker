"""
Text Processing Configuration Classes.

This module contains configuration classes for text processing:
- ZawgyiConfig: Zawgyi detection and conversion thresholds
- ToneConfig: Tone disambiguation settings
- StemmerConfig: Stemmer/lemmatizer settings
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
