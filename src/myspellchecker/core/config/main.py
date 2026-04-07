"""
Main SpellChecker Configuration.

This module contains the main SpellCheckerConfig class that aggregates
all other configuration classes into a unified configuration object.
"""

from __future__ import annotations

import warnings
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from myspellchecker.core.config.algorithm_configs import (
    AlgorithmCacheConfig,
    BrokenCompoundStrategyConfig,
    CompoundResolverConfig,
    FrequencyGuardConfig,
    NeuralRerankerConfig,
    NgramContextConfig,
    PhoneticConfig,
    RankerConfig,
    ReduplicationConfig,
    SemanticConfig,
    SymSpellConfig,
    TokenRefinementConfig,
)
from myspellchecker.core.config.grammar_configs import GrammarEngineConfig
from myspellchecker.core.config.tagger_configs import (
    JointConfig,
    POSTaggerConfig,
)
from myspellchecker.core.config.validation_configs import ProviderConfig, ValidationConfig
from myspellchecker.core.exceptions import InvalidConfigError
from myspellchecker.providers import DictionaryProvider
from myspellchecker.segmenters import Segmenter
from myspellchecker.text.ner_config import NERConfig


class SpellCheckerConfig(BaseModel):
    """
    Main configuration for the SpellChecker.

    Provides a centralized way to manage spell checking parameters,
    organized into logical groups for easier management and extensibility.

    The configuration supports nested config objects for fine-grained
    control over each component of the spell checking pipeline.

    Core Parameters:
        - **max_edit_distance**: Controls suggestion quality vs speed tradeoff.
          Lower values (1) are faster but may miss suggestions.
          Higher values (3) catch more typos but are slower.

        - **max_suggestions**: Number of correction suggestions per error.
          More suggestions provide options but increase response size.

        - **use_phonetic**: Enables phonetic similarity matching.
          Helps catch words that sound similar but are spelled differently.

        - **use_context_checker**: Enables N-gram context validation.
          Detects unlikely word sequences using bigram/trigram probabilities.

    Nested Configurations:
        - **symspell**: SymSpell algorithm settings (prefix_length, beam_width, etc.)
        - **ngram_context**: Context checker thresholds (bigram/trigram cutoffs)
        - **phonetic**: Phonetic hashing algorithm settings
        - **semantic**: AI-powered semantic checking (model paths, thresholds)
        - **pos_tagger**: POS tagging configuration (tagger type, model)
        - **joint**: Joint segmentation+tagging settings
        - **validation**: Error detection thresholds and confidence levels
        - **provider_config**: Database provider settings (cache size, pool config)
        - **cache**: Unified cache configuration
        - **ranker**: Suggestion ranking weights and strategy

    Attributes:
        segmenter: Segmentation engine (handles syllable/word breaks).
        provider: Dictionary provider (for validation and frequencies).
        max_edit_distance: Max edit distance for suggestions (1-3, default: 2).
        max_suggestions: Max suggestions per error (default: 5).
        use_phonetic: Enable phonetic similarity matching (default: True).
        use_context_checker: Enable context-aware checking (default: True).
        use_ner: Enable Named Entity Recognition (default: True).
        use_rule_based_validation: Enable rule-based syllable validation (default: True).
        word_engine: Word segmentation engine ("myword", "crf", "transformer", default: "myword").
        fallback_to_empty_provider: Allow empty provider if database not found (default: False).

    Example:
        >>> from myspellchecker.core.config import (
        ...     SpellCheckerConfig,
        ...     SymSpellConfig,
        ...     NgramContextConfig,
        ... )
        >>>
        >>> # Basic configuration
        >>> config = SpellCheckerConfig(
        ...     max_edit_distance=2,
        ...     max_suggestions=5,
        ... )
        >>>
        >>> # Detailed configuration with nested configs
        >>> config = SpellCheckerConfig(
        ...     symspell=SymSpellConfig(prefix_length=8, beam_width=100),
        ...     ngram_context=NgramContextConfig(
        ...         bigram_threshold=0.001,
        ...         trigram_threshold=0.0001,
        ...     ),
        ...     max_suggestions=10,
        ...     use_phonetic=True,
        ...     use_context_checker=True,
        ... )
        >>>
        >>> # Fast configuration (disable expensive features)
        >>> config = SpellCheckerConfig(
        ...     use_phonetic=False,
        ...     use_context_checker=False,
        ...     use_ner=False,
        ...     max_edit_distance=1,
        ... )
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow Segmenter and DictionaryProvider instances
    )

    # Core dependencies (runtime objects - not serializable to JSON schema)
    segmenter: Segmenter | None = Field(
        default=None,
        description="Segmentation engine for syllable/word breaks (runtime object)",
    )
    provider: DictionaryProvider | None = Field(
        default=None,
        description="Dictionary provider for validation and frequencies (runtime object)",
    )

    # Top-level settings
    max_edit_distance: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Maximum edit distance for spell correction suggestions (1-3)",
    )
    max_suggestions: int = Field(
        default=5,
        ge=1,
        description="Maximum number of suggestions to return per error",
    )
    max_text_length: int = Field(
        default=100_000,
        ge=1,
        description="Maximum input text length in characters. Prevents resource exhaustion.",
    )
    use_phonetic: bool = Field(
        default=True,
        description="Enable phonetic similarity matching for suggestions",
    )
    use_context_checker: bool = Field(
        default=True,
        description="Enable context-aware N-gram validation",
    )
    use_ner: bool = Field(
        default=True,
        description="Enable Named Entity Recognition heuristics",
    )
    use_rule_based_validation: bool = Field(
        default=True,
        description="Enable rule-based syllable validation",
    )
    word_engine: Literal["myword", "crf", "transformer"] = Field(
        default="myword",
        description="Word segmentation engine backend",
    )
    seg_model: str | None = Field(
        default=None,
        description=(
            "Custom model name/path for transformer word segmentation engine. "
            "Only used when word_engine='transformer'. "
            "Default: chuuhtetnaing/myanmar-text-segmentation-model"
        ),
    )
    seg_device: int = Field(
        default=-1,
        description=(
            "Device for transformer word segmentation inference. "
            "-1 for CPU, 0+ for GPU index. Only used when word_engine='transformer'."
        ),
    )
    fallback_to_empty_provider: bool = Field(
        default=False,
        description=(
            "If True, silently fall back to empty MemoryProvider when database not found. "
            "If False (default), raise MissingDatabaseError. Set to True only for testing "
            "or when you explicitly want to handle missing databases gracefully."
        ),
    )

    # Nested configuration objects
    symspell: SymSpellConfig = Field(
        default_factory=SymSpellConfig,
        description="SymSpell algorithm configuration",
    )
    ngram_context: NgramContextConfig = Field(
        default_factory=NgramContextConfig,
        description="N-gram context checker configuration",
    )
    phonetic: PhoneticConfig = Field(
        default_factory=PhoneticConfig,
        description="Phonetic matching configuration",
    )
    semantic: SemanticConfig = Field(
        default_factory=SemanticConfig,
        description="Semantic model configuration",
    )
    pos_tagger: POSTaggerConfig = Field(
        default_factory=POSTaggerConfig,
        description="Pluggable POS tagger configuration",
    )
    joint: JointConfig = Field(
        default_factory=JointConfig,
        description="Joint segmentation-tagging configuration",
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Error validation configuration",
    )
    provider_config: ProviderConfig = Field(
        default_factory=ProviderConfig,
        description="Provider caching and query configuration",
    )
    cache: AlgorithmCacheConfig = Field(
        default_factory=AlgorithmCacheConfig,
        description="Unified cache configuration for all caching layers",
    )
    ranker: RankerConfig = Field(
        default_factory=RankerConfig,
        description="Suggestion ranker configuration (type and scoring weights)",
    )
    frequency_guards: FrequencyGuardConfig = Field(
        default_factory=FrequencyGuardConfig,
        description="Centralized frequency thresholds for validators and strategies",
    )
    compound_resolver: CompoundResolverConfig = Field(
        default_factory=CompoundResolverConfig,
        description="Compound word resolution configuration",
    )
    reduplication: ReduplicationConfig = Field(
        default_factory=ReduplicationConfig,
        description="Reduplication detection configuration",
    )
    neural_reranker: NeuralRerankerConfig = Field(
        default_factory=NeuralRerankerConfig,
        description="Neural MLP suggestion reranker configuration",
    )
    broken_compound_strategy: BrokenCompoundStrategyConfig = Field(
        default_factory=BrokenCompoundStrategyConfig,
        description="Broken compound detection strategy configuration",
    )
    token_refinement: TokenRefinementConfig = Field(
        default_factory=TokenRefinementConfig,
        description="Token boundary refinement scoring configuration",
    )
    grammar_engine: GrammarEngineConfig = Field(
        default_factory=GrammarEngineConfig,
        description="Grammar engine configuration (rule priorities, checker confidence thresholds)",
    )
    ner: NERConfig | None = Field(
        default=None,
        description=(
            "NER model configuration. When provided with enabled=True, overrides "
            "use_ner=True and uses the specified NER model (heuristic or transformer). "
            "When None, use_ner=True falls back to heuristic NER for backward compat."
        ),
    )

    @model_validator(mode="after")
    def validate_dependencies_and_types(self) -> "SpellCheckerConfig":
        """Validate dependency types and warn about optional dependencies."""
        # If NERConfig is provided and enabled, ensure use_ner is True.
        # Guard against recursive validation by checking current value first.
        if self.ner is not None and self.ner.enabled and not self.use_ner:
            self.use_ner = True

        # Type validation for segmenter
        if self.segmenter is not None and not isinstance(self.segmenter, Segmenter):
            raise InvalidConfigError(
                f"segmenter must be an instance of Segmenter, got {type(self.segmenter).__name__}"
            )

        # Type validation for provider
        if self.provider is not None and not isinstance(self.provider, DictionaryProvider):
            raise InvalidConfigError(
                "provider must be an instance of DictionaryProvider, "
                f"got {type(self.provider).__name__}"
            )

        # Warn about optional dependencies for non-default word engines
        if self.word_engine == "crf":
            try:
                import pycrfsuite  # noqa: F401
            except ImportError:
                warnings.warn(
                    "word_engine='crf' requires pycrfsuite. "
                    "Install with: pip install python-crfsuite",
                    UserWarning,
                    stacklevel=2,
                )

        if self.word_engine == "transformer":
            try:
                import transformers  # noqa: F401
            except ImportError:
                warnings.warn(
                    "word_engine='transformer' requires transformers and torch. "
                    "Install with: pip install myspellchecker[transformers]",
                    UserWarning,
                    stacklevel=2,
                )

        return self
