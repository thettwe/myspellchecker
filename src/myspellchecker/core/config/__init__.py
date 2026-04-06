"""
Configuration Package for mySpellChecker.

This package provides centralized configuration management for the spell
checker, organized into focused modules for better maintainability and clarity.

Quick Start:
    >>> from myspellchecker.core.config import SpellCheckerConfig
    >>> config = SpellCheckerConfig(max_suggestions=10)

    >>> # Or use presets for common use cases
    >>> from myspellchecker.core.config import get_profile
    >>> config = get_profile("fast")  # Speed-optimized
    >>> config = get_profile("accurate")  # Quality-optimized

Module Organization:
    - **main.py**: Main SpellCheckerConfig aggregator
    - **algorithm_configs.py**: Algorithm settings (SymSpell, N-gram, Phonetic, etc.)
    - **tagger_configs.py**: POS tagging configurations (POSTagger, Joint)
    - **grammar_configs.py**: Grammar engine configurations
    - **validation_configs.py**: Validation and provider configurations
    - **text_configs.py**: Text processing configurations (Zawgyi, Tone, Stemmer)
    - **profiles.py**: Pre-defined configuration profiles
    - **loader.py**: Environment-based configuration loading

Exported Classes:
    Main Configuration:
        - SpellCheckerConfig: Primary configuration class

    Algorithm Configurations:
        - SymSpellConfig: SymSpell algorithm parameters
        - NgramContextConfig: Bigram/trigram thresholds
        - PhoneticConfig: Phonetic matching settings
        - SemanticConfig: AI model configuration
        - AlgorithmCacheConfig: Cache size and TTL settings
        - RankerConfig: Suggestion ranking weights

    Tagger Configurations:
        - POSTaggerConfig: POS tagger type and model
        - JointConfig: Joint segmentation+tagging

    Grammar Configurations:
        - GrammarEngineConfig: Grammar checker settings
        - AspectCheckerConfig, ClassifierCheckerConfig, etc.

    Validation Configurations:
        - ValidationConfig: Error detection thresholds
        - ProviderConfig: Database provider settings
        - ConnectionPoolConfig: Connection pool settings

    Text Configurations:
        - ZawgyiConfig: Zawgyi detection/conversion
        - ToneConfig: Tone mark handling
        - StemmerConfig: Suffix stripping settings

    Profiles & Loaders:
        - get_profile(): Get predefined configuration profile
        - ProfileName: Profile name literals
        - ConfigLoader: Environment-based loader
        - load_config(): Load config from environment

Example:
    >>> from myspellchecker.core.config import (
    ...     SpellCheckerConfig,
    ...     SymSpellConfig,
    ...     NgramContextConfig,
    ... )
    >>>
    >>> # Custom configuration
    >>> config = SpellCheckerConfig(
    ...     symspell=SymSpellConfig(prefix_length=8),
    ...     ngram_context=NgramContextConfig(bigram_threshold=0.001),
    ...     max_suggestions=10,
    ... )
    >>>
    >>> # Load from environment variables
    >>> from myspellchecker.core.config import load_config
    >>> config = load_config()  # Reads MYSPELL_* env vars
"""

# Algorithm configurations
from myspellchecker.core.config.algorithm_configs import (
    AlgorithmCacheConfig,
    BrokenCompoundStrategyConfig,
    CompoundResolverConfig,
    ConfusableSemanticConfig,
    FrequencyGuardConfig,
    HomophoneStrategyConfig,
    MorphologyConfig,
    NeuralRerankerConfig,
    NgramContextConfig,
    NgramStrategyConfig,
    PhoneticConfig,
    RankerConfig,
    ReduplicationConfig,
    ResourceConfig,
    SemanticConfig,
    SemanticStrategyConfig,
    SymSpellConfig,
    TokenRefinementConfig,
    ToneStrategyConfig,
    TransformerSegmenterConfig,
)

# Grammar configurations
from myspellchecker.core.config.grammar_configs import (
    AspectCheckerConfig,
    ClassifierCheckerConfig,
    CompoundCheckerConfig,
    GrammarEngineConfig,
    MergedWordCheckerConfig,
    NegationCheckerConfig,
    ParticleCheckerConfig,
    RegisterCheckerConfig,
    TenseAgreementCheckerConfig,
)

# Configuration loader
from myspellchecker.core.config.loader import ConfigLoader, load_config

# Main configuration
from myspellchecker.core.config.main import SpellCheckerConfig

# Profiles and presets
from myspellchecker.core.config.profiles import ProfileName, get_profile

# Tagger configurations
from myspellchecker.core.config.tagger_configs import (
    JointConfig,
    POSTaggerConfig,
)

# Text processing configurations
from myspellchecker.core.config.text_configs import (
    StemmerConfig,
    ToneConfig,
    ZawgyiConfig,
)

# Validation configurations
from myspellchecker.core.config.validation_configs import (
    ConnectionPoolConfig,
    ProviderConfig,
    ValidationConfig,
)

# NER configuration (re-exported for convenience)
from myspellchecker.text.ner_config import NERConfig

__all__ = [
    # Main configuration
    "SpellCheckerConfig",
    # NER configuration
    "NERConfig",
    # Algorithm configurations
    "AlgorithmCacheConfig",
    "BrokenCompoundStrategyConfig",
    "CompoundResolverConfig",
    "ConfusableSemanticConfig",
    "FrequencyGuardConfig",
    "HomophoneStrategyConfig",
    "MorphologyConfig",
    "NeuralRerankerConfig",
    "NgramContextConfig",
    "NgramStrategyConfig",
    "PhoneticConfig",
    "RankerConfig",
    "ReduplicationConfig",
    "ResourceConfig",
    "SemanticConfig",
    "SemanticStrategyConfig",
    "SymSpellConfig",
    "TokenRefinementConfig",
    "ToneStrategyConfig",
    "TransformerSegmenterConfig",
    # Text processing configurations
    "ZawgyiConfig",
    "ToneConfig",
    "StemmerConfig",
    # Grammar configurations
    "GrammarEngineConfig",
    "AspectCheckerConfig",
    "ClassifierCheckerConfig",
    "CompoundCheckerConfig",
    "MergedWordCheckerConfig",
    "NegationCheckerConfig",
    "ParticleCheckerConfig",
    "RegisterCheckerConfig",
    "TenseAgreementCheckerConfig",
    # Tagger configurations
    "POSTaggerConfig",
    "JointConfig",
    # Validation configurations
    "ValidationConfig",
    "ProviderConfig",
    "ConnectionPoolConfig",
    # Profiles
    "get_profile",
    "ProfileName",
    # Loader
    "ConfigLoader",
    "load_config",
]
