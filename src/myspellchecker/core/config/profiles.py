"""
Configuration Profiles and Presets.

This module provides pre-configured profiles optimized for different use cases:
- development: Fast iteration, minimal validation
- production: Full validation, optimized performance
- testing: Deterministic, reproducible results
- fast: Maximum speed, reduced accuracy
- accurate: Maximum accuracy, slower performance

Each profile returns a fully configured SpellCheckerConfig with settings
tuned for the specific use case.

Example:
    >>> from myspellchecker.core.config.profiles import get_profile
    >>> config = get_profile("production")
    >>> checker = SpellChecker(config=config)
"""

from typing import Literal

from myspellchecker.core.config.algorithm_configs import (
    NgramContextConfig,
    SemanticConfig,
    SymSpellConfig,
)
from myspellchecker.core.config.main import SpellCheckerConfig
from myspellchecker.core.config.tagger_configs import (
    POSTaggerConfig,
)
from myspellchecker.core.config.validation_configs import (
    ProviderConfig,
    ValidationConfig,
)
from myspellchecker.core.exceptions import InvalidConfigError

ProfileName = Literal["development", "production", "testing", "fast", "accurate"]


def get_development_profile() -> SpellCheckerConfig:
    """
    Development profile: Fast iteration with minimal validation.

    Optimized for:
    - Fast startup time
    - Quick feedback during development
    - Reduced resource usage

    Trade-offs:
    - Lower accuracy (rule-based POS only)
    - Minimal context checking
    - Smaller caches

    Returns:
        SpellCheckerConfig optimized for development.
    """
    return SpellCheckerConfig(
        # Fast validation
        max_edit_distance=2,
        max_suggestions=3,
        use_phonetic=True,
        use_context_checker=False,  # Disable for speed
        use_ner=False,  # Disable for speed
        use_rule_based_validation=True,
        word_engine="myword",
        # Fast SymSpell settings
        symspell=SymSpellConfig(
            prefix_length=5,  # Smaller prefix for less memory
            count_threshold=100,  # Higher threshold filters noise
            beam_width=30,  # Smaller beam for speed
            compound_max_suggestions=3,
            damerau_cache_size=2048,  # Smaller cache
        ),
        # Minimal N-gram checking
        ngram_context=NgramContextConfig(
            bigram_threshold=0.001,
            trigram_threshold=0.001,
            candidate_limit=20,  # Fewer candidates
        ),
        # No semantic checking (expensive)
        semantic=SemanticConfig(
            use_proactive_scanning=False,
            use_semantic_refinement=False,
        ),
        # Fast rule-based POS tagger
        pos_tagger=POSTaggerConfig(
            tagger_type="rule_based",
            cache_size=5000,  # Smaller cache
        ),
        # Minimal validation
        validation=ValidationConfig(
            strict_validation=False,
            use_zawgyi_detection=True,
            use_zawgyi_conversion=True,
        ),
        # Small provider cache
        provider_config=ProviderConfig(
            cache_size=512,  # Smaller cache
            pool_max_size=3,  # Fewer connections
        ),
    )


def get_production_profile() -> SpellCheckerConfig:
    """
    Production profile: Full validation with optimized performance.

    Optimized for:
    - High accuracy
    - Balanced performance
    - Production reliability

    Features:
    - Full context validation
    - Viterbi POS tagging for accuracy
    - Larger caches for throughput
    - NER enabled

    Returns:
        SpellCheckerConfig optimized for production.
    """
    return SpellCheckerConfig(
        # Balanced settings
        max_edit_distance=2,
        max_suggestions=5,
        use_phonetic=True,
        use_context_checker=True,
        use_ner=True,
        use_rule_based_validation=True,
        word_engine="myword",
        # Standard SymSpell settings
        symspell=SymSpellConfig(
            prefix_length=7,
            count_threshold=50,
            beam_width=50,
            compound_max_suggestions=5,
            damerau_cache_size=4096,
        ),
        # Full N-gram checking
        ngram_context=NgramContextConfig(
            bigram_threshold=0.0001,
            trigram_threshold=0.0001,
            candidate_limit=50,
        ),
        # Semantic refinement enabled (no proactive scanning)
        semantic=SemanticConfig(
            use_proactive_scanning=False,
            use_semantic_refinement=True,
            proactive_confidence_threshold=0.85,
        ),
        # Viterbi POS tagger for better accuracy
        pos_tagger=POSTaggerConfig(
            tagger_type="viterbi",
            beam_width=10,
            cache_size=10000,
        ),
        # Standard validation
        validation=ValidationConfig(
            strict_validation=True,
            use_zawgyi_detection=True,
            use_zawgyi_conversion=True,
        ),
        # Optimized provider cache
        provider_config=ProviderConfig(
            cache_size=1024,
            pool_max_size=5,
        ),
    )


def get_testing_profile() -> SpellCheckerConfig:
    """
    Testing profile: Deterministic and reproducible results.

    Optimized for:
    - Reproducible results
    - Comprehensive validation
    - Test coverage

    Features:
    - Deterministic settings
    - All features enabled
    - Minimal caching (for consistency)
    - Strict validation

    Returns:
        SpellCheckerConfig optimized for testing.
    """
    return SpellCheckerConfig(
        # Standard settings
        max_edit_distance=2,
        max_suggestions=5,
        use_phonetic=True,
        use_context_checker=True,
        use_ner=True,
        use_rule_based_validation=True,
        word_engine="myword",
        # Standard SymSpell
        symspell=SymSpellConfig(
            prefix_length=7,
            count_threshold=50,
            beam_width=50,
            compound_max_suggestions=5,
            damerau_cache_size=1024,  # Smaller for determinism
        ),
        # Standard N-gram
        ngram_context=NgramContextConfig(
            bigram_threshold=0.0001,
            trigram_threshold=0.0001,
            candidate_limit=50,
        ),
        # No semantic (for determinism)
        semantic=SemanticConfig(
            use_proactive_scanning=False,
            use_semantic_refinement=False,
        ),
        # Rule-based POS for determinism
        pos_tagger=POSTaggerConfig(
            tagger_type="rule_based",
            cache_size=1000,  # Smaller cache
        ),
        # Strict validation
        validation=ValidationConfig(
            strict_validation=True,
            use_zawgyi_detection=True,
            use_zawgyi_conversion=True,
        ),
        # Minimal cache for consistency
        provider_config=ProviderConfig(
            cache_size=256,  # Small cache
            pool_max_size=3,
        ),
    )


def get_fast_profile() -> SpellCheckerConfig:
    """
    Fast profile: Maximum speed with acceptable accuracy.

    Optimized for:
    - Sub-50ms response times
    - Low resource usage
    - Real-time applications

    Trade-offs:
    - Lower accuracy (no context checking)
    - Fewer suggestions
    - Minimal validation

    Returns:
        SpellCheckerConfig optimized for speed.
    """
    return SpellCheckerConfig(
        # Minimal settings
        max_edit_distance=1,  # Only 1 edit distance
        max_suggestions=3,  # Fewer suggestions
        use_phonetic=False,  # Disable phonetic
        use_context_checker=False,  # Disable context
        use_ner=False,  # Disable NER
        use_rule_based_validation=True,
        word_engine="myword",
        # Fast SymSpell
        symspell=SymSpellConfig(
            prefix_length=5,
            count_threshold=200,  # High threshold
            beam_width=20,  # Small beam
            compound_max_suggestions=2,
            damerau_cache_size=1024,
        ),
        # No N-gram checking
        ngram_context=NgramContextConfig(
            bigram_threshold=0.01,  # Very high threshold
            trigram_threshold=0.01,
            candidate_limit=10,
        ),
        # No semantic
        semantic=SemanticConfig(
            use_proactive_scanning=False,
            use_semantic_refinement=False,
        ),
        # Fast rule-based POS
        pos_tagger=POSTaggerConfig(
            tagger_type="rule_based",
            cache_size=2000,
        ),
        # Minimal validation
        validation=ValidationConfig(
            strict_validation=False,
            use_zawgyi_detection=False,  # Disable for speed
            use_zawgyi_conversion=False,
        ),
        # Minimal provider cache
        provider_config=ProviderConfig(
            cache_size=256,
            pool_max_size=2,
        ),
    )


def get_accurate_profile() -> SpellCheckerConfig:
    """
    Accurate profile: Maximum accuracy with slower performance.

    Optimized for:
    - Highest possible accuracy
    - Comprehensive error detection
    - Batch processing

    Features:
    - Maximum edit distance
    - Full context validation
    - Semantic proactive scanning
    - Large caches
    - Viterbi POS tagging with large beam
    - Candidate fusion (calibrated Noisy-OR)

    Returns:
        SpellCheckerConfig optimized for accuracy.
    """
    return SpellCheckerConfig(
        # Maximum accuracy
        max_edit_distance=3,  # Maximum edit distance
        max_suggestions=10,  # More suggestions
        use_phonetic=True,
        use_context_checker=True,
        use_ner=True,
        use_rule_based_validation=True,
        word_engine="myword",
        # Comprehensive SymSpell
        symspell=SymSpellConfig(
            prefix_length=8,  # Longer prefix
            count_threshold=10,  # Lower threshold
            beam_width=100,  # Large beam
            compound_max_suggestions=10,
            damerau_cache_size=8192,  # Large cache
        ),
        # Comprehensive N-gram
        ngram_context=NgramContextConfig(
            bigram_threshold=0.00001,  # Very low threshold
            trigram_threshold=0.00001,
            candidate_limit=100,  # Many candidates
        ),
        # Full semantic validation
        semantic=SemanticConfig(
            use_proactive_scanning=True,  # Enable proactive
            use_semantic_refinement=True,
            proactive_confidence_threshold=0.85,  # Higher precision default
        ),
        # Viterbi POS with large beam
        pos_tagger=POSTaggerConfig(
            tagger_type="viterbi",  # Transformer requires separate opt-in via POSTaggerConfig
            beam_width=15,  # Larger beam
            cache_size=20000,  # Large cache
        ),
        # Strict validation with candidate fusion
        validation=ValidationConfig(
            strict_validation=True,
            use_zawgyi_detection=True,
            use_zawgyi_conversion=True,
            use_candidate_fusion=True,
        ),
        # Large caches
        provider_config=ProviderConfig(
            cache_size=4096,  # Large cache
            pool_max_size=10,
        ),
    )


def get_profile(name: ProfileName = "production") -> SpellCheckerConfig:
    """
    Get a pre-configured profile by name.

    Args:
        name: Profile name. Options:
            - "development": Fast iteration, minimal validation
            - "production": Balanced accuracy and performance (default)
            - "testing": Deterministic, reproducible results
            - "fast": Maximum speed, reduced accuracy
            - "accurate": Maximum accuracy, slower performance

    Returns:
        SpellCheckerConfig for the specified profile.

    Raises:
        ValueError: If profile name is not recognized.

    Example:
        >>> config = get_profile("production")
        >>> checker = SpellChecker(config=config)
    """
    profiles = {
        "development": get_development_profile,
        "production": get_production_profile,
        "testing": get_testing_profile,
        "fast": get_fast_profile,
        "accurate": get_accurate_profile,
    }

    if name not in profiles:
        raise InvalidConfigError(
            f"Unknown profile: {name}. Available profiles: {', '.join(profiles.keys())}"
        )

    return profiles[name]()
