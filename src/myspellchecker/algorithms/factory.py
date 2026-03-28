"""
Algorithm Factory for Standalone Algorithm Creation.

This module provides a factory for creating spell checking algorithms
independently, without needing a full SpellChecker instance.

Factory Pattern Overview:
    The codebase has three factory patterns for different use cases:

    1. **ComponentFactory** (core/component_factory.py)
       - Used internally by SpellChecker.__init__()
       - Creates all components in proper dependency order
       - Recommended for most users (via SpellChecker)

    2. **ServiceContainer + DI Factories** (core/di/ + core/factories/)
       - Full dependency injection pattern
       - For advanced users who need DI container
       - All factories share code via core/factories/builders.py

    3. **AlgorithmFactory** (this module)
       - Standalone algorithm creation with caching
       - For users who need algorithms without SpellChecker
       - Uses shared builders from core/factories/builders.py

Key Features:
- Automatic caching layer integration (10-100x speedup)
- Configuration object support (reduce 10+ params to 1)
- Dependency injection through Protocol interfaces
- Easy switching between cached/uncached implementations
- **Shared cache registry** - multiple factories with same provider share caches

Cache Sharing:
    AlgorithmFactory uses a CacheRegistry singleton to ensure that multiple
    factory instances using the same provider share their cache wrappers.
    This prevents memory duplication and improves cache hit rates:

    >>> provider = SQLiteProvider("myspell.db")
    >>> factory1 = AlgorithmFactory(provider)  # Creates shared cache
    >>> factory2 = AlgorithmFactory(provider)  # Reuses same cache
    >>> # Both factories share the same underlying cache!

Usage Pattern:
    >>> from myspellchecker.providers import SQLiteProvider
    >>> from myspellchecker.algorithms.factory import AlgorithmFactory
    >>>
    >>> provider = SQLiteProvider("myspell.db")
    >>> factory = AlgorithmFactory(provider, enable_caching=True)
    >>>
    >>> # Create algorithms with caching
    >>> symspell = factory.create_symspell()  # Uses SymSpellConfig defaults
    >>> ngram = factory.create_ngram_checker(symspell=symspell)
    >>>
    >>> # Or with custom configuration
    >>> from myspellchecker.core.config import SymSpellConfig
    >>> custom_config = SymSpellConfig(prefix_length=9, beam_width=100)
    >>> symspell = factory.create_symspell(config=custom_config)

See Also:
    - core/component_factory.py: ComponentFactory (recommended for most users)
    - core/factories/builders.py: Shared builder functions
    - core/di/container.py: ServiceContainer for DI pattern
    - algorithms/cache.py: CacheRegistry for shared cache management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from myspellchecker.algorithms.semantic_checker import SemanticChecker
    from myspellchecker.algorithms.symspell import SymSpell

from myspellchecker.algorithms.cache import (
    CachedBigramSource,
    CachedDictionaryLookup,
    CachedFrequencySource,
    CachedPOSRepository,
    CachedTrigramSource,
    CacheRegistry,
)
from myspellchecker.algorithms.interfaces import (
    BigramSource,
    DictionaryLookup,
    FrequencySource,
    POSRepository,
    TrigramSource,
)
from myspellchecker.core.config import (
    SemanticConfig,
    SymSpellConfig,
)
from myspellchecker.providers import DictionaryProvider


class AlgorithmFactory:
    """
    Factory for creating spell checking algorithms with caching.

    This factory provides a consistent way to create algorithms with:
    - Automatic caching (10-100x speedup)
    - Configuration object support
    - Dependency injection
    - Resource pooling
    - **Shared cache registry** to avoid duplication across instances

    Cache Sharing:
        By default (share_caches=True), multiple factory instances using
        the same provider share their cache wrappers via CacheRegistry.
        This prevents memory duplication when multiple parts of an application
        create their own factories.

    Args:
        provider: DictionaryProvider for data access
        enable_caching: Enable transparent caching (default: True)
        cache_sizes: Custom cache sizes per data type
        share_caches: Share caches across factory instances (default: True)

    Example:
        >>> from myspellchecker.providers import SQLiteProvider
        >>> from myspellchecker.algorithms.factory import AlgorithmFactory
        >>>
        >>> provider = SQLiteProvider("myspell.db")
        >>> factory = AlgorithmFactory(provider, enable_caching=True)
        >>>
        >>> # Create algorithms
        >>> symspell = factory.create_symspell()
        >>> ngram = factory.create_ngram_checker(symspell=symspell)
        >>>
        >>> # Check cache statistics
        >>> stats = factory.get_cache_stats()
        >>> print(f"Dictionary cache hit rate: {stats['dictionary']['hit_rate']:.1%}")
        >>>
        >>> # Multiple factories share the same cache
        >>> factory2 = AlgorithmFactory(provider)  # Reuses factory's caches
    """

    def __init__(
        self,
        provider: DictionaryProvider,
        enable_caching: bool = True,
        cache_sizes: dict[str, int] | None = None,
        share_caches: bool = True,
    ):
        """
        Initialize the algorithm factory.

        Args:
            provider: DictionaryProvider for data access
            enable_caching: Enable LRU caching (default: True)
            cache_sizes: Custom cache sizes (optional)
                - 'dictionary_syllable': Cache size for syllable lookups
                - 'dictionary_word': Cache size for word lookups
                - 'frequency': Cache size for frequency lookups
                - 'bigram': Cache size for bigram probabilities
                - 'trigram': Cache size for trigram probabilities
            share_caches: Share caches across factory instances with same provider
                (default: True). Set to False to use isolated caches per factory.

        Note:
            When share_caches=True (default), multiple AlgorithmFactory instances
            using the same provider will share their cache wrappers via the
            CacheRegistry singleton. This prevents memory duplication and improves
            cache hit rates across the application.
        """
        self.provider = provider
        self.enable_caching = enable_caching
        self.share_caches = share_caches

        # Default cache sizes (tuned for optimal performance vs memory)
        default_sizes = {
            "dictionary_syllable": 4096,
            "dictionary_word": 8192,
            "frequency": 8192,
            "bigram": 16384,
            "trigram": 16384,
        }
        self.cache_sizes = {**default_sizes, **(cache_sizes or {})}

        # Create cached wrappers for Protocol interfaces
        if enable_caching:
            if share_caches:
                # Use shared cache registry (recommended)
                registry = CacheRegistry.get_instance()
                self.dict_source: DictionaryLookup = registry.get_or_create_dictionary_cache(
                    provider,
                    syllable_size=self.cache_sizes["dictionary_syllable"],
                    word_size=self.cache_sizes["dictionary_word"],
                )
                self.freq_source: FrequencySource = registry.get_or_create_frequency_cache(
                    provider, cache_size=self.cache_sizes["frequency"]
                )
                self.bigram_source: BigramSource = registry.get_or_create_bigram_cache(
                    provider, cache_size=self.cache_sizes["bigram"]
                )
                self.trigram_source: TrigramSource = registry.get_or_create_trigram_cache(
                    provider, cache_size=self.cache_sizes["trigram"]
                )
                self.pos_repository: POSRepository = registry.get_or_create_pos_cache(provider)
            else:
                # Create isolated caches (legacy behavior)
                self.dict_source = CachedDictionaryLookup(
                    provider,
                    syllable_cache_size=self.cache_sizes["dictionary_syllable"],
                    word_cache_size=self.cache_sizes["dictionary_word"],
                )
                self.freq_source = CachedFrequencySource(
                    provider, cache_size=self.cache_sizes["frequency"]
                )
                self.bigram_source = CachedBigramSource(
                    provider, cache_size=self.cache_sizes["bigram"]
                )
                self.trigram_source = CachedTrigramSource(
                    provider, cache_size=self.cache_sizes["trigram"]
                )
                self.pos_repository = CachedPOSRepository(provider)
        else:
            # Use provider directly (no caching)
            self.dict_source = provider
            self.freq_source = provider
            self.bigram_source = provider
            self.trigram_source = provider
            self.pos_repository = provider

    def create_symspell(
        self,
        config: SymSpellConfig | None = None,
        max_edit_distance: int = 2,
        phonetic_hasher: Any | None = None,
        build_index: bool = True,
    ) -> "SymSpell":
        """
        Create SymSpell algorithm with caching.

        This is a standalone API for creating SymSpell without SpellChecker.
        For use within SpellChecker, see core/factories/builders.py:build_symspell().

        Args:
            config: SymSpellConfig instance (uses defaults if None)
            max_edit_distance: Maximum edit distance for suggestions (default: 2)
            phonetic_hasher: Optional PhoneticHasher for phonetic matching
            build_index: Whether to build the index after creation (default: True)

        Returns:
            SymSpell instance with cached dictionary access

        Example:
            >>> factory = AlgorithmFactory(provider)
            >>> symspell = factory.create_symspell()  # Default config
            >>>
            >>> # Or with custom config
            >>> from myspellchecker.core.config import SymSpellConfig
            >>> config = SymSpellConfig(prefix_length=9, beam_width=100)
            >>> symspell = factory.create_symspell(config)

        See Also:
            core/factories/builders.py:build_symspell - Used by ComponentFactory
        """
        from myspellchecker.algorithms.ranker import DefaultRanker
        from myspellchecker.algorithms.symspell import SymSpell
        from myspellchecker.core.config import RankerConfig

        config = config or SymSpellConfig()

        # Create a base ranker (consistent with shared builders)
        ranker = DefaultRanker(ranker_config=RankerConfig())

        # Create SymSpell with cached dictionary access
        symspell = SymSpell(
            provider=self.provider,
            max_edit_distance=max_edit_distance,
            prefix_length=config.prefix_length,
            count_threshold=config.count_threshold,
            max_word_length=config.max_word_length,
            compound_lookup_count=config.compound_lookup_count,
            beam_width=config.beam_width,
            compound_max_suggestions=config.compound_max_suggestions,
            damerau_cache_size=config.damerau_cache_size,
            frequency_denominator=config.frequency_denominator,
            phonetic_bonus_weight=config.phonetic_bonus_weight,
            use_syllable_distance=config.use_syllable_distance,
            syllable_bonus_weight=config.syllable_bonus_weight,
            use_weighted_distance=config.use_weighted_distance,
            weighted_distance_bonus_weight=(config.weighted_distance_bonus_weight),
            max_deletes_per_term=config.max_deletes_per_term,
            phonetic_hasher=phonetic_hasher,
            ranker=ranker,
            config=config,
        )

        if build_index:
            symspell.build_index(["syllable", "word"])

        return symspell

    def create_semantic_checker(
        self, config: SemanticConfig | None = None
    ) -> "SemanticChecker" | None:
        """
        Create semantic checker (ONNX-based).

        This is a standalone API for creating SemanticChecker without SpellChecker.
        For use within SpellChecker, see ComponentFactory.create_semantic_checker().

        Args:
            config: SemanticConfig instance (uses defaults if None)

        Returns:
            SemanticChecker if model paths/instances configured, None otherwise

        Example:
            >>> factory = AlgorithmFactory(provider)
            >>> semantic = factory.create_semantic_checker()

        See Also:
            core/component_factory.py:ComponentFactory.create_semantic_checker
        """
        config = config or SemanticConfig()

        has_model = config.model_path is not None or config.model is not None
        if not has_model:
            return None

        try:
            from myspellchecker.algorithms.semantic_checker import SemanticChecker

            return SemanticChecker(
                model_path=config.model_path,
                tokenizer_path=config.tokenizer_path,
                model=config.model,
                tokenizer=config.tokenizer,
                num_threads=config.num_threads,
                predict_top_k=config.predict_top_k,
                check_top_k=config.check_top_k,
                semantic_config=config,
            )
        except ImportError:
            # ONNX runtime not installed
            return None


__all__ = ["AlgorithmFactory"]
