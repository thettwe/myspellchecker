"""
SpellChecker Builder Pattern Implementation.

This module provides a fluent builder API for constructing SpellChecker
instances with complex configurations. It simplifies initialization
by providing sensible defaults and a step-by-step configuration approach.

Example:
    >>> from myspellchecker.core.builder import SpellCheckerBuilder
    >>> checker = (
    ...     SpellCheckerBuilder()
    ...     .with_phonetic(True)
    ...     .with_context_checking(True)
    ...     .with_max_suggestions(10)
    ...     .build()
    ... )
"""

from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING, Any, Literal

# Lazy imports to avoid loading heavy dependencies (pycrfsuite, etc.) at module import time
# These are imported inside methods that need them

if TYPE_CHECKING:
    from myspellchecker.core.config import SpellCheckerConfig
    from myspellchecker.core.spellchecker import SpellChecker
    from myspellchecker.providers import DictionaryProvider
    from myspellchecker.segmenters import Segmenter

__all__ = [
    "SpellCheckerBuilder",
]


class SpellCheckerBuilder:
    """
    Builder for constructing SpellChecker instances with fluent API.

    Provides a readable and maintainable way to configure SpellChecker
    with many options, using method chaining for clean configuration.

    Example:
        >>> checker = (
        ...     SpellCheckerBuilder()
        ...     .with_phonetic(True)
        ...     .with_context_checking(True)
        ...     .with_ner(True)
        ...     .with_max_edit_distance(2)
        ...     .build()
        ... )

        >>> # Or with custom components
        >>> checker = (
        ...     SpellCheckerBuilder()
        ...     .with_provider(custom_provider)
        ...     .with_segmenter(custom_segmenter)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize builder with default configuration."""
        from myspellchecker.core.config import SpellCheckerConfig

        self._config = SpellCheckerConfig()
        self._provider: "DictionaryProvider" | None = None
        self._segmenter: "Segmenter" | None = None

    def with_config(self, config: "SpellCheckerConfig") -> "SpellCheckerBuilder":
        """
        Set the full configuration object.

        This replaces any previously set configuration options.

        Args:
            config: SpellCheckerConfig instance.

        Returns:
            Self for method chaining.
        """
        self._config = config
        return self

    def with_provider(self, provider: "DictionaryProvider") -> "SpellCheckerBuilder":
        """
        Set a custom dictionary provider.

        Args:
            provider: DictionaryProvider instance.

        Returns:
            Self for method chaining.
        """
        self._provider = provider
        return self

    def with_segmenter(self, segmenter: "Segmenter") -> "SpellCheckerBuilder":
        """
        Set a custom text segmenter.

        Args:
            segmenter: Segmenter instance.

        Returns:
            Self for method chaining.
        """
        self._segmenter = segmenter
        return self

    # --- Feature toggles ---

    def with_phonetic(self, enabled: bool = True) -> "SpellCheckerBuilder":
        """
        Enable or disable phonetic similarity matching.

        Args:
            enabled: Whether to enable phonetic matching (default: True).

        Returns:
            Self for method chaining.
        """
        self._config.use_phonetic = enabled
        return self

    def with_context_checking(self, enabled: bool = True) -> "SpellCheckerBuilder":
        """
        Enable or disable N-gram context checking.

        Args:
            enabled: Whether to enable context checking (default: True).

        Returns:
            Self for method chaining.
        """
        self._config.use_context_checker = enabled
        return self

    def with_ner(self, enabled: bool = True) -> "SpellCheckerBuilder":
        """
        Enable or disable Named Entity Recognition heuristics.

        Args:
            enabled: Whether to enable NER (default: True).

        Returns:
            Self for method chaining.
        """
        self._config.use_ner = enabled
        return self

    def with_rule_based_validation(self, enabled: bool = True) -> "SpellCheckerBuilder":
        """
        Enable or disable rule-based syllable validation.

        Args:
            enabled: Whether to enable rule-based validation (default: True).

        Returns:
            Self for method chaining.
        """
        self._config.use_rule_based_validation = enabled
        return self

    def with_candidate_fusion(
        self,
        enabled: bool = True,
        calibration_path: str | None = None,
        fusion_threshold: float = 0.5,
    ) -> "SpellCheckerBuilder":
        """
        Enable or disable calibrated Noisy-OR candidate fusion.

        When enabled, all strategies may fire at every position and the
        arbiter uses calibrated confidence fusion to determine which
        errors to emit. Replaces the default mutex-based selection.

        Args:
            enabled: Whether to enable candidate fusion (default: True).
            calibration_path: Optional path to a YAML file with per-strategy
                calibration breakpoints (produced by train_calibrators.py).
            fusion_threshold: Minimum fused confidence to emit an error
                (default: 0.5).

        Returns:
            Self for method chaining.
        """
        self._config.validation.use_candidate_fusion = enabled
        if calibration_path is not None:
            self._config.validation.calibration_path = calibration_path
        self._config.validation.fusion_confidence_threshold = fusion_threshold
        return self

    # --- Performance tuning ---

    def with_max_edit_distance(self, distance: int) -> "SpellCheckerBuilder":
        """
        Set maximum edit distance for suggestions.

        Args:
            distance: Max edit distance (1-3).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If distance is not between 1 and 3.
        """
        if not isinstance(distance, int):
            raise TypeError(f"max_edit_distance must be an integer, got {type(distance).__name__}")
        if distance < 1 or distance > 3:
            raise ValueError(f"max_edit_distance must be between 1 and 3, got {distance}")
        self._config.max_edit_distance = distance
        return self

    def with_max_suggestions(self, count: int) -> "SpellCheckerBuilder":
        """
        Set maximum number of suggestions per error.

        Args:
            count: Max suggestions (>= 1).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If count is less than 1.
        """
        if count < 1:
            raise ValueError(f"max_suggestions must be >= 1, got {count}")
        self._config.max_suggestions = count
        return self

    def with_symspell_prefix_length(self, length: int) -> "SpellCheckerBuilder":
        """
        Set SymSpell prefix length for performance optimization.

        Shorter prefix = faster but less accurate.
        Longer prefix = more accurate but slower.

        Args:
            length: Prefix length (typically 5-10).

        Returns:
            Self for method chaining.
        """
        self._config.symspell.prefix_length = length
        return self

    def with_cache_size(self, size: int) -> "SpellCheckerBuilder":
        """
        Set provider cache size for memory optimization.

        Args:
            size: Cache size (number of entries).

        Returns:
            Self for method chaining.
        """
        self._config.provider_config.cache_size = size
        return self

    # --- Context checking thresholds ---

    def with_bigram_threshold(self, threshold: float) -> "SpellCheckerBuilder":
        """
        Set probability threshold for flagging bigram errors.

        Lower threshold = more sensitive (more errors flagged).
        Higher threshold = less sensitive (fewer errors flagged).

        Args:
            threshold: Probability threshold (0.0-1.0).

        Returns:
            Self for method chaining.
        """
        self._config.ngram_context.bigram_threshold = threshold
        return self

    def with_trigram_threshold(self, threshold: float) -> "SpellCheckerBuilder":
        """
        Set probability threshold for flagging trigram errors.

        Args:
            threshold: Probability threshold (0.0-1.0).

        Returns:
            Self for method chaining.
        """
        self._config.ngram_context.trigram_threshold = threshold
        return self

    # --- Semantic model ---

    def with_semantic_model(
        self,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> "SpellCheckerBuilder":
        """
        Configure semantic checking model.

        Provide either paths to load model from disk, or pre-loaded
        model/tokenizer instances.

        Args:
            model_path: Path to semantic model file.
            tokenizer_path: Path to tokenizer file.
            model: Pre-loaded model instance.
            tokenizer: Pre-loaded tokenizer instance.

        Returns:
            Self for method chaining.
        """
        # Only update fields that were explicitly provided (non-None)
        # to avoid overwriting previously set values
        update: dict[str, Any] = {}
        if model_path is not None:
            update["model_path"] = model_path
        if tokenizer_path is not None:
            update["tokenizer_path"] = tokenizer_path
        if model is not None:
            update["model"] = model
        if tokenizer is not None:
            update["tokenizer"] = tokenizer
        if update:
            self._config.semantic = self._config.semantic.model_copy(update=update)
        return self

    # --- Word engine ---

    def with_word_engine(
        self, engine: Literal["myword", "crf", "transformer"]
    ) -> "SpellCheckerBuilder":
        """
        Set the word segmentation engine.

        Args:
            engine: Engine name ("myword", "crf", or "transformer").

        Returns:
            Self for method chaining.
        """
        self._config.word_engine = engine
        return self

    # --- Build ---

    def build(self) -> "SpellChecker":
        """
        Construct SpellChecker with all configured options.

        Resolves defaults for provider and segmenter if not explicitly set,
        then constructs the SpellChecker with all components.

        Resolution Order:
            1. Uses explicitly set provider/segmenter via ``with_provider()``/``with_segmenter()``
            2. Falls back to auto-detected SQLiteProvider (searches standard paths)
            3. If no database found and ``fallback_to_empty_provider=True``, uses MemoryProvider

        Returns:
            Fully configured SpellChecker instance ready for use.

        Raises:
            MissingDatabaseError: If no database is found and
                ``fallback_to_empty_provider=False`` (default).

        Example:
            >>> # Basic build with defaults
            >>> checker = SpellCheckerBuilder().build()
            >>>
            >>> # Build with custom configuration
            >>> checker = (
            ...     SpellCheckerBuilder()
            ...     .with_phonetic(True)
            ...     .with_max_edit_distance(2)
            ...     .with_max_suggestions(10)
            ...     .with_context_checking(True)
            ...     .build()
            ... )
            >>>
            >>> # Build with custom provider
            >>> from myspellchecker.providers import SQLiteProvider
            >>> checker = (
            ...     SpellCheckerBuilder()
            ...     .with_provider(SQLiteProvider("custom.db"))
            ...     .build()
            ... )
        """
        # Lazy imports to avoid loading heavy dependencies until needed
        from myspellchecker.core.exceptions import MissingDatabaseError
        from myspellchecker.core.spellchecker import SpellChecker
        from myspellchecker.providers import MemoryProvider, SQLiteProvider
        from myspellchecker.segmenters import DefaultSegmenter

        # Resolve provider
        provider = self._provider
        if provider is None:
            try:
                provider = SQLiteProvider(
                    database_path=self._config.provider_config.database_path,
                    cache_size=self._config.provider_config.cache_size,
                    pool_min_size=self._config.provider_config.pool_min_size,
                    pool_max_size=self._config.provider_config.pool_max_size,
                    pool_timeout=self._config.provider_config.pool_timeout,
                    pool_max_connection_age=self._config.provider_config.pool_max_connection_age,
                    curated_min_frequency=self._config.provider_config.curated_min_frequency,
                )
            except MissingDatabaseError as e:
                # Check if fallback is allowed
                if self._config.fallback_to_empty_provider:
                    warnings.warn(
                        "Default database not found. Using empty MemoryProvider.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    provider = MemoryProvider()
                else:
                    raise MissingDatabaseError(
                        message="No database available for spell checking.",
                        suggestion=(
                            "No bundled database is included. Build one first:\n"
                            "  myspellchecker build --sample\n"
                            "Then pass the path:\n"
                            "  SpellCheckerBuilder().with_provider("
                            "SQLiteProvider(database_path='mySpellChecker-default.db'))\n"
                            "Or set fallback_to_empty_provider=True in SpellCheckerConfig."
                        ),
                    ) from e

        # Resolve segmenter
        segmenter = self._segmenter
        if segmenter is None:
            segmenter = DefaultSegmenter(
                word_engine=self._config.word_engine,
                allow_extended_myanmar=self._config.validation.allow_extended_myanmar,
                seg_model=self._config.seg_model,
                seg_device=self._config.seg_device,
            )

        return SpellChecker(
            config=self._config.model_copy(deep=True),
            provider=provider,
            segmenter=segmenter,
        )


# --- Configuration Presets ---


# Metaclass for lazy preset loading - must be defined before ConfigPresets
class _ConfigPresetsMeta(type):
    """
    Metaclass for lazy preset loading.

    Thread Safety:
        Returns deep copies of cached configs to prevent mutation of
        shared instances. Users can safely modify returned configs without
        affecting other accesses.
    """

    _cache: dict = {}
    _lock = threading.Lock()

    def __getattr__(cls, name: str) -> Any:
        if name in ("DEFAULT", "FAST", "ACCURATE", "MINIMAL", "STRICT"):
            with cls._lock:
                if name not in cls._cache:
                    method = getattr(cls, name.lower())
                    cls._cache[name] = method()
                # Return a deep copy to prevent mutation of cached instance
                # Uses Pydantic's model_copy(deep=True) for proper nested copying
                return cls._cache[name].model_copy(deep=True)
        raise AttributeError(f"type object '{cls.__name__}' has no attribute '{name}'")


class ConfigPresets(metaclass=_ConfigPresetsMeta):
    """
    Pre-configured SpellCheckerConfig instances for common use cases.

    Provides sensible defaults for different scenarios without
    requiring detailed knowledge of all configuration options.

    Class attributes (DEFAULT, FAST, ACCURATE, MINIMAL, STRICT) are
    lazily evaluated on first access to avoid import overhead.

    Available Presets:
        - **DEFAULT**: Balanced configuration with all features enabled.
          Good starting point for most applications.

        - **FAST**: Speed-optimized with expensive features disabled.
          Use for real-time applications, chat, or high-throughput scenarios.
          ~3x faster than DEFAULT but may miss some errors.

        - **ACCURATE**: Quality-optimized with lower detection thresholds.
          Use for document editing, publishing, or when accuracy is critical.
          ~2x slower than DEFAULT but catches more errors.

        - **MINIMAL**: Basic syllable validation only.
          Use for resource-constrained environments or when only basic
          checking is needed. Fastest option.

        - **STRICT**: Conservative thresholds that flag more potential errors.
          Use for formal documents where false positives are acceptable.
          May have higher false positive rate.

    Thread Safety:
        Each access returns a deep copy of the preset configuration,
        allowing safe modification without affecting other users.

    Example:
        >>> from myspellchecker import SpellChecker
        >>> from myspellchecker.core.builder import ConfigPresets
        >>>
        >>> # Use a preset directly
        >>> checker = SpellChecker(config=ConfigPresets.FAST)
        >>>
        >>> # Customize a preset
        >>> config = ConfigPresets.ACCURATE
        >>> config.max_suggestions = 10  # Safe - this is a copy
        >>> checker = SpellChecker(config=config)
        >>>
        >>> # Compare presets
        >>> print(f"FAST edit distance: {ConfigPresets.FAST.max_edit_distance}")
        >>> print(f"ACCURATE edit distance: {ConfigPresets.ACCURATE.max_edit_distance}")
    """

    @staticmethod
    def default() -> "SpellCheckerConfig":
        """
        Default configuration with all features enabled.

        Balanced between accuracy and performance.
        """
        from myspellchecker.core.config import SpellCheckerConfig

        return SpellCheckerConfig()

    @staticmethod
    def fast() -> "SpellCheckerConfig":
        """
        Fast configuration optimized for speed.

        Delegates to ``get_fast_profile()`` for a single source of truth.
        Disables expensive features like context checking and NER.
        Uses rule-based POS, smaller caches, and fewer semantic checks.
        Suitable for real-time applications with high throughput needs.
        """
        from myspellchecker.core.config.profiles import get_fast_profile

        return get_fast_profile()

    @staticmethod
    def accurate() -> "SpellCheckerConfig":
        """
        Accurate configuration optimized for quality.

        Delegates to ``get_accurate_profile()`` for a single source of truth.
        Enables all features with higher edit distance, lower thresholds,
        larger beams, and more semantic checks for catching more errors.
        """
        from myspellchecker.core.config.profiles import get_accurate_profile

        return get_accurate_profile()

    @staticmethod
    def minimal() -> "SpellCheckerConfig":
        """
        Minimal configuration with only basic syllable validation.

        Suitable for environments with limited resources or
        when only basic checking is needed.
        """
        from myspellchecker.core.config import SpellCheckerConfig, ValidationConfig

        return SpellCheckerConfig(
            use_phonetic=False,
            use_context_checker=False,
            use_ner=False,
            use_rule_based_validation=False,
            max_edit_distance=1,
            max_suggestions=3,
            validation=ValidationConfig(
                use_broken_compound_detection=False,
                use_homophone_detection=False,
                use_confusable_semantic=False,
            ),
        )

    @staticmethod
    def strict() -> "SpellCheckerConfig":
        """
        Strict configuration with conservative thresholds.

        Catches more potential errors, may have more false positives.
        Suitable for formal documents where accuracy is critical.
        """
        from myspellchecker.core.config import NgramContextConfig, SpellCheckerConfig

        return SpellCheckerConfig(
            max_edit_distance=2,
            use_phonetic=True,
            use_context_checker=True,
            use_ner=True,
            use_rule_based_validation=True,
            max_suggestions=10,
            ngram_context=NgramContextConfig(
                bigram_threshold=0.001,
                trigram_threshold=0.0001,
            ),
        )
