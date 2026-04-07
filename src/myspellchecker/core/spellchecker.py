"""
SpellChecker Core Module

This module implements the main SpellChecker class that orchestrates
multi-layered spell checking for Myanmar (Burmese) text.

Architecture:
    - Layer 1: Syllable validation (fast, catches most structural errors)
    - Layer 2: Word validation (comprehensive, checks combinations)
    - Layer 3: Context validation (N-gram probability checking)

Integrated Features:
    - **Particle Typo Detection**: Catches common particle errors (ကို→ကို့, etc.)
      using PARTICLE_TYPO_PATTERNS with 0.90-0.95 confidence.
    - **Medial Confusion Detection**: Context-aware correction of ျ vs ြ medial
      confusion using MEDIAL_CONFUSION_PATTERNS.
    - **Morphology OOV Recovery**: Extracts roots from unknown words via suffix
      stripping to improve suggestion quality for out-of-vocabulary words.
    - **POS-Based Validation**: Uses ViterbiTagger output to detect invalid
      POS sequences (e.g., V-V, P-P patterns).
    - **Question Word Detection**: Identifies sentence types and validates
      question particle usage.
    - **Unified Suggestion Ranking**: Prioritizes suggestions by source
      (particle_typo > semantic > symspell) with deduplication.
    - **Proactive Semantic Scanning**: AI-powered error detection using
      language models (disabled by default, enable via config.semantic).
    - **Joint Segmentation+Tagging**: Unified Viterbi decoder for simultaneous
      word segmentation and POS tagging (disabled by default, enable via config.joint).
    - **Tone Disambiguation**: Context-aware tone mark correction for commonly
      confused Myanmar words.

The class uses dependency injection to accept custom Segmenter and
DictionaryProvider implementations.

Configuration:
    Most features can be enabled/disabled via SpellCheckerConfig:
    - config.semantic.use_proactive_scanning (default: False)
    - config.semantic.proactive_confidence_threshold (default: 0.5)
    - config.joint.enabled (default: False)
    - config.use_context_checker (default: True)
    - config.use_ner (default: True)

Import Strategy:
    Some imports are deferred (lazy) to:
    1. Avoid circular import issues (builder.py <-> spellchecker.py)
    2. Defer heavy dependencies (transformers, stemmer) until needed
    TYPE_CHECKING guards provide type hints without runtime imports.
"""

from __future__ import annotations

import asyncio
import threading
import time
import warnings
from typing import TYPE_CHECKING, Any, cast

# Type hints for lazy-loaded modules
# These modules are imported lazily to avoid circular imports and defer
# heavy dependencies. TYPE_CHECKING enables IDE/mypy support.
if TYPE_CHECKING:
    from myspellchecker.algorithms.joint_segment_tagger import JointSegmentTagger
    from myspellchecker.algorithms.viterbi import ViterbiTagger

from myspellchecker.algorithms import NgramContextChecker
from myspellchecker.algorithms.semantic_checker import SemanticChecker
from myspellchecker.algorithms.symspell import SymSpell
from myspellchecker.core.check_options import CheckOptions
from myspellchecker.core.component_factory import ComponentFactory, ComponentFactoryProtocol
from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.constants import ValidationLevel
from myspellchecker.core.constants.detector_thresholds import (
    DEFAULT_COMPOUND_THRESHOLDS,
    DEFAULT_PARTICLE_THRESHOLDS,
)
from myspellchecker.core.context_validator import ContextValidator
from myspellchecker.core.correction_utils import (
    generate_corrected_text,
)
from myspellchecker.core.detection_rules import DetectionRules, RerankRulesData
from myspellchecker.core.detectors.post_normalization import PostNormalizationDetectorsMixin
from myspellchecker.core.detectors.pre_normalization import PreNormalizationDetectorsMixin
from myspellchecker.core.detectors.sentence_detectors import SentenceDetectorsMixin
from myspellchecker.core.error_suppression import ErrorSuppressionMixin
from myspellchecker.core.exceptions import (
    MissingDatabaseError,
    ProcessingError,
    ValidationError,
)
from myspellchecker.core.response import (
    Error,
    GrammarError,
    Response,
)
from myspellchecker.core.response_builder import build_response_metadata
from myspellchecker.core.suggestion_pipeline import SuggestionPipelineMixin
from myspellchecker.core.syllable_rules import SyllableRuleValidator
from myspellchecker.core.validators import (
    SyllableValidator,
    WordValidator,
)
from myspellchecker.providers import DictionaryProvider, MemoryProvider, SQLiteProvider
from myspellchecker.segmenters import DefaultSegmenter, Segmenter
from myspellchecker.text.ner import NameHeuristic
from myspellchecker.text.normalize import normalize, normalize_for_lookup
from myspellchecker.text.phonetic import PhoneticHasher
from myspellchecker.utils.logging_utils import get_logger

__all__ = ["SpellChecker"]


class SpellChecker(
    PreNormalizationDetectorsMixin,
    PostNormalizationDetectorsMixin,
    SentenceDetectorsMixin,
    SuggestionPipelineMixin,
    ErrorSuppressionMixin,
):
    """
    Core spell checker for Myanmar (Burmese) text.

    Implements a "Syllable-First" architecture with multi-layered validation:

    - **Layer 1 (Syllable)**: Fast validation using `is_valid_syllable()`
      - Catches invalid syllables (typos, diacritic errors)
      - Default checking mode (level=ValidationLevel.SYLLABLE)
      - Performance: <10ms per sentence

    - **Layer 2 (Word)**: Validates multi-syllable word combinations
      - Uses `is_valid_word()` to check if syllable combinations form real words
      - Only checks multi-syllable words
      - Enabled with level=ValidationLevel.WORD

    - **Layer 3 (Context)**: Statistical N-gram validation
      - Uses `get_bigram_probability()` to detect unlikely sequences
      - Flags low-probability word pairs
      - Particle typo and medial confusion detection
      - POS sequence validation (V-V, P-P pattern detection)
      - Optional proactive semantic scanning (AI-powered)
      - Enabled with level=ValidationLevel.WORD

    Additional Features:
        - **Morphology Recovery**: OOV root extraction for better suggestions
        - **Unified Ranking**: Source-weighted suggestion ranking
        - **Tone Disambiguation**: Context-aware tone mark correction
        - **Joint Tagger**: Optional unified segmentation+POS tagging

    Example:
        >>> from myspellchecker import SpellChecker
        >>> checker = SpellChecker()
        >>> result = checker.check("မြန်မာနိုင်ငံ")
        >>> print(result.has_errors)
        False

        >>> # Async usage
        >>> import asyncio
        >>> async def main():
        ...     result = await checker.check_async("မြန်မာ")
        ...     print(result.corrected_text)

        >>> # Joint segmentation and POS tagging
        >>> words, tags = checker.segment_and_tag("သူစားတယ်")
        >>> print(list(zip(words, tags)))
    """

    # Class-level defaults for confidence thresholds (overridden per-instance
    # from ValidationConfig in __init__).
    _CONFIDENCE_THRESHOLDS: dict[str, float] = {"confusable_error": 0.75}
    _SECONDARY_CONFIDENCE_THRESHOLDS: dict[str, float] = {"semantic_error": 0.85}

    # Type annotations for instance attributes
    config: SpellCheckerConfig
    provider: DictionaryProvider
    segmenter: Segmenter
    viterbi_tagger: "ViterbiTagger" | None
    joint_segment_tagger: "JointSegmentTagger" | None
    syntactic_rule_checker: Any | None
    syllable_validator: SyllableValidator
    word_validator: WordValidator
    context_validator: ContextValidator

    def __init__(
        self,
        config: SpellCheckerConfig | None = None,
        segmenter: Segmenter | None = None,
        provider: DictionaryProvider | None = None,
        syllable_validator: SyllableValidator | None = None,
        word_validator: WordValidator | None = None,
        context_validator: ContextValidator | None = None,
        factory: ComponentFactoryProtocol | None = None,
    ):
        """
        Initialize SpellChecker with configuration and optional custom components.

        Parameter Precedence:
            Parameters are resolved in the following order (highest to lowest):

            1. **Explicit constructor arguments** (e.g., `provider=my_provider`)
            2. **Config object attributes** (e.g., `config.provider`)
            3. **Environment variables** (e.g., `MYSPELL_DATABASE_PATH`)
            4. **Default values** (e.g., auto-detect database location)

            Example precedence:
                >>> # provider arg takes precedence over config.provider
                >>> checker = SpellChecker(config=config, provider=custom_provider)
                >>> assert checker.provider is custom_provider  # Not config.provider

        Initialization Flow:
            1. Resolve config (create default if None)
            2. Resolve provider: arg > config.provider > auto-detect SQLiteProvider
            3. Resolve segmenter: arg > config.segmenter > DefaultSegmenter
            4. Create all algorithm components via ComponentFactory
            5. Resolve validators: arg > factory-created validators

        For simpler initialization, consider using the factory methods:
            - SpellChecker.create_fast() - Optimized for speed
            - SpellChecker.create_accurate() - Optimized for accuracy
            - SpellChecker.create_minimal() - Basic syllable validation only

        Or use SpellCheckerBuilder for fluent configuration:
            >>> from myspellchecker.core.builder import SpellCheckerBuilder
            >>> checker = SpellCheckerBuilder().with_phonetic(True).build()

        Args:
            config: SpellCheckerConfig with all settings. If None, creates a
                default config. Settings can also be configured via environment
                variables (see config/loader.py for supported variables).
            segmenter: Custom Segmenter instance for text tokenization.
                If None, uses config.segmenter or creates DefaultSegmenter.
            provider: Custom DictionaryProvider for dictionary data.
                If None, uses config.provider or auto-detects SQLiteProvider.
            syllable_validator: Custom SyllableValidator (advanced use).
                Overrides factory-created validator for testing/customization.
            word_validator: Custom WordValidator (advanced use).
                Overrides factory-created validator for testing/customization.
            context_validator: Custom ContextValidator (advanced use).
                Overrides factory-created validator for testing/customization.
            factory: Custom ComponentFactory for dependency injection.
                Enables unit testing with mock factory or custom component creation.
                If None, uses default ComponentFactory.

        Raises:
            TypeError: If config is provided but not a SpellCheckerConfig instance.
            MissingDatabaseError: If no database found and fallback_to_empty_provider
                is False (default).

        Example:
            >>> # Basic usage with defaults
            >>> checker = SpellChecker()

            >>> # Custom configuration
            >>> config = SpellCheckerConfig(max_suggestions=10, use_phonetic=True)
            >>> checker = SpellChecker(config=config)

            >>> # Custom provider
            >>> provider = SQLiteProvider("custom.db")
            >>> checker = SpellChecker(provider=provider)

            >>> # Full customization
            >>> checker = SpellChecker(
            ...     config=config,
            ...     provider=provider,
            ...     segmenter=custom_segmenter,
            ... )
        """
        self.logger = get_logger(__name__)

        # 1. Resolve Config
        if config is None:
            config = SpellCheckerConfig(
                segmenter=segmenter,
                provider=provider,
            )
        elif not isinstance(config, SpellCheckerConfig):
            raise TypeError(
                f"config must be an instance of SpellCheckerConfig, got {type(config).__name__}"
            )

        self.config = config

        # 2. Initialize subsystems
        self._init_provider(provider, segmenter, factory)
        self._init_validators(syllable_validator, word_validator, context_validator)
        self._init_context()
        self._init_detectors()

        self.logger.info(f"Initialized SpellChecker with provider: {type(self.provider).__name__}")

    def _init_provider(
        self,
        provider: DictionaryProvider | None,
        segmenter: Segmenter | None,
        factory: ComponentFactoryProtocol | None,
    ) -> None:
        """Resolve provider, segmenter, and create algorithm components via factory."""
        # Override class-level confidence thresholds from config
        self._CONFIDENCE_THRESHOLDS = self.config.validation.output_confidence_thresholds
        self._SECONDARY_CONFIDENCE_THRESHOLDS = (
            self.config.validation.secondary_confidence_thresholds
        )
        # Override suggestion pipeline constants from config
        _sc = self.config.symspell
        self._MORPHEME_PROMOTE_MAX_ERR_LEN = _sc.morpheme_promote_max_err_len
        self._MORPHEME_COMPOUND_CTX = _sc.morpheme_compound_ctx
        self._MORPHEME_COMPOUND_MIN_LEN = _sc.morpheme_compound_min_len
        self._MORPHEME_COMPOUND_MAX_SUGG = _sc.morpheme_compound_max_sugg
        self._ASAT_CONTEXT_WINDOW = _sc.asat_context_window
        self._DISTANCE_RERANK_MIN_GAP = _sc.distance_rerank_min_gap
        self._DISTANCE_RERANK_MAX_BASE_DISTANCE = _sc.distance_rerank_max_base_distance
        self._DISTANCE_RERANK_MAX_PROMOTE_DISTANCE = _sc.distance_rerank_max_promote_distance
        self._SPAN_LENGTH_MIN_ERROR_LEN = _sc.span_length_min_error_len
        self._SPAN_LENGTH_PENALTY_WEIGHT = _sc.span_length_penalty_weight
        self._enable_targeted_rerank_hints = self.config.ranker.enable_targeted_rerank_hints
        self._enable_targeted_candidate_injections = (
            self.config.ranker.enable_targeted_candidate_injections
        )

        # Resolve Core Components
        self.provider = provider or self.config.provider or self._get_default_provider()

        self.segmenter = (
            segmenter
            or self.config.segmenter
            or DefaultSegmenter(
                word_engine=self.config.word_engine,
                allow_extended_myanmar=self.config.validation.allow_extended_myanmar,
            )
        )

        # Inject word repository into segmenter for syllable-reassembly fallback.
        # The segmenter is constructed before the provider in the DI graph, so we
        # wire the provider in after both are resolved via a setter.
        if isinstance(self.segmenter, DefaultSegmenter) and hasattr(
            self.segmenter, "set_word_repository"
        ):
            self.segmenter.set_word_repository(self.provider)

        # Use ComponentFactory for all other components (supports DI)
        self._factory = factory or ComponentFactory(self.config)
        self._components = self._factory.create_all(self.provider, self.segmenter)

        # Store algorithm components
        self.viterbi_tagger = self._components["viterbi_tagger"]
        self.joint_segment_tagger = self._components["joint_segment_tagger"]
        self.syntactic_rule_checker = self._components["syntactic_rule_checker"]
        self._semantic_checker = self._components["semantic_checker"]
        self._context_checker = self._components["context_checker"]
        self._name_heuristic = self._components["name_heuristic"]
        self._ner_model = self._components.get("ner_model")
        self._phonetic_hasher = self._components["phonetic_hasher"]
        self._pos_disambiguator = self._components.get("pos_disambiguator")
        self._cached_sources = self._components.get("cached_sources", {})

        # Initialize neural reranker (graceful degradation)
        self._neural_reranker = self._create_neural_reranker()
        self._neural_reranker_gap_threshold = self.config.neural_reranker.confidence_gap_threshold

    def _init_validators(
        self,
        syllable_validator: SyllableValidator | None,
        word_validator: WordValidator | None,
        context_validator: ContextValidator | None,
    ) -> None:
        """Resolve validators, allowing injection for testing."""
        self.syllable_validator = syllable_validator or self._components["syllable_validator"]
        self.word_validator = word_validator or self._components["word_validator"]
        self.context_validator = context_validator or self._components["context_validator"]
        # Per-check mutable state stored in threading.local() to prevent
        # race conditions when check_batch_async runs concurrent threads.
        self._thread_local = threading.local()
        self._rerank_telemetry_lock = threading.Lock()
        self._last_rerank_rule_telemetry: dict[str, dict[str, int]] = {}

    def _init_context(self) -> None:
        """Load detection rules from YAML (shadows class-level hardcoded dicts)."""
        self._detection_rules = DetectionRules()
        self._MEDIAL_CONFUSION_UNCONDITIONAL = self._detection_rules.medial_confusion_unconditional
        self._MEDIAL_CONFUSION_CONTEXTUAL = self._detection_rules.medial_confusion_contextual
        self._AUKMYIT_CONTEXT = self._detection_rules.aukmyit_context
        self._EXTRA_AUKMYIT_CONTEXT = self._detection_rules.extra_aukmyit_context
        self._VOWEL_REORDER_ERRORS = self._detection_rules.vowel_reorder_errors
        self._COLLOQUIAL_CONTRACTIONS = self._detection_rules.colloquial_contractions
        self._STACKING_COMPLETIONS = self._detection_rules.stacking_completions
        self._HA_HTOE_COMPOUNDS = self._detection_rules.ha_htoe_compounds
        self._ASPIRATED_COMPOUNDS = self._detection_rules.aspirated_compounds
        self._CONSONANT_CONFUSION_COMPOUNDS = self._detection_rules.consonant_confusion_compounds
        self._PARTICLE_CONFUSION = self._detection_rules.particle_confusion
        self._SEQUENTIAL_PARTICLE_LEFT_CONTEXT = (
            self._detection_rules.sequential_particle_left_context
        )
        self._HA_HTOE_PARTICLES = self._detection_rules.ha_htoe_particles
        self._HA_HTOE_EXCLUSIONS = self._detection_rules.ha_htoe_exclusions
        self._HONORIFIC_PREFIXES = self._detection_rules.honorific_prefixes
        self._DANGLING_PARTICLES = self._detection_rules.dangling_particles
        self._KEEP_ATTACHED_SUFFIXES = self._detection_rules.suppression_suffixes
        self._MISSING_ASAT_PARTICLES = self._detection_rules.missing_asat_particles
        self._MISSING_VISARGA_SUFFIXES = self._detection_rules.missing_visarga_suffixes
        self._SYLLABLE_FINALS = self._detection_rules.syllable_finals
        self._VISARGA_EXCLUDE_PRONOUNS = self._detection_rules.visarga_exclude_pronouns
        self._SUBJECT_PRONOUNS = self._detection_rules.subject_pronouns
        self._PARTICLE_MISUSE_RULES = self._detection_rules.particle_misuse_rules
        self._LOCATIVE_EXEMPT_PREFIXES = self._detection_rules.locative_exempt_prefixes
        self._HOMOPHONE_LEFT_CONTEXT = self._detection_rules.homophone_left_context
        self._HOMOPHONE_LEFT_SUFFIXES = self._detection_rules.homophone_left_suffixes
        self._COLLOCATION_RULES = self._detection_rules.collocation_rules

    def _init_detectors(self) -> None:
        """Set up detector thresholds and rerank rules."""
        # Use frozen dataclass singletons for detector thresholds.
        # These are internal engineering parameters, not user-facing config.
        self._compound_thresholds = DEFAULT_COMPOUND_THRESHOLDS
        self._particle_thresholds = DEFAULT_PARTICLE_THRESHOLDS

        # Load rerank rules from YAML
        self._rerank_data = RerankRulesData()

    def cache_stats(self) -> dict[str, Any]:
        """
        Get unified cache statistics from all components.

        Returns:
            Dictionary with cache stats per component (dictionary, frequency,
            bigram, trigram, plus algorithm-specific caches when available).
        """
        stats: dict[str, Any] = {}

        # Provider-level cached sources
        for name, source in self._cached_sources.items():
            if hasattr(source, "cache_info"):
                stats[name] = source.cache_info()

        # Algorithm-level caches
        if self.joint_segment_tagger and hasattr(self.joint_segment_tagger, "cache_stats"):
            stats["joint_tagger"] = self.joint_segment_tagger.cache_stats()

        if self._semantic_checker and hasattr(self._semantic_checker, "cache_stats"):
            stats["semantic"] = self._semantic_checker.cache_stats()

        if self.viterbi_tagger and hasattr(self.viterbi_tagger, "cache_stats"):
            stats["viterbi"] = self.viterbi_tagger.cache_stats()

        return stats

    @classmethod
    def create_default(cls) -> "SpellChecker":
        """
        Create SpellChecker with default settings.

        All features enabled with balanced performance/accuracy.

        Returns:
            Configured SpellChecker instance.

        Example:
            >>> checker = SpellChecker.create_default()
        """
        from myspellchecker.core.builder import ConfigPresets

        return cls(config=ConfigPresets.DEFAULT)

    @classmethod
    def create_fast(cls) -> "SpellChecker":
        """
        Create SpellChecker optimized for speed.

        Disables expensive features (context checking, NER, phonetic).
        Suitable for real-time applications with high throughput needs.

        Returns:
            Speed-optimized SpellChecker instance.

        Example:
            >>> checker = SpellChecker.create_fast()
        """
        from myspellchecker.core.builder import ConfigPresets

        return cls(config=ConfigPresets.FAST)

    @classmethod
    def create_accurate(cls) -> "SpellChecker":
        """
        Create SpellChecker optimized for accuracy.

        Enables all features with higher edit distance and
        lower thresholds for catching more errors.

        Returns:
            Accuracy-optimized SpellChecker instance.

        Example:
            >>> checker = SpellChecker.create_accurate()
        """
        from myspellchecker.core.builder import ConfigPresets

        return cls(config=ConfigPresets.ACCURATE)

    @classmethod
    def create_minimal(cls) -> "SpellChecker":
        """
        Create SpellChecker with minimal features.

        Only basic syllable validation, no context checking or NER.
        Suitable for limited resource environments.

        Returns:
            Minimal SpellChecker instance.

        Example:
            >>> checker = SpellChecker.create_minimal()
        """
        from myspellchecker.core.builder import ConfigPresets

        return cls(config=ConfigPresets.MINIMAL)

    def _get_default_provider(self) -> DictionaryProvider:
        """Get the default dictionary provider (SQLite or Memory fallback)."""
        try:
            # Create POS tagger from config
            pos_tagger = None
            if self.config.pos_tagger:
                from myspellchecker.algorithms.pos_tagger_factory import POSTaggerFactory

                try:
                    # Use config unpacking instead of explicit params
                    pos_tagger_params = self.config.pos_tagger.model_dump(
                        exclude_none=True, exclude_unset=True
                    )
                    pos_tagger = POSTaggerFactory.create(**pos_tagger_params)
                except (RuntimeError, ValueError, ImportError, OSError, TypeError) as e:
                    self.logger.warning(f"Failed to initialize POS tagger: {e}")
                    self.logger.info("Falling back to default rule-based tagger")
                    # Create fallback rule-based tagger
                    from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger

                    pos_tagger = RuleBasedPOSTagger(
                        use_morphology_fallback=True,
                        cache_size=self.config.pos_tagger.cache_size,
                        unknown_tag=self.config.pos_tagger.unknown_tag,
                    )

            return SQLiteProvider(
                database_path=self.config.provider_config.database_path,
                cache_size=self.config.provider_config.cache_size,
                pos_tagger=pos_tagger,
                pool_min_size=self.config.provider_config.pool_min_size,
                pool_max_size=self.config.provider_config.pool_max_size,
                pool_timeout=self.config.provider_config.pool_timeout,
                pool_max_connection_age=self.config.provider_config.pool_max_connection_age,
                curated_min_frequency=self.config.provider_config.curated_min_frequency,
            )
        except MissingDatabaseError as e:
            if self.config.fallback_to_empty_provider:
                msg = (
                    "Default database not found. Using empty MemoryProvider. "
                    "WARNING: Spell checking will not work without dictionary data! "
                    "Build the database using 'myspellchecker build --sample' or "
                    "'myspellchecker build --input <corpus.txt>'."
                )
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                self.logger.warning(msg)
                return MemoryProvider()
            else:
                raise MissingDatabaseError(
                    message="No database available for spell checking.",
                    suggestion=(
                        "No bundled database is included. Build one first:\n"
                        "  myspellchecker build --sample\n"
                        "Then pass the path:\n"
                        "  SpellChecker(provider=SQLiteProvider("
                        "database_path='mySpellChecker-default.db'))\n"
                        "Or set fallback_to_empty_provider=True in SpellCheckerConfig."
                    ),
                ) from e

    def check(
        self,
        text: str,
        level: ValidationLevel = ValidationLevel.SYLLABLE,
        use_semantic: bool | None = None,
        options: CheckOptions | None = None,
    ) -> Response:
        """
        Check Myanmar text for spelling errors.

        This is the main entry point for spell checking. The method
        applies validation layers based on the specified level.

        Validation Levels:
            - **SYLLABLE** (default): Fast validation (~10ms). Checks individual
              syllables against dictionary and structural rules. Catches ~90%
              of common typos with minimal overhead.

            - **WORD**: Comprehensive validation (~50-200ms). Includes syllable
              validation plus word-level checking, context analysis (N-gram),
              and optional semantic refinement.

        Args:
            text: Myanmar text to check (can include punctuation, spaces).
                Unicode-normalized internally; Zawgyi encoding is detected
                and optionally converted based on config settings.
            level: Validation level (ValidationLevel.SYLLABLE or ValidationLevel.WORD).
                Use SYLLABLE for real-time applications, WORD for thorough checking.
            use_semantic: Override semantic checking. If None, uses config setting.
                If True, enables semantic refinement (requires model).
                If False, disables semantic refinement.

        Returns:
            Response object containing:
                - text: Original input text
                - corrected_text: Text with top suggestions applied
                - has_errors: Boolean indicating if any errors were found
                - errors: List of Error objects with positions and suggestions
                - metadata: Processing statistics (time, layer counts, etc.)

        Raises:
            TypeError: If text is not a string.
            ValueError: If level is not a valid ValidationLevel enum.

        Example:
            >>> checker = SpellChecker()
            >>>
            >>> # Basic syllable-level check (fast)
            >>> result = checker.check("မြန်မာနိုင်ငံ")
            >>> print(result.has_errors)
            False
            >>>
            >>> # Word-level check with context validation
            >>> result = checker.check("မြန်မာနိုင်ငံ", level=ValidationLevel.WORD)
            >>> print(result.metadata["layers_applied"])
            ['syllable', 'word', 'context']
            >>>
            >>> # Check with explicit semantic override
            >>> result = checker.check(text, use_semantic=True)
            >>>
            >>> # Access error details
            >>> for error in result.errors:
            ...     print(f"{error.text} at {error.position}: {error.suggestions[:3]}")
        """
        # Merge per-request options into effective values.
        if options is not None:
            if options.use_semantic is not None:
                use_semantic = options.use_semantic

        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")

        if len(text) > self.config.max_text_length:
            raise ProcessingError(
                f"text length ({len(text)}) exceeds max_text_length "
                f"({self.config.max_text_length}). Use StreamingChecker for large texts."
            )

        start_time = time.perf_counter()
        level = self._validate_level(level)

        # Phase 1: Prepare text -- Zawgyi detection, normalization,
        # pre-normalization error detection, zero-width detection.
        prepared = self._prepare_text(text, level)

        # Early return for empty text after normalization
        if prepared["normalized_text"] is None:
            pre_errors = prepared["pre_norm_errors"]
            return Response(
                text=text,
                corrected_text=text,
                has_errors=bool(pre_errors),
                level=level.value,
                errors=pre_errors,
                metadata={
                    "processing_time": time.perf_counter() - start_time,
                    "layers_applied": [],
                },
            )

        normalized_text = prepared["normalized_text"]

        # If context_checking is explicitly disabled via options, skip context
        # validation by forcing syllable-level for the validation pipeline.
        effective_level = level
        if options is not None and options.context_checking is False:
            effective_level = ValidationLevel.SYLLABLE

        # Phase 2: Run validation pipeline on normalized text.
        errors, layers_applied = self._run_validation(
            normalized_text, effective_level, use_semantic, prepared
        )

        # Apply per-request grammar filtering.
        if options is not None and options.grammar_checking is False:
            errors = [e for e in errors if not isinstance(e, GrammarError)]

        # Phase 3: Post-processing, dedup, and response building.
        response = self._finalize_response(
            text=text,
            normalized_text=normalized_text,
            errors=errors,
            layers_applied=layers_applied,
            start_time=start_time,
            zawgyi_warning=prepared["zawgyi_warning"],
            level=level,
        )

        # Apply per-request max_suggestions limit.
        if options is not None and options.max_suggestions is not None:
            max_s = options.max_suggestions
            for error in response.errors:
                error.suggestions = error.suggestions[:max_s]

        return response

    def _prepare_text(self, text: str, level: ValidationLevel) -> dict[str, Any]:
        """Zawgyi detection, normalization, length validation, zero-width detection.

        Runs all pre-normalization detectors that must examine raw text before
        normalization alters or removes evidence.

        Returns:
            Dictionary with keys:
                - normalized_text: str or None (None if text is empty after normalization)
                - zawgyi_config: Zawgyi detection config
                - zawgyi_warning: Zawgyi warning (or None)
                - pre_norm_errors: combined list of all pre-normalization errors
                - zawgyi_errors, zero_width_errors, broken_virama_errors,
                  medial_order_errors, duplicate_diacritic_errors,
                  leading_vowel_e_errors, vowel_reorder_errors,
                  vowel_medial_reorder_errors, incomplete_stacking_errors,
                  vowel_after_dotbelow_errors: individual error lists for merging
        """
        self._thread_local.strategy_debug_telemetry = {}
        self._thread_local.last_strategy_debug_telemetry = {}

        # Clear per-check session caches on the SymSpell instance to avoid
        # stale results leaking across sentences.  Session caches are
        # thread-local, so each thread clears only its own caches.
        _symspell = self.symspell
        if _symspell is not None and hasattr(_symspell, "clear_session_cache"):
            _symspell.clear_session_cache()

        self.logger.debug(f"Checking text (len={len(text)}) at level {level.value}")

        # Detect Zawgyi-encoded words BEFORE normalization (which may convert them)
        zawgyi_config, zawgyi_warning = self._detect_zawgyi(text)
        zawgyi_errors = self._detect_zawgyi_words(text)

        # Detect zero-width characters BEFORE normalization removes them
        zero_width_errors = self._detect_zero_width_chars(text)

        # Detect broken virama (U+1039 + vowel) BEFORE normalization converts
        # virama to asat (U+103A), destroying the evidence.
        broken_virama_errors = self._detect_broken_virama(text)

        # Detect medial ordering errors BEFORE normalization reorders them
        medial_order_errors = self._detect_medial_order_errors(text)

        # Detect duplicate diacritics BEFORE normalization may alter them
        duplicate_diacritic_errors = self._detect_duplicate_diacritics(text)

        # Detect leading vowel-e BEFORE normalization reorders it
        leading_vowel_e_errors = self._detect_leading_vowel_e(text)

        # Detect vowel reordering errors BEFORE normalization fixes them silently
        vowel_reorder_errors = self._detect_vowel_reorder_errors(text)

        # Detect vowel-before-medial reorder BEFORE normalization fixes it silently
        vowel_medial_reorder_errors = self._detect_vowel_medial_reorder(text)

        # Detect incomplete stacking BEFORE normalization reorders virama+vowel
        incomplete_stacking_errors = self._detect_incomplete_stacking(text)

        # Detect vowel after dot-below BEFORE normalization reorders vowels
        vowel_after_dotbelow_errors = self._detect_vowel_after_dotbelow(text)

        normalized_text = self._normalize_text(text, zawgyi_config)

        # Build position map from original to normalized text so that
        # pre-normalization error positions can be remapped before
        # generate_corrected_text (which runs on normalized text).
        from myspellchecker.core.correction_utils import build_orig_to_norm_map

        orig_to_norm_map = build_orig_to_norm_map(text) if normalized_text else []

        # Build combined pre-normalization error list
        pre_norm_errors = (
            zawgyi_errors
            + zero_width_errors
            + broken_virama_errors
            + medial_order_errors
            + duplicate_diacritic_errors
            + leading_vowel_e_errors
            + incomplete_stacking_errors
            + vowel_reorder_errors
            + vowel_medial_reorder_errors
            + vowel_after_dotbelow_errors
        )

        return {
            "normalized_text": normalized_text
            if (normalized_text and normalized_text.strip())
            else None,
            "orig_to_norm_map": orig_to_norm_map,
            "zawgyi_config": zawgyi_config,
            "zawgyi_warning": zawgyi_warning,
            "pre_norm_errors": pre_norm_errors,
            "zawgyi_errors": zawgyi_errors,
            "zero_width_errors": zero_width_errors,
            "broken_virama_errors": broken_virama_errors,
            "medial_order_errors": medial_order_errors,
            "duplicate_diacritic_errors": duplicate_diacritic_errors,
            "leading_vowel_e_errors": leading_vowel_e_errors,
            "vowel_reorder_errors": vowel_reorder_errors,
            "vowel_medial_reorder_errors": vowel_medial_reorder_errors,
            "incomplete_stacking_errors": incomplete_stacking_errors,
            "vowel_after_dotbelow_errors": vowel_after_dotbelow_errors,
        }

    def _run_validation(
        self,
        normalized_text: str,
        level: ValidationLevel,
        use_semantic: bool | None,
        prepared: dict[str, Any],
    ) -> tuple[list[Error], list[str]]:
        """Run validation pipeline and merge pre-normalization errors.

        Executes validation layers on normalized text then merges in the
        pre-normalization errors collected during _prepare_text().

        Returns:
            Tuple of (errors, layers_applied).
        """
        # Run validation layers on normalized text
        errors, layers_applied = self._run_validation_layers(normalized_text, level, use_semantic)

        # Remap pre-normalization error positions from original-text offsets
        # to normalized-text offsets so they align with generate_corrected_text.
        offset_map = prepared.get("orig_to_norm_map", [])
        if offset_map:
            from myspellchecker.core.correction_utils import remap_pre_norm_error

            for key in (
                "zawgyi_errors",
                "zero_width_errors",
                "broken_virama_errors",
                "medial_order_errors",
                "duplicate_diacritic_errors",
                "leading_vowel_e_errors",
                "incomplete_stacking_errors",
                "vowel_reorder_errors",
                "vowel_medial_reorder_errors",
                "vowel_after_dotbelow_errors",
            ):
                for err in prepared.get(key, []):
                    remap_pre_norm_error(err, offset_map)

        # Merge pre-normalization errors into post-normalization error list.
        # Each type uses a different merge strategy based on its semantics.
        self._merge_pre_norm_errors(
            errors, prepared["zawgyi_errors"], replace_mode="if_strictly_longer"
        )
        self._merge_pre_norm_errors(errors, prepared["zero_width_errors"], replace_mode="add_only")
        self._merge_pre_norm_errors(errors, prepared["broken_virama_errors"], replace_mode="always")
        self._merge_pre_norm_errors(errors, prepared["medial_order_errors"])
        self._merge_duplicate_diacritic_errors(errors, prepared["duplicate_diacritic_errors"])
        self._merge_pre_norm_errors(errors, prepared["leading_vowel_e_errors"])
        self._merge_pre_norm_errors(errors, prepared["incomplete_stacking_errors"])
        self._merge_pre_norm_errors(errors, prepared["vowel_reorder_errors"])
        self._merge_pre_norm_errors(errors, prepared["vowel_medial_reorder_errors"])
        self._merge_pre_norm_errors(errors, prepared["vowel_after_dotbelow_errors"])

        # Final span-overlap dedup for pre-normalization errors
        errors = self._dedup_pre_norm_overlaps(errors)

        return errors, layers_applied

    def _finalize_response(
        self,
        *,
        text: str,
        normalized_text: str,
        errors: list[Error],
        layers_applied: list[str],
        start_time: float,
        zawgyi_warning: Any,
        level: ValidationLevel,
    ) -> Response:
        """Post-processing, dedup, and response building.

        Applies suggestion expansion, reranking, suppression, confidence
        gating, and assembles the final Response object.
        """
        # Post-processing: suggestion expansion, reranking, suppression,
        # and confidence gating.
        errors, rerank_telemetry = self._apply_error_post_processing(errors, normalized_text)

        # Build and return the final Response object.
        return self._build_check_response(
            text=text,
            normalized_text=normalized_text,
            errors=errors,
            layers_applied=layers_applied,
            start_time=start_time,
            zawgyi_warning=zawgyi_warning,
            rerank_telemetry=rerank_telemetry,
            level=level,
        )

    def _apply_error_post_processing(
        self,
        errors: list[Error],
        normalized_text: str,
    ) -> tuple[list[Error], dict[str, dict[str, int]]]:
        """Apply suggestion expansion, reranking, suppression, and confidence gating.

        This is the post-processing phase after validation layers and
        pre-normalization error merging.  It mutates the *errors* list
        in-place (suggestion expansion, reranking, suppression) and then
        returns a filtered copy after confidence gating.

        Returns:
            A tuple of (filtered_errors, rerank_telemetry).
        """
        # Extend morpheme-level suggestions with compound suffixes from sentence context
        self._extend_suggestions_with_sentence_context(errors, normalized_text)

        # Extract valid-word morphemes from compound suggestions so that
        # morpheme-level gold corrections can be found (appended after existing).
        self._append_morpheme_subwords(errors)
        with self._rerank_telemetry_lock:
            self._last_rerank_rule_telemetry = {}
            self._rerank_detector_suggestions_by_distance(errors, sentence_text=normalized_text)
            rerank_telemetry = dict(self._last_rerank_rule_telemetry)

        # Suggestion expansion/reranking can re-introduce low-value
        # semantic/confusable variants. Re-apply only those filters on
        # final candidate lists to avoid widening suppression scope.
        self._suppress_low_value_confusable_errors(errors, text=normalized_text)
        self._suppress_low_value_semantic_errors(errors, text=normalized_text)
        self._suppress_low_value_word_errors(errors, text=normalized_text)

        # Confidence-gated output filter: suppress error types where the
        # system has low precision unless confidence exceeds a per-type
        # threshold.  This prevents noisy detections (grammar, complex
        # contextual) from confusing users while keeping reliable types
        # (spelling, syllable, word) at full sensitivity.
        #
        # confusable_error: FPs cluster at confidence 0.72, TPs at 0.88+.
        # Threshold 0.75 cleanly separates them (empirically calibrated).
        errors = [
            e
            for e in errors
            if getattr(e, "confidence", 1.0) >= self._CONFIDENCE_THRESHOLDS.get(e.error_type, 0.0)
        ]

        # Secondary gating: when the sentence already has multiple errors,
        # suppress low-confidence detections of FP-prone types when a
        # higher-confidence error of a DIFFERENT type exists.  This
        # prevents cascade FPs (semantic model detecting valid words near
        # a real error) while allowing multiple genuine same-type
        # detections to coexist.
        if len(errors) > 1 and self._SECONDARY_CONFIDENCE_THRESHOLDS:
            # Build per-type max confidence for "different type" comparison.
            type_max_conf: dict[str, float] = {}
            for e in errors:
                c = getattr(e, "confidence", 0.0)
                if c > type_max_conf.get(e.error_type, 0.0):
                    type_max_conf[e.error_type] = c
            errors = [
                e
                for e in errors
                if not (
                    e.error_type in self._SECONDARY_CONFIDENCE_THRESHOLDS
                    and getattr(e, "confidence", 1.0)
                    < self._SECONDARY_CONFIDENCE_THRESHOLDS[e.error_type]
                    and any(
                        c > getattr(e, "confidence", 1.0)
                        for t, c in type_max_conf.items()
                        if t != e.error_type
                    )
                )
            ]

        # MLM post-filter: suppress invalid_word/dangling_word FPs when the
        # semantic model confirms the word is contextually plausible.
        if self._semantic_checker is not None:
            self._suppress_invalid_word_via_mlm(errors, normalized_text)

        return errors, rerank_telemetry

    def _build_check_response(
        self,
        *,
        text: str,
        normalized_text: str,
        errors: list[Error],
        layers_applied: list[str],
        start_time: float,
        zawgyi_warning: Any,
        rerank_telemetry: dict[str, dict[str, int]],
        level: ValidationLevel,
    ) -> Response:
        """Sort errors, generate corrected text, assemble metadata, and return Response."""
        errors.sort(key=lambda e: e.position)
        corrected_text = generate_corrected_text(normalized_text, errors)
        processing_time = time.perf_counter() - start_time

        metadata = build_response_metadata(errors, layers_applied, processing_time, zawgyi_warning)
        if rerank_telemetry:
            metadata["rerank_rule_telemetry"] = {
                rule_id: {
                    "fires": int(rule_stats.get("fires", 0)),
                    "top1_changes": int(rule_stats.get("top1_changes", 0)),
                }
                for rule_id, rule_stats in rerank_telemetry.items()
            }
        _telemetry = getattr(self._thread_local, "last_strategy_debug_telemetry", {})
        if _telemetry:
            metadata["strategy_debug_telemetry"] = _telemetry

        return Response(
            text=text,
            corrected_text=corrected_text,
            has_errors=len(errors) > 0,
            level=level.value,
            errors=errors,
            metadata=metadata,
        )

    def _validate_level(self, level: ValidationLevel) -> ValidationLevel:
        """Validate and normalize the validation level parameter."""
        if isinstance(level, ValidationLevel):
            return level
        # For backward compatibility, allow string input
        try:
            return ValidationLevel(level)
        except ValueError as e:
            raise ValidationError(
                f"Invalid level: '{level}'. "
                f"Must be '{ValidationLevel.SYLLABLE.value}' or '{ValidationLevel.WORD.value}'."
            ) from e

    def _normalize_text(self, text: str, zawgyi_config: Any) -> str:
        """Apply comprehensive normalization to text."""
        return normalize_for_lookup(
            text,
            convert_zawgyi=self.config.validation.use_zawgyi_conversion,
            config=zawgyi_config,
        )

    def _run_validation_layers(
        self,
        normalized_text: str,
        level: ValidationLevel,
        use_semantic: bool | None,
    ) -> tuple[list[Error], list[str]]:
        """Run all validation layers and return errors and applied layers.

        Execution order:
        1. Syllable validation + cascade/Pali suppression
        2. Post-normalization detectors (see ``detection_registry.py`` for sequence)
        3. Bare-consonant suppression
        4. Word + context validation (if level=WORD)
        5. Suggestion reconstruction + dedup pipeline + NER filtering
        """
        from myspellchecker.core.detection_registry import POST_NORM_DETECTOR_SEQUENCE
        from myspellchecker.core.detectors.context import DetectorContext
        from myspellchecker.core.detectors.tokenized_text import TokenizedText

        errors: list[Error] = []
        layers_applied: list[str] = []

        # Pre-compute tokenization once for all detectors (replaces ~25
        # duplicated text.split() + text.find(token, cursor) loops).
        # Stored in thread-local to prevent races in check_batch_async.
        tokenized = TokenizedText.from_text(normalized_text)
        self._thread_local.current_tokenized = tokenized

        # Build explicit dependency context for detectors.
        self._thread_local.detector_ctx = DetectorContext(
            provider=self.provider,
            segmenter=self.segmenter,
            symspell=getattr(self, "symspell", None),
            semantic_checker=getattr(self, "_semantic_checker", None),
            config=self.config,
            tokenized=tokenized,
        )

        # Layer 1: Syllable validation + cascade suppression
        self._validate_syllables(normalized_text, errors, layers_applied)
        self._suppress_cascade_syllable_errors(errors, normalized_text)
        self._suppress_pali_stacking_errors(errors, normalized_text)

        # Run all post-normalization detectors in registry order
        for entry in POST_NORM_DETECTOR_SEQUENCE:
            getattr(self, entry.method_name)(normalized_text, errors)

        self._suppress_bare_consonant_near_text_errors(errors)

        # Layer 2 & 3: Word and Context Validation
        if level == ValidationLevel.WORD:
            words = self.segmenter.segment_words(normalized_text)
            self._validate_words(normalized_text, errors, layers_applied)
            self._filter_syllable_errors_in_valid_words(normalized_text, errors, words)
            self._validate_context(normalized_text, errors, layers_applied, use_semantic)
            self._suppress_generic_pos_sequence_errors(errors)

        # Suggestion reconstruction + dedup pipeline
        self._reconstruct_compound_suggestions(normalized_text, errors)
        self._reconstruct_particle_compound_suggestions(normalized_text, errors)
        self._inject_asat_visarga_candidates(normalized_text, errors)
        self._reconstruct_morpheme_in_compound(normalized_text, errors)
        self._dedup_errors_by_position(errors)
        self._dedup_errors_by_span(errors)
        self._suppress_tense_adjacent_syntax(errors)
        self._suppress_low_value_syllable_errors(errors, text=normalized_text)
        self._suppress_low_value_syntax_errors(errors, text=normalized_text)
        self._suppress_low_value_pos_sequence_errors(errors)
        self._suppress_low_value_context_probability(errors, text=normalized_text)
        self._suppress_low_value_confusable_errors(errors, text=normalized_text)
        self._suppress_low_value_semantic_errors(errors, text=normalized_text)
        self._suppress_known_entity_errors(errors, text=normalized_text)
        self._filter_ner_entities(errors, normalized_text)

        return errors, layers_applied

    def _validate_syllables(
        self, text: str, errors: list[Error], layers_applied: list[str]
    ) -> None:
        """Run syllable validation layer."""
        if self.syllable_validator:
            syllable_errors = self.syllable_validator.validate(text)
            errors.extend(syllable_errors)
            layers_applied.append(ValidationLevel.SYLLABLE.value)

    def _validate_words(self, text: str, errors: list[Error], layers_applied: list[str]) -> None:
        """Run word validation layer."""
        if self.word_validator:
            word_errors = self.word_validator.validate(text)
            errors.extend(word_errors)
            layers_applied.append(ValidationLevel.WORD.value)

    def _validate_context(
        self,
        text: str,
        errors: list[Error],
        layers_applied: list[str],
        use_semantic: bool | None,
    ) -> None:
        """Run context validation layer with optional semantic checking."""
        if not self.context_validator:
            return

        # Compute effective semantic setting locally without
        # modifying config (thread-safe approach)
        effective_semantic_enabled = (
            use_semantic
            if use_semantic is not None
            else (self.config.semantic.use_semantic_refinement)
        )

        # Skip semantic strategy when user explicitly opts out, avoiding
        # unnecessary ONNX inference latency.
        exclude_types: frozenset[type] | None = None
        if not effective_semantic_enabled:
            from myspellchecker.core.validation_strategies.semantic_strategy import (
                SemanticValidationStrategy,
            )

            exclude_types = frozenset({SemanticValidationStrategy})

        context_errors = self.context_validator.validate(text, exclude_strategy_types=exclude_types)
        errors.extend(context_errors)
        layers_applied.append("context")
        if self.config.validation.enable_strategy_debug:
            self._thread_local.last_strategy_debug_telemetry = (
                self.context_validator.get_last_strategy_debug_telemetry()
            )
        else:
            self._thread_local.last_strategy_debug_telemetry = {}

        # N-gram re-ranking: use n-gram context to re-order suggestions.
        # Applied before semantic reranking so that the MLM model can
        # further refine the n-gram-based order.
        self._apply_ngram_reranking(text, errors)

        # Semantic re-ranking: use MLM to re-rank suggestions on existing errors.
        # The semantic model masks each flagged word and checks which suggestion
        # the model predicts — boosting it to the top of the suggestion list.
        # This is additive-only: it never adds new errors, only improves suggestion order.
        if effective_semantic_enabled and self.semantic_checker is not None:
            self._apply_semantic_reranking(text, errors)
            layers_applied.append("semantic")

        # Neural MLP re-ranking: final reranking step using a trained MLP.
        # Runs AFTER both n-gram and semantic reranking to get the final say.
        # Only modifies suggestion order — never adds/removes errors.
        if self._neural_reranker is not None:
            self._apply_neural_reranking(text, errors)

    async def check_async(
        self,
        text: str,
        level: ValidationLevel = ValidationLevel.SYLLABLE,
        use_semantic: bool | None = None,
        options: CheckOptions | None = None,
    ) -> Response:
        """
        Asynchronously check Myanmar text for spelling errors.

        This method runs the CPU-bound `check` method in a separate thread
        to avoid blocking the event loop, making it suitable for usage in
        asynchronous web frameworks like FastAPI.

        Args:
            text: Myanmar text to check.
            level: Validation level.
            use_semantic: Override semantic checking. If None, uses config setting.
            options: Per-request overrides (see :class:`CheckOptions`).

        Returns:
            Response object containing results.
        """
        return await asyncio.to_thread(self.check, text, level, use_semantic, options)

    def check_batch(
        self,
        texts: list[str],
        level: ValidationLevel = ValidationLevel.SYLLABLE,
        use_semantic: bool | None = None,
        options: CheckOptions | None = None,
    ) -> list[Response]:
        """
        Check multiple texts for spelling errors in batch.

        This method is a convenience wrapper that processes multiple texts
        sequentially. For concurrent processing, use ``check_batch_async()``.

        Args:
            texts: List of Myanmar texts to check. Each text is processed
                independently with the same validation level.
            level: Validation level for all texts (default: SYLLABLE).
            use_semantic: Override semantic checking. If None, uses config setting.
            options: Per-request overrides (see :class:`CheckOptions`).

        Returns:
            List of Response objects in the same order as input texts.
            Each Response contains the validation results for the
            corresponding input text.

        Raises:
            TypeError: If texts is not a list.
            ValueError: If texts is empty or contains non-string elements.

        Example:
            >>> checker = SpellChecker()
            >>> texts = ["မြန်မာ", "နိုင်ငံ", "အင်္ဂလန်"]
            >>> results = checker.check_batch(texts)
            >>> for text, result in zip(texts, results):
            ...     status = "valid" if not result.has_errors else "errors found"
            ...     print(f"{text}: {status}")
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts must be a list, got {type(texts).__name__}")

        if not texts:
            raise ProcessingError("texts list cannot be empty")

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ProcessingError(f"texts[{i}] must be a string, got {type(text).__name__}")

        return [
            self.check(text, level=level, use_semantic=use_semantic, options=options)
            for text in texts
        ]

    async def check_batch_async(
        self,
        texts: list[str],
        level: ValidationLevel = ValidationLevel.SYLLABLE,
        max_concurrency: int = 4,
        use_semantic: bool | None = None,
        options: CheckOptions | None = None,
    ) -> list[Response]:
        """
        Async batch spell checking with configurable concurrency.

        Runs multiple spell checks concurrently using a semaphore to
        control parallelism.

        Args:
            texts: List of Myanmar texts to check.
            level: Validation level for all texts.
            max_concurrency: Maximum concurrent checks (default: 4).
                Higher values use more CPU but complete faster.
            use_semantic: Override semantic checking. If None, uses config setting.
            options: Per-request overrides (see :class:`CheckOptions`).

        Returns:
            List of Response objects in same order as input.

        Example:
            >>> import asyncio
            >>>
            >>> async def process():
            ...     checker = SpellChecker()
            ...     texts = ["မြန်မာစာ", "ကျောင်းသား", "သင်ယူ"]
            ...     results = await checker.check_batch_async(texts, max_concurrency=2)
            ...     for text, result in zip(texts, results):
            ...         print(f"{text}: {'✓' if not result.has_errors else '✗'}")
            ...
            >>> asyncio.run(process())
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts must be a list, got {type(texts).__name__}")

        if not texts:
            raise ProcessingError("texts list cannot be empty")

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ProcessingError(f"texts[{i}] must be a string, got {type(text).__name__}")

        if max_concurrency < 1:
            raise ProcessingError(f"max_concurrency must be >= 1, got {max_concurrency}")
        semaphore = asyncio.Semaphore(max_concurrency)

        async def check_with_semaphore(text: str) -> Response:
            async with semaphore:
                return await self.check_async(text, level, use_semantic, options)

        return list(await asyncio.gather(*[check_with_semaphore(text) for text in texts]))

    @property
    def symspell(self) -> SymSpell | None:
        """Return the SymSpell instance, or None if disabled."""
        if self.syllable_validator:
            return self.syllable_validator.symspell
        if self.word_validator:
            return self.word_validator.symspell
        return None

    @property
    def context_checker(self) -> NgramContextChecker | None:
        """Return the NgramContextChecker instance, or None if disabled."""
        return cast("NgramContextChecker | None", self._context_checker)

    @property
    def syllable_rule_validator(self) -> SyllableRuleValidator | None:
        """Return the SyllableRuleValidator instance, or None if disabled."""
        if self.syllable_validator:
            return self.syllable_validator.syllable_rule_validator
        return None

    @property
    def ner_model(self) -> Any | None:
        """Return the NERModel instance, or None if NER is disabled."""
        return self._ner_model

    @property
    def name_heuristic(self) -> NameHeuristic | None:
        """Return the NameHeuristic instance, or None if NER is disabled."""
        return cast("NameHeuristic | None", self._name_heuristic)

    @property
    def semantic_checker(self) -> SemanticChecker | None:
        """Return the SemanticChecker instance, or None if unconfigured."""
        return cast("SemanticChecker | None", self._semantic_checker)

    @property
    def phonetic_hasher(self) -> PhoneticHasher | None:
        """Return the PhoneticHasher instance, or None if disabled."""
        return cast("PhoneticHasher | None", self._phonetic_hasher)

    def _create_neural_reranker(self) -> Any | None:
        """Create NeuralReranker if configured.

        Returns:
            NeuralReranker instance or None if not configured or
            initialization failed.  Provides graceful degradation --
            the pipeline continues without neural reranking on failure.
        """
        nr_config = self.config.neural_reranker
        if not nr_config.enabled or not nr_config.model_path:
            return None

        try:
            from myspellchecker.algorithms.neural_reranker import NeuralReranker

            reranker = NeuralReranker(
                model_path=nr_config.model_path,
                stats_path=nr_config.stats_path,
            )
            self.logger.info("Initialized NeuralReranker")
            return reranker
        except (ImportError, OSError, RuntimeError, ValueError, FileNotFoundError) as e:
            self.logger.warning(
                "NeuralReranker initialization failed, continuing without "
                "neural reranking. Error: %s",
                e,
            )
            return None

    def close(self) -> None:
        """
        Close underlying resources and release connections.

        This method should be called when the SpellChecker is no longer needed
        to properly release database connections and other resources. It is
        automatically called when using the SpellChecker as a context manager.

        The method is idempotent - calling it multiple times is safe.

        Example:
            >>> # Manual resource management
            >>> checker = SpellChecker()
            >>> try:
            ...     result = checker.check("မြန်မာ")
            ... finally:
            ...     checker.close()
            >>>
            >>> # Using context manager (preferred)
            >>> with SpellChecker() as checker:
            ...     result = checker.check("မြန်မာ")
            ... # Resources automatically released
        """
        if self.provider and hasattr(self.provider, "close"):
            try:
                self.provider.close()
            except (RuntimeError, OSError) as e:
                self.logger.warning(f"Error closing provider: {e}")

        # Close inference sessions held by validation strategies
        if self.context_validator and hasattr(self.context_validator, "close"):
            try:
                self.context_validator.close()
            except (RuntimeError, OSError) as e:
                self.logger.warning(f"Error closing context validator: {e}")

        # Close directly-held semantic checker (may share session with strategy)
        if self._semantic_checker and hasattr(self._semantic_checker, "close"):
            try:
                self._semantic_checker.close()
            except (RuntimeError, OSError) as e:
                self.logger.warning(f"Error closing semantic checker: {e}")

    def get_pos_tags(self, text: str = "", words: list[str] | None = None) -> list[str]:
        """
        Get the most likely sequence of POS tags for a given text or pre-segmented words.

        Args:
            text: The input text to tag. (Optional if `words` is provided)
            words: A list of pre-segmented words. (Optional if `text` is provided)

        Returns:
            A list of POS tags, one for each word in the sequence.
        """
        if not text and not words:
            return []

        if words:
            processed_words = words
        else:
            normalized_text = normalize(text)
            if not normalized_text or not normalized_text.strip():
                return []
            processed_words = self.segmenter.segment_words(normalized_text)

        if self.viterbi_tagger is None:
            return []
        return self.viterbi_tagger.tag_sequence(processed_words)

    def segment_and_tag(self, text: str) -> tuple[list[str], list[str]]:
        """
        Perform joint word segmentation and POS tagging.

        When joint mode is enabled (config.joint.enabled=True), this uses a
        unified Viterbi decoder that simultaneously optimizes word boundaries
        and POS tags, potentially achieving better accuracy than sequential
        segmentation followed by tagging.

        When joint mode is disabled (default), falls back to sequential processing:
        first segment the text, then tag the resulting words.

        Mode Selection:
            - **Sequential (default)**: Faster (~15K words/sec), lower memory,
              recommended for production use
            - **Joint**: Better for ambiguous segmentation and OOV-heavy text,
              ~3x slower, requires bigram/trigram probability tables

        To enable joint mode:
            >>> config = SpellCheckerConfig(joint=JointConfig(enabled=True))
            >>> checker = SpellChecker(config=config)

        See Also:
            - https://docs.myspellchecker.com/features/pos-tagging#joint-segmentation-and-tagging
              for detailed comparison and when to use each mode

        Args:
            text: The input Myanmar text to segment and tag.

        Returns:
            Tuple of (words, tags) where:
            - words: List of segmented words
            - tags: List of POS tags (same length as words)

        Example:
            >>> checker = SpellChecker()
            >>> words, tags = checker.segment_and_tag("မြန်မာနိုင်ငံ")
            >>> print(list(zip(words, tags)))
            [('မြန်မာ', 'N'), ('နိုင်ငံ', 'N')]
        """
        if not text:
            return [], []

        normalized_text = normalize(text)
        if not normalized_text or not normalized_text.strip():
            return [], []

        # Use joint model if available
        if self.joint_segment_tagger is not None:
            return self.joint_segment_tagger.segment_and_tag(normalized_text)

        # Fallback to sequential: segment then tag
        words = self.segmenter.segment_words(normalized_text)
        if self.viterbi_tagger is None:
            return words, []
        tags = self.viterbi_tagger.tag_sequence(words)
        return words, tags

    def __enter__(self) -> "SpellChecker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
