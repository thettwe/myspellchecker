"""
Component Factory for SpellChecker.

This module provides centralized component construction logic
with improved testability.

Factory Pattern Overview:
    The codebase has three factory patterns for different use cases:

    1. **ComponentFactory** (this module) - RECOMMENDED
       - Used internally by SpellChecker.__init__()
       - Creates all components in proper dependency order
       - Supports custom factory via SpellChecker(factory=...)
       - Uses shared builders from core/factories/builders.py

    2. **ServiceContainer + DI Factories** (core/di/ + core/factories/)
       - Full dependency injection pattern
       - For advanced users who need DI container
       - All factories share code via core/factories/builders.py

    3. **AlgorithmFactory** (algorithms/factory.py)
       - Standalone algorithm creation with caching
       - For users who need algorithms without SpellChecker
       - Takes individual config objects (SymSpellConfig, etc.)

The factory handles:
- Algorithm component construction (SymSpell, ViterbiTagger, etc.)
- Validator construction (Syllable, Word, Context validators)
- Default component resolution

Dependency Injection:
    The ComponentFactoryProtocol defines the interface for component creation,
    allowing users to provide custom factory implementations for testing or
    specialized component construction.

    Example:
        >>> # Using a custom factory for testing
        >>> class MockFactory:
        ...     def create_symspell(self, provider, ...):
        ...         return MockSymSpell()
        ...     # ... implement other methods
        >>> checker = SpellChecker(factory=MockFactory())

See Also:
    - core/factories/builders.py: Shared builder functions
    - algorithms/factory.py: Standalone AlgorithmFactory
    - core/di/container.py: ServiceContainer for DI pattern
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from myspellchecker.algorithms.cache import (
        CachedBigramSource,
        CachedDictionaryLookup,
        CachedFrequencySource,
        CachedTrigramSource,
    )
    from myspellchecker.core.homophones import HomophoneChecker

from myspellchecker.core.config import SpellCheckerConfig
from myspellchecker.core.exceptions import InferenceError, ModelLoadError
from myspellchecker.utils.logging_utils import get_logger


@runtime_checkable
class ComponentFactoryProtocol(Protocol):
    """
    Protocol for component factory implementations.

    Defines the interface for creating SpellChecker components.
    Implement this protocol to provide custom component construction
    for testing, alternative implementations, or specialized configurations.

    Example:
        >>> class CustomFactory:
        ...     def __init__(self, config):
        ...         self.config = config
        ...
        ...     def create_all(self, provider, segmenter):
        ...         # Custom component creation logic
        ...         return {"symspell": CustomSymSpell(), ...}
        ...
        >>> checker = SpellChecker(factory=CustomFactory(config))

    Notes:
        - All methods should be implemented for complete replacement
        - For partial customization, extend ComponentFactory instead
        - The create_all method is the minimum required for SpellChecker integration
    """

    config: SpellCheckerConfig

    def create_phonetic_hasher(self) -> Any | None:
        """Create PhoneticHasher if phonetic matching is enabled."""
        ...

    def create_symspell(
        self,
        provider: Any,
        phonetic_hasher: Any | None = None,
        build_index: bool = True,
    ) -> Any:
        """Create and configure SymSpell instance."""
        ...

    def create_pos_probabilities(
        self, provider: Any
    ) -> tuple[
        dict[str, float] | None,
        dict[tuple[str, str], float] | None,
        dict[tuple[str, str, str], float] | None,
    ]:
        """Load POS probability dictionaries from provider."""
        ...

    def create_viterbi_tagger(
        self,
        provider: Any,
        bigram_probs: dict[tuple[str, str], float] | None,
        trigram_probs: dict[tuple[str, str, str], float] | None,
        unigram_probs: dict[str, float] | None = None,
    ) -> Any:
        """Create ViterbiTagger for POS sequence tagging."""
        ...

    def create_context_checker(
        self,
        provider: Any,
        symspell: Any,
        unigram_probs: dict[str, float] | None,
        bigram_probs: dict[tuple[str, str], float] | None,
    ) -> Any | None:
        """Create NgramContextChecker if context checking is enabled."""
        ...

    def create_syllable_validator(
        self,
        segmenter: Any,
        provider: Any,
        symspell: Any,
        syllable_rule_validator: Any | None,
    ) -> Any:
        """Create SyllableValidator for syllable-level validation."""
        ...

    def create_word_validator(
        self,
        segmenter: Any,
        provider: Any,
        symspell: Any,
        context_checker: Any | None = None,
        suggestion_strategy: Any | None = None,
    ) -> Any:
        """Create WordValidator for word-level validation."""
        ...

    def create_context_validator(
        self,
        segmenter: Any,
        provider: Any,
        context_checker: Any | None,
        name_heuristic: Any | None,
        phonetic_hasher: Any | None,
        semantic_checker: Any | None,
        syntactic_rule_checker: Any,
        homophone_checker: Any,
        viterbi_tagger: Any | None = None,
        tone_disambiguator: Any | None = None,
        ner_model: Any | None = None,
        pos_disambiguator: Any | None = None,
        symspell: Any | None = None,
    ) -> Any:
        """Create ContextValidator using strategy pattern."""
        ...

    def create_all(
        self,
        provider: Any,
        segmenter: Any,
    ) -> dict[str, Any]:
        """Create all SpellChecker components in proper order."""
        ...


class ComponentFactory:
    """
    Factory for constructing SpellChecker components.

    Centralizes component construction logic to simplify SpellChecker.__init__
    and improve testability through dependency injection.

    Example:
        >>> config = SpellCheckerConfig()
        >>> factory = ComponentFactory(config)
        >>> components = factory.create_all(provider, segmenter)
        >>> # components contains all validators and algorithm instances
    """

    def __init__(self, config: SpellCheckerConfig):
        """
        Initialize the factory with configuration.

        Args:
            config: SpellCheckerConfig instance with all settings.
        """
        self.config = config
        self.logger = get_logger(__name__)
        # Cached sources (initialized in create_cached_sources)
        self._cached_dict_source: "CachedDictionaryLookup" | None = None
        self._cached_freq_source: "CachedFrequencySource" | None = None
        self._cached_bigram_source: "CachedBigramSource" | None = None
        self._cached_trigram_source: "CachedTrigramSource" | None = None

    def create_cached_sources(self, provider: Any) -> dict[str, Any]:
        """
        Create cached wrappers for provider using AlgorithmCacheConfig settings.

        This method wraps the provider with LRU caching layers to improve
        performance (10-100x speedup for repeated lookups).

        Args:
            provider: DictionaryProvider to wrap with caching.

        Returns:
            Dictionary with cached sources:
            - 'dictionary': CachedDictionaryLookup
            - 'frequency': CachedFrequencySource
            - 'bigram': CachedBigramSource
            - 'trigram': CachedTrigramSource

        Note:
            Uses cache sizes from self.config.cache (AlgorithmCacheConfig).
        """
        from myspellchecker.algorithms.cache import (
            CachedBigramSource,
            CachedDictionaryLookup,
            CachedFrequencySource,
            CachedTrigramSource,
        )

        cache_config = self.config.cache

        self._cached_dict_source = CachedDictionaryLookup(
            provider,
            syllable_cache_size=cache_config.syllable_cache_size,
            word_cache_size=cache_config.word_cache_size,
        )
        self._cached_freq_source = CachedFrequencySource(
            provider,
            cache_size=cache_config.frequency_cache_size,
        )
        self._cached_bigram_source = CachedBigramSource(
            provider,
            cache_size=cache_config.bigram_cache_size,
        )
        self._cached_trigram_source = CachedTrigramSource(
            provider,
            cache_size=cache_config.trigram_cache_size,
        )

        self.logger.debug(
            f"Created cached sources with sizes: "
            f"syllable={cache_config.syllable_cache_size}, "
            f"word={cache_config.word_cache_size}, "
            f"frequency={cache_config.frequency_cache_size}, "
            f"bigram={cache_config.bigram_cache_size}, "
            f"trigram={cache_config.trigram_cache_size}"
        )

        return {
            "dictionary": self._cached_dict_source,
            "frequency": self._cached_freq_source,
            "bigram": self._cached_bigram_source,
            "trigram": self._cached_trigram_source,
        }

    def create_phonetic_hasher(self) -> Any | None:
        """
        Create PhoneticHasher if phonetic matching is enabled.

        Returns:
            PhoneticHasher instance or None if disabled.
        """
        if not self.config.use_phonetic:
            self.logger.debug(
                "Phonetic matching is disabled (use_phonetic=False). "
                "Suggestions will be based on edit distance only. "
                "Enable phonetic matching for better Myanmar-specific suggestions "
                "for homophones and similar-sounding words."
            )
            return None

        from myspellchecker.text.phonetic import PhoneticHasher

        hasher = PhoneticHasher(config=self.config.phonetic)
        self.logger.info(
            f"Phonetic matching enabled with max_code_length={self.config.phonetic.max_code_length}"
        )
        return hasher

    def create_symspell(
        self,
        provider: Any,
        phonetic_hasher: Any | None = None,
        build_index: bool = True,
        segmenter: Any | None = None,
    ) -> Any:
        """
        Create and configure SymSpell instance.

        Args:
            provider: DictionaryProvider for dictionary lookups.
            phonetic_hasher: Optional PhoneticHasher for phonetic matching.
            build_index: Whether to build the SymSpell index (default: True).
            segmenter: Optional Segmenter for syllable-aligned compound DP.

        Returns:
            Configured SymSpell instance.

        Note:
            SymSpell uses a base ranker (never UnifiedRanker) to avoid
            double-normalizing scores. UnifiedRanker is only used at the
            composite pipeline level via CompositeSuggestionStrategy.
        """
        from myspellchecker.core.factories.builders import build_symspell

        return build_symspell(
            provider=provider,
            config=self.config,
            phonetic_hasher=phonetic_hasher,
            build_index=build_index,
            segmenter=segmenter,
        )

    def create_pos_probabilities(
        self, provider: Any
    ) -> tuple[
        dict[str, float] | None,
        dict[tuple[str, str], float] | None,
        dict[tuple[str, str, str], float] | None,
    ]:
        """
        Load POS probability dictionaries from provider.

        Args:
            provider: DictionaryProvider with POS probability data.

        Returns:
            Tuple of (unigram_probs, bigram_probs, trigram_probs).
            Returns empty dicts if loading fails.
        """
        if not self.config.use_context_checker:
            return None, None, None

        try:
            unigram_probs = provider.get_pos_unigram_probabilities()
            bigram_probs = provider.get_pos_bigram_probabilities()
            trigram_probs = provider.get_pos_trigram_probabilities()
            return unigram_probs, bigram_probs, trigram_probs
        except (OSError, RuntimeError, KeyError, ValueError) as e:
            self.logger.warning(f"Could not load POS probabilities: {e}")
            return {}, {}, {}

    def create_viterbi_tagger(
        self,
        provider: Any,
        bigram_probs: dict[tuple[str, str], float] | None,
        trigram_probs: dict[tuple[str, str, str], float] | None,
        unigram_probs: dict[str, float] | None = None,
    ) -> Any:
        """
        Create ViterbiTagger for POS sequence tagging.

        Args:
            provider: DictionaryProvider for word lookups.
            bigram_probs: POS bigram probabilities.
            trigram_probs: POS trigram probabilities.
            unigram_probs: POS unigram probabilities (emission prior).

        Returns:
            Configured ViterbiTagger instance.
        """
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        # Use pos_tagger config
        pos_config = self.config.pos_tagger

        return ViterbiTagger(
            provider,
            bigram_probs or {},
            trigram_probs or {},
            pos_unigram_probs=unigram_probs,
            unknown_word_tag=pos_config.unknown_tag,
            beam_width=pos_config.viterbi_beam_width,
            emission_weight=pos_config.viterbi_emission_weight,
            min_prob=pos_config.viterbi_min_prob,
            config=pos_config,
        )

    def create_context_checker(
        self,
        provider: Any,
        symspell: Any,
        unigram_probs: dict[str, float] | None,
        bigram_probs: dict[tuple[str, str], float] | None,
    ) -> Any | None:
        """
        Create NgramContextChecker if context checking is enabled.

        Args:
            provider: DictionaryProvider for n-gram lookups.
            symspell: SymSpell instance for suggestions.
            unigram_probs: POS unigram probabilities.
            bigram_probs: POS bigram probabilities.

        Returns:
            NgramContextChecker instance or None if disabled.
        """
        from myspellchecker.core.factories.builders import build_ngram_context_checker

        context_checker = build_ngram_context_checker(
            provider=provider,
            config=self.config,
            symspell=symspell,
            pos_unigram_probs=unigram_probs,
            pos_bigram_probs=bigram_probs,
        )

        # Log warning if context checking was requested but couldn't be created
        if self.config.use_context_checker and context_checker is None:
            self.logger.warning(
                "Context checking was requested (use_context_checker=True) but "
                "NgramContextChecker could not be created. Context validation and "
                "context-aware ranking will be unavailable. Ensure N-gram probability "
                "data exists in the database."
            )

        return context_checker

    def create_syllable_rule_validator(self) -> Any | None:
        """
        Create SyllableRuleValidator if rule-based validation is enabled.

        Returns:
            SyllableRuleValidator instance or None if disabled.
        """
        if not self.config.use_rule_based_validation:
            return None

        from pathlib import Path

        from myspellchecker.core.detection_rules import load_stacking_pairs
        from myspellchecker.core.syllable_rules import SyllableRuleValidator

        raw_path = self.config.validation.stacking_pairs_path
        stacking_path = Path(raw_path) if isinstance(raw_path, str) else None
        stacking_pairs = load_stacking_pairs(stacking_path)

        return SyllableRuleValidator(
            max_syllable_length=self.config.validation.max_syllable_length,
            corruption_threshold=self.config.validation.syllable_corruption_threshold,
            strict=self.config.validation.strict_validation,
            allow_extended_myanmar=self.config.validation.allow_extended_myanmar,
            stacking_pairs=stacking_pairs,
        )

    def create_syntactic_rule_checker(self, provider: Any) -> Any:
        """
        Create SyntacticRuleChecker for syntactic validation.

        Args:
            provider: DictionaryProvider for lookups.

        Returns:
            SyntacticRuleChecker instance.
        """
        from myspellchecker.grammar.engine import SyntacticRuleChecker

        return SyntacticRuleChecker(provider, grammar_config=self.config.grammar_engine)

    def create_joint_segment_tagger(
        self,
        provider: Any,
        bigram_probs: dict[tuple[str, str], float] | None,
        trigram_probs: dict[tuple[str, str, str], float] | None,
        unigram_probs: dict[str, float] | None = None,
        word_tag_probs: dict[str, dict[str, float]] | None = None,
    ) -> Any | None:
        """
        Create JointSegmentTagger for unified segmentation and POS tagging.

        Args:
            provider: DictionaryProvider for word lookups.
            bigram_probs: POS bigram probabilities.
            trigram_probs: POS trigram probabilities.
            unigram_probs: POS unigram probabilities (emission prior).
            word_tag_probs: P(tag|word) emission probabilities.

        Returns:
            JointSegmentTagger instance or None if disabled.
        """
        if not self.config.joint.enabled:
            return None

        from myspellchecker.algorithms import JointSegmentTagger

        return JointSegmentTagger(
            provider=provider,
            pos_bigram_probs=bigram_probs or {},
            pos_trigram_probs=trigram_probs or {},
            pos_unigram_probs=unigram_probs,
            cache_config=self.config.cache,
            word_tag_probs=word_tag_probs,
            min_prob=self.config.joint.min_prob,
            max_word_length=self.config.joint.max_word_length,
            beam_width=self.config.joint.beam_width,
            emission_weight=self.config.joint.emission_weight,
            word_score_weight=self.config.joint.word_score_weight,
            use_morphology_fallback=self.config.joint.use_morphology_fallback,
        )

    def create_ner_model(self, segmenter: Any) -> Any | None:
        """
        Create NERModel based on configuration.

        If config.ner is provided and enabled, uses NERFactory.create() to build
        the appropriate NER model (heuristic, transformer, or hybrid).
        If config.ner is None but config.use_ner is True, creates HeuristicNER
        for backward compatibility.

        Args:
            segmenter: Segmenter for word tokenization (used by HeuristicNER).

        Returns:
            NERModel instance or None if NER is disabled.
        """
        if not self.config.use_ner:
            return None

        if self.config.ner is not None and self.config.ner.enabled:
            from myspellchecker.text.ner_model import NERFactory

            try:
                ner_model = NERFactory.create(
                    self.config.ner,
                    allow_extended_myanmar=self.config.validation.allow_extended_myanmar,
                )
                self.logger.info(
                    f"NER model created: {type(ner_model).__name__} "
                    f"(model_type={self.config.ner.model_type})"
                )
                return ner_model
            except (ImportError, OSError, RuntimeError, ValueError) as e:
                self.logger.warning(f"NER model creation failed, falling back to heuristic: {e}")
                # Fall through to heuristic

        # Backward compat: use_ner=True without explicit NERConfig → HeuristicNER
        from myspellchecker.text.ner_model import HeuristicNER

        return HeuristicNER(
            segmenter=segmenter,
            allow_extended_myanmar=self.config.validation.allow_extended_myanmar,
        )

    def create_name_heuristic(self) -> Any | None:
        """
        Create NameHeuristic if NER is enabled.

        Returns:
            NameHeuristic instance or None if disabled.
        """
        if not self.config.use_ner:
            return None

        from myspellchecker.text.ner import NameHeuristic

        return NameHeuristic()

    def create_semantic_checker(self) -> Any | None:
        """
        Create SemanticChecker if model paths are configured.

        Returns:
            SemanticChecker instance or None if not configured or initialization failed.

        Note:
            This method provides graceful degradation - if semantic checker initialization
            fails, it logs a warning and returns None, allowing the spell checker to
            continue working with basic features.
        """
        has_paths = self.config.semantic.model_path and self.config.semantic.tokenizer_path
        has_instances = self.config.semantic.model and self.config.semantic.tokenizer

        if not (has_paths or has_instances):
            return None

        try:
            from myspellchecker.algorithms.semantic_checker import SemanticChecker

            checker = SemanticChecker(
                model_path=self.config.semantic.model_path,
                tokenizer_path=self.config.semantic.tokenizer_path,
                model=self.config.semantic.model,
                tokenizer=self.config.semantic.tokenizer,
                num_threads=self.config.semantic.num_threads,
                predict_top_k=self.config.semantic.predict_top_k,
                check_top_k=self.config.semantic.check_top_k,
                allow_extended_myanmar=self.config.validation.allow_extended_myanmar,
                cache_config=self.config.cache,
                semantic_config=self.config.semantic,
            )
            self.logger.info("Initialized SemanticChecker")
            return checker
        except (
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
            FileNotFoundError,
            ModelLoadError,
            InferenceError,
        ) as e:
            # If model paths were explicitly configured, surface the error prominently.
            # Silent failures when the user explicitly requested semantic checking
            # are confusing and hide missing dependencies.
            if has_paths:
                self.logger.error(
                    f"SemanticChecker initialization FAILED with explicitly configured "
                    f"model paths. This likely means required packages are missing. "
                    f"Install via: pip install transformers onnxruntime\n"
                    f"  model_path: {self.config.semantic.model_path}\n"
                    f"  tokenizer_path: {self.config.semantic.tokenizer_path}\n"
                    f"  Error: {e}"
                )
            else:
                self.logger.warning(
                    f"SemanticChecker initialization failed, continuing without "
                    f"semantic checking. Error: {e}"
                )
            return None

    def create_suggestion_strategy(
        self,
        symspell: Any,
        provider: Any,
        context_checker: Any | None = None,
        compound_resolver: Any | None = None,
        reduplication_engine: Any | None = None,
    ) -> Any | None:
        """
        Create CompositeSuggestionStrategy for unified suggestion generation.

        Creates a composite strategy that aggregates suggestions from multiple
        sources (SymSpell, morphology, compound, morpheme) and applies unified
        ranking. If context checking is enabled and a context_checker is provided,
        wraps the composite in ContextSuggestionStrategy for context-aware
        re-ranking.

        Args:
            symspell: SymSpell instance for base suggestions.
            provider: DictionaryProvider for dictionary lookups.
            context_checker: Optional NgramContextChecker for context-aware suggestions.
            compound_resolver: Optional CompoundResolver for morpheme-level suggestions.
            reduplication_engine: Optional ReduplicationEngine for morpheme-level suggestions.

        Returns:
            SuggestionStrategy instance or None if SymSpell not available.
        """
        from myspellchecker.core.factories.builders import build_suggestion_strategy

        return build_suggestion_strategy(
            symspell=symspell,
            provider=provider,
            config=self.config,
            context_checker=context_checker,
            compound_resolver=compound_resolver,
            reduplication_engine=reduplication_engine,
        )

    def create_syllable_validator(
        self,
        segmenter: Any,
        provider: Any,
        symspell: Any,
        syllable_rule_validator: Any | None,
    ) -> Any:
        """
        Create SyllableValidator for syllable-level validation.

        Args:
            segmenter: Segmenter for syllable segmentation.
            provider: DictionaryProvider for validation.
            symspell: SymSpell for suggestions.
            syllable_rule_validator: Optional rule validator.

        Returns:
            Configured SyllableValidator instance.
        """
        from myspellchecker.core.validators import SyllableValidator

        return SyllableValidator(
            config=self.config,
            segmenter=segmenter,
            repository=provider,  # Provider implements SyllableRepository protocol
            symspell=symspell,
            syllable_rule_validator=syllable_rule_validator,
        )

    def create_word_validator(
        self,
        segmenter: Any,
        provider: Any,
        symspell: Any,
        context_checker: Any | None = None,
        suggestion_strategy: Any | None = None,
        reduplication_engine: Any | None = None,
        compound_resolver: Any | None = None,
    ) -> Any:
        """
        Create WordValidator for word-level validation.

        Args:
            segmenter: Segmenter for word segmentation.
            provider: DictionaryProvider for validation.
            symspell: SymSpell for suggestions.
            context_checker: Optional NgramContextChecker for context-aware suggestions.
            suggestion_strategy: Optional SuggestionStrategy for unified suggestions.
            reduplication_engine: Optional ReduplicationEngine for reduplication validation.
            compound_resolver: Optional CompoundResolver for compound synthesis.

        Returns:
            Configured WordValidator instance.
        """
        from myspellchecker.core.validators import WordValidator

        return WordValidator(
            config=self.config,
            segmenter=segmenter,
            word_repository=provider,  # Provider implements WordRepository protocol
            syllable_repository=provider,  # Provider implements SyllableRepository protocol
            symspell=symspell,
            context_checker=context_checker,
            suggestion_strategy=suggestion_strategy,
            reduplication_engine=reduplication_engine,
            compound_resolver=compound_resolver,
        )

    def create_homophone_checker(self, provider: object | None = None) -> "HomophoneChecker":
        """
        Create HomophoneChecker.

        Args:
            provider: Optional dictionary provider with ``get_confusable_pairs()``
                for DB-driven confusable pair lookup.

        Returns:
            HomophoneChecker instance.
        """
        from myspellchecker.core.homophones import HomophoneChecker

        return HomophoneChecker(provider=provider)

    def create_tone_disambiguator(self) -> Any:
        """
        Create ToneDisambiguator for Myanmar tone mark validation.

        Loads tone rules from tone_rules.yaml via GrammarRuleConfig and
        wires them into the ToneDisambiguator via ToneConfig.

        Returns:
            ToneDisambiguator instance with YAML-based rules if available.
        """
        from myspellchecker.core.config.text_configs import ToneConfig
        from myspellchecker.grammar.config import get_grammar_config
        from myspellchecker.text.tone import ToneDisambiguator

        # Load grammar config which parses tone_rules.yaml
        grammar_config = get_grammar_config()

        # Create ToneConfig with YAML-loaded maps if available
        tone_config = ToneConfig(
            tone_ambiguous_map=grammar_config.tone_ambiguous_map or None,
            tone_errors_map=grammar_config.tone_errors_map or None,
        )

        return ToneDisambiguator(config=tone_config)

    def create_pos_disambiguator(self) -> Any:
        """
        Create POSDisambiguator for resolving ambiguous multi-POS words.

        Uses R1-R5 disambiguation rules to resolve words with multiple
        possible POS tags based on their sentence context.

        Returns:
            POSDisambiguator instance.
        """
        from myspellchecker.algorithms.pos_disambiguator import POSDisambiguator

        return POSDisambiguator()

    def create_context_validator(
        self,
        segmenter: Any,
        provider: Any,
        context_checker: Any | None,
        name_heuristic: Any | None,
        phonetic_hasher: Any | None,
        semantic_checker: Any | None,
        syntactic_rule_checker: Any,
        homophone_checker: Any,
        viterbi_tagger: Any | None = None,
        tone_disambiguator: Any | None = None,
        ner_model: Any | None = None,
        pos_disambiguator: Any | None = None,
        symspell: Any | None = None,
    ) -> Any:
        """
        Create ContextValidator using strategy pattern.

        Creates validation strategies based on available components and
        returns a strategy-based ContextValidator that orchestrates them.

        Strategies created (in priority order):
        1. ToneValidationStrategy (10) - Tone mark disambiguation
        2. OrthographyValidationStrategy (15) - Medial order and compatibility
        3. SyntacticValidationStrategy (20) - Grammar rules
        4. POSSequenceValidationStrategy (30) - POS patterns
        5. QuestionStructureValidationStrategy (40) - Question particles
        6. HomophoneValidationStrategy (45) - Homophone detection
        7. NgramContextValidationStrategy (50) - Statistical context
        8. SemanticValidationStrategy (70) - AI-powered semantic checking

        Args:
            segmenter: Segmenter for word segmentation.
            provider: DictionaryProvider for validation.
            context_checker: NgramContextChecker instance.
            name_heuristic: NameHeuristic instance.
            phonetic_hasher: PhoneticHasher instance.
            semantic_checker: SemanticChecker instance.
            syntactic_rule_checker: SyntacticRuleChecker instance.
            homophone_checker: HomophoneChecker instance.
            viterbi_tagger: ViterbiTagger instance for POS sequence validation.
            tone_disambiguator: ToneDisambiguator instance for tone validation.
            pos_disambiguator: POSDisambiguator instance for multi-POS resolution.

        Returns:
            Strategy-based ContextValidator instance.
        """
        from myspellchecker.core.context_validator import ContextValidator
        from myspellchecker.core.factories.builders import build_context_validation_strategies

        # Build strategies using shared builder
        strategies = build_context_validation_strategies(
            config=self.config,
            provider=provider,
            tone_disambiguator=tone_disambiguator,
            syntactic_rule_checker=syntactic_rule_checker,
            viterbi_tagger=viterbi_tagger,
            context_checker=context_checker,
            homophone_checker=homophone_checker,
            semantic_checker=semantic_checker,
            pos_disambiguator=pos_disambiguator,
            symspell=symspell,
        )

        # Note: ContextValidator no longer requires provider directly (ISP).
        # Each strategy receives only the interfaces it needs via factory injection.
        return ContextValidator(
            config=self.config,
            segmenter=segmenter,
            strategies=strategies,
            name_heuristic=name_heuristic,
            ner_model=ner_model,
            viterbi_tagger=viterbi_tagger,
        )

    def create_all(
        self,
        provider: Any,
        segmenter: Any,
    ) -> dict[str, Any]:
        """
        Create all SpellChecker components in proper order.

        This method handles the complex dependency resolution between
        components, constructing them in the correct order.

        Args:
            provider: DictionaryProvider for all lookups.
            segmenter: Segmenter for text segmentation.

        Returns:
            Dictionary containing all constructed components:
            - phonetic_hasher
            - symspell
            - viterbi_tagger
            - joint_segment_tagger
            - context_checker
            - syllable_rule_validator
            - syntactic_rule_checker
            - homophone_checker
            - name_heuristic
            - semantic_checker
            - suggestion_strategy
            - syllable_validator
            - word_validator
            - context_validator
            - cached_sources: Dict with cached wrappers using AlgorithmCacheConfig settings
              (dictionary, frequency, bigram, trigram)
        """
        # 0. Initialize cached sources using AlgorithmCacheConfig settings
        # This provides cached wrappers for frequently accessed provider methods.
        # Cache sizes are configured via self.config.cache (AlgorithmCacheConfig).
        cached_sources = self.create_cached_sources(provider)
        self.logger.debug("Initialized cached sources from AlgorithmCacheConfig")

        # 1. Create algorithm components
        phonetic_hasher = self.create_phonetic_hasher()

        # Skip SymSpell initialization if configured (for POS-only use cases)
        if self.config.symspell.skip_init:
            symspell = None
        else:
            symspell = self.create_symspell(provider, phonetic_hasher, segmenter=segmenter)

        # 2. Load POS probabilities
        unigram_probs, bigram_probs, trigram_probs = self.create_pos_probabilities(provider)

        # 3. Create taggers and checkers
        viterbi_tagger = self.create_viterbi_tagger(
            provider, bigram_probs, trigram_probs, unigram_probs
        )
        joint_segment_tagger = self.create_joint_segment_tagger(
            provider, bigram_probs, trigram_probs, unigram_probs
        )
        context_checker = self.create_context_checker(
            provider, symspell, unigram_probs, bigram_probs
        )

        # 4. Create rule validators
        syllable_rule_validator = self.create_syllable_rule_validator()
        syntactic_rule_checker = self.create_syntactic_rule_checker(provider)
        homophone_checker = self.create_homophone_checker(provider=provider)
        name_heuristic = self.create_name_heuristic()
        ner_model = self.create_ner_model(segmenter)
        semantic_checker = self.create_semantic_checker()
        tone_disambiguator = self.create_tone_disambiguator()
        pos_disambiguator = self.create_pos_disambiguator()

        # 4.5. Create morphological synthesis engines
        reduplication_engine = None
        compound_resolver = None
        if self.config.validation.use_reduplication_validation:
            from myspellchecker.text.reduplication import ReduplicationEngine

            reduplication_engine = ReduplicationEngine(
                segmenter=segmenter,
                min_base_frequency=self.config.validation.reduplication_min_base_frequency,
                cache_size=self.config.validation.reduplication_cache_size,
                config=self.config.reduplication,
            )
        if self.config.validation.use_compound_synthesis:
            from myspellchecker.text.compound_resolver import CompoundResolver

            compound_resolver = CompoundResolver(
                segmenter=segmenter,
                min_morpheme_frequency=self.config.validation.compound_min_morpheme_frequency,
                max_parts=self.config.validation.compound_max_parts,
                cache_size=self.config.validation.compound_cache_size,
                config=self.config.compound_resolver,
            )

        # 4.6. Create suggestion strategy (unified pipeline for suggestions)
        # Pass context_checker for optional context-aware re-ranking
        suggestion_strategy = self.create_suggestion_strategy(
            symspell,
            provider,
            context_checker,
            compound_resolver=compound_resolver,
            reduplication_engine=reduplication_engine,
        )

        # 5. Create validators
        syllable_validator = self.create_syllable_validator(
            segmenter, provider, symspell, syllable_rule_validator
        )
        word_validator = self.create_word_validator(
            segmenter,
            provider,
            symspell,
            context_checker,
            suggestion_strategy,
            reduplication_engine=reduplication_engine,
            compound_resolver=compound_resolver,
        )
        context_validator = self.create_context_validator(
            segmenter,
            provider,
            context_checker,
            name_heuristic,
            phonetic_hasher,
            semantic_checker,
            syntactic_rule_checker,
            homophone_checker,
            viterbi_tagger,
            tone_disambiguator,
            ner_model,
            pos_disambiguator,
            symspell=symspell,
        )

        return {
            "phonetic_hasher": phonetic_hasher,
            "symspell": symspell,
            "viterbi_tagger": viterbi_tagger,
            "joint_segment_tagger": joint_segment_tagger,
            "context_checker": context_checker,
            "syllable_rule_validator": syllable_rule_validator,
            "syntactic_rule_checker": syntactic_rule_checker,
            "homophone_checker": homophone_checker,
            "name_heuristic": name_heuristic,
            "pos_disambiguator": pos_disambiguator,
            "ner_model": ner_model,
            "semantic_checker": semantic_checker,
            "suggestion_strategy": suggestion_strategy,
            "syllable_validator": syllable_validator,
            "word_validator": word_validator,
            "context_validator": context_validator,
            # Cached sources for cache stats and optimized lookups
            "cached_sources": cached_sources,
        }
