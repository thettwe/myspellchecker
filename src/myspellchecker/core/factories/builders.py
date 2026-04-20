"""
Shared builder functions for component creation.

This module contains the core creation logic used by both DI factories
and ComponentFactory. Extracting this logic eliminates duplication and
ensures consistent component configuration across all creation paths.

Each builder function takes explicit dependencies and configuration,
making them testable and reusable.

Factory Pattern Overview:
    The codebase has three factory patterns that all use these builders:

    1. **ComponentFactory** (core/component_factory.py) - RECOMMENDED
       - Calls these builders internally
       - Used by SpellChecker.__init__()

    2. **DI Factory Functions** (core/factories/*.py)
       - Wrap these builders for ServiceContainer
       - Return Callable[[ServiceContainer], T]

    3. **AlgorithmFactory** (algorithms/factory.py)
       - Standalone algorithm creation
       - Similar logic but takes individual configs

Builder Functions:
    - build_symspell(): Create SymSpell with consistent configuration
    - build_ngram_context_checker(): Create NgramContextChecker
    - build_suggestion_strategy(): Create CompositeSuggestionStrategy
    - build_context_validation_strategies(): Create validation strategy list

See Also:
    - core/component_factory.py: ComponentFactory
    - core/di/container.py: ServiceContainer
    - algorithms/factory.py: AlgorithmFactory
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms import NgramContextChecker, SymSpell
    from myspellchecker.algorithms.ranker import SuggestionRanker
    from myspellchecker.algorithms.semantic_checker import SemanticChecker
    from myspellchecker.algorithms.suggestion_strategy import SuggestionStrategy
    from myspellchecker.algorithms.viterbi import ViterbiTagger
    from myspellchecker.core.config import SpellCheckerConfig
    from myspellchecker.core.homophones import HomophoneChecker
    from myspellchecker.core.validation_strategies import ValidationStrategy
    from myspellchecker.grammar.engine import SyntacticRuleChecker
    from myspellchecker.providers import DictionaryProvider
    from myspellchecker.segmenters import Segmenter
    from myspellchecker.text.phonetic import PhoneticHasher
    from myspellchecker.text.tone import ToneDisambiguator

logger = get_logger(__name__)


# =============================================================================
# SymSpell Builder
# =============================================================================


def build_symspell(
    provider: DictionaryProvider,
    config: SpellCheckerConfig,
    phonetic_hasher: PhoneticHasher | None = None,
    ranker: SuggestionRanker | None = None,
    build_index: bool = True,
    segmenter: Segmenter | None = None,
) -> SymSpell:
    """
    Build a SymSpell instance with consistent configuration.

    This is the canonical SymSpell creation logic used by both
    DI factories and ComponentFactory.

    Args:
        provider: DictionaryProvider for dictionary data.
        config: SpellCheckerConfig with symspell and phonetic settings.
        phonetic_hasher: Optional PhoneticHasher for phonetic suggestions.
        ranker: Optional base ranker (created from config if None).
        build_index: Whether to build the index after creation.

    Returns:
        Configured SymSpell instance.

    Example:
        >>> from myspellchecker.core.factories.builders import build_symspell
        >>> symspell = build_symspell(provider, config, phonetic_hasher)
    """
    from myspellchecker.algorithms import SymSpell
    from myspellchecker.core.factories.ranker_factory import create_base_ranker

    # Use provided ranker or create from config
    if ranker is None:
        ranker = create_base_ranker(config.ranker)

    # Use data-driven frequency denominator when config has default value
    freq_denom = config.symspell.frequency_denominator
    if freq_denom == 10000.0:
        computed = SymSpell.compute_frequency_denominator(provider, level="syllable")
        if isinstance(computed, (int, float)) and computed != 10000.0:
            freq_denom = computed
            logger.debug("Using data-driven frequency_denominator: %.1f", freq_denom)

    symspell = SymSpell(
        provider=provider,
        max_edit_distance=config.max_edit_distance,
        prefix_length=config.symspell.prefix_length,
        count_threshold=config.symspell.count_threshold,
        phonetic_hasher=phonetic_hasher,
        max_word_length=config.symspell.max_word_length,
        compound_lookup_count=config.symspell.compound_lookup_count,
        beam_width=config.symspell.beam_width,
        compound_max_suggestions=config.symspell.compound_max_suggestions,
        damerau_cache_size=config.symspell.damerau_cache_size,
        frequency_denominator=freq_denom,
        phonetic_bonus_weight=config.symspell.phonetic_bonus_weight,
        use_weighted_distance=config.symspell.use_weighted_distance,
        use_syllable_distance=config.symspell.use_syllable_distance,
        max_deletes_per_term=config.symspell.max_deletes_per_term,
        syllable_bonus_weight=config.symspell.syllable_bonus_weight,
        weighted_distance_bonus_weight=(config.symspell.weighted_distance_bonus_weight),
        ranker=ranker,
        phonetic_bypass_threshold=(config.phonetic.phonetic_bypass_threshold),
        phonetic_extra_distance=(config.phonetic.phonetic_extra_distance),
        syllable_segmenter=segmenter,
        config=config.symspell,
    )

    if build_index:
        symspell.build_index(["syllable", "word"])

    return symspell


# =============================================================================
# NgramContextChecker Builder
# =============================================================================


def build_ngram_context_checker(
    provider: DictionaryProvider,
    config: SpellCheckerConfig,
    symspell: SymSpell | None = None,
    pos_unigram_probs: dict[str, float] | None = None,
    pos_bigram_probs: dict[tuple[str, str], float] | None = None,
) -> NgramContextChecker | None:
    """
    Build an NgramContextChecker instance with consistent configuration.

    This is the canonical NgramContextChecker creation logic used by both
    DI factories and ComponentFactory.

    Args:
        provider: DictionaryProvider for N-gram probabilities.
        config: SpellCheckerConfig with ngram_context settings.
        symspell: Optional SymSpell for generating suggestions.
        pos_unigram_probs: POS unigram probabilities (loaded from provider if None).
        pos_bigram_probs: POS bigram probabilities (loaded from provider if None).

    Returns:
        NgramContextChecker instance, or None if context checking is disabled.

    Example:
        >>> checker = build_ngram_context_checker(provider, config, symspell)
    """
    if not config.use_context_checker:
        return None

    from myspellchecker.algorithms import NgramContextChecker

    # Load POS probabilities from provider if not provided
    if pos_unigram_probs is None:
        try:
            pos_unigram_probs = provider.get_pos_unigram_probabilities()
        except (OSError, RuntimeError, KeyError, ValueError):
            pos_unigram_probs = {}
    if pos_bigram_probs is None:
        try:
            pos_bigram_probs = provider.get_pos_bigram_probabilities()
        except (OSError, RuntimeError, KeyError, ValueError):
            pos_bigram_probs = {}

    return NgramContextChecker(
        provider=provider,
        config=config.ngram_context,
        symspell=symspell,
        pos_unigram_probs=pos_unigram_probs,
        pos_bigram_probs=pos_bigram_probs,
    )


# =============================================================================
# Suggestion Strategy Builder
# =============================================================================


def build_suggestion_strategy(
    symspell: SymSpell,
    provider: DictionaryProvider,
    config: SpellCheckerConfig,
    context_checker: NgramContextChecker | None = None,
    compound_resolver: Any | None = None,
    reduplication_engine: Any | None = None,
) -> SuggestionStrategy | None:
    """
    Build a CompositeSuggestionStrategy with consistent configuration.

    Creates a composite strategy that aggregates suggestions from multiple
    sources (SymSpell, morphology, compound, morpheme) and applies unified ranking.
    If context checking is enabled, wraps in ContextSuggestionStrategy.

    This is the canonical suggestion strategy creation logic used by both
    DI factories and ComponentFactory.

    Args:
        symspell: SymSpell instance for base suggestions.
        provider: DictionaryProvider for dictionary lookups.
        config: SpellCheckerConfig with strategy settings.
        context_checker: Optional NgramContextChecker for context-aware suggestions.
        compound_resolver: Optional CompoundResolver for morpheme-level suggestions.
        reduplication_engine: Optional ReduplicationEngine for morpheme-level suggestions.

    Returns:
        SuggestionStrategy instance, or None if symspell is None.

    Example:
        >>> strategy = build_suggestion_strategy(symspell, provider, config)
    """
    if symspell is None:
        return None

    from myspellchecker.algorithms.ranker import UnifiedRanker
    from myspellchecker.algorithms.suggestion_strategy import (
        CompositeSuggestionStrategy,
        CompoundSuggestionStrategy,
        ContextSuggestionStrategy,
        MorphologySuggestionStrategy,
        SymSpellSuggestionStrategy,
    )

    # Build dictionary check function
    def dictionary_check(word: str) -> bool:
        return provider.is_valid_word(word) or provider.is_valid_syllable(word)

    # Create strategies list
    strategies: list[Any] = []

    # 1. SymSpell strategy (always included as base)
    symspell_strategy = SymSpellSuggestionStrategy(
        symspell=symspell,
        max_suggestions=config.max_suggestions * 2,  # Fetch more for filtering
        max_edit_distance=config.max_edit_distance,
        use_phonetic=config.use_phonetic,
        validation_level="word",
    )
    strategies.append(symspell_strategy)

    # 2. Morphology strategy (for OOV recovery)
    morphology_strategy = MorphologySuggestionStrategy(
        symspell=symspell,
        dictionary_check=dictionary_check,
        max_suggestions=3,  # Limit morphological suggestions
        use_phonetic=config.use_phonetic,
        allow_extended_myanmar=config.validation.allow_extended_myanmar,
    )
    strategies.append(morphology_strategy)

    # 3. Compound strategy (for word splitting/joining)
    compound_strategy = CompoundSuggestionStrategy(
        symspell=symspell,
        max_suggestions=3,  # Limit compound suggestions
        max_edit_distance=config.max_edit_distance,
        include_spaced_variants=True,
    )
    strategies.append(compound_strategy)

    # 4. Morpheme-level strategy (for compound/reduplication typo correction)
    if compound_resolver is not None or reduplication_engine is not None:
        from myspellchecker.algorithms.morpheme_suggestion_strategy import (
            MorphemeSuggestionStrategy,
        )

        morpheme_strategy = MorphemeSuggestionStrategy(
            compound_resolver=compound_resolver,
            reduplication_engine=reduplication_engine,
            symspell=symspell,
            dictionary_check=dictionary_check,
            max_suggestions=3,
        )
        strategies.append(morpheme_strategy)

    # 5. Medial swap strategy (for ျ↔ြ, ှ insertion — #1 error type)
    from myspellchecker.algorithms.medial_swap_strategy import (
        MedialSwapSuggestionStrategy,
    )

    medial_swap_strategy = MedialSwapSuggestionStrategy(
        dictionary_check=dictionary_check,
        get_frequency=lambda w: provider.get_word_frequency(w),
        max_suggestions=3,
    )
    strategies.append(medial_swap_strategy)

    # Unified pipeline always uses UnifiedRanker
    ranker = UnifiedRanker(ranker_config=config.ranker)

    # Create base composite strategy
    # Use a higher internal cap to give rerankers more candidates to work with.
    # The final user-facing truncation happens later in the pipeline.
    internal_max = max(config.max_suggestions, 8)
    base_strategy: SuggestionStrategy = CompositeSuggestionStrategy(
        strategies=strategies,
        ranker=ranker,
        max_suggestions=internal_max,
        deduplicate=True,
    )

    logger.debug(f"Created CompositeSuggestionStrategy with {len(strategies)} strategies")

    # Wrap with ContextSuggestionStrategy if context checking is enabled
    if config.use_context_checker:
        if context_checker is not None:
            ngram_config = config.ngram_context
            base_strategy = ContextSuggestionStrategy(
                context_checker=context_checker,
                base_strategy=base_strategy,
                max_suggestions=config.max_suggestions,
                # Context reranking weights (from config):
                left_weight=ngram_config.rerank_left_weight,
                right_weight=ngram_config.rerank_right_weight,
            )
            logger.debug(
                f"Wrapped strategy with ContextSuggestionStrategy "
                f"(left_weight={ngram_config.rerank_left_weight}, "
                f"right_weight={ngram_config.rerank_right_weight})"
            )
        else:
            logger.warning(
                "Context checking is enabled (use_context_checker=True) but "
                "context_checker is None. Context-aware suggestion ranking will be "
                "disabled. Check that N-gram data is available in the database."
            )

    return base_strategy


# =============================================================================
# Context Validation Strategies Builder
# =============================================================================


def build_context_validation_strategies(
    config: SpellCheckerConfig,
    provider: DictionaryProvider,
    tone_disambiguator: ToneDisambiguator | None = None,
    syntactic_rule_checker: SyntacticRuleChecker | None = None,
    viterbi_tagger: ViterbiTagger | None = None,
    context_checker: NgramContextChecker | None = None,
    homophone_checker: HomophoneChecker | None = None,
    semantic_checker: SemanticChecker | None = None,
    pos_disambiguator: Any | None = None,
    symspell: SymSpell | None = None,
) -> list[ValidationStrategy]:
    """
    Build context validation strategies with consistent priority ordering.

    Creates a list of validation strategies based on configuration and
    available dependencies. Strategies are ordered by priority:

    1. ToneValidationStrategy (priority 10)
    2. OrthographyValidationStrategy (priority 15)
    3. SyntacticValidationStrategy (priority 20)
    4. SyllableWindowOOVStrategy (priority 22, needs symspell)
    5. HiddenCompoundStrategy (priority 23)
    6. StatisticalConfusableStrategy (priority 24)
    7. BrokenCompoundStrategy (priority 25)
    8. POSSequenceValidationStrategy (priority 30)
    9. QuestionStructureValidationStrategy (priority 40)
    10. HomophoneValidationStrategy (priority 45)
    11. ConfusableCompoundClassifierStrategy (priority 47)
    12. ConfusableSemanticStrategy (priority 48)
    13. NgramContextValidationStrategy (priority 50)
    14. SemanticValidationStrategy (priority 70)

    This is the canonical strategy building logic used by both
    DI factories and ComponentFactory.

    Args:
        config: SpellCheckerConfig with validation settings.
        provider: DictionaryProvider for lookups.
        tone_disambiguator: Optional ToneDisambiguator.
        syntactic_rule_checker: Optional SyntacticRuleChecker.
        viterbi_tagger: Optional ViterbiTagger for POS tagging.
        context_checker: Optional NgramContextChecker.
        homophone_checker: Optional HomophoneChecker.
        semantic_checker: Optional SemanticChecker.
        pos_disambiguator: Optional POSDisambiguator.

    Returns:
        List of ValidationStrategy instances ordered by priority.

    Example:
        >>> strategies = build_context_validation_strategies(
        ...     config, provider, tone_disambiguator=tone_dis
        ... )
    """
    from myspellchecker.core.validation_strategies import (
        BrokenCompoundStrategy,
        ConfusableSemanticStrategy,
        HomophoneValidationStrategy,
        NgramContextValidationStrategy,
        OrthographyValidationStrategy,
        POSSequenceValidationStrategy,
        QuestionStructureValidationStrategy,
        SemanticValidationStrategy,
        SyntacticValidationStrategy,
        ToneValidationStrategy,
    )

    strategies: list[ValidationStrategy] = []
    validation_config = config.validation

    # Priority 10: Tone Validation
    if tone_disambiguator:
        strategies.append(
            ToneValidationStrategy(
                tone_disambiguator=tone_disambiguator,
                confidence_threshold=validation_config.tone_validation_confidence,
                provider=provider,
            )
        )
        logger.debug("Added ToneValidationStrategy (priority 10)")

    # Priority 15: Orthography Validation (medial order and compatibility)
    strategies.append(OrthographyValidationStrategy(provider=provider))
    logger.debug("Added OrthographyValidationStrategy (priority 15)")

    # Priority 20: Syntactic Validation
    if syntactic_rule_checker:
        strategies.append(
            SyntacticValidationStrategy(
                syntactic_rule_checker=syntactic_rule_checker,
                confidence=validation_config.syntactic_validation_confidence,
            )
        )
        logger.debug("Added SyntacticValidationStrategy (priority 20)")

    # Priority 16: Visarga/Aukmyit/Asat Corrections
    from myspellchecker.core.validation_strategies.visarga_strategy import (
        VisargaStrategy,
    )

    strategies.append(VisargaStrategy(provider=provider))
    logger.debug("Added VisargaStrategy (priority 16)")

    # Priority 18: Loan Word Transliteration Detection
    if getattr(validation_config, "use_loan_word_detection", True):
        from myspellchecker.core.validation_strategies.loan_word_strategy import (
            LoanWordValidationStrategy,
        )

        strategies.append(
            LoanWordValidationStrategy(
                provider=provider,
            )
        )
        logger.debug("Added LoanWordValidationStrategy (priority 18)")

    # Priority 25: Broken Compound Detection
    if validation_config.use_broken_compound_detection:
        strategies.append(
            BrokenCompoundStrategy(
                provider=provider,
                rare_threshold=validation_config.broken_compound_rare_threshold,
                compound_min_frequency=validation_config.broken_compound_min_frequency,
                compound_ratio=validation_config.broken_compound_ratio,
                confidence=validation_config.broken_compound_confidence,
            )
        )
        logger.debug("Added BrokenCompoundStrategy (priority 25)")

    # Priority 22: Syllable-Window OOV Detection
    # Runs first in the structural phase to surface multi-syllable compound
    # typos that the segmenter over-split into individually valid syllables.
    # Does not populate existing_errors so HiddenCompound (23) can still
    # fire at overlapping positions with its own error type.
    if validation_config.use_syllable_window_oov and symspell is not None:
        from myspellchecker.core.validation_strategies.syllable_window_oov_strategy import (
            SyllableWindowOOVStrategy,
        )

        strategies.append(
            SyllableWindowOOVStrategy(
                provider=provider,
                symspell=symspell,
                enabled=True,
                window_sizes=tuple(validation_config.syllable_window_sizes),
                min_frequency=validation_config.syllable_window_min_frequency,
                confidence_floor=validation_config.syllable_window_confidence_floor,
                max_edit_distance=validation_config.syllable_window_max_edit_distance,
                require_typo_prone=validation_config.syllable_window_require_typo_prone,
                skip_names=validation_config.syllable_window_skip_names,
                require_valid_source_words=(
                    validation_config.syllable_window_require_valid_source_words
                ),
            )
        )
        logger.debug("Added SyllableWindowOOVStrategy (priority 22)")

    # Priority 22: Tone Safety Net (missing / extra trailing tone marks).
    # Probes trailing tone insert / delete candidates against the
    # dictionary to recover real-word confusions the
    # ``mined_confusable_pair`` strategy misses when its freq_ratio gate
    # filters out rare-form pairs.
    if validation_config.use_tone_safety_net:
        from myspellchecker.core.validation_strategies.tone_safety_net_strategy import (
            ToneSafetyNetStrategy,
        )

        strategies.append(
            ToneSafetyNetStrategy(
                provider=provider,
                enabled=True,
                min_frequency=validation_config.tone_safety_net_min_frequency,
                freq_ratio=validation_config.tone_safety_net_freq_ratio,
                skip_above_freq=validation_config.tone_safety_net_skip_above_freq,
                confidence=validation_config.tone_safety_net_confidence,
            )
        )
        logger.debug("Added ToneSafetyNetStrategy (priority 22)")

    # Priority 23: Pre-Segmenter Raw-Token SymSpell Probe
    # Runs SymSpell.lookup(raw_token, level='word') on unsegmented Myanmar spans
    # BEFORE any segmentation-dependent strategy looks at the sentence. Catches
    # compound typos the segmenter would otherwise fragment into piecewise-valid
    # subtokens.
    # Registered before HiddenCompoundStrategy so that on priority ties the
    # probe fires first and HiddenCompound sees the already-claimed position.
    if validation_config.use_pre_segmenter_raw_probe and symspell is not None:
        from myspellchecker.core.validation_strategies.pre_segmenter_raw_probe_strategy import (
            PreSegmenterRawProbeStrategy,
        )

        strategies.append(
            PreSegmenterRawProbeStrategy(
                symspell=symspell,
                provider=provider,
                enabled=True,
                max_edit_distance=validation_config.pre_segmenter_raw_probe_max_ed,
                min_frequency=validation_config.pre_segmenter_raw_probe_min_freq,
                max_token_length=validation_config.pre_segmenter_raw_probe_max_length,
                max_length_diff=validation_config.pre_segmenter_raw_probe_max_length_diff,
                confidence=validation_config.pre_segmenter_raw_probe_confidence,
            )
        )
        logger.debug("Added PreSegmenterRawProbeStrategy (priority 23)")

    # Priority 23: Hidden Compound Typo Detection
    # Runs before StatisticalConfusable (24) and BrokenCompound (25) in the
    # structural phase (priority <= 25 survives the fast-path cutoff).
    if validation_config.use_hidden_compound_detection:
        from myspellchecker.core.validation_strategies.hidden_compound_strategy import (
            HiddenCompoundStrategy,
        )
        from myspellchecker.text.phonetic import PhoneticHasher

        strategies.append(
            HiddenCompoundStrategy(
                provider=provider,
                hasher=PhoneticHasher(),
                enabled=True,
                max_token_syllables=validation_config.hidden_compound_max_token_syllables,
                max_variants_per_token=validation_config.hidden_compound_max_variants_per_token,
                compound_min_frequency=validation_config.hidden_compound_min_frequency,
                confidence_floor=validation_config.hidden_compound_confidence_floor,
                enable_trigram_lookahead=validation_config.hidden_compound_enable_trigram_lookahead,
                variant_cache_size=validation_config.hidden_compound_variant_cache_size,
                require_typo_prone_chars=validation_config.hidden_compound_require_typo_prone_chars,
                curated_only=validation_config.hidden_compound_curated_only,
            )
        )
        logger.debug("Added HiddenCompoundStrategy (priority 23)")

    # Priority 30: POS Sequence Validation
    if viterbi_tagger and validation_config.use_pos_sequence:
        strategies.append(
            POSSequenceValidationStrategy(
                viterbi_tagger=viterbi_tagger,
                confidence=validation_config.pos_sequence_confidence,
                pos_disambiguator=pos_disambiguator,
                provider=provider,
            )
        )
        logger.debug("Added POSSequenceValidationStrategy (priority 30)")

    # Priority 40: Question Structure Validation (always added)
    strategies.append(
        QuestionStructureValidationStrategy(
            confidence=validation_config.question_structure_confidence,
        )
    )
    logger.debug("Added QuestionStructureValidationStrategy (priority 40)")

    # Priority 45: Homophone Validation
    freq_guards = config.frequency_guards
    if homophone_checker and validation_config.use_homophone_detection:
        strategies.append(
            HomophoneValidationStrategy(
                homophone_checker=homophone_checker,
                provider=provider,
                context_checker=context_checker,
                confidence=validation_config.homophone_confidence,
            )
        )
        logger.debug("Added HomophoneValidationStrategy (priority 45)")

    # Priority 24: Statistical Confusable Gate (bigram ratio)
    if validation_config.use_statistical_confusable_gate:
        from myspellchecker.core.detection_rules import load_confusable_pairs
        from myspellchecker.core.validation_strategies.statistical_confusable_strategy import (
            StatisticalConfusableStrategy,
        )

        curated_pairs, near_synonym_pairs = load_confusable_pairs()
        merged_map: dict[str, set[str]] = {}
        for src in (curated_pairs, near_synonym_pairs):
            for word, variants in src.items():
                merged_map.setdefault(word, set()).update(variants)

        # Share the curated homophone map so StatisticalConfusableStrategy
        # can promote detections that also appear there.
        homophone_pairs_map = homophone_checker.homophone_map if homophone_checker else None

        if merged_map:
            strategies.append(
                StatisticalConfusableStrategy(
                    provider=provider,
                    confusable_map=merged_map,
                    threshold=validation_config.statistical_confusable_threshold,
                    homophone_map=homophone_pairs_map,
                )
            )
            logger.debug("Added StatisticalConfusableStrategy (priority 24)")

    # Priority 47b: MLP-based Confusable/Compound Classifier
    classifier_model_path = getattr(
        validation_config,
        "confusable_compound_classifier_path",
        None,
    )
    if classifier_model_path:
        import pathlib as _pl

        _classifier_exists = _pl.Path(classifier_model_path).exists()
    else:
        _classifier_exists = False
    if _classifier_exists:
        from myspellchecker.core.validation_strategies import (
            confusable_compound_classifier_strategy as _cc_mod,
        )

        ConfusableCompoundClassifierStrategy = _cc_mod.ConfusableCompoundClassifierStrategy

        strategies.append(
            ConfusableCompoundClassifierStrategy(
                provider=provider,
                model_path=str(classifier_model_path),
                threshold=getattr(
                    validation_config,
                    "confusable_compound_classifier_threshold",
                    0.5,
                ),
            )
        )
        logger.debug("Added ConfusableCompoundClassifierStrategy (priority 47)")

    # Priority 46: MLM span-mask candidate generator.
    # Wraps semantic-v2.4 as a candidate generator for real-word confusions,
    # the highest-ceiling lever among the no-train prototypes per the
    # 2026-04-18 probe (+297 TP simulated at 8% position-level FP). Must run
    # before ConfusableCompoundClassifier so the MLM's proposal is seen by
    # the classifier's downstream decision.
    if semantic_checker and validation_config.use_mlm_span_mask_candgen:
        from myspellchecker.core.validation_strategies.mlm_span_mask_candgen_strategy import (
            MLMSpanMaskCandGenStrategy,
        )

        strategies.append(
            MLMSpanMaskCandGenStrategy(
                semantic_checker=semantic_checker,
                provider=provider,
                enabled=True,
                top_k=validation_config.mlm_candgen_top_k,
                margin=validation_config.mlm_candgen_margin,
                max_edit_distance=validation_config.mlm_candgen_max_ed,
                skip_above_freq=validation_config.mlm_candgen_skip_above_freq,
                min_token_length=validation_config.mlm_candgen_min_token_length,
                confidence=validation_config.mlm_candgen_confidence,
            )
        )
        logger.debug("Added MLMSpanMaskCandGenStrategy (priority 46)")

    # Priority 48: Confusable Semantic Detection (MLM-enhanced)
    if semantic_checker and validation_config.use_confusable_semantic:
        # Load curated confusable pairs from YAML
        from myspellchecker.core.detection_rules import load_confusable_pairs

        curated_pairs, near_synonym_pairs = load_confusable_pairs()

        strategies.append(
            ConfusableSemanticStrategy(
                semantic_checker=semantic_checker,
                provider=provider,
                confidence=validation_config.confusable_semantic_confidence,
                top_k=validation_config.confusable_semantic_top_k,
                logit_diff_threshold=validation_config.confusable_semantic_logit_diff,
                logit_diff_threshold_medial=validation_config.confusable_semantic_logit_diff_medial,
                logit_diff_threshold_current_in_topk=validation_config.confusable_semantic_logit_diff_current_in_topk,
                high_freq_threshold=freq_guards.semantic_high_freq_protection,
                high_freq_logit_diff=validation_config.confusable_semantic_high_freq_logit_diff,
                freq_ratio_penalty_high=validation_config.confusable_semantic_freq_ratio_penalty_high,
                freq_ratio_penalty_mid=validation_config.confusable_semantic_freq_ratio_penalty_mid,
                visarga_penalty=validation_config.confusable_semantic_visarga_penalty,
                sentence_final_penalty=validation_config.confusable_semantic_sentence_final_penalty,
                homophone_map=homophone_checker.homophone_map if homophone_checker else None,
                max_threshold=validation_config.confusable_semantic_max_threshold,
                reverse_ratio_min_freq=validation_config.confusable_semantic_reverse_ratio_min_freq,
                visarga_high_freq_hard_block=validation_config.confusable_semantic_visarga_high_freq_hard_block,
                curated_pairs=curated_pairs,
                near_synonym_pairs=near_synonym_pairs,
            )
        )
        logger.debug("Added ConfusableSemanticStrategy (priority 48)")

    # Priority 49: Mined Confusable Pair Strategy
    # Flags real-word confusables via mined ed-1 pair list with MLM margin gate.
    if validation_config.use_mined_confusable_pair and semantic_checker is not None:
        from myspellchecker.core.validation_strategies.mined_confusable_pair_strategy import (
            MinedConfusablePairStrategy,
        )

        strategies.append(
            MinedConfusablePairStrategy(
                provider=provider,
                semantic_checker=semantic_checker,
                enabled=True,
                yaml_path=validation_config.mined_pair_yaml_path,
                low_freq_min=validation_config.mined_pair_low_freq_min,
                freq_ratio=validation_config.mined_pair_freq_ratio,
                mlm_margin=validation_config.mined_pair_mlm_margin,
                backend=validation_config.mined_pair_backend,
                classifier_path=validation_config.mined_pair_classifier_path,
            )
        )
        logger.debug(
            "Added MinedConfusablePairStrategy (priority 49, backend=%s)",
            validation_config.mined_pair_backend,
        )

    # Priority 50: N-gram Context Validation
    if context_checker and config.use_context_checker and validation_config.use_ngram_context:
        strategies.append(
            NgramContextValidationStrategy(
                context_checker=context_checker,
                provider=provider,
                confidence_high=validation_config.context_error_confidence_high,
                confidence_low=validation_config.context_error_confidence_low,
                max_suggestions=config.max_suggestions,
                edit_distance=config.ngram_context.edit_distance,
            )
        )
        logger.debug("Added NgramContextValidationStrategy (priority 50)")

    # Priority 70: Semantic Validation
    # Always add when semantic_checker is available — animacy detection runs
    # regardless of proactive scanning setting.
    if semantic_checker:
        strategies.append(
            SemanticValidationStrategy(
                semantic_checker=semantic_checker,
                provider=provider,
                use_proactive_scanning=config.semantic.use_proactive_scanning,
                proactive_confidence_threshold=config.semantic.proactive_confidence_threshold,
                min_word_length=2,
            )
        )
        logger.debug("Added SemanticValidationStrategy (priority 70)")

    # Priority 80: ByT5 safety net (last-chance structural-typo rescue).
    # Fires only when every other strategy declined; gated by MLM margin.
    if validation_config.use_byt5_safety_net and validation_config.byt5_safety_net_model_path:
        from myspellchecker.core.validation_strategies.byt5_safety_net_strategy import (
            ByT5SafetyNetStrategy,
        )

        strategies.append(
            ByT5SafetyNetStrategy(
                provider=provider,
                model_path=validation_config.byt5_safety_net_model_path,
                semantic_checker=semantic_checker,
                enabled=True,
                mlm_gate_margin=validation_config.byt5_safety_net_mlm_gate_margin,
                min_typo_prone_chars=validation_config.byt5_safety_net_min_typo_prone_chars,
                max_sentence_chars=validation_config.byt5_safety_net_max_sentence_chars,
                confidence=validation_config.byt5_safety_net_confidence,
            )
        )
        logger.debug("Added ByT5SafetyNetStrategy (priority 80)")

    logger.info(
        f"Built {len(strategies)} validation strategies: "
        f"{[s.__class__.__name__ for s in strategies]}"
    )

    return strategies
