"""
Factory functions for optional services.

These services are not required for basic spell checking but enable
advanced validation strategies like tone disambiguation, syntactic
rule checking, and semantic validation.

All factories follow the pattern of checking if dependencies are available
and returning None if the service cannot be created (graceful degradation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from myspellchecker.core.di.service_names import SERVICE_PROVIDER, SERVICE_SEGMENTER
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.semantic_checker import SemanticChecker
    from myspellchecker.algorithms.viterbi import ViterbiTagger
    from myspellchecker.core.di.container import ServiceContainer
    from myspellchecker.core.homophones import HomophoneChecker
    from myspellchecker.grammar.engine import SyntacticRuleChecker
    from myspellchecker.text.ner import NameHeuristic
    from myspellchecker.text.ner_model import NERModel
    from myspellchecker.text.tone import ToneDisambiguator

    # Type aliases for factory functions
    ToneDisambiguatorFactory = Callable[[ServiceContainer], ToneDisambiguator | None]
    SyntacticRuleCheckerFactory = Callable[[ServiceContainer], SyntacticRuleChecker | None]
    ViterbiTaggerFactory = Callable[[ServiceContainer], ViterbiTagger | None]
    HomophoneCheckerFactory = Callable[[ServiceContainer], HomophoneChecker | None]
    SemanticCheckerFactory = Callable[[ServiceContainer], SemanticChecker | None]
    NameHeuristicFactory = Callable[[ServiceContainer], NameHeuristic | None]
    NERModelFactory = Callable[[ServiceContainer], NERModel | None]

logger = get_logger(__name__)


# =============================================================================
# Tone Disambiguator Factory
# =============================================================================


def create_tone_disambiguator(container: "ServiceContainer") -> "ToneDisambiguator" | None:
    """
    Create ToneDisambiguator for tone mark validation.

    The ToneDisambiguator helps identify and correct tone mark errors
    based on context (e.g., distinguishing between ့ and း).

    Args:
        container: Service container for configuration access.

    Returns:
        ToneDisambiguator instance, or None if creation fails.
    """
    try:
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
    except ImportError as e:
        logger.debug("ToneDisambiguator dependency not available: %s", e)
        return None
    except Exception as e:
        logger.warning("ToneDisambiguator init failed: %s: %s", type(e).__name__, e)
        return None


def create_tone_disambiguator_factory() -> ToneDisambiguatorFactory:
    """Create factory function for ToneDisambiguator."""
    return create_tone_disambiguator


# =============================================================================
# Syntactic Rule Checker Factory
# =============================================================================


def create_syntactic_rule_checker(
    container: "ServiceContainer",
) -> "SyntacticRuleChecker" | None:
    """
    Create SyntacticRuleChecker for grammar rule validation.

    The SyntacticRuleChecker validates Myanmar grammar rules including
    particle placement, POS sequence patterns, and sentence structure.

    Args:
        container: Service container for dependency resolution.

    Returns:
        SyntacticRuleChecker instance, or None if creation fails.
    """
    try:
        from myspellchecker.grammar.engine import SyntacticRuleChecker

        provider = container.get(SERVICE_PROVIDER)
        config = container.get_config()
        grammar_config = getattr(config, "grammar_engine", None)
        return SyntacticRuleChecker(provider, grammar_config=grammar_config)
    except ImportError as e:
        logger.debug("SyntacticRuleChecker dependency not available: %s", e)
        return None
    except Exception as e:
        logger.warning("SyntacticRuleChecker init failed: %s: %s", type(e).__name__, e)
        return None


def create_syntactic_rule_checker_factory() -> SyntacticRuleCheckerFactory:
    """Create factory function for SyntacticRuleChecker."""
    return create_syntactic_rule_checker


# =============================================================================
# Viterbi Tagger Factory
# =============================================================================


def create_viterbi_tagger(container: ServiceContainer) -> ViterbiTagger | None:
    """
    Create ViterbiTagger for POS tagging.

    The ViterbiTagger uses HMM-based sequence tagging with trigram
    probabilities for context-aware POS tagging.

    Args:
        container: Service container for dependency resolution.

    Returns:
        ViterbiTagger instance, or None if creation fails.
    """
    try:
        from myspellchecker.algorithms.viterbi import ViterbiTagger

        provider = container.get(SERVICE_PROVIDER)
        if provider is None:
            logger.debug("ViterbiTagger requires provider, but none available")
            return None

        # Load probability tables from provider
        bigram_probs: dict[tuple[str, str], float] = {}
        trigram_probs: dict[tuple[str, str, str], float] = {}
        unigram_probs: dict[str, float] = {}

        if hasattr(provider, "get_pos_bigram_probabilities"):
            bigram_probs = provider.get_pos_bigram_probabilities() or {}
        if hasattr(provider, "get_pos_trigram_probabilities"):
            trigram_probs = provider.get_pos_trigram_probabilities() or {}
        if hasattr(provider, "get_pos_unigram_probabilities"):
            unigram_probs = provider.get_pos_unigram_probabilities() or {}

        # Only create if we have probability data
        if not bigram_probs and not trigram_probs:
            logger.debug("ViterbiTagger requires probability tables, but none available")
            return None

        config = container.get_config()
        pos_config = getattr(config, "pos_tagger", None)

        beam_width = 10
        emission_weight = 1.2
        if pos_config:
            beam_width = getattr(pos_config, "beam_width", 10)
            emission_weight = getattr(pos_config, "emission_weight", 1.2)

        return ViterbiTagger(
            provider=provider,
            pos_bigram_probs=bigram_probs,
            pos_trigram_probs=trigram_probs,
            pos_unigram_probs=unigram_probs,
            beam_width=beam_width,
            emission_weight=emission_weight,
            config=pos_config,
        )
    except ImportError as e:
        logger.debug("ViterbiTagger dependency not available: %s", e)
        return None
    except Exception as e:
        logger.warning("ViterbiTagger init failed: %s: %s", type(e).__name__, e)
        return None


def create_viterbi_tagger_factory() -> ViterbiTaggerFactory:
    """Create factory function for ViterbiTagger."""
    return create_viterbi_tagger


# =============================================================================
# Homophone Checker Factory
# =============================================================================


def create_homophone_checker(container: "ServiceContainer") -> "HomophoneChecker" | None:
    """
    Create HomophoneChecker for homophone detection.

    The HomophoneChecker identifies potential homophone confusions
    (words that sound similar but have different meanings/spellings).
    Merges YAML curated pairs with DB confusable_pairs (corpus-mined).

    Args:
        container: Service container for provider lookup.

    Returns:
        HomophoneChecker instance, or None if creation fails.
    """
    try:
        from myspellchecker.core.homophones import HomophoneChecker

        provider = container.get(SERVICE_PROVIDER)
        return HomophoneChecker(provider=provider)
    except ImportError as e:
        logger.debug("HomophoneChecker dependency not available: %s", e)
        return None
    except Exception as e:
        logger.warning("HomophoneChecker init failed: %s: %s", type(e).__name__, e)
        return None


def create_homophone_checker_factory() -> HomophoneCheckerFactory:
    """Create factory function for HomophoneChecker."""
    return create_homophone_checker


# =============================================================================
# Semantic Checker Factory
# =============================================================================


def create_semantic_checker(container: ServiceContainer) -> SemanticChecker | None:
    """
    Create SemanticChecker for AI-powered semantic validation.

    The SemanticChecker uses ONNX or PyTorch models for context-aware
    semantic validation. This is an optional heavy dependency.

    Note:
        This factory only creates the checker if:
        1. The semantic model path is configured
        2. Required dependencies (onnxruntime or torch) are available

    Args:
        container: Service container for configuration access.

    Returns:
        SemanticChecker instance, or None if not configured/available.
    """
    try:
        config = container.get_config()
        semantic_config = getattr(config, "semantic", None)

        if not semantic_config:
            logger.debug("SemanticChecker not configured (no semantic config)")
            return None

        model_path = getattr(semantic_config, "model_path", None)
        if not model_path:
            logger.debug("SemanticChecker not configured (no model_path)")
            return None

        from myspellchecker.algorithms.semantic_checker import SemanticChecker

        tokenizer_path = getattr(semantic_config, "tokenizer_path", None)
        use_pytorch = getattr(semantic_config, "use_pytorch", False)
        num_threads = getattr(semantic_config, "num_threads", 1)

        # Get allow_extended_myanmar from validation config
        validation_config = getattr(config, "validation", None)
        allow_extended = (
            getattr(validation_config, "allow_extended_myanmar", False)
            if validation_config
            else False
        )

        return SemanticChecker(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            use_pytorch=use_pytorch,
            num_threads=num_threads,
            predict_top_k=getattr(semantic_config, "predict_top_k", 5),
            check_top_k=getattr(semantic_config, "check_top_k", 10),
            allow_extended_myanmar=allow_extended,
            cache_config=getattr(config, "cache", None),
            semantic_config=semantic_config,
        )
    except ImportError as e:
        logger.debug("SemanticChecker dependency not available: %s", e)
        return None
    except Exception as e:
        logger.warning("SemanticChecker init failed: %s: %s", type(e).__name__, e)
        return None


def create_semantic_checker_factory() -> SemanticCheckerFactory:
    """Create factory function for SemanticChecker."""
    return create_semantic_checker


# =============================================================================
# Name Heuristic Factory
# =============================================================================


def create_name_heuristic(container: "ServiceContainer") -> "NameHeuristic" | None:
    """
    Create NameHeuristic for proper noun detection.

    The NameHeuristic identifies proper nouns, foreign words, and
    special tokens that should not be spell-checked.

    Args:
        container: Service container (currently unused, for future config).

    Returns:
        NameHeuristic instance, or None if creation fails.
    """
    try:
        from myspellchecker.text.ner import NameHeuristic

        return NameHeuristic()
    except ImportError as e:
        logger.debug("NameHeuristic dependency not available: %s", e)
        return None
    except Exception as e:
        logger.warning("NameHeuristic init failed: %s: %s", type(e).__name__, e)
        return None


def create_name_heuristic_factory() -> NameHeuristicFactory:
    """Create factory function for NameHeuristic."""
    return create_name_heuristic


# =============================================================================
# NER Model Factory
# =============================================================================


def create_ner_model(container: "ServiceContainer") -> "NERModel" | None:
    """
    Create NERModel based on configuration.

    If config.ner is provided and enabled, uses NERFactory to create the
    appropriate model (heuristic, transformer, or hybrid).
    If config.ner is None but config.use_ner is True, creates HeuristicNER.

    Args:
        container: Service container for configuration access.

    Returns:
        NERModel instance, or None if NER is disabled.
    """
    try:
        config = container.get_config()

        if not getattr(config, "use_ner", True):
            return None

        ner_config = getattr(config, "ner", None)
        validation_config = getattr(config, "validation", None)
        allow_extended = (
            getattr(validation_config, "allow_extended_myanmar", False)
            if validation_config
            else False
        )

        if ner_config is not None and ner_config.enabled:
            from myspellchecker.text.ner_model import NERFactory

            return NERFactory.create(ner_config, allow_extended_myanmar=allow_extended)

        # Backward compat: use_ner=True without explicit NERConfig → HeuristicNER
        from myspellchecker.text.ner_model import HeuristicNER

        segmenter = container.get(SERVICE_SEGMENTER)
        return HeuristicNER(segmenter=segmenter, allow_extended_myanmar=allow_extended)
    except ImportError as e:
        logger.debug("NER model dependency not available: %s", e)
        return None
    except Exception as e:
        logger.warning("NER model init failed: %s: %s", type(e).__name__, e)
        return None
