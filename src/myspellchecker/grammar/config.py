"""
Grammar Rule Configuration Loader.

This module provides YAML-based configuration for grammar rules,
allowing easy customization and expansion of grammatical validation rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import yaml

from myspellchecker.utils.logging_utils import get_logger
from myspellchecker.utils.singleton import Singleton

logger = get_logger(__name__)

__all__ = [
    "GrammarRuleConfig",
    "get_grammar_config",
]

# Module-level singleton for GrammarRuleConfig to avoid redundant YAML loading.
# Without this, 13 independent instantiation sites each load 15 YAML files = 195 reads.
_grammar_config_singleton: Singleton["GrammarRuleConfig"] = Singleton()


def get_grammar_config(config_path: str | None = None) -> "GrammarRuleConfig":
    """Get or create the shared GrammarRuleConfig instance.

    Uses a module-level singleton to avoid redundant YAML loading across
    the 13+ components that need grammar configuration. The first call
    creates the instance (loading 15 YAML files); subsequent calls return
    the cached instance.

    Args:
        config_path: Optional path to grammar rules YAML. Only used on
            first call (when the singleton is created). Ignored on
            subsequent calls.

    Returns:
        Shared GrammarRuleConfig instance.
    """
    already_created = GrammarRuleConfig in _grammar_config_singleton._instances
    instance = _grammar_config_singleton.get(
        GrammarRuleConfig,
        factory=lambda: GrammarRuleConfig(config_path=config_path),
    )
    if config_path is not None and already_created:
        logger.warning(
            "get_grammar_config called with config_path=%s but singleton already initialized; "
            "using cached instance (new path ignored)",
            config_path,
        )
    return instance


def _load_yaml_config(
    path: Path,
    config_name: str,
    parse_func: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any] | None:
    """Load a YAML config file with standardized error handling.

    After successful loading, the data is optionally validated against
    a matching JSON Schema (if one exists in ``schemas/``).  Validation
    issues are logged at WARNING level but never raise — this keeps
    backwards compatibility with existing YAML files that may drift
    slightly from the schema.

    Args:
        path: Path to the YAML config file.
        config_name: Human-readable name for logging messages.
        parse_func: Optional callable to process the loaded config.

    Returns:
        Loaded config dict, or None if loading failed.
    """
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config:
                # Optional schema validation — never blocks loading.
                _validate_config_schema(config, path, config_name)

                if parse_func:
                    parse_func(config)
                logger.debug(f"Loaded {config_name} from {path}")
                return cast(dict[str, Any], config)
    except (yaml.YAMLError, OSError) as e:
        logger.warning(f"Failed to load {config_name} from {path}: {e}")
    except (KeyError, ValueError, TypeError, AttributeError, IndexError) as e:
        logger.warning(f"Failed to parse {config_name} from {path}: {type(e).__name__}: {e}")
    return None


def _validate_config_schema(
    config: dict[str, Any],
    yaml_path: Path,
    config_name: str,
) -> None:
    """Run optional JSON Schema validation on loaded YAML data.

    Looks for a schema file matching the YAML filename (e.g.
    ``particles.yaml`` -> ``schemas/particles.schema.json``).  If found,
    validates *config* against it and logs any issues as warnings.

    This function never raises; all errors are caught and logged so that
    YAML loading is never interrupted.
    """
    try:
        from myspellchecker.utils.yaml_schema_validator import (
            get_schema_path_for_yaml,
            validate_yaml_against_schema,
        )

        schema_path = get_schema_path_for_yaml(yaml_path.name)
        if schema_path is None:
            return

        errors = validate_yaml_against_schema(config, schema_path)
        if errors:
            logger.warning(
                "Schema validation issues in %s (%s): %s",
                config_name,
                yaml_path.name,
                "; ".join(errors),
            )
    except Exception:  # noqa: BLE001
        # Never let validation machinery break config loading.
        logger.debug("Schema validation skipped for %s due to an unexpected error", config_name)


# Default path for grammar rules YAML
DEFAULT_GRAMMAR_RULES_PATH = Path(__file__).parent.parent / "rules" / "grammar_rules.yaml"
DEFAULT_TYPO_CORRECTIONS_PATH = Path(__file__).parent.parent / "rules" / "typo_corrections.yaml"
DEFAULT_PARTICLES_PATH = Path(__file__).parent.parent / "rules" / "particles.yaml"
DEFAULT_PRONOUNS_PATH = Path(__file__).parent.parent / "rules" / "pronouns.yaml"
DEFAULT_CLASSIFIERS_PATH = Path(__file__).parent.parent / "rules" / "classifiers.yaml"
DEFAULT_REGISTER_PATH = Path(__file__).parent.parent / "rules" / "register.yaml"
DEFAULT_HOMOPHONES_PATH = Path(__file__).parent.parent / "rules" / "homophones.yaml"
DEFAULT_COMPOUNDS_PATH = Path(__file__).parent.parent / "rules" / "compounds.yaml"
DEFAULT_ASPECTS_PATH = Path(__file__).parent.parent / "rules" / "aspects.yaml"
DEFAULT_POS_INFERENCE_PATH = Path(__file__).parent.parent / "rules" / "pos_inference.yaml"
DEFAULT_AMBIGUOUS_WORDS_PATH = Path(__file__).parent.parent / "rules" / "ambiguous_words.yaml"
DEFAULT_TONE_RULES_PATH = Path(__file__).parent.parent / "rules" / "tone_rules.yaml"
DEFAULT_NEGATION_PATH = Path(__file__).parent.parent / "rules" / "negation.yaml"
DEFAULT_MORPHOLOGY_PATH = Path(__file__).parent.parent / "rules" / "morphology.yaml"
DEFAULT_MORPHOTACTICS_PATH = Path(__file__).parent.parent / "rules" / "morphotactics.yaml"


class GrammarRuleConfig:
    """
    Load and manage grammar rules from YAML configuration.

    This class provides a structured interface to grammar rules,
    supporting:
    - Particle typo mappings
    - Medial confusion patterns
    - Verb and noun particles
    - Invalid POS sequences
    - Sentence-final particles

    Attributes:
        particle_typos: Dict mapping typos to corrections with context.
        medial_confusions: Dict of medial confusion patterns keyed by pattern string.
        verb_particles: Set of valid verb-following particles.
        noun_particles: Set of valid noun-following particles.
        invalid_pos_sequences: List of invalid POS sequence rules.
        sentence_final_particles: Set of sentence-ending particles.
    """

    def __init__(
        self,
        config_path: str | None = None,
        typo_path: str | None = None,
        particles_path: str | None = None,
        pronouns_path: str | None = None,
        classifiers_path: str | None = None,
        register_path: str | None = None,
        homophones_path: str | None = None,
        compounds_path: str | None = None,
        aspects_path: str | None = None,
        pos_inference_path: str | None = None,
        ambiguous_words_path: str | None = None,
        tone_rules_path: str | None = None,
        negation_path: str | None = None,
        morphology_path: str | None = None,
    ) -> None:
        """
        Initialize the grammar rule configuration.

        Args:
            config_path: Optional path to grammar rules YAML config file.
                        If not provided, uses default path.
            typo_path: Optional path to typo corrections YAML config file.
                      If not provided, uses default path.
            particles_path: Optional path to particles YAML config file.
                          If not provided, uses default path.
            pronouns_path: Optional path to pronouns YAML config file.
                          If not provided, uses default path.
            classifiers_path: Optional path to classifiers YAML config file.
                          If not provided, uses default path.
            register_path: Optional path to register YAML config file.
                          If not provided, uses default path.
            homophones_path: Optional path to homophones YAML config file.
                          If not provided, uses default path.
            compounds_path: Optional path to compounds YAML config file.
                          If not provided, uses default path.
            aspects_path: Optional path to aspects YAML config file.
                          If not provided, uses default path.
            pos_inference_path: Optional path to pos inference YAML config file.
                          If not provided, uses default path.
            ambiguous_words_path: Optional path to ambiguous words YAML config file.
                          If not provided, uses default path.
            tone_rules_path: Optional path to tone rules YAML config file.
                          If not provided, uses default path.
            negation_path: Optional path to negation YAML config file.
                          If not provided, uses default path.
        """
        self.particle_typos: dict[str, dict[str, Any]] = {}
        self.medial_confusions: dict[str, dict[str, str]] = {}
        self.verb_particles: set[str] = set()
        self.noun_particles: set[str] = set()
        self.invalid_pos_sequences: list[dict[str, Any]] = []
        self.sentence_final_particles: set[str] = set()
        self.word_corrections: dict[str, dict[str, Any]] = {}
        self.question_particle_corrections: list[dict[str, Any]] = []
        self.particle_constraints: dict[str, dict[str, Any]] = {}
        self.particle_tags: dict[str, str] = {}
        self.particle_metadata: dict[str, dict[str, Any]] = {}
        self.pronouns: list[dict[str, Any]] = []
        self.sentence_start_constraints: list[dict[str, Any]] = []
        self.sentence_end_constraints: list[dict[str, Any]] = []
        self.classifiers: dict[str, list[dict[str, Any]]] = {}
        self.register_config: dict[str, Any] = {}
        self.homophones_map: dict[str, set[str]] = {}
        self.compounds_config: dict[str, Any] = {}
        self.aspects_config: dict[str, Any] = {}
        self.pos_inference_config: dict[str, Any] = {}
        self.ambiguous_words_map: dict[str, set[str]] = {}
        self.tone_rules_config: dict[str, Any] = {}
        self.tone_ambiguous_map: dict[str, dict[str, dict[str, Any]]] = {}
        self.tone_errors_map: dict[str, str] = {}
        self.negation_config: dict[str, Any] = {}
        self.morphology_config: dict[str, Any] = {}
        self.morphotactics_config: dict[str, Any] = {}

        # Grammar rules from grammar_rules.yaml
        self.particle_chains_valid: list[dict[str, Any]] = []
        self.particle_chains_invalid: list[dict[str, Any]] = []
        self.clause_linkage: list[dict[str, Any]] = []
        self.negation_rules_config: list[dict[str, Any]] = []
        self.classifier_rules_config: list[dict[str, Any]] = []

        # Load configuration
        self._load_config(
            config_path,
            typo_path,
            particles_path,
            pronouns_path,
            classifiers_path,
            register_path,
            homophones_path,
            compounds_path,
            aspects_path,
            pos_inference_path,
            ambiguous_words_path,
            tone_rules_path,
            negation_path,
            morphology_path,
        )

    def _load_config(
        self,
        config_path: str | None = None,
        typo_path: str | None = None,
        particles_path: str | None = None,
        pronouns_path: str | None = None,
        classifiers_path: str | None = None,
        register_path: str | None = None,
        homophones_path: str | None = None,
        compounds_path: str | None = None,
        aspects_path: str | None = None,
        pos_inference_path: str | None = None,
        ambiguous_words_path: str | None = None,
        tone_rules_path: str | None = None,
        negation_path: str | None = None,
        morphology_path: str | None = None,
    ) -> None:
        """Load all grammar and linguistic configurations."""
        grammar_path = Path(config_path) if config_path else DEFAULT_GRAMMAR_RULES_PATH
        typo_rules_path = Path(typo_path) if typo_path else DEFAULT_TYPO_CORRECTIONS_PATH
        particles_rules_path = Path(particles_path) if particles_path else DEFAULT_PARTICLES_PATH
        pronouns_rules_path = Path(pronouns_path) if pronouns_path else DEFAULT_PRONOUNS_PATH
        classifiers_rules_path = (
            Path(classifiers_path) if classifiers_path else DEFAULT_CLASSIFIERS_PATH
        )
        register_rules_path = Path(register_path) if register_path else DEFAULT_REGISTER_PATH
        homophones_rules_path = (
            Path(homophones_path) if homophones_path else DEFAULT_HOMOPHONES_PATH
        )
        compounds_rules_path = Path(compounds_path) if compounds_path else DEFAULT_COMPOUNDS_PATH
        aspects_rules_path = Path(aspects_path) if aspects_path else DEFAULT_ASPECTS_PATH
        pos_inference_rules_path = (
            Path(pos_inference_path) if pos_inference_path else DEFAULT_POS_INFERENCE_PATH
        )
        ambiguous_words_rules_path = (
            Path(ambiguous_words_path) if ambiguous_words_path else DEFAULT_AMBIGUOUS_WORDS_PATH
        )
        tone_rules_config_path = (
            Path(tone_rules_path) if tone_rules_path else DEFAULT_TONE_RULES_PATH
        )
        negation_rules_path = Path(negation_path) if negation_path else DEFAULT_NEGATION_PATH
        morphology_rules_path = (
            Path(morphology_path) if morphology_path else DEFAULT_MORPHOLOGY_PATH
        )
        # Define config sources: (path, name, parser)
        config_sources = [
            (grammar_path, "grammar rules", self._parse_grammar_config),
            (typo_rules_path, "typo corrections", self._parse_typo_config),
            (particles_rules_path, "particles", self._parse_particles_config),
            (pronouns_rules_path, "pronouns", self._parse_pronouns_config),
            (classifiers_rules_path, "classifiers", self._parse_classifiers_config),
            (register_rules_path, "register rules", self._parse_register_config),
            (homophones_rules_path, "homophones", self._parse_homophones_config),
            (compounds_rules_path, "compounds", self._parse_compounds_config),
            (aspects_rules_path, "aspects", self._parse_aspects_config),
            (pos_inference_rules_path, "pos inference", self._parse_pos_inference_config),
            (ambiguous_words_rules_path, "ambiguous words", self._parse_ambiguous_words_config),
            (tone_rules_config_path, "tone rules", self._parse_tone_rules_config),
            (negation_rules_path, "negation rules", self._parse_negation_config),
            (morphology_rules_path, "morphology rules", self._parse_morphology_config),
            (DEFAULT_MORPHOTACTICS_PATH, "morphotactics", self._parse_morphotactics_config),
        ]

        # Load all config files
        loaded_configs = [
            _load_yaml_config(path, name, parser) for path, name, parser in config_sources
        ]
        loaded_any = any(cfg is not None for cfg in loaded_configs)

        # Fall back to built-in defaults only if nothing was loaded
        if not loaded_any:
            logger.debug("Using built-in default grammar rules")
            self._load_defaults()

    # ------------------------------------------------------------------ #
    # Parser delegates — thin wrappers that forward to grammar.parsers.*
    # ------------------------------------------------------------------ #

    def _parse_grammar_config(self, config: dict[str, Any]) -> None:
        """Parse grammar_rules.yaml configuration."""
        from myspellchecker.grammar.parsers.grammar_parser import parse_grammar_config

        result = parse_grammar_config(
            config,
            sentence_start_constraints=self.sentence_start_constraints,
            sentence_end_constraints=self.sentence_end_constraints,
            invalid_pos_sequences=self.invalid_pos_sequences,
            noun_particles=self.noun_particles,
            sentence_final_particles=self.sentence_final_particles,
            particle_chains_valid=self.particle_chains_valid,
            particle_chains_invalid=self.particle_chains_invalid,
            clause_linkage=self.clause_linkage,
            negation_rules_config=self.negation_rules_config,
            classifier_rules_config=self.classifier_rules_config,
        )
        # Apply reassigned values from the parser
        if "particle_chains_valid" in result:
            self.particle_chains_valid = result["particle_chains_valid"]
        if "particle_chains_invalid" in result:
            self.particle_chains_invalid = result["particle_chains_invalid"]

    def _parse_typo_config(self, config: dict[str, Any]) -> None:
        """Parse typo_corrections.yaml configuration."""
        from myspellchecker.grammar.parsers.typo_parser import parse_typo_config

        parse_typo_config(
            config,
            particle_typos=self.particle_typos,
            medial_confusions=self.medial_confusions,
            word_corrections=self.word_corrections,
            question_particle_corrections=self.question_particle_corrections,
        )

    def _parse_particles_config(self, config: dict[str, Any]) -> None:
        """Parse particles.yaml configuration."""
        from myspellchecker.grammar.parsers.particle_parser import parse_particles_config

        parse_particles_config(
            config,
            verb_particles=self.verb_particles,
            noun_particles=self.noun_particles,
            sentence_final_particles=self.sentence_final_particles,
            particle_constraints=self.particle_constraints,
            particle_tags=self.particle_tags,
            particle_metadata=self.particle_metadata,
        )

    def _parse_pronouns_config(self, config: dict[str, Any]) -> None:
        """Parse pronouns.yaml configuration."""
        from myspellchecker.grammar.parsers.morphology_parser import parse_pronouns_config

        parse_pronouns_config(config, pronouns=self.pronouns)

    def _parse_classifiers_config(self, config: dict[str, Any]) -> None:
        """Parse classifiers.yaml configuration."""
        from myspellchecker.grammar.parsers.morphology_parser import parse_classifiers_config

        parse_classifiers_config(config, classifiers=self.classifiers)

    def _parse_register_config(self, config: dict[str, Any]) -> None:
        """Parse register.yaml configuration."""
        from myspellchecker.grammar.parsers.morphology_parser import parse_register_config

        self.register_config = parse_register_config(config)

    def _parse_homophones_config(self, config: dict[str, Any]) -> None:
        """Parse homophones.yaml configuration."""
        from myspellchecker.grammar.parsers.homophone_parser import parse_homophones_config

        parse_homophones_config(config, homophones_map=self.homophones_map)

    def _parse_compounds_config(self, config: dict[str, Any]) -> None:
        """Parse compounds.yaml configuration."""
        from myspellchecker.grammar.parsers.morphology_parser import parse_compounds_config

        self.compounds_config = parse_compounds_config(config)

    def _parse_aspects_config(self, config: dict[str, Any]) -> None:
        """Parse aspects.yaml configuration."""
        from myspellchecker.grammar.parsers.morphology_parser import parse_aspects_config

        self.aspects_config = parse_aspects_config(config)

    def _parse_ambiguous_words_config(self, config: dict[str, Any]) -> None:
        """Parse ambiguous_words.yaml configuration."""
        from myspellchecker.grammar.parsers.morphology_parser import parse_ambiguous_words_config

        parse_ambiguous_words_config(config, ambiguous_words_map=self.ambiguous_words_map)

    def _parse_pos_inference_config(self, config: dict[str, Any]) -> None:
        """Parse pos_inference.yaml configuration."""
        from myspellchecker.grammar.parsers.morphology_parser import parse_pos_inference_config

        self.pos_inference_config = parse_pos_inference_config(config)

    def _parse_tone_rules_config(self, config: dict[str, Any]) -> None:
        """Parse tone_rules.yaml configuration."""
        from myspellchecker.grammar.parsers.morphology_parser import parse_tone_rules_config

        self.tone_rules_config = parse_tone_rules_config(
            config,
            tone_ambiguous_map=self.tone_ambiguous_map,
            tone_errors_map=self.tone_errors_map,
        )

    def _parse_negation_config(self, config: dict[str, Any]) -> None:
        """Parse negation.yaml configuration."""
        from myspellchecker.grammar.parsers.morphology_parser import parse_negation_config

        self.negation_config = parse_negation_config(config)

    def _parse_morphology_config(self, config: dict[str, Any]) -> None:
        """Parse morphology.yaml configuration."""
        from myspellchecker.grammar.parsers.morphology_parser import parse_morphology_config

        self.morphology_config = parse_morphology_config(config)

    def _parse_morphotactics_config(self, config: dict[str, Any]) -> None:
        """Parse morphotactics.yaml configuration."""
        from myspellchecker.grammar.parsers.morphology_parser import parse_morphotactics_config

        self.morphotactics_config = parse_morphotactics_config(config)

    # ------------------------------------------------------------------ #
    # Default rules (fallback when no YAML files are found)
    # ------------------------------------------------------------------ #

    def _load_defaults(self) -> None:
        """Load built-in default rules when YAML is not available."""
        # Default particle typos
        # Note: "လာ" (to come) was removed - it's a valid common verb, not a typo for "လှာ" (tongue)
        self.particle_typos = {
            "မာ": {
                "correction": "မှာ",
                "context": "after_noun",
                "meaning": "location particle",
                "excluded_pos": ["ADJ"],  # "မာ" as adjective ("hard") is valid
            },
            "နာ": {
                "correction": "နှာ",
                "context": "context_dependent",
                "meaning": "nose",
                "followed_by": ["ခေါင်း"],
            },
            "ဂ": {"correction": "က", "context": "after_noun", "meaning": "subject marker"},
            # Note: Bare "ည" -> "ညှ" mapping was intentionally removed.
            # "ည" (U+100A, nya) is a valid consonant/word meaning "night".
            # "ညှ" alone is not a standard word -- the word for "squeeze" is "ညှစ်".
        }

        # Default medial confusions (dict keyed by pattern for O(1) lookup)
        self.medial_confusions = {
            "ကျောင်း": {
                "pattern": "ကျောင်း",
                "correction": "ကြောင်း",
                "context": "after_verb",
                "meaning": "because/cause",
            },
            "ကျည်": {
                "pattern": "ကျည်",
                "correction": "ကြည့်",
                "context": "after_verb",
                "meaning": "look/watch",
            },
            "ကြေးဇူး": {
                "pattern": "ကြေးဇူး",
                "correction": "ကျေးဇူး",
                "meaning": "thanks",
            },
        }

        # Default verb particles
        self.verb_particles = {
            "ခဲ့",
            "မယ်",
            "နေ",
            "မည်",
            "လိမ့်မည်",
            "တယ်",
            "ပါတယ်",
            "သည်",
            "ပါသည်",
            "ကြ",
            "နိုင်",
            "ရ",
            "တတ်",
            "တတ်တယ်",
            "လျက်",
            "လေ",
            "ပေ",
            "ပြီ",
            "ပြီး",
            "ရင်",
            "လျှင်",
            "သော်လည်း",
            "သော်",
            "သော",
            "ခြင်း",
            "မှု",
            "ရန်",
            "ဖို့",
            "အောင်",
            "လို့",
            "ကြောင့်",
            "ကြောင်း",
        }

        # Default noun particles
        self.noun_particles = {
            "က",
            "ကို",
            "သည်",
            "မှာ",
            "မှ",
            "နဲ့",
            "နှင့်",
            "ရဲ့",
            "၏",
            "အတွက်",
            "လို",
            "လိုလို",
            "လိုပဲ",
            "ထက်",
            "ထဲ",
            "အထဲ",
            "ပေါ်",
            "အပေါ်",
            "အောက်",
            "ရှေ့",
            "အရှေ့",
            "နောက်",
            "အနောက်",
            "ဘေး",
            "အနား",
            "တွင်",
            "၌",
            "များ",
            "တို့",
            "ဟာ",
            "တော့",
            "ပဲ",
            "ပေါ့",
            "လေ",
        }

        # Default invalid POS sequences
        self.invalid_pos_sequences = [
            {
                "sequence": "V-V",
                "severity": "error",
                "message": "Two consecutive verbs without particle",
            },
            {"sequence": "P-P", "severity": "error", "message": "Two consecutive particles"},
            {
                "sequence": "N-N",
                "severity": "warning",
                "message": "May need particle between nouns",
            },
        ]

        # Default sentence-final particles
        self.sentence_final_particles = {
            "တယ်",
            "ပါတယ်",
            "သည်",
            "ပါသည်",
            "မယ်",
            "ပါမယ်",
            "မည်",
            "ပါမည်",
            "ပြီ",
            "ပါပြီ",
            # Removed: "ပေ့" -- not a recognized particle (see patterns.py)
            "လေ",
            "နော်",
            "လား",
            "သလား",
            "လဲ",
            "မလဲ",
            "ပါ",
            "ပါစေ",
            "ပဲ",
            "ပေါ့",
            "စေ",
            "ဘူး",
            "ပါဘူး",
        }

        # Default word corrections
        self.word_corrections = {
            "ကြေးဇူး": {"correction": "ကျေးဇူး", "meaning": "thanks"},
            "ပါတယ": {"correction": "ပါတယ်", "meaning": "polite ending"},
            "တယ": {"correction": "တယ်", "meaning": "statement ending"},
            "သည": {"correction": "သည်", "meaning": "formal ending"},
        }

    # ------------------------------------------------------------------ #
    # Public accessor methods
    # ------------------------------------------------------------------ #

    def get_medial_confusion(self, word: str) -> dict[str, str] | None:
        """
        Get medial confusion info for a word if it exists.

        Args:
            word: The word to check.

        Returns:
            Dict with correction info or None.
        """
        return self.medial_confusions.get(word)

    def get_particle_typo(self, word: str) -> dict[str, Any] | None:
        """
        Get particle typo info for a word if it exists.

        Args:
            word: The word to check.

        Returns:
            Dict with correction info or None.
        """
        return self.particle_typos.get(word)

    def get_word_correction(self, word: str) -> dict[str, Any] | None:
        """
        Get word correction info if it exists.

        Args:
            word: The word to check.

        Returns:
            Dict with correction info or None.
        """
        return self.word_corrections.get(word)

    def get_question_particle_correction(
        self, word: str, prev_pos: str | None, is_sentence_final: bool
    ) -> dict[str, Any] | None:
        """
        Get question particle correction if applicable.

        Checks position-dependent rules for question particle corrections
        (e.g., "လာ" -> "လား" at sentence end after verb/particle).

        Args:
            word: Current word to check.
            prev_pos: POS tag of preceding word.
            is_sentence_final: Whether word is at sentence end.

        Returns:
            Dict with correction info or None if no match.
        """
        for rule in self.question_particle_corrections:
            if rule.get("incorrect") != word:
                continue

            # Check position constraint
            pos_constraint = rule.get("pos_constraint", {})
            position_req = pos_constraint.get("position", "")

            if position_req == "sentence_final" and not is_sentence_final:
                continue

            # Check POS constraint
            required_preceding = pos_constraint.get("preceding", [])
            if required_preceding and prev_pos:
                # Check if previous POS matches any required tag (exact match
                # on pipe-separated components to avoid prefix false-matches
                # like "P" matching "PRON")
                prev_tags = set(prev_pos.split("|"))
                matches = any(tag in prev_tags for tag in required_preceding)
                if not matches:
                    continue

            # All constraints satisfied
            return rule

        return None

    def is_verb_particle(self, word: str) -> bool:
        """Check if word is a valid verb particle."""
        return word in self.verb_particles

    def is_noun_particle(self, word: str) -> bool:
        """Check if word is a valid noun particle."""
        return word in self.noun_particles

    def is_sentence_final(self, word: str) -> bool:
        """Check if word is a sentence-final particle."""
        return word in self.sentence_final_particles

    def get_invalid_sequence_error(self, prev_tag: str, curr_tag: str) -> dict[str, Any] | None:
        """
        Check if a POS sequence is invalid.

        Args:
            prev_tag: Previous word's POS tag.
            curr_tag: Current word's POS tag.

        Returns:
            Dict with error info or None.
        """
        sequence = f"{prev_tag}-{curr_tag}"
        for rule in self.invalid_pos_sequences:
            if rule.get("sequence") == sequence:
                return rule
        return None

    # Accessor methods for grammar sections

    def get_invalid_particle_chain(self, particles: tuple[str, ...]) -> dict[str, Any] | None:
        """
        Check if a particle sequence is an invalid chain.

        Args:
            particles: Tuple of consecutive particles to check.

        Returns:
            Dict with chain rule info if invalid, None if valid or unknown.
        """
        for chain in self.particle_chains_invalid:
            chain_particles = chain.get("particles", [])
            if tuple(chain_particles) == particles:
                return chain
        return None

    def get_clause_linkers(self) -> set[str]:
        """
        Extract clause linker words from YAML config patterns.

        Patterns in grammar_rules.yaml are in format "V-linker-V" (e.g., "V-linker-V").
        This method extracts the middle element (the actual linker word).

        Returns:
            Set of clause linker words extracted from config patterns.
        """
        linkers: set[str] = set()
        for rule in self.clause_linkage:
            pattern = rule.get("pattern", "")
            if "-" in pattern:
                parts = pattern.split("-")
                # Extract linker (middle element in V-linker-V patterns)
                if len(parts) >= 2:
                    # For patterns like "V-linker-V", the linker is parts[1]
                    linkers.add(parts[1])
        return linkers

    def get_clause_linkage_rule(self, linker: str) -> dict[str, Any] | None:
        """
        Get clause linkage rule for a connector word.

        Args:
            linker: The clause linking word/particle.

        Returns:
            Dict with linkage rule info if found, None otherwise.
        """
        for rule in self.clause_linkage:
            # Pattern format is "V-linker-V" — only match the linker position (index 1),
            # not POS labels like "V" which appear at other positions
            pattern = rule.get("pattern", "")
            pattern_parts = pattern.split("-")
            if len(pattern_parts) >= 3 and pattern_parts[1] == linker:
                return rule
        return None
