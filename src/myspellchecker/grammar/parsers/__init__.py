"""Grammar YAML configuration parsers.

Re-exports all parser functions for convenient access.
"""

from myspellchecker.grammar.parsers.grammar_parser import parse_grammar_config
from myspellchecker.grammar.parsers.homophone_parser import parse_homophones_config
from myspellchecker.grammar.parsers.morphology_parser import (
    parse_ambiguous_words_config,
    parse_aspects_config,
    parse_classifiers_config,
    parse_compounds_config,
    parse_morphology_config,
    parse_morphotactics_config,
    parse_negation_config,
    parse_pos_inference_config,
    parse_pronouns_config,
    parse_register_config,
    parse_tone_rules_config,
)
from myspellchecker.grammar.parsers.particle_parser import parse_particles_config
from myspellchecker.grammar.parsers.typo_parser import parse_typo_config

__all__ = [
    "parse_ambiguous_words_config",
    "parse_aspects_config",
    "parse_classifiers_config",
    "parse_compounds_config",
    "parse_grammar_config",
    "parse_homophones_config",
    "parse_morphology_config",
    "parse_morphotactics_config",
    "parse_negation_config",
    "parse_particles_config",
    "parse_pos_inference_config",
    "parse_pronouns_config",
    "parse_register_config",
    "parse_tone_rules_config",
    "parse_typo_config",
]
