"""Parsers for morphology, morphotactics, and other simple YAML configs.

Handles parsing of pronouns, classifiers, register, compounds, aspects,
ambiguous words, POS inference, tone rules, negation, morphology, and
morphotactics YAML configurations.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "parse_ambiguous_words_config",
    "parse_aspects_config",
    "parse_classifiers_config",
    "parse_compounds_config",
    "parse_morphology_config",
    "parse_morphotactics_config",
    "parse_negation_config",
    "parse_pos_inference_config",
    "parse_pronouns_config",
    "parse_register_config",
    "parse_tone_rules_config",
]


def parse_pronouns_config(
    config: dict[str, Any],
    *,
    pronouns: list[dict[str, Any]],
) -> None:
    """Parse pronouns.yaml configuration.

    Args:
        config: Raw YAML dict from pronouns.yaml.
        pronouns: List to populate with pronoun entries.
    """
    pronouns_data = config.get("pronouns", {})

    # Iterate over all sections (first_person, second_person, etc.)
    for category, pronoun_list in pronouns_data.items():
        for p in pronoun_list:
            if "word" in p:
                # Add category info
                p["category"] = category
                pronouns.append(p)


def parse_classifiers_config(
    config: dict[str, Any],
    *,
    classifiers: dict[str, list[dict[str, Any]]],
) -> None:
    """Parse classifiers.yaml configuration.

    Args:
        config: Raw YAML dict from classifiers.yaml.
        classifiers: Dict to populate with category -> classifier list.
    """
    classifiers_data = config.get("classifiers", {})

    # Iterate over all categories (people, animals, etc.)
    for category, classifier_list in classifiers_data.items():
        classifiers[category] = classifier_list


def parse_register_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Parse register.yaml configuration.

    Args:
        config: Raw YAML dict from register.yaml.

    Returns:
        The entire register config structure.
    """
    return config


def parse_compounds_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Parse compounds.yaml configuration.

    Args:
        config: Raw YAML dict from compounds.yaml.

    Returns:
        The entire compounds config structure.
    """
    return config


def parse_aspects_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Parse aspects.yaml configuration.

    Args:
        config: Raw YAML dict from aspects.yaml.

    Returns:
        The entire aspects config structure.
    """
    return config


def parse_ambiguous_words_config(
    config: dict[str, Any],
    *,
    ambiguous_words_map: dict[str, set[str]],
) -> None:
    """Parse ambiguous_words.yaml configuration.

    Args:
        config: Raw YAML dict from ambiguous_words.yaml.
        ambiguous_words_map: Dict to populate with word -> set of tags.
    """
    ambiguous_words = config.get("ambiguous_words", {})

    for word, tags_list in ambiguous_words.items():
        if isinstance(tags_list, list):
            ambiguous_words_map[word] = set(tags_list)


def parse_pos_inference_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Parse pos_inference.yaml configuration.

    Args:
        config: Raw YAML dict from pos_inference.yaml.

    Returns:
        The POS inference config dict.
    """
    return config


def parse_tone_rules_config(
    config: dict[str, Any],
    *,
    tone_ambiguous_map: dict[str, dict[str, dict[str, Any]]],
    tone_errors_map: dict[str, str],
) -> dict[str, Any]:
    """Parse tone_rules.yaml configuration.

    Args:
        config: Raw YAML dict from tone_rules.yaml.
        tone_ambiguous_map: Dict to populate with ambiguous word map.
        tone_errors_map: Dict to populate with tone error mappings.

    Returns:
        The entire tone rules config dict.
    """
    # Populate ambiguous words map
    if "ambiguous_words" in config:
        tone_ambiguous_map.update(config["ambiguous_words"])

    # Populate tone errors map
    if "tone_errors" in config:
        tone_errors_map.update(config["tone_errors"])

    return config


def parse_negation_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Parse negation.yaml configuration.

    Args:
        config: Raw YAML dict from negation.yaml.

    Returns:
        The entire negation config dict.
    """
    return config


def parse_morphology_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Parse morphology.yaml configuration.

    Args:
        config: Raw YAML dict from morphology.yaml.

    Returns:
        The entire morphology config dict.
    """
    return config


def parse_morphotactics_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Parse morphotactics.yaml configuration.

    Args:
        config: Raw YAML dict from morphotactics.yaml.

    Returns:
        The entire morphotactics config dict.
    """
    return config
