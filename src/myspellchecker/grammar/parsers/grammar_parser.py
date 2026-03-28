"""Parser for grammar_rules.yaml configuration.

Extracts sentence boundary constraints, invalid POS sequences,
required particles, sentence-final particles, particle chains,
register rules, clause linkage, negation rules, and classifier rules.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "parse_grammar_config",
]


def parse_grammar_config(
    config: dict[str, Any],
    *,
    sentence_start_constraints: list[dict[str, Any]],
    sentence_end_constraints: list[dict[str, Any]],
    invalid_pos_sequences: list[dict[str, Any]],
    noun_particles: set[str],
    sentence_final_particles: set[str],
    particle_chains_valid: list[dict[str, Any]],
    particle_chains_invalid: list[dict[str, Any]],
    clause_linkage: list[dict[str, Any]],
    negation_rules_config: list[dict[str, Any]],
    classifier_rules_config: list[dict[str, Any]],
) -> dict[str, Any]:
    """Parse grammar_rules.yaml configuration.

    Mutates the provided mutable containers in-place and returns a dict
    with any newly-assigned values for immutable fields.

    Args:
        config: Raw YAML dict from grammar_rules.yaml.
        sentence_start_constraints: List to populate with start constraints.
        sentence_end_constraints: List to populate with end constraints.
        invalid_pos_sequences: List to populate with invalid POS sequences.
        noun_particles: Set to populate with noun particles.
        sentence_final_particles: Set to populate with sentence-final particles.
        particle_chains_valid: List to populate with valid chains.
        particle_chains_invalid: List to populate with invalid chains.
        clause_linkage: List to populate with clause linkage rules.
        negation_rules_config: List to populate with negation rules.
        classifier_rules_config: List to populate with classifier rules.

    Returns:
        Dict with keys for fields that need reassignment:
        ``register_rules_config``, ``particle_chains_valid``,
        ``particle_chains_invalid``.
    """
    rules = config.get("rules", {})
    result: dict[str, Any] = {}

    # Parse sentence boundary constraints
    if "sentence_start_constraints" in rules:
        sentence_start_constraints.extend(rules["sentence_start_constraints"])

    if "sentence_end_constraints" in rules:
        sentence_end_constraints.extend(rules["sentence_end_constraints"])

    # Parse invalid POS sequences
    if "invalid_sequences" in rules:
        for rule in rules["invalid_sequences"]:
            # Convert to internal format if needed
            if "pattern" in rule:
                parts = rule["pattern"].split("-")
                if len(parts) == 2:
                    rule["sequence"] = rule["pattern"]  # Keep for compatibility
            invalid_pos_sequences.append(rule)

    # Parse required particles (populate verb/noun particles)
    if "required_particles" in rules:
        for rule in rules["required_particles"]:
            if "required_one_of" in rule:
                pattern = rule.get("pattern", "")
                if pattern == "N-V":
                    # These are noun particles (follow noun)
                    noun_particles.update(rule["required_one_of"])

    # Parse sentence-final particles
    if "sentence_final_required" in rules:
        for rule in rules["sentence_final_required"]:
            if "required_one_of" in rule:
                sentence_final_particles.update(rule["required_one_of"])

    # Parse particle chains (valid and invalid sequences of particles)
    if "particle_chains" in rules:
        chains = rules["particle_chains"]
        result["particle_chains_valid"] = chains.get("valid_chains", [])
        result["particle_chains_invalid"] = chains.get("invalid_chains", [])

    # Parse clause linkage rules (how clauses connect)
    if "clause_linkage" in rules:
        clause_linkage.extend(rules["clause_linkage"])

    # Parse negation rules (negation pattern validation)
    if "negation_rules" in rules:
        negation_rules_config.extend(rules["negation_rules"])

    # Parse classifier rules (noun+classifier patterns)
    if "classifier_rules" in rules:
        classifier_rules_config.extend(rules["classifier_rules"])

    return result
