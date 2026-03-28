"""Parser for typo_corrections.yaml configuration.

Extracts particle typos, medial confusions, missing asat marks,
missing e-vowel patterns, classifier corrections, negation corrections,
visual/OCR errors, and question particle corrections.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "parse_typo_config",
]


def parse_typo_config(
    config: dict[str, Any],
    *,
    particle_typos: dict[str, dict[str, Any]],
    medial_confusions: dict[str, dict[str, str]],
    word_corrections: dict[str, dict[str, Any]],
    question_particle_corrections: list[dict[str, Any]],
) -> None:
    """Parse typo_corrections.yaml configuration.

    Mutates the provided containers in-place.

    Args:
        config: Raw YAML dict from typo_corrections.yaml.
        particle_typos: Dict to populate with particle typo mappings.
        medial_confusions: Dict to populate with medial confusion patterns.
        word_corrections: Dict to populate with word correction mappings.
        question_particle_corrections: List to populate with question particle rules.
    """
    corrections = config.get("corrections", {})

    # Parse particle typos
    if "particles" in corrections:
        for item in corrections["particles"]:
            typo = item.get("incorrect")
            if typo:
                entry: dict[str, Any] = {
                    "correction": item.get("correct"),
                    "context": item.get("context", "any"),
                    "meaning": item.get("meaning", ""),
                    "confidence": item.get("confidence", 0.9),
                }
                if "excluded_pos" in item:
                    entry["excluded_pos"] = item["excluded_pos"]
                if "followed_by" in item:
                    entry["followed_by"] = item["followed_by"]
                particle_typos[typo] = entry

    # Parse medial confusions
    if "medial_confusions" in corrections:
        for item in corrections["medial_confusions"]:
            pattern = item.get("incorrect")
            item["pattern"] = pattern  # Map for internal usage
            item["correction"] = item.get("correct")
            if pattern:
                medial_confusions[pattern] = item

    # Parse missing asat marks
    if "missing_asat" in corrections:
        for item in corrections["missing_asat"]:
            typo = item.get("incorrect")
            if typo:
                # Treat as word correction or particle typo depending on length/usage
                # For now, add to word corrections
                word_corrections[typo] = {
                    "correction": item.get("correct"),
                    "meaning": item.get("meaning", ""),
                    "confidence": item.get("confidence", 0.95),
                }
                # Also add to particle typos if it looks like a particle
                if len(typo) <= 3:
                    particle_typos[typo] = {
                        "correction": item.get("correct"),
                        "context": "any",
                        "meaning": item.get("meaning", ""),
                        "confidence": item.get("confidence", 0.95),
                    }

    # Parse missing e-vowel patterns (e.g., ကာင်း -> ကောင်း)
    if "missing_e_vowel" in corrections:
        for item in corrections["missing_e_vowel"]:
            typo = item.get("incorrect")
            if typo:
                entry = {
                    "correction": item.get("correct"),
                    "meaning": item.get("meaning", ""),
                    "confidence": item.get("confidence", 0.95),
                    "error_type": item.get("error_type", "missing_e_vowel"),
                }
                if "context" in item:
                    entry["context"] = item["context"]
                if "excluded_pos" in item:
                    entry["excluded_pos"] = item["excluded_pos"]
                word_corrections[typo] = entry
                # Also add to particle typos if short
                if len(typo) <= 3:
                    particle_typos[typo] = {
                        "correction": item.get("correct"),
                        "context": item.get("context", "any"),
                        "meaning": item.get("meaning", ""),
                        "confidence": item.get("confidence", 0.95),
                    }

    # Parse classifiers
    if "classifiers" in corrections:
        for item in corrections["classifiers"]:
            typo = item.get("incorrect")
            if typo:
                word_corrections[typo] = {
                    "correction": item.get("correct"),
                    "meaning": item.get("meaning", ""),
                    "confidence": item.get("confidence", 0.9),
                }

    # Parse negation
    if "negation" in corrections:
        for item in corrections["negation"]:
            typo = item.get("incorrect")
            if typo:
                word_corrections[typo] = {
                    "correction": item.get("correct"),
                    "meaning": item.get("meaning", ""),
                    "confidence": item.get("confidence", 0.98),
                }

    # Parse visual/OCR errors
    if "visual_errors" in corrections:
        for item in corrections["visual_errors"]:
            typo = item.get("incorrect")
            if typo:
                entry = {
                    "correction": item.get("correct"),
                    "meaning": item.get("meaning", ""),
                    "confidence": item.get("confidence", 0.95),
                    "error_type": item.get("error_type", "visual"),
                }
                if "excluded_pos" in item:
                    entry["excluded_pos"] = item["excluded_pos"]
                if "context" in item:
                    entry["context"] = item["context"]
                word_corrections[typo] = entry
                # If it's a short token, it might be a particle typo too
                if len(typo) <= 2:
                    particle_typos[typo] = {
                        "correction": item.get("correct"),
                        "context": item.get("context", "any"),
                        "meaning": item.get("meaning", ""),
                        "confidence": item.get("confidence", 0.95),
                    }

    # Parse question particle corrections (position-dependent)
    if "question_particles" in corrections:
        for item in corrections["question_particles"]:
            question_particle_corrections.append(item)
