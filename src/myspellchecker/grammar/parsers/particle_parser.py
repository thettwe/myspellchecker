"""Parser for particles.yaml configuration.

Extracts verb particles, noun particles, interrogative particles,
politeness particles, negation particles, modifiers, and their
associated constraints, POS tags, and metadata.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "parse_particles_config",
]


def parse_particles_config(
    config: dict[str, Any],
    *,
    verb_particles: set[str],
    noun_particles: set[str],
    sentence_final_particles: set[str],
    particle_constraints: dict[str, dict[str, Any]],
    particle_tags: dict[str, str],
    particle_metadata: dict[str, dict[str, Any]],
) -> None:
    """Parse particles.yaml configuration.

    Mutates the provided containers in-place.

    Args:
        config: Raw YAML dict from particles.yaml.
        verb_particles: Set to populate with verb-following particles.
        noun_particles: Set to populate with noun-following particles.
        sentence_final_particles: Set to populate with sentence-ending particles.
        particle_constraints: Dict to populate with particle constraints.
        particle_tags: Dict to populate with POS tag mappings.
        particle_metadata: Dict to populate with full particle metadata.
    """
    particles = config.get("particles", {})

    def _add_constraints(p_data: dict[str, Any]) -> None:
        """Helper to extract and store constraints.

        Merges tags and constraints for particles that appear in multiple
        categories (e.g., a particle that is both comitative and prohibition).
        """
        if "particle" not in p_data:
            return
        p_text = p_data["particle"]

        # Store full metadata (last-write-wins is OK for metadata)
        particle_metadata[p_text] = p_data

        # Merge POS tag mapping (pipe-separated for multi-role particles)
        if "pos_tag" in p_data:
            new_tag = p_data["pos_tag"]
            if p_text in particle_tags:
                existing = particle_tags[p_text]
                existing_tags = set(existing.split("|"))
                if new_tag not in existing_tags:
                    particle_tags[p_text] = f"{existing}|{new_tag}"
            else:
                particle_tags[p_text] = new_tag

        # Merge constraints (union required_preceding lists, OR sentence_final)
        constraints: dict[str, Any] = {}
        if "required_preceding" in p_data:
            constraints["required_preceding"] = p_data["required_preceding"]
        if "sentence_final" in p_data:
            constraints["sentence_final"] = p_data["sentence_final"]

        if constraints:
            if p_text in particle_constraints:
                existing_cons = particle_constraints[p_text]
                # Merge required_preceding lists
                if "required_preceding" in constraints and "required_preceding" in existing_cons:
                    merged = list(
                        set(existing_cons["required_preceding"])
                        | set(constraints["required_preceding"])
                    )
                    existing_cons["required_preceding"] = merged
                elif "required_preceding" in constraints:
                    existing_cons["required_preceding"] = constraints["required_preceding"]
                # OR sentence_final (if any role is sentence-final, keep it)
                if "sentence_final" in constraints:
                    existing_cons["sentence_final"] = (
                        existing_cons.get("sentence_final", False) or constraints["sentence_final"]
                    )
            else:
                particle_constraints[p_text] = constraints

    # Parse verb particles
    if "verbs" in particles:
        for _category_name, particle_list in particles["verbs"].items():
            for p in particle_list:
                if "particle" in p:
                    verb_particles.add(p["particle"])
                    _add_constraints(p)
                    if p.get("type") and "ending" in p["type"]:
                        sentence_final_particles.add(p["particle"])

        # Specifically check statement_endings
        if "statement_endings" in particles["verbs"]:
            for p in particles["verbs"]["statement_endings"]:
                if "particle" in p:
                    sentence_final_particles.add(p["particle"])

    # Parse noun particles
    if "nouns" in particles:
        for _category_name, particle_list in particles["nouns"].items():
            for p in particle_list:
                if "particle" in p:
                    noun_particles.add(p["particle"])
                    _add_constraints(p)

    # Parse interrogative/sentence final
    if "interrogative" in particles:
        for p in particles["interrogative"]:
            if "particle" in p:
                # Often also considered verb particles or sentence final
                verb_particles.add(p["particle"])  # Many Q-particles follow verbs
                _add_constraints(p)
                if p.get("sentence_final", False):
                    sentence_final_particles.add(p["particle"])

    # Parse politeness
    if "politeness" in particles:
        for p in particles["politeness"]:
            if "particle" in p:
                verb_particles.add(p["particle"])
                _add_constraints(p)
                if p.get("sentence_final", False):
                    sentence_final_particles.add(p["particle"])

    # Parse negation
    if "negation" in particles:
        for p in particles["negation"]:
            if "particle" in p:
                # Negation particles can be verb-following or prefix
                # We add them to verb_particles if they are endings
                if p.get("type", "").endswith("ending") or p.get("type") == "prohibition":
                    verb_particles.add(p["particle"])
                _add_constraints(p)
                if p.get("sentence_final", False):
                    sentence_final_particles.add(p["particle"])

    # Parse modifiers (some might be considered verb/noun particles depending on usage)
    if "modifiers" in particles:
        for category_name, particle_list in particles["modifiers"].items():
            for p in particle_list:
                if "particle" in p:
                    # Add to generic lists if appropriate, or just keep specialized
                    # For now, treat relative clause markers as verb particles
                    # (follow verbs)
                    if category_name == "relative_clause":
                        verb_particles.add(p["particle"])
                    _add_constraints(p)
