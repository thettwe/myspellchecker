"""POS tag utility mixin for SyntacticRuleChecker.

Provides lightweight helpers for inspecting pipe-separated POS tag
strings and mapping particle-like tags to the generic ``P`` tag.

Extracted from ``engine.py`` to reduce file size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myspellchecker.grammar.config import GrammarRuleConfig

# HuggingFace transformer particle-like tags that should map to generic 'P'
HF_PARTICLE_TAGS: frozenset[str] = frozenset({"PPM", "PART"})


class POSTagMixin:
    """Mixin providing POS tag utility methods.

    These methods only use ``self.config`` (``GrammarRuleConfig``).
    """

    # --- Type stubs for attributes provided by SyntacticRuleChecker ---
    config: "GrammarRuleConfig"

    def _has_tag(self, pos_string: str | None, target_tag: str) -> bool:
        """Check if the pipe-separated POS string contains the target tag."""
        if not pos_string:
            return False
        return target_tag in pos_string.split("|")

    def _get_all_tags(self, pos_string: str | None) -> set[str]:
        """Get all POS tags from a pipe-separated string."""
        if not pos_string:
            return set()
        return set(pos_string.split("|"))

    def _is_particle_like_tag(self, tag: str) -> bool:
        """
        Check if a tag represents a particle-like POS.

        This includes:
        - Granular particle tags (P_SUBJ, P_OBJ, P_SENT, etc.)
        - HuggingFace transformer particle tags (PPM, PART)

        Used for mapping to generic 'P' tag in fallback rule matching.
        """
        if not tag:
            return False
        return tag.startswith("P_") or tag in HF_PARTICLE_TAGS

    def _get_primary_tag(self, pos_string: str | None, word: str | None = None) -> str:
        """
        Get the primary (first) POS tag.

        If word is provided, looks up specific particle tag from config
        to refine generic 'P' tags.
        """
        # Try to get specific tag from config first
        if word and hasattr(self.config, "particle_tags"):
            specific_tag = self.config.particle_tags.get(word)
            if specific_tag:
                return specific_tag

        if not pos_string:
            return ""

        tags = pos_string.split("|")
        primary = tags[0] if tags else ""

        # Normalize granular particle tags (P_SUBJ, P_OBJ, etc.) to P
        # ONLY if we didn't get a specific one from config (which we check above)
        # Return granular tags (P_SUBJ, P_OBJ, etc.) when available so
        # sequence rules like P_SENT-P_SENT work. Legacy rules (N-V) map
        # these back to P internally.
        return primary

    def _is_particle_tag(self, tag: str) -> bool:
        """
        Check if a POS tag represents a particle.

        Handles pipe-separated multi-tags (e.g., "PPM|N") by splitting
        on ``|`` and checking each component individually.

        Args:
            tag: POS tag to check (may be pipe-separated).

        Returns:
            True if any component tag is a particle tag.
        """
        if not tag:
            return False
        for component in tag.split("|"):
            if (
                component.startswith("P_")  # Granular particle tags (P_SUBJ, P_OBJ, etc.)
                or component == "P"  # Generic particle
                or component in HF_PARTICLE_TAGS  # HuggingFace particle tags (PPM, PART)
            ):
                return True
        return False
