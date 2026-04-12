"""Config-driven grammar rule mixin for SyntacticRuleChecker.

Provides methods that validate against YAML-configured grammar rules:
particle chains, register mixing, clause linkage, negation patterns,
and classifier constructions.

Extracted from ``engine.py`` to reduce file size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myspellchecker.core.config import GrammarEngineConfig
    from myspellchecker.grammar.config import GrammarRuleConfig
    from myspellchecker.providers import DictionaryProvider


class ConfigGrammarMixin:
    """Mixin providing config-driven grammar validation methods.

    Each method checks word sequences against rules defined in
    ``grammar_rules.yaml`` sections (particle_chains, register_rules,
    clause_linkage, negation_rules, classifier_rules).
    """

    # --- Type stubs for attributes provided by SyntacticRuleChecker ---
    config: "GrammarRuleConfig"
    grammar_config: "GrammarEngineConfig"
    provider: "DictionaryProvider"

    # --- Type stubs for methods provided by POSTagMixin (resolved via MRO) ---
    def _has_tag(self, pos_string: str | None, target_tag: str) -> bool:
        raise NotImplementedError("provided by POSTagMixin via MRO")

    def _is_particle_tag(self, tag: str) -> bool:
        raise NotImplementedError("provided by POSTagMixin via MRO")

    def _check_particle_chains(
        self, words: list[str], pos_tags: list[str | None]
    ) -> list[tuple[int, str, str, float]]:
        """
        Check for invalid particle chain sequences.

        Validates consecutive particle sequences against the particle_chains
        rules in grammar_rules.yaml.

        Args:
            words: List of words in the sentence.
            pos_tags: List of POS tags corresponding to words.

        Returns:
            List of (index, error_word, suggestion, confidence) tuples.
        """
        errors: list[tuple[int, str, str, float]] = []

        if len(words) < 2 or not pos_tags or len(pos_tags) != len(words):
            return errors

        # Find consecutive particle sequences
        i = 0
        while i < len(words) - 1:
            curr_tag = pos_tags[i]
            if curr_tag and self._is_particle_tag(curr_tag):
                # Collect consecutive particles
                chain_start = i
                chain_words = [words[i]]

                j = i + 1
                while j < len(words):
                    next_tag = pos_tags[j]
                    if next_tag is None or not self._is_particle_tag(next_tag):
                        break
                    chain_words.append(words[j])
                    j += 1

                if len(chain_words) >= 2:
                    chain_tuple = tuple(chain_words)
                    invalid_chain = self.config.get_invalid_particle_chain(chain_tuple)

                    if invalid_chain:
                        # Report error at the start of the chain
                        suggestion = invalid_chain.get("suggestion", "")
                        confidence = invalid_chain.get("confidence", 0.75)
                        errors.append((chain_start, " ".join(chain_words), suggestion, confidence))

                    i = j  # Skip past the chain
                    continue

            i += 1

        return errors

    def _check_clause_linkage_from_config(
        self, words: list[str], pos_tags: list[str | None]
    ) -> list[tuple[int, str, str, float]]:
        """
        Check clause linkage patterns using grammar_rules.yaml clause_linkage rules.

        Validates that clause linking particles have proper context based on
        the YAML pattern specification:
        - V-<linker>-V: requires verb before AND after
        - N-<linker>: requires noun before, no requirement after
        - V-<linker>: requires verb before, no requirement after

        Pattern-aware validation respects YAML-defined context.

        Args:
            words: List of words in the sentence.
            pos_tags: List of POS tags corresponding to words.

        Returns:
            List of (index, error_word, suggestion, confidence) tuples.
        """
        errors: list[tuple[int, str, str, float]] = []

        if len(words) < 2 or not pos_tags or len(pos_tags) != len(words):
            return errors

        clause_linkers = self.config.get_clause_linkers()
        if not clause_linkers:
            return errors

        for i, word in enumerate(words):
            if word not in clause_linkers:
                continue

            rule = self.config.get_clause_linkage_rule(word)
            if not rule:
                continue

            # Parse pattern to determine expected context
            # Pattern format: "LEFT-<linker>[-RIGHT]" e.g., "V-word-V", "N-word"
            pattern = rule.get("pattern", "")
            parts = pattern.split("-") if pattern else []

            # Determine expected POS from pattern
            left_expected = parts[0] if len(parts) >= 1 else None
            # Right expected only if 3 parts (LEFT-linker-RIGHT)
            right_expected = parts[2] if len(parts) == 3 else None

            # Check for expected POS in preceding words
            has_left_match = False
            if left_expected and i > 0:
                for j in range(i - 1, -1, -1):
                    prev_tag = pos_tags[j]
                    if prev_tag and self._has_tag(prev_tag, left_expected):
                        has_left_match = True
                        break
                    # Stop if we hit a sentence boundary marker
                    if prev_tag and prev_tag in {"PUNCT", "SB"}:
                        break
            elif not left_expected:
                # No left requirement
                has_left_match = True

            # Check for expected POS in following words (only if pattern specifies)
            has_right_match = False
            if right_expected and i < len(words) - 1:
                for j in range(i + 1, len(words)):
                    next_tag = pos_tags[j]
                    if next_tag and self._has_tag(next_tag, right_expected):
                        has_right_match = True
                        break
                    # Stop if we hit a sentence boundary marker
                    if next_tag and next_tag in {"PUNCT", "SB"}:
                        break
            elif not right_expected:
                # No right requirement (e.g., N-word pattern)
                has_right_match = True

            # Flag if missing expected context, but only for "error" severity rules.
            severity = rule.get("severity", "info")
            if severity == "error" and (not has_left_match or not has_right_match):
                confidence = rule.get("confidence", 0.5)
                message = rule.get("message", f"Clause linker '{word}' missing context")
                errors.append((i, word, message, confidence))

        return errors
