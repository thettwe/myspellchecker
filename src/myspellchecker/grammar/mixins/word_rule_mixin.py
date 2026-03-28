"""Word-level rule check mixin for SyntacticRuleChecker.

Provides methods that check individual words against config-driven
rules (particle typos, POS sequences, medial confusions, verb-particle
agreement, particle constraints, and config patterns).

Extracted from ``engine.py`` to reduce file size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myspellchecker.core.config import GrammarEngineConfig
    from myspellchecker.grammar.config import GrammarRuleConfig


class WordRuleMixin:
    """Mixin providing word-level grammar rule check methods.

    All methods use ``self.config`` (``GrammarRuleConfig``) and
    ``self.grammar_config`` (``GrammarEngineConfig``), plus POS tag
    helper methods from ``POSTagMixin``.
    """

    # --- Type stubs for attributes provided by SyntacticRuleChecker ---
    config: "GrammarRuleConfig"
    grammar_config: "GrammarEngineConfig"

    # --- Type stubs for methods provided by POSTagMixin (resolved via MRO) ---
    def _has_tag(self, pos_string: str | None, target_tag: str) -> bool: ...  # type: ignore[empty-body]
    def _get_all_tags(self, pos_string: str | None) -> set[str]: ...  # type: ignore[empty-body]
    def _get_primary_tag(self, pos_string: str | None, word: str | None = None) -> str: ...  # type: ignore[empty-body]

    def _check_particle_typos(
        self, word: str, prev_pos: str | None
    ) -> tuple[str, str, float] | None:
        """
        Check for common particle typing errors.

        Returns:
            Tuple of (correction, reason, confidence) or None if no error found.
        """
        typo_info = self.config.get_particle_typo(word)
        if not typo_info:
            return None

        correction = typo_info["correction"]
        confidence = typo_info.get("confidence", 0.9)
        desc = typo_info.get("meaning", "")
        context = typo_info.get("context", "any")
        excluded_pos = typo_info.get("excluded_pos", [])

        # Check exclusions: skip if preceding word's POS is in the exclusion list
        if prev_pos and excluded_pos:
            prev_tags = self._get_all_tags(prev_pos)
            if prev_tags & set(excluded_pos):
                return None

        # Context-specific validation
        if context == "after_verb":
            if self._has_tag(prev_pos, "V"):
                return (correction, f"After verb, should be {correction} ({desc})", confidence)
            return None
        elif context == "after_noun":
            if self._has_tag(prev_pos, "N"):
                return (correction, f"Likely {correction} ({desc})", confidence)
            return None

        # Default: return the mapping
        return (correction, desc, confidence)

    def _check_verb_particle_agreement(
        self, word: str, prev_pos: str | None, prev_word: str | None = None
    ) -> tuple[str, str, float] | None:
        """
        Check if particle correctly follows a verb.

        Some particles only make sense after verbs. This method checks:
        1. Verb particles appearing after non-verbs (potential typo)
        2. Common verb particle typos based on context
        3. Missing particles after verbs

        Args:
            word: Current word to check.
            prev_pos: POS tag of previous word.
            prev_word: Previous word (for context).

        Returns:
            Tuple of (correction, reason, confidence) or None if no error.
        """
        typo_info = self.config.get_particle_typo(word)
        if typo_info and typo_info.get("context") == "after_verb":
            if self._has_tag(prev_pos, "V"):
                correction = typo_info["correction"]
                meaning = typo_info.get("meaning", "")
                return (
                    correction,
                    f"After verb: {correction} ({meaning})",
                    self.grammar_config.medium_confidence,
                )

        # Check if word is a verb particle appearing after non-verb
        if self.config.is_verb_particle(word):
            if prev_pos and not self._has_tag(prev_pos, "V"):
                # Only flag certain high-confidence cases
                # Tense markers like "ခဲ့", "မယ်", "နေ" should follow verbs
                tense_markers = {"ခဲ့", "မယ်", "နေ", "မည်"}
                if word in tense_markers:
                    # These almost always follow verbs
                    if self._has_tag(prev_pos, "N"):
                        # Might be missing a verb between noun and tense marker
                        return (
                            word,
                            f"Tense marker '{word}' usually follows a verb",
                            self.grammar_config.tense_marker_confidence,
                        )

        word_correction = self.config.get_word_correction(word)
        if word_correction:
            # Respect excluded_pos from YAML (e.g., don't correct after verbs)
            excluded_pos = word_correction.get("excluded_pos", [])
            if prev_pos and excluded_pos:
                prev_tags = self._get_all_tags(prev_pos)
                if prev_tags & set(excluded_pos):
                    return None
            # Respect context requirements
            context = word_correction.get("context", "")
            if context == "context_dependent":
                return None  # Skip context-dependent entries without enough info
            correction = word_correction["correction"]
            meaning = word_correction.get("meaning", "")
            confidence = word_correction.get("confidence", self.grammar_config.medium_confidence)
            return (
                correction,
                f"Common error: {correction} ({meaning})",
                confidence,
            )

        return None

    def _check_medial_confusions(
        self, word: str, prev_pos: str | None, context_words: list[str] | None = None
    ) -> tuple[str, str, float] | None:
        """
        Check for medial character confusions.

        Common Myanmar typing errors involve swapping:
        - ya-pin and ya-yit
        - wa-hswe and ha-htoe
        - Missing asat/tone marks

        Uses configuration-based rules for comprehensive coverage.

        Args:
            word: Current word to check.
            prev_pos: POS tag of previous word.
            context_words: Surrounding words for context-aware checking.

        Returns:
            Tuple of (correction, reason, confidence) or None if no error.
        """
        confusion_info = self.config.get_medial_confusion(word)
        if confusion_info:
            correction = confusion_info.get("correction", "")
            context = confusion_info.get("context", "")
            meaning = confusion_info.get("meaning", "")

            # Context-specific validation
            if context == "after_verb":
                # Only fire when prev word is unambiguously a verb.
                # Words with both N and V tags (e.g., N|V)
                # are ambiguous -- flagging causes false positives.
                if self._has_tag(prev_pos, "V") and not self._has_tag(prev_pos, "N"):
                    return (
                        correction,
                        f"After verb: {correction} ({meaning})",
                        self.grammar_config.high_confidence,
                    )
                # Don't flag if not after verb (or ambiguous)
                return None
            elif context == "after_noun":
                if self._has_tag(prev_pos, "N"):
                    return (
                        correction,
                        f"After noun: {correction} ({meaning})",
                        self.grammar_config.medium_confidence,
                    )
                return None
            elif context == "context_dependent":
                # Lower confidence for context-dependent corrections
                return (
                    correction,
                    f"Possible: {correction} ({meaning})",
                    self.grammar_config.context_confidence_threshold,
                )
            else:
                # No specific context - apply with moderate confidence
                return (
                    correction,
                    f"Common error: {correction} ({meaning})",
                    self.grammar_config.pos_sequence_confidence,
                )

        return None

    def _check_particle_constraints(
        self, word: str, prev_pos: str | None, is_sentence_final: bool
    ) -> tuple[str, str, float] | None:
        """
        Check dynamic particle constraints loaded from config.

        Args:
            word: The particle to check.
            prev_pos: POS tag of the preceding word.
            is_sentence_final: Whether this word is at the end of the sentence.

        Returns:
            Tuple of (correction, reason, confidence) or None.
        """
        if not hasattr(self.config, "particle_constraints"):
            return None

        constraints = self.config.particle_constraints.get(word)
        if not constraints:
            return None

        # Check required preceding POS
        if "required_preceding" in constraints:
            required = constraints["required_preceding"]

            if prev_pos is None:
                # Position 0: no preceding word, so the constraint fails
                return (
                    word,
                    f"Particle '{word}' usually follows {required}",
                    self.grammar_config.verb_particle_confidence,
                )

            prev_tag = self._get_primary_tag(prev_pos)

            # Allow "P" match if any P_* tag is present, or exact match
            match = prev_tag in required

            # Special case: If required is ["V"] but we have "P" (verb particle),
            # that's often okay. e.g., V + P_part + P_sent is valid
            if not match and "V" in required and prev_tag == "P":
                match = True

            if not match:
                return (
                    word,
                    f"Particle '{word}' usually follows {required}, not {prev_tag}",
                    self.grammar_config.verb_particle_confidence,
                )

        # Check sentence final constraint
        if "sentence_final" in constraints:
            must_be_final = constraints["sentence_final"]
            if must_be_final and not is_sentence_final:
                return (
                    word,
                    f"Particle '{word}' is usually sentence-final",
                    self.grammar_config.sentence_final_confidence,
                )

        return None

    def _check_config_patterns(
        self,
        curr_word: str,
        prev_word: str | None,
        prev_pos: str | None,
        i: int,
        word_count: int,
    ) -> tuple[int, str, str, float] | None:
        """
        Check word patterns for common errors using YAML config.

        All patterns are defined in typo_corrections.yaml:
        - medial_confusions: Character-level typos
        - missing_asat: Words missing final asat (killer mark)
        - question_particles: Position-dependent question particle errors

        Returns (position, word, correction, confidence) or None if no error found.
        """
        # Skip medial confusions — already handled at higher priority
        # by _check_medial_confusions (Rule 2) with proper context guards.
        # Processing them again here produces duplicate lower-priority corrections.
        if self.config.get_medial_confusion(curr_word) is not None:
            return None

        # Check word corrections (e.g., missing asat patterns)
        word_correction = self.config.get_word_correction(curr_word)
        if word_correction and "correction" in word_correction:
            conf = word_correction.get("confidence", self.grammar_config.medium_confidence)
            # Respect excluded_pos from YAML
            excluded_pos = word_correction.get("excluded_pos", [])
            if prev_pos and excluded_pos:
                prev_tags = self._get_all_tags(prev_pos)
                if prev_tags & set(excluded_pos):
                    pass  # Skip this correction
                else:
                    context = word_correction.get("context", "")
                    if context != "context_dependent":
                        return (i, curr_word, word_correction["correction"], conf)
            else:
                context = word_correction.get("context", "")
                if context != "context_dependent":
                    return (i, curr_word, word_correction["correction"], conf)

        # Check question particle corrections (position-dependent)
        is_sentence_final = i == word_count - 1
        q_correction = self.config.get_question_particle_correction(
            curr_word, prev_pos, is_sentence_final
        )
        if q_correction:
            conf = q_correction.get("confidence", self.grammar_config.medium_confidence)
            return (i, curr_word, q_correction["correct"], conf)

        # Rules 9-11: Context-dependent patterns
        # Double particles, etc.
        # Flagged by other strategies (homophone, n-gram) -- not auto-corrected here

        return None
