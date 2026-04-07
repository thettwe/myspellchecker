"""
Myanmar Particle Context Validator.

This module implements validation for Myanmar particle usage given
verb/noun context. Myanmar particles (postpositions) must agree with
the verb type and syntactic role of surrounding words.

Common misuse patterns addressed:
    - Subject marker က used where object marker ကို is needed
    - Static locative မှာ used with motion verbs (should be ကို/သို့)
    - Sequential ပြီး confused with sentence-final completion ပြီ
    - Negated sentences ending with affirmative တယ် instead of ဘူး
    - Register mismatches (colloquial လို့ in formal text, etc.)

Features:
    - Particle confusion pair detection from YAML rules
    - Verb-particle frame compatibility checking
    - POS-tag-aware validation (with fallback heuristics)
    - Configurable confidence thresholds per rule
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from myspellchecker.core.constants import ET_PARTICLE_MISUSE
from myspellchecker.core.response import GrammarError
from myspellchecker.utils.logging_utils import get_logger
from myspellchecker.utils.singleton import ThreadSafeSingleton

logger = get_logger(__name__)

__all__ = [
    "ParticleChecker",
    "ParticleContextError",
    "get_particle_checker",
]

# Singleton registry for ParticleChecker
_singleton: ThreadSafeSingleton["ParticleChecker"] = ThreadSafeSingleton()

# Default path to the particle_contexts YAML rule file.
DEFAULT_PARTICLE_CONTEXTS_PATH = (
    Path(__file__).parent.parent.parent / "rules" / "particle_contexts.yaml"
)

# POS tags that identify verbs (including multi-tag entries like V|N).
_VERB_POS_TAGS = frozenset({"V"})

# POS tags that identify nouns.
_NOUN_POS_TAGS = frozenset({"N", "TN"})

# POS tags that identify particles.
_PARTICLE_POS_TAGS = frozenset({"PPM", "PART", "P", "SFP", "CONJ", "P_LOC", "P_OBJ", "P_SUBJ"})


@dataclass
class ParticleContextError(GrammarError):
    """
    Represents a particle context error.

    Extends GrammarError to integrate with the spell checker's error hierarchy.

    Attributes:
        text: The erroneous particle (inherited from Error).
        position: Index of the error in the word list (inherited from Error).
        suggestions: List of suggested corrections (inherited from Error).
        error_type: Type of error (particle_misuse).
        confidence: Confidence score (0.0-1.0) (inherited from Error).
        reason: Human-readable explanation (inherited from GrammarError).
        word: Alias for 'text' (inherited from GrammarError).
        suggestion: First suggestion (inherited from GrammarError).
    """

    # Override default error_type
    error_type: str = field(default=ET_PARTICLE_MISUSE)


# ---- Internal data structures ----


@dataclass
class _ConfusionRule:
    """A single particle confusion pair loaded from YAML."""

    particle: str
    confused_with: str
    context: str
    description: str
    confidence: float


@dataclass
class _VerbFrame:
    """A verb-particle frame loaded from YAML."""

    verbs: frozenset[str]
    required_particles: frozenset[str]
    incompatible_particles: frozenset[str]
    note: str


class ParticleChecker:
    """
    Validates Myanmar particle usage in verb/noun context.

    This checker identifies:
    - Particle confusion pairs (e.g., က/ကို, မှာ/ကို, ပြီ/ပြီး)
    - Verb-particle frame violations (motion verb + static locative)
    - Register-mismatched particle choices

    Attributes:
        confusion_lookup: Mapping from particle -> list of confusion rules.
        verb_frames: List of verb-particle frame rules.
        all_particles: Set of all particles known to this checker.
    """

    def __init__(self, config_path: str | None = None) -> None:
        """
        Initialize the particle checker.

        Args:
            config_path: Optional path to particle_contexts.yaml.
                Defaults to the bundled rules file.
        """
        self.confusion_lookup: dict[str, list[_ConfusionRule]] = {}
        self.verb_frames: list[_VerbFrame] = []
        self.all_particles: set[str] = set()

        # Reverse lookup: confused_with -> list of rules (for detecting
        # when the *wrong* particle is present and the *right* one is needed).
        self._reverse_confusion: dict[str, list[_ConfusionRule]] = {}

        # verb -> list of frame indices (for fast lookup)
        self._verb_to_frames: dict[str, list[int]] = {}

        path = Path(config_path) if config_path else DEFAULT_PARTICLE_CONTEXTS_PATH
        self._load_config(path)

    def _load_config(self, path: Path) -> None:
        """Load particle context rules from YAML."""
        if not path.exists():
            logger.warning("Particle contexts config not found at %s", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data: dict[str, Any] = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as e:
            logger.warning("Failed to load particle contexts from %s: %s", path, e)
            return

        # Parse confusion pairs
        for item in data.get("particle_confusions", []):
            rule = _ConfusionRule(
                particle=item["particle"],
                confused_with=item["confused_with"],
                context=item.get("context", ""),
                description=item.get("description", ""),
                confidence=float(item.get("confidence", 0.70)),
            )
            self.confusion_lookup.setdefault(rule.particle, []).append(rule)
            self._reverse_confusion.setdefault(rule.confused_with, []).append(rule)
            self.all_particles.add(rule.particle)
            self.all_particles.add(rule.confused_with)

        # Parse verb-particle frames
        for idx, item in enumerate(data.get("verb_particle_frames", [])):
            verbs = frozenset(item.get("verbs", []))
            frame = _VerbFrame(
                verbs=verbs,
                required_particles=frozenset(item.get("required_particles", [])),
                incompatible_particles=frozenset(item.get("incompatible_particles", [])),
                note=item.get("note", ""),
            )
            self.verb_frames.append(frame)
            for verb in verbs:
                self._verb_to_frames.setdefault(verb, []).append(idx)
                self.all_particles.update(frame.required_particles)
                self.all_particles.update(frame.incompatible_particles)

        logger.debug(
            "Loaded %d confusion rules, %d verb-particle frames",
            sum(len(v) for v in self.confusion_lookup.values()),
            len(self.verb_frames),
        )

    # ---- POS-tag helpers ----

    @staticmethod
    def _has_tag(pos: str | None, tag_set: frozenset[str]) -> bool:
        """Check if a POS tag string contains any tag in *tag_set*.

        Handles multi-tags like ``V|N`` by splitting on ``|``.
        """
        if not pos:
            return False
        return bool(set(pos.split("|")) & tag_set)

    @staticmethod
    def _is_verb_pos(pos: str | None) -> bool:
        """Check whether *pos* indicates a verb."""
        return ParticleChecker._has_tag(pos, _VERB_POS_TAGS)

    @staticmethod
    def _is_noun_pos(pos: str | None) -> bool:
        """Check whether *pos* indicates a noun."""
        return ParticleChecker._has_tag(pos, _NOUN_POS_TAGS)

    @staticmethod
    def _is_particle_pos(pos: str | None) -> bool:
        """Check whether *pos* indicates a particle."""
        return ParticleChecker._has_tag(pos, _PARTICLE_POS_TAGS)

    # ---- Word-list heuristics (fallback when POS tags unavailable) ----

    # Common motion verbs (used as heuristic when POS tags are absent).
    _MOTION_VERBS: frozenset[str] = frozenset({"သွား", "လာ", "ပြန်", "ထွက်", "ဝင်", "တက်", "ဆင်း"})

    # Common stative verbs.
    _STATIVE_VERBS: frozenset[str] = frozenset({"ရှိ", "နေ", "ထိုင်", "အိပ်", "ရပ်", "တည်"})

    # Negation prefix.
    _NEGATION_PREFIX: str = "မ"

    def _find_preceding_verb(
        self,
        words: list[str],
        pos_tags: list[str | None] | None,
        particle_index: int,
    ) -> str | None:
        """Find the verb immediately preceding the particle at *particle_index*.

        Searches backward up to 2 positions (to skip auxiliary particles).
        Uses POS tags if available, otherwise falls back to heuristic verb
        sets and suffix matching.

        Returns:
            The verb string, or None if no verb is found.
        """
        for offset in range(1, min(3, particle_index + 1)):
            idx = particle_index - offset
            word = words[idx]
            if pos_tags and idx < len(pos_tags):
                if self._is_verb_pos(pos_tags[idx]):
                    return word
                # Skip particles between verb and current particle
                if self._is_particle_pos(pos_tags[idx]):
                    continue
                # Hit a non-verb, non-particle — stop
                break
            else:
                # Heuristic: check known verb sets and common suffixes
                if word in self._MOTION_VERBS or word in self._STATIVE_VERBS:
                    return word
                # Check any verb frame verb set
                if word in self._verb_to_frames:
                    return word
                break
        return None

    def _find_preceding_noun(
        self,
        words: list[str],
        pos_tags: list[str | None] | None,
        particle_index: int,
    ) -> str | None:
        """Find the noun immediately preceding the particle.

        Returns:
            The noun string, or None.
        """
        if particle_index == 0:
            return None
        idx = particle_index - 1
        if pos_tags and idx < len(pos_tags):
            if self._is_noun_pos(pos_tags[idx]):
                return words[idx]
        else:
            # Without POS tags, assume the word before a particle is a noun
            # unless it's a known verb.
            word = words[idx]
            if word not in self._MOTION_VERBS and word not in self._STATIVE_VERBS:
                return word
        return None

    # ---- Main validation ----

    def validate_sequence(
        self,
        words: list[str],
        pos_tags: list[str | None] | None = None,
    ) -> list[ParticleContextError]:
        """
        Validate particle usage in a word sequence.

        Checks for:
        1. Particle confusion pairs (wrong particle for context)
        2. Verb-particle frame violations (incompatible verb+particle)

        Args:
            words: List of words in the sentence.
            pos_tags: Optional POS tags corresponding to each word.

        Returns:
            List of ParticleContextError objects.

        Examples:
            >>> checker = ParticleChecker()
            >>> errors = checker.validate_sequence(
            ...     ["ကျောင်း", "ကို", "ရှိ", "တယ်"]
            ... )
            >>> len(errors) > 0  # ကို with stative verb ရှိ
            True
        """
        errors: list[ParticleContextError] = []
        flagged_positions: set[int] = set()

        for i, word in enumerate(words):
            if i in flagged_positions:
                continue

            # --- Check 1: Verb-particle frame violations ---
            # Look for a verb preceding this particle and check frame compat.
            if word in self.all_particles:
                verb = self._find_preceding_verb(words, pos_tags, i)
                if verb:
                    frame_error = self._check_verb_frame(verb, word, i)
                    if frame_error:
                        errors.append(frame_error)
                        flagged_positions.add(i)
                        continue

            # --- Check 2: Particle confusion detection ---
            # If the word is the *confused_with* side of a rule, check
            # whether the context suggests the correct particle instead.
            if word in self._reverse_confusion:
                error = self._check_confusion(words, pos_tags, i)
                if error:
                    errors.append(error)
                    flagged_positions.add(i)

        return errors

    def _check_verb_frame(
        self,
        verb: str,
        particle: str,
        position: int,
    ) -> ParticleContextError | None:
        """Check if *particle* is compatible with *verb* per frame rules.

        Returns an error if the particle is in the verb's incompatible set.
        """
        frame_indices = self._verb_to_frames.get(verb, [])
        for idx in frame_indices:
            frame = self.verb_frames[idx]
            if particle in frame.incompatible_particles:
                # Suggest the first required particle as correction.
                suggestion = next(iter(frame.required_particles), particle)
                return ParticleContextError(
                    text=particle,
                    position=position,
                    suggestions=[suggestion],
                    confidence=0.70,
                    reason=(
                        f"Particle '{particle}' is incompatible with verb "
                        f"'{verb}'. {frame.note} "
                        f"Consider using '{suggestion}'."
                    ),
                )
        return None

    def _check_confusion(
        self,
        words: list[str],
        pos_tags: list[str | None] | None,
        position: int,
    ) -> ParticleContextError | None:
        """Check whether the particle at *position* matches a confusion rule.

        The word at *position* is the ``confused_with`` side. If surrounding
        context suggests that the correct particle should be used instead,
        return an error.
        """
        word = words[position]
        rules = self._reverse_confusion.get(word, [])

        for rule in rules:
            if self._context_matches(words, pos_tags, position, rule):
                return ParticleContextError(
                    text=word,
                    position=position,
                    suggestions=[rule.particle],
                    confidence=rule.confidence,
                    reason=(
                        f"{rule.description}: '{word}' should be '{rule.particle}' in this context."
                    ),
                )

        return None

    def _context_matches(
        self,
        words: list[str],
        pos_tags: list[str | None] | None,
        position: int,
        rule: _ConfusionRule,
    ) -> bool:
        """Determine whether the surrounding context matches the confusion rule.

        Uses POS tags when available, otherwise falls back to heuristic
        word-list checks.
        """
        context = rule.context

        # --- Negation + affirmative SFP ---
        if context == "negative_sentence_ending":
            # Check if any preceding word starts with negation prefix.
            # Exclude known particles that happen to start with မ
            # (e.g., မှာ "at", မှ "from") — those are not negation.
            for j in range(max(0, position - 3), position):
                w = words[j]
                if (
                    w.startswith(self._NEGATION_PREFIX)
                    and len(w) > 1
                    and w not in self.all_particles
                ):
                    return True
            return False

        # --- Motion verb + static particle ---
        if context == "static_location":
            verb = self._find_preceding_verb(words, pos_tags, position)
            if verb and verb in self._MOTION_VERBS:
                # Motion verb context — static particle is wrong
                return True
            # Stative verb context — static particle is correct → no error
            if verb and verb in self._STATIVE_VERBS:
                return False
            return False

        # --- Directional motion ---
        if context == "directional_motion":
            verb = self._find_next_verb(words, pos_tags, position)
            if verb and verb in self._MOTION_VERBS:
                return True
            return False

        # --- Source / origin ---
        if context == "source_origin":
            verb = self._find_next_verb(words, pos_tags, position)
            if verb and verb in self._MOTION_VERBS:
                return True
            return False

        # --- Sequential action (ပြီ where ပြီး needed) ---
        if context == "sequential_action":
            # If there is a following verb, the speaker likely means
            # "after doing X" which needs ပြီး.
            if position + 1 < len(words):
                next_word = words[position + 1]
                if pos_tags and position + 1 < len(pos_tags):
                    if self._is_verb_pos(pos_tags[position + 1]):
                        return True
                else:
                    # Heuristic: if next word is a known verb
                    if next_word in self._verb_to_frames or next_word in self._MOTION_VERBS:
                        return True
            return False

        # --- Sentence-final completion (ပြီး where ပြီ needed) ---
        if context == "sentence_final_completion":
            # If this is the last word or followed only by punctuation.
            if position == len(words) - 1:
                return True
            if position + 1 < len(words) and words[position + 1] in {"။", ""}:
                return True
            return False

        # --- Object noun context ---
        if context == "after_object_noun":
            # Check if a transitive verb follows this particle.
            verb = self._find_next_verb(words, pos_tags, position)
            if verb:
                return True
            return False

        # --- Subject noun context ---
        if context == "after_subject_noun":
            # Check if the preceding word is a noun and following is
            # an intransitive verb.
            noun = self._find_preceding_noun(words, pos_tags, position)
            if noun:
                verb = self._find_next_verb(words, pos_tags, position)
                if verb and verb in self._STATIVE_VERBS:
                    return True
            return False

        # For other contexts (formal_location, formal_reason_clause, etc.)
        # we do not have enough signal to auto-detect — return False
        # to avoid false positives.
        return False

    def _find_next_verb(
        self,
        words: list[str],
        pos_tags: list[str | None] | None,
        particle_index: int,
    ) -> str | None:
        """Find the verb following the particle at *particle_index*.

        Searches forward up to 3 positions.
        """
        for offset in range(1, min(4, len(words) - particle_index)):
            idx = particle_index + offset
            word = words[idx]
            if pos_tags and idx < len(pos_tags):
                if self._is_verb_pos(pos_tags[idx]):
                    return word
                if self._is_particle_pos(pos_tags[idx]):
                    continue
                break
            else:
                if word in self._MOTION_VERBS or word in self._STATIVE_VERBS:
                    return word
                if word in self._verb_to_frames:
                    return word
                break
        return None


# Module-level singleton
def get_particle_checker() -> ParticleChecker:
    """
    Get the default ParticleChecker singleton.

    Returns:
        ParticleChecker instance.
    """
    return _singleton.get(ParticleChecker)
