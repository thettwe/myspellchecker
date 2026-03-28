"""
Syntactic/Grammar Rule Checker for Myanmar spell checking.

This module implements a rule-based checker that uses Part-of-Speech (POS) tags
to validate word sequences. It operates at Layer 2.5 (Syntactic) of the pipeline,
catching errors that N-grams might miss due to data sparsity.

Myanmar Grammar Reference:
- Core POS tags: N (Noun), V (Verb), ADJ (Adjective), ADV (Adverb),
                 CONJ (Conjunction), INT (Interjection), NUM (Number)
- HuggingFace transformer tags: PPM (Postpositional Marker), PART (Particle)
- Granular particle tags: P_SUBJ, P_OBJ, P_LOC, P_SENT, P_POSS, etc.
- Particles follow their head words (Noun+P, Verb+P)
- SOV word order: Subject + Object + Verb

Tag Mapping:
    HuggingFace tags (PPM, PART) are mapped to granular tags (P_SUBJ, P_OBJ, etc.)
    via the particle_tags config. When no specific mapping exists, they fall back
    to the generic 'P' tag for rule matching.

Method groups are factored into mixins under ``grammar.mixins``:
- POSTagMixin: POS tag utility helpers
- WordRuleMixin: word-level rule checks
- CheckerDelegationMixin: specialized checker delegation
- SentenceStructureMixin: sentence-level structure checks
- ConfigGrammarMixin: YAML config-driven grammar rules
"""

from __future__ import annotations

from dataclasses import dataclass

from myspellchecker.core.config import GrammarEngineConfig
from myspellchecker.grammar.checkers.aspect import AspectChecker
from myspellchecker.grammar.checkers.classifier import ClassifierChecker
from myspellchecker.grammar.checkers.compound import CompoundChecker
from myspellchecker.grammar.checkers.merged_word import MergedWordChecker
from myspellchecker.grammar.checkers.negation import NegationChecker
from myspellchecker.grammar.checkers.register import RegisterChecker
from myspellchecker.grammar.config import get_grammar_config
from myspellchecker.grammar.mixins import (
    CheckerDelegationMixin,
    ConfigGrammarMixin,
    POSTagMixin,
    SentenceStructureMixin,
    WordRuleMixin,
)
from myspellchecker.providers import DictionaryProvider

__all__ = [
    "RuleMatch",
    "RulePriority",
    "SyntacticRuleChecker",
]


@dataclass
class RuleMatch:
    """
    Represents a grammar rule match with priority for conflict resolution.

    Attributes:
        position: Word position in sentence.
        word: The matched word.
        suggestion: Suggested correction.
        priority: Rule priority (higher = more important).
        rule_name: Name of the matching rule for debugging.
        confidence: Confidence score for the match.
    """

    position: int
    word: str
    suggestion: str
    priority: int
    rule_name: str
    confidence: float = 0.8


# Default Grammar Engine configuration (module-level singleton)
_default_grammar_config = GrammarEngineConfig()

# Rule priority constants
# Higher priority = checked first, wins in case of conflicts
# Priority ordering: specific rules > general rules > fallback rules
#
# NOTE: These class-level defaults are overridden per-instance from
# GrammarEngineConfig.rule_priorities in SyntacticRuleChecker.__init__.
class RulePriority:
    """Rule priority constants for conflict resolution."""

    SENTENCE_BOUNDARY = 100
    MEDIAL_CONFUSION = 90
    PARTICLE_TYPO = 85
    VERB_PARTICLE_AGREEMENT = 70
    POS_SEQUENCE = 65
    CONFIG_PATTERN = 50
    CLASSIFIER = 45
    MERGED_WORD = 42
    COMPOUND = 40
    REGISTER = 20
    ASPECT = 15
    NEGATION = 10

    @classmethod
    def from_config(cls, priorities: dict[str, int]) -> "RulePriority":
        """Create a RulePriority with values from config dict."""
        instance = cls()
        for key, value in priorities.items():
            attr_name = key.upper()
            if hasattr(instance, attr_name):
                setattr(instance, attr_name, value)
        return instance


class SyntacticRuleChecker(
    POSTagMixin,
    WordRuleMixin,
    CheckerDelegationMixin,
    SentenceStructureMixin,
    ConfigGrammarMixin,
):
    """
    Rule-based syntactic spell checker using POS tags.

    Implements grammar rules for Myanmar spell checking:
    - Particle typo corrections
    - POS-based particle validation
    - Medial confusion detection (ျ vs ြ, ွ vs ှ)
    - Sentence structure validation
    - Verb-particle agreement checks

    Attributes:
        provider: DictionaryProvider for POS tag access.
        config: GrammarRuleConfig for customizable grammar rules.
        grammar_config: GrammarEngineConfig for confidence thresholds.
    """

    def __init__(
        self,
        provider: DictionaryProvider,
        config_path: str | None = None,
        grammar_config: GrammarEngineConfig | None = None,
    ) -> None:
        """
        Initialize the syntactic rule checker.

        Args:
            provider: Dictionary provider for POS tag lookup.
            config_path: Optional path to custom grammar rules YAML.
            grammar_config: GrammarEngineConfig for confidence settings.
        """
        self.provider = provider
        self.config = get_grammar_config(config_path)
        self.grammar_config = grammar_config or _default_grammar_config
        # Override rule priorities from config
        self.rule_priority = RulePriority.from_config(self.grammar_config.rule_priorities)
        self.aspect_checker = AspectChecker()
        self.classifier_checker = ClassifierChecker()
        self.compound_checker = CompoundChecker()
        self.merged_word_checker = MergedWordChecker()
        self.negation_checker = NegationChecker()
        self.register_checker = RegisterChecker(provider=provider)

    def check_sequence(self, words: list[str]) -> list[tuple[int, str, str, float]]:
        """
        Check a sequence of words for grammatical errors.

        Implements comprehensive grammar checking:
        1. Particle typo detection (YAML config-driven)
        2. Medial confusion detection (ျ vs ြ, ွ vs ှ)
        3. Verb-particle agreement validation
        4. POS sequence validation
        5. Sentence structure checks
        6. Common error patterns

        Args:
            words: List of words in the sentence.

        Returns:
            List of (index, error_word, suggestion, confidence) tuples.
        """
        if not words:
            return []

        corrections: list[tuple[int, str, str, float]] = []
        pos_tags = [self.provider.get_word_pos(w) for w in words]

        # Rule 0: Check sentence boundary constraints
        self._add_boundary_errors(words, pos_tags, corrections)

        # Rules 1-11: Word-level checks
        for i in range(len(words)):
            correction = self._check_word_at_position(words, pos_tags, i)
            if correction:
                corrections.append(correction)

        # Rules 13-18: Checker-based validation
        self._add_checker_errors(words, pos_tags, corrections)

        return corrections

    def _add_boundary_errors(
        self,
        words: list[str],
        pos_tags: list[str | None],
        corrections: list[tuple[int, str, str, float]],
    ) -> None:
        """Add sentence boundary constraint errors to corrections list."""
        boundary_errors = self._check_sentence_boundaries(words, pos_tags)
        for idx, err_word, suggestion, confidence in boundary_errors:
            if confidence >= self.grammar_config.default_confidence_threshold:
                corrections.append((idx, err_word, suggestion, confidence))

    def _check_word_at_position(
        self,
        words: list[str],
        pos_tags: list[str | None],
        i: int,
    ) -> tuple[int, str, str, float] | None:
        """
        Check a single word at position i for errors.

        Uses priority-based conflict resolution to select the best
        correction when multiple rules match the same word.

        Returns correction tuple (index, word, suggestion, confidence) or None.
        """
        curr_word = words[i]
        prev_word = words[i - 1] if i > 0 else None
        prev_pos = pos_tags[i - 1] if i > 0 else None

        # Collect all matching rules with priorities
        matches: list[RuleMatch] = []

        # Rule 1: Check particle typos (priority 85)
        typo_result = self._check_particle_typos(curr_word, prev_pos)
        if typo_result:
            # Check followed_by constraint: only flag if the next word matches
            typo_info = self.config.get_particle_typo(curr_word)
            followed_by = typo_info.get("followed_by") if typo_info else None
            if followed_by:
                next_word = words[i + 1] if i + 1 < len(words) else None
                if next_word not in followed_by:
                    typo_result = None
            if typo_result:
                matches.append(
                    RuleMatch(
                        position=i,
                        word=curr_word,
                        suggestion=typo_result[0],
                        confidence=typo_result[2] if len(typo_result) > 2 else 0.8,
                        priority=self.rule_priority.PARTICLE_TYPO,
                        rule_name="particle_typo",
                    )
                )

        # Rule 2: Check medial confusions (priority 90 - higher than particle typo)
        medial_result = self._check_medial_confusions(curr_word, prev_pos, words)
        if medial_result:
            matches.append(
                RuleMatch(
                    position=i,
                    word=curr_word,
                    suggestion=medial_result[0],
                    confidence=medial_result[2] if len(medial_result) > 2 else 0.8,
                    priority=self.rule_priority.MEDIAL_CONFUSION,
                    rule_name="medial_confusion",
                )
            )

        # Rule 3: Check verb-particle agreement (priority 70)
        verb_result = self._check_verb_particle_agreement(curr_word, prev_pos, prev_word)
        if verb_result:
            matches.append(
                RuleMatch(
                    position=i,
                    word=curr_word,
                    suggestion=verb_result[0],
                    confidence=verb_result[2] if len(verb_result) > 2 else 0.8,
                    priority=self.rule_priority.VERB_PARTICLE_AGREEMENT,
                    rule_name="verb_particle_agreement",
                )
            )

        # Rule 3.5: Check dynamic particle constraints (metadata-based)
        is_final = i == len(words) - 1
        constraint_result = self._check_particle_constraints(curr_word, prev_pos, is_final)
        if constraint_result:
            matches.append(
                RuleMatch(
                    position=i,
                    word=curr_word,
                    suggestion=constraint_result[0],
                    priority=self.rule_priority.VERB_PARTICLE_AGREEMENT,
                    rule_name="particle_constraint",
                    confidence=constraint_result[2] if len(constraint_result) > 2 else 0.7,
                )
            )

        # Rules 4-11: Check config-driven patterns (priority 50)
        pattern_result = self._check_config_patterns(curr_word, prev_word, prev_pos, i, len(words))
        if pattern_result:
            matches.append(
                RuleMatch(
                    position=pattern_result[0],
                    word=pattern_result[1],
                    suggestion=pattern_result[2],
                    confidence=pattern_result[3] if len(pattern_result) > 3 else 0.8,
                    priority=self.rule_priority.CONFIG_PATTERN,
                    rule_name="config_pattern",
                )
            )

        # Conflict resolution: select highest priority match, filter by confidence
        if matches:
            # Filter out matches below the confidence threshold
            threshold = self.grammar_config.default_confidence_threshold
            valid_matches = [m for m in matches if m.confidence >= threshold]
            if valid_matches:
                best_match = max(valid_matches, key=lambda m: m.priority)
                return (
                    best_match.position,
                    best_match.word,
                    best_match.suggestion,
                    best_match.confidence,
                )

        return None

    def _add_checker_errors(
        self,
        words: list[str],
        pos_tags: list[str | None],
        corrections: list[tuple[int, str, str, float]],
    ) -> None:
        """
        Add errors from specialized checkers (rules 13-18+).

        Collects errors from classifier, structure, register, negation,
        aspect, compound, merged word checkers, and config-based rules
        from grammar_rules.yaml.
        """
        threshold = self.grammar_config.default_confidence_threshold

        # Collect all checker errors
        all_errors: list[tuple[int, str, str, float]] = []
        all_errors.extend(self._check_classifiers(words, pos_tags))
        all_errors.extend(self._check_sentence_structure(words, pos_tags))
        all_errors.extend(self._check_register(words))
        all_errors.extend(self._check_negation(words))
        all_errors.extend(self._check_aspect(words, pos_tags))
        all_errors.extend(self._check_compound(words))

        # Merged word detection (requires POS tags)
        all_errors.extend(self._check_merged_words(words, pos_tags))

        # Config-based validation from grammar_rules.yaml
        all_errors.extend(self._check_particle_chains(words, pos_tags))
        all_errors.extend(self._check_clause_linkage_from_config(words, pos_tags))
        # NOTE: _check_register_from_config, _check_negation_from_config, and
        # _check_classifier_from_config removed — their functionality is fully
        # covered by the dedicated checkers (RegisterChecker, NegationChecker,
        # ClassifierChecker) called above via _check_register/_check_negation/
        # _check_classifiers. Running both produced overlapping errors.

        # Tense agreement: future time word + past tense marker = error
        all_errors.extend(self._check_tense_agreement(words))

        # Filter by confidence and avoid duplicates
        # Deduplicate by (position, suggestion) to allow multiple different
        # errors at the same position (e.g., classifier + negation issues)
        existing_errors = {(c[0], c[2]) for c in corrections}  # (position, suggestion)
        for idx, err_word, suggestion, confidence in all_errors:
            error_key = (idx, suggestion)
            if confidence >= threshold and error_key not in existing_errors:
                corrections.append((idx, err_word, suggestion, confidence))
                existing_errors.add(error_key)
