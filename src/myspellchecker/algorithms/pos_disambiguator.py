"""
POS Disambiguator for Myanmar Text.

This module provides context-based Part-of-Speech (POS) disambiguation
for ambiguous words that have multiple possible POS tags (e.g., N|V, ADJ|N).

The disambiguator implements five rules (R1-R5) derived from Myanmar
linguistic research to resolve ambiguous words based on their context:

- R1: Noun after Verb
- R2: Adjective before Noun
- R3: Verb before Particle
- R4: Noun after Determiner
- R5: Verb after Adverb

Example:
    >>> disambiguator = POSDisambiguator()
    >>> # Word "ကြီး" can be ADJ|N|V
    >>> result = disambiguator.disambiguate_in_context(
    ...     word="ကြီး",
    ...     word_pos_tags=frozenset({"ADJ", "N", "V"}),
    ...     prev_word_pos="V",
    ...     next_word_pos="PPM",
    ... )
    >>> print(result.resolved_pos)  # "N" (Rule R1: after verb)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum

from myspellchecker.core.config.tagger_configs import POSTaggerConfig
from myspellchecker.grammar.config import get_grammar_config


class DisambiguationRule(Enum):
    """Disambiguation rules applied to resolve multi-POS words."""

    R1_NOUN_AFTER_VERB = "R1"
    R2_ADJ_BEFORE_NOUN = "R2"
    R3_VERB_BEFORE_PARTICLE = "R3"
    R4_NOUN_AFTER_DETERMINER = "R4"
    R5_VERB_AFTER_ADVERB = "R5"
    NO_RULE = "none"


@dataclass
class DisambiguationResult:
    """Result of POS disambiguation for a word.

    Attributes:
        word: The disambiguated word.
        original_pos_tags: Original set of possible POS tags.
        resolved_pos: The resolved single POS tag.
        rule_applied: Which disambiguation rule was applied.
        confidence: Confidence in the disambiguation (0.0-1.0).
        context_used: Description of the context that triggered the rule.
    """

    word: str
    original_pos_tags: frozenset[str]
    resolved_pos: str
    rule_applied: DisambiguationRule
    confidence: float
    context_used: str

    def __repr__(self) -> str:
        return (
            f"DisambiguationResult(word='{self.word}', "
            f"pos='{self.resolved_pos}', "
            f"rule={self.rule_applied.value}, "
            f"conf={self.confidence:.2f})"
        )


# Determiners and demonstratives that indicate noun context
DETERMINERS = frozenset(
    {
        "ဤ",  # this
        "ယင်း",  # that
        "ထို",  # that
        "ဒီ",  # this (colloquial)
        "အဲဒီ",  # that (colloquial)
        "တစ်",  # one/a
        "တချို့",  # some
        "အချို့",  # some
        "အားလုံး",  # all
        "တစ်ခု",  # one (thing)
    }
)

# Manner adverb markers that indicate verb context
ADVERB_MARKERS = frozenset(
    {
        "လျင်မြန်စွာ",  # quickly
        "ဖြည်းဖြည်း",  # slowly
        "အလွန်",  # very (degree adverb)
        "အရမ်း",  # very much (degree adverb)
        "ကောင်းစွာ",  # well
        "စွာ",  # -ly (adverb suffix)
        "ချက်ချင်း",  # immediately
        "အမြန်",  # fast
        "နှေးနှေး",  # slowly
    }
)

# Degree adverbs that typically modify adjectives (e.g., အလွန် ကြီး = very big)
DEGREE_ADVERBS = frozenset(
    {
        "အလွန်",  # very
        "အရမ်း",  # very much
    }
)

# Particle POS tags that indicate preceding word is verb
PARTICLE_POS_TAGS = frozenset(
    {
        "P_SENT",  # Sentence-final particle
        "P_MOD",  # Modifying particle
        "PPM",  # Post-positional marker
    }
)

# POS tags that indicate verb context
VERB_TAGS = frozenset({"V", "AUX"})

# POS tags that indicate adjective can modify
MODIFIABLE_TAGS = frozenset({"N", "PRON"})


class POSDisambiguator:
    """
    Context-based POS disambiguator using R1-R5 rules.

    This class applies linguistic rules to resolve ambiguous words
    that have multiple possible POS tags based on their sentence context.

    Example:
        >>> disambiguator = POSDisambiguator()
        >>> # "ကြီး" can be ADJ, N, or V
        >>> result = disambiguator.disambiguate_in_context(
        ...     word="ကြီး",
        ...     word_pos_tags=frozenset({"ADJ", "N", "V"}),
        ...     prev_word_pos="V",
        ...     next_word_pos=None,
        ... )
        >>> print(result.resolved_pos)  # "N" (R1: noun after verb)
    """

    def __init__(
        self,
        config_path: str | None = None,
        pos_tagger_config: POSTaggerConfig | None = None,
    ):
        """Initialize the disambiguator.

        Args:
            config_path: Path to grammar configuration YAML file.
            pos_tagger_config: Optional POSTaggerConfig for centralized
                disambiguation confidence values.
        """
        self.grammar_config = get_grammar_config(config_path)
        self._pos_config = pos_tagger_config or POSTaggerConfig()

        # Rule priority order
        self._rules = [
            (DisambiguationRule.R3_VERB_BEFORE_PARTICLE, self._apply_r3),
            (DisambiguationRule.R5_VERB_AFTER_ADVERB, self._apply_r5),
            (DisambiguationRule.R1_NOUN_AFTER_VERB, self._apply_r1),
            (DisambiguationRule.R2_ADJ_BEFORE_NOUN, self._apply_r2),
            (DisambiguationRule.R4_NOUN_AFTER_DETERMINER, self._apply_r4),
        ]

    def disambiguate_in_context(
        self,
        word: str,
        word_pos_tags: frozenset[str],
        prev_word: str | None = None,
        prev_word_pos: str | None = None,
        next_word: str | None = None,
        next_word_pos: str | None = None,
    ) -> DisambiguationResult:
        """
        Disambiguate a multi-POS word based on context.

        Args:
            word: The word to disambiguate.
            word_pos_tags: Set of possible POS tags for the word.
            prev_word: Previous word in the sentence (optional).
            prev_word_pos: POS tag of previous word (optional).
            next_word: Next word in the sentence (optional).
            next_word_pos: POS tag of next word (optional).

        Returns:
            DisambiguationResult with the resolved POS tag.

        Example:
            >>> result = disambiguator.disambiguate_in_context(
            ...     word="ကြီး",
            ...     word_pos_tags=frozenset({"ADJ", "N", "V"}),
            ...     next_word_pos="P_SENT",
            ... )
            >>> print(result.resolved_pos)  # "V" (R3: before particle)
        """
        if not word_pos_tags or len(word_pos_tags) == 1:
            # No ambiguity
            resolved = next(iter(word_pos_tags)) if word_pos_tags else None
            return DisambiguationResult(
                word=word,
                original_pos_tags=word_pos_tags,
                resolved_pos=resolved or "",
                rule_applied=DisambiguationRule.NO_RULE,
                confidence=(
                    self._pos_config.disambiguation_confidence_resolved
                    if resolved
                    else self._pos_config.disambiguation_confidence_unresolved
                ),
                context_used="unambiguous",
            )

        # Apply rules in priority order
        for _rule, rule_func in self._rules:
            result = rule_func(
                word=word,
                word_pos_tags=word_pos_tags,
                prev_word=prev_word,
                prev_word_pos=prev_word_pos,
                next_word=next_word,
                next_word_pos=next_word_pos,
            )
            if result is not None:
                return result

        # No rule matched - return default (most common POS)
        default_pos = self._select_default_pos(word_pos_tags)
        return DisambiguationResult(
            word=word,
            original_pos_tags=word_pos_tags,
            resolved_pos=default_pos,
            rule_applied=DisambiguationRule.NO_RULE,
            confidence=self._pos_config.disambiguation_confidence_fallback,
            context_used="default (no context match)",
        )

    def _select_default_pos(self, tags: frozenset[str]) -> str:
        """Select default POS when no rule matches (based on frequency)."""
        # Priority: N > V > ADJ > others
        priority = ["N", "V", "ADJ", "ADV", "PRON", "NUM", "CONJ", "INT"]
        for pos in priority:
            if pos in tags:
                return pos
        return next(iter(sorted(tags)))

    def _apply_r1(
        self,
        word: str,
        word_pos_tags: frozenset[str],
        prev_word: str | None,
        prev_word_pos: str | None,
        next_word: str | None,
        next_word_pos: str | None,
    ) -> DisambiguationResult | None:
        """
        Rule R1: Noun after Verb.

        If the previous word is a verb and the current word is ambiguous,
        the current word is likely a noun (object of the verb).

        Example: tokens=["ပြော", "ကြီး", "ကို"] → ကြီး = N (after V "ပြော")
        """
        if prev_word_pos not in VERB_TAGS:
            return None

        if "N" not in word_pos_tags:
            return None

        return DisambiguationResult(
            word=word,
            original_pos_tags=word_pos_tags,
            resolved_pos="N",
            rule_applied=DisambiguationRule.R1_NOUN_AFTER_VERB,
            confidence=self._pos_config.disambiguation_r1_confidence,
            context_used=f"after verb '{prev_word}' ({prev_word_pos})",
        )

    def _apply_r2(
        self,
        word: str,
        word_pos_tags: frozenset[str],
        prev_word: str | None,
        prev_word_pos: str | None,
        next_word: str | None,
        next_word_pos: str | None,
    ) -> DisambiguationResult | None:
        """
        Rule R2: Adjective before Noun.

        If the next word is a noun and the current word is ambiguous
        with ADJ as an option, select ADJ.

        Example: "ကြီး သော အိမ်" → ကြီး = ADJ (before N)
        """
        if next_word_pos not in MODIFIABLE_TAGS:
            return None

        if "ADJ" not in word_pos_tags:
            return None

        return DisambiguationResult(
            word=word,
            original_pos_tags=word_pos_tags,
            resolved_pos="ADJ",
            rule_applied=DisambiguationRule.R2_ADJ_BEFORE_NOUN,
            confidence=self._pos_config.disambiguation_r2_confidence,
            context_used=f"before noun '{next_word}' ({next_word_pos})",
        )

    def _apply_r3(
        self,
        word: str,
        word_pos_tags: frozenset[str],
        prev_word: str | None,
        prev_word_pos: str | None,
        next_word: str | None,
        next_word_pos: str | None,
    ) -> DisambiguationResult | None:
        """
        Rule R3: Verb before Particle.

        If the next word is a sentence-final or modifying particle,
        the current ambiguous word is likely a verb.

        Example: tokens=["စား", "ကြီး", "ပြီ"] → ကြီး = V (before particle "ပြီ")
        """
        is_known_particle = False
        if next_word_pos in PARTICLE_POS_TAGS:
            is_known_particle = True
        elif next_word and next_word in self.grammar_config.particle_tags:
            is_known_particle = True

        if not is_known_particle:
            return None

        if "V" not in word_pos_tags:
            return None

        return DisambiguationResult(
            word=word,
            original_pos_tags=word_pos_tags,
            resolved_pos="V",
            rule_applied=DisambiguationRule.R3_VERB_BEFORE_PARTICLE,
            confidence=self._pos_config.disambiguation_r3_confidence,
            context_used=f"before particle '{next_word}' ({next_word_pos})",
        )

    def _apply_r4(
        self,
        word: str,
        word_pos_tags: frozenset[str],
        prev_word: str | None,
        prev_word_pos: str | None,
        next_word: str | None,
        next_word_pos: str | None,
    ) -> DisambiguationResult | None:
        """
        Rule R4: Noun after Determiner.

        If the previous word is a determiner/demonstrative and the
        current word is ambiguous, select N.

        Example: "ဤ ကြီး ကို" → ကြီး = N (after determiner)
        """
        if prev_word not in DETERMINERS:
            return None

        if "N" not in word_pos_tags:
            return None

        return DisambiguationResult(
            word=word,
            original_pos_tags=word_pos_tags,
            resolved_pos="N",
            rule_applied=DisambiguationRule.R4_NOUN_AFTER_DETERMINER,
            confidence=self._pos_config.disambiguation_r4_confidence,
            context_used=f"after determiner '{prev_word}'",
        )

    def _apply_r5(
        self,
        word: str,
        word_pos_tags: frozenset[str],
        prev_word: str | None,
        prev_word_pos: str | None,
        next_word: str | None,
        next_word_pos: str | None,
    ) -> DisambiguationResult | None:
        """
        Rule R5: Adjective or Verb after Adverb.

        If the previous word is an adverb marker, the current
        ambiguous word is likely an adjective (for degree adverbs
        like အလွန်) or a verb (for manner adverbs like လျင်မြန်စွာ).

        Example: tokens=["လျင်မြန်စွာ", "ရေး", "ပြီးသည်"] → ရေး = V (after adverb)
        """
        if prev_word not in ADVERB_MARKERS and prev_word_pos != "ADV":
            return None

        # Degree adverbs (အလွန်, အရမ်း) typically modify adjectives,
        # e.g., "အလွန် ကြီး" = "very big" (ADJ), not "very grow" (V).
        if prev_word in DEGREE_ADVERBS and "ADJ" in word_pos_tags:
            return DisambiguationResult(
                word=word,
                original_pos_tags=word_pos_tags,
                resolved_pos="ADJ",
                rule_applied=DisambiguationRule.R5_VERB_AFTER_ADVERB,
                confidence=self._pos_config.disambiguation_r5_confidence,
                context_used=f"after degree adverb '{prev_word}'",
            )

        if "V" not in word_pos_tags:
            return None

        return DisambiguationResult(
            word=word,
            original_pos_tags=word_pos_tags,
            resolved_pos="V",
            rule_applied=DisambiguationRule.R5_VERB_AFTER_ADVERB,
            confidence=self._pos_config.disambiguation_r5_confidence,
            context_used=f"after adverb '{prev_word}'",
        )


# Module-level singleton with thread-safe initialization
_disambiguator: POSDisambiguator | None = None
_lock = threading.Lock()


def get_disambiguator() -> POSDisambiguator:
    """Get the module-level POSDisambiguator singleton (thread-safe)."""
    global _disambiguator
    if _disambiguator is None:
        with _lock:
            # Double-check pattern to avoid multiple initializations
            if _disambiguator is None:
                _disambiguator = POSDisambiguator()
    return _disambiguator


def disambiguate(
    word: str,
    word_pos_tags: frozenset[str],
    prev_word_pos: str | None = None,
    next_word_pos: str | None = None,
) -> str:
    """
    Convenience function to disambiguate a single word.

    Args:
        word: The word to disambiguate.
        word_pos_tags: Set of possible POS tags.
        prev_word_pos: POS of previous word.
        next_word_pos: POS of next word.

    Returns:
        The resolved POS tag.

    Example:
        >>> from myspellchecker.algorithms.pos_disambiguator import disambiguate
        >>> pos = disambiguate("ကြီး", frozenset({"ADJ", "N", "V"}), next_word_pos="N")
        >>> print(pos)  # "ADJ"
    """
    result = get_disambiguator().disambiguate_in_context(
        word=word,
        word_pos_tags=word_pos_tags,
        prev_word_pos=prev_word_pos,
        next_word_pos=next_word_pos,
    )
    return result.resolved_pos
