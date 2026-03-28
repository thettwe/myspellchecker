"""
Myanmar Merged Word Detection and Validation.

This module detects words that the segmenter may have incorrectly merged
from a particle + verb sequence into a single compound word.

Problem:
    Myanmar word segmenters sometimes merge adjacent tokens when the
    concatenation forms a valid dictionary word. For example:

    Input text: "သူက စားသောကြောင့်"
    Intended:   သူ + က (subject particle) + စား (eat) + သောကြောင့် (because)
    Segmented:  သူ + ကစား (play) + သောကြောင့် (because)

    "ကစား" is a valid word meaning "play", so the segmenter merges
    "က" + "စား" into "ကစား". This changes the sentence meaning from
    "because he/she eats..." to "because he/she plays...".

Detection Strategy:
    A merged word is flagged ONLY when ALL of the following conditions hold:
    1. The word is in the known ambiguous-merge set (e.g., "ကစား")
    2. The preceding word is a NOUN or PRONOUN (POS: N, PRON)
       — because "က" is a subject particle that follows nouns/pronouns
    3. The following word is a clause-linking particle or verb-final marker
       (e.g., သောကြောင့်, ၍, သဖြင့်, တယ်, သည်, ပါတယ်, etc.)
       — because the decomposed verb "စား" needs a clause continuation

    This three-way evidence requirement prevents false positives on
    legitimate uses like "ကစားတယ်" (he/she plays).

Confidence:
    The confidence is set conservatively (0.80) since this is a heuristic
    that cannot be 100% certain without semantic understanding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from myspellchecker.core.config.grammar_configs import MergedWordCheckerConfig
from myspellchecker.core.constants import ET_MERGED_WORD
from myspellchecker.core.response import GrammarError
from myspellchecker.utils.singleton import Singleton

__all__ = [
    "MergedWordChecker",
    "MergedWordError",
]

# Singleton registry for MergedWordChecker
_singleton: Singleton["MergedWordChecker"] = Singleton()


@dataclass
class MergedWordError(GrammarError):
    """
    Represents a word that may have been incorrectly merged by the segmenter.

    Extends GrammarError to integrate with the spell checker's error hierarchy.

    Attributes:
        text: The merged word (e.g., "ကစား").
        position: Index of the error in the word list.
        suggestions: List containing the decomposed suggestion (e.g., "က စား").
        error_type: Always "merged_word".
        confidence: Confidence score (0.0-1.0).
        reason: Human-readable explanation.
        decomposition: Tuple of (particle, verb) that the word decomposes into.
    """

    error_type: str = field(default=ET_MERGED_WORD)
    decomposition: tuple[str, str] = field(default=("", ""))

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"MergedWordError(pos={self.position}, {self.text} -> {' '.join(self.decomposition)})"
        )


# Type alias for a merge rule entry
# (particle, verb, particle_pos_requirement, description)
MergeRule = tuple[str, str, str, str]

# ============================================================
# Ambiguous merge rules
#
# Each entry maps a merged word to its possible decomposition:
#   merged_word -> (particle, verb, particle_type, description)
#
# particle_type indicates what POS the particle is:
#   "P_SUBJ" = subject marker (follows nouns/pronouns)
#   "P_OBJ"  = object marker (follows nouns/pronouns)
#
# Only add entries where the merged form is genuinely common
# AND the decomposition is linguistically plausible.
# ============================================================
AMBIGUOUS_MERGES: dict[str, MergeRule] = {
    # "ကစား" (to play) vs "က" (subject particle) + "စား" (to eat)
    "ကစား": ("က", "စား", "P_SUBJ", "subject marker + eat"),
}

# POS tags that indicate a noun/pronoun (can precede subject particle "က")
NOUN_LIKE_TAGS: frozenset[str] = frozenset(
    {
        "N",
        "PRON",
        "NP",
        "NOUN",
    }
)

# Clause-linking particles that provide STRONG evidence for the
# decomposed form (particle + verb).
#
# IMPORTANT: We intentionally EXCLUDE sentence-final markers (တယ်, ပါတယ်,
# သည်, etc.) and verb-modifying particles (နေ, ခဲ့, နိုင်, etc.) because
# these follow verbs in general -- they would fire on legitimate uses
# of the merged word (e.g., "ကလေး ကစားတယ်" = "the child plays").
#
# Only clause-linking markers that form subordinate clauses are included,
# because a bare verb like "စား" (eat) before a clause linker is a strong
# signal that the segmenter merged the subject particle into the verb.
#
# Categories:
#   - Causal: သောကြောင့်, ကြောင့်, လို့, သဖြင့်
#   - Sequential/conjunctive: ၍
#   - Conditional: ရင်, လျှင်
#   - Concessive: သော်လည်း, ပေမယ့်
#   - Relative clause: သော, တဲ့
CLAUSE_LINKING_MARKERS: frozenset[str] = frozenset(
    {
        # Causal (strongest evidence)
        "သောကြောင့်",
        "ကြောင့်",
        "လို့",
        "သဖြင့်",
        # Sequential / conjunctive (literary)
        "၍",
        # Conditional
        "ရင်",
        "လျှင်",
        # Concessive
        "သော်လည်း",
        "ပေမယ့်",
        # Relative clause markers
        # "ကစားသော" (which plays) is valid but "က စားသော" (which eats) is
        # equally valid when preceded by a noun -- strong disambiguation signal
        "သော",
        "တဲ့",
    }
)


class MergedWordChecker:
    """
    Detects words that the segmenter may have incorrectly merged.

    This checker identifies cases where a valid dictionary word is
    actually a mis-merge of a particle + verb, based on surrounding
    POS context.

    The check requires three-way evidence:
    1. The word is in AMBIGUOUS_MERGES
    2. The preceding word has a noun-like POS tag
    3. The following word is a clause continuation marker

    This conservative approach prevents false positives on legitimate
    uses of the merged word (e.g., "ကစားတယ်" = "plays").
    """

    def __init__(
        self,
        confidence: float | None = None,
        checker_config: MergedWordCheckerConfig | None = None,
    ):
        """
        Initialize the merged word checker.

        Args:
            confidence: Default confidence for merged word detections.
                       Set conservatively since this is a heuristic.
                       Deprecated: use checker_config instead.
            checker_config: MergedWordCheckerConfig for configurable confidence.
        """
        self.checker_config = checker_config or MergedWordCheckerConfig()
        self.confidence = (
            confidence if confidence is not None else self.checker_config.default_confidence
        )
        self.merge_rules = AMBIGUOUS_MERGES

    def validate_sequence(
        self,
        words: list[str],
        pos_tags: list[str | None] | None = None,
    ) -> list[MergedWordError]:
        """
        Validate a word sequence for incorrectly merged words.

        Args:
            words: List of words in the sentence.
            pos_tags: Optional list of POS tags for each word.
                     If not provided, the checker cannot detect merges
                     (returns empty list).

        Returns:
            List of MergedWordError objects for detected merges.
        """
        errors: list[MergedWordError] = []

        if not words or not pos_tags or len(pos_tags) != len(words):
            return errors

        for i, word in enumerate(words):
            if word not in self.merge_rules:
                continue

            particle, verb, particle_type, description = self.merge_rules[word]

            # Condition 1: Preceding word must be noun-like
            if i == 0:
                # No preceding word -- cannot be a particle merge
                continue

            prev_pos = pos_tags[i - 1]
            if not prev_pos or not self._has_noun_like_tag(prev_pos):
                continue

            # Condition 2: Following word must be a clause continuation marker
            if i + 1 >= len(words):
                # End of sentence -- less likely to be a merge error
                # (though "သူ ကစား" at sentence end could still be wrong,
                #  we require clause continuation for safety)
                continue

            next_word = words[i + 1]
            if next_word not in CLAUSE_LINKING_MARKERS:
                continue

            # All three conditions met: flag as potential merge error
            suggestion = f"{particle} {verb}"
            prev_word = words[i - 1]

            errors.append(
                MergedWordError(
                    text=word,
                    position=i,
                    suggestions=[suggestion],
                    confidence=self.confidence,
                    reason=(
                        f"'{prev_word} {word}' may be '{prev_word}{particle} {verb}' "
                        f"({description}); "
                        f"'{word}' could be segmenter merge of '{particle}' + '{verb}'"
                    ),
                    decomposition=(particle, verb),
                )
            )

        return errors

    def _has_noun_like_tag(self, pos_string: str) -> bool:
        """
        Check if a POS tag string contains a noun-like tag.

        Args:
            pos_string: Pipe-separated POS tag string (e.g., "N|V").

        Returns:
            True if any tag in the string is noun-like.
        """
        tags = set(pos_string.split("|"))
        return bool(tags & NOUN_LIKE_TAGS)
