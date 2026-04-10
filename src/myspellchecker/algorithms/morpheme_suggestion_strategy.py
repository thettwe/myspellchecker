"""
Morpheme-level suggestion strategy for compound and reduplication typos.

When an OOV word is a near-miss compound (one morpheme has a typo) or
near-miss reduplication, this strategy corrects the individual morpheme
and reconstructs the valid compound/reduplication.

Example:
    Input: "ကျောင်ငသား" (typo in first morpheme)
    → Correct first morpheme via SymSpell: "ကျောင်း"
    → Reconstruct: "ကျောင်းသား" (student)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.algorithms.ranker import SuggestionData
from myspellchecker.algorithms.suggestion_strategy import (
    BaseSuggestionStrategy,
    SuggestionContext,
    SuggestionResult,
)
from myspellchecker.text.types import DictionaryCheck
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.symspell import SymSpell
    from myspellchecker.text.compound_resolver import CompoundResolver
    from myspellchecker.text.reduplication import ReduplicationEngine

logger = get_logger(__name__)

__all__ = [
    "MorphemeSuggestionStrategy",
]


class MorphemeSuggestionStrategy(BaseSuggestionStrategy):
    """Strategy for suggesting corrections to individual morphemes in compounds.

    When a word can be split into morphemes where exactly one is invalid,
    this strategy corrects that morpheme and reconstructs the compound.

    Args:
        compound_resolver: CompoundResolver for splitting words.
        reduplication_engine: ReduplicationEngine for reduplication detection.
        symspell: SymSpell for morpheme-level correction.
        dictionary_check: Callable to check if a word is valid.
        max_suggestions: Maximum suggestions to return.
    """

    def __init__(
        self,
        compound_resolver: CompoundResolver | None,
        reduplication_engine: ReduplicationEngine | None,
        symspell: SymSpell,
        dictionary_check: DictionaryCheck,
        max_suggestions: int = 3,
    ):
        super().__init__(max_suggestions=max_suggestions, max_edit_distance=1)
        self._compound_resolver = compound_resolver
        self._reduplication_engine = reduplication_engine
        self._symspell = symspell
        self._dictionary_check = dictionary_check

    @property
    def name(self) -> str:
        return "morpheme"

    def supports_context(self) -> bool:
        return False

    def suggest(
        self,
        term: str,
        context: SuggestionContext | None = None,
    ) -> SuggestionResult:
        """Generate morpheme-level suggestions.

        Tries to split the term into morphemes, correct exactly one
        invalid morpheme, and reconstruct a valid compound.

        Args:
            term: The misspelled word.
            context: Optional context (unused by this strategy).

        Returns:
            SuggestionResult with morpheme-corrected suggestions.
        """
        suggestions: list[SuggestionData] = []

        # Try compound morpheme correction
        compound_suggestions = self._suggest_compound_correction(term)
        suggestions.extend(compound_suggestions)

        # Try reduplication correction
        redup_suggestions = self._suggest_reduplication_correction(term)
        suggestions.extend(redup_suggestions)

        return self._create_result(suggestions, metadata={"strategy": "morpheme"})

    def _try_correct_part(
        self,
        valid_part: str,
        invalid_part: str,
        is_left_invalid: bool,
        word: str,
    ) -> list[SuggestionData]:
        """Generate suggestions by correcting the invalid morpheme.

        Args:
            valid_part: The morpheme that passed dictionary lookup.
            invalid_part: The morpheme that failed dictionary lookup.
            is_left_invalid: If True, the invalid part is on the left side.
            word: The original compound word (unused, kept for diagnostics).

        Returns:
            List of SuggestionData with source="morpheme".
        """
        corrections = self._correct_morpheme(invalid_part)
        results: list[SuggestionData] = []
        for corrected, dist, freq in corrections:
            if is_left_invalid:
                reconstructed = corrected + valid_part
            else:
                reconstructed = valid_part + corrected
            if self._dictionary_check(reconstructed) or self._dictionary_check(corrected):
                results.append(
                    SuggestionData(
                        term=reconstructed,
                        edit_distance=dist,
                        frequency=freq,
                        source="morpheme",
                        confidence=0.85,
                    )
                )
        return results

    def _suggest_compound_correction(self, term: str) -> list[SuggestionData]:
        """Try to correct a compound word with one bad morpheme.

        Uses two strategies:
        1. Binary splits: try all 2-way split points (original)
        2. Ternary splits: try prefix + error_middle + suffix where
           prefix and suffix are valid dictionary words and the middle
           part gets corrected via SymSpell. This handles cases where
           the error word is embedded inside a longer compound token.
        """
        if self._compound_resolver is None:
            return []

        segmenter = self._compound_resolver.segmenter
        syllables = segmenter.segment_syllables(term)
        n = len(syllables)

        if n < 2:
            return []

        suggestions: list[SuggestionData] = []

        # Strategy 1: Binary splits (original)
        for i in range(1, n):
            left = "".join(syllables[:i])
            right = "".join(syllables[i:])

            left_valid = self._dictionary_check(left)
            right_valid = self._dictionary_check(right)

            # Exactly one part must be invalid
            if left_valid == right_valid:
                continue

            if left_valid and not right_valid:
                suggestions.extend(
                    self._try_correct_part(left, right, is_left_invalid=False, word=term)
                )
            elif right_valid and not left_valid:
                suggestions.extend(
                    self._try_correct_part(right, left, is_left_invalid=True, word=term)
                )

        # Strategy 2: Ternary splits for longer compounds (3+ syllables)
        # Try all (prefix, middle, suffix) splits where prefix and suffix
        # are valid and middle gets corrected.
        if n >= 3 and len(suggestions) < self._max_suggestions:
            suggestions.extend(self._suggest_ternary_splits(syllables, term))

        return suggestions[: self._max_suggestions]

    def _suggest_ternary_splits(
        self, syllables: list[str], term: str
    ) -> list[SuggestionData]:
        """Try ternary (prefix + error + suffix) splits.

        For compounds like "ခုန်ကျစရိတ်" where the error "ခုန်ကျ" is
        sandwiched between valid prefix "" and suffix "စရိတ်", binary
        splits won't find the correction because neither binary half is
        fully valid. Ternary splits identify the invalid middle segment
        and correct it.
        """
        n = len(syllables)
        results: list[SuggestionData] = []
        seen: set[str] = set()

        for i in range(n):
            for j in range(i + 1, min(i + 4, n)):  # middle up to 3 syllables
                prefix = "".join(syllables[:i]) if i > 0 else ""
                middle = "".join(syllables[i:j])
                suffix = "".join(syllables[j:]) if j < n else ""

                # Prefix must be valid (or empty)
                if prefix and not self._dictionary_check(prefix):
                    continue
                # Suffix must be valid (or empty)
                if suffix and not self._dictionary_check(suffix):
                    continue
                # Middle must be invalid (the error part)
                if self._dictionary_check(middle):
                    continue
                # Skip if middle is the entire term (that's just the original)
                if not prefix and not suffix:
                    continue

                corrections = self._correct_morpheme(middle)
                for corrected, dist, freq in corrections:
                    reconstructed = prefix + corrected + suffix
                    if reconstructed in seen:
                        continue
                    if self._dictionary_check(reconstructed) or self._dictionary_check(corrected):
                        seen.add(reconstructed)
                        results.append(
                            SuggestionData(
                                term=reconstructed,
                                edit_distance=dist,
                                frequency=freq,
                                source="morpheme",
                                confidence=0.80,
                            )
                        )
                        if len(results) >= self._max_suggestions:
                            return results

        return results

    def _suggest_reduplication_correction(self, term: str) -> list[SuggestionData]:
        """Try to correct a near-miss reduplication."""
        if self._reduplication_engine is None:
            return []

        segmenter = self._reduplication_engine.segmenter
        syllables = segmenter.segment_syllables(term)
        n = len(syllables)

        if n < 2 or n % 2 != 0:
            return []

        suggestions: list[SuggestionData] = []

        # For AA pattern: check if correcting one half makes a valid reduplication
        half = n // 2
        first_half = "".join(syllables[:half])
        second_half = "".join(syllables[half:])

        if first_half == second_half:
            # Already identical - not a typo
            return []

        # Try correcting second half to match first (if first is valid)
        if self._dictionary_check(first_half):
            reconstructed = first_half + first_half
            if reconstructed != term:
                suggestions.append(
                    SuggestionData(
                        term=reconstructed,
                        edit_distance=1,
                        frequency=0,
                        source="morpheme",
                        confidence=0.80,
                    )
                )

        # Try correcting first half to match second (if second is valid)
        if self._dictionary_check(second_half):
            reconstructed = second_half + second_half
            if reconstructed != term and reconstructed not in [s.term for s in suggestions]:
                suggestions.append(
                    SuggestionData(
                        term=reconstructed,
                        edit_distance=1,
                        frequency=0,
                        source="morpheme",
                        confidence=0.80,
                    )
                )

        return suggestions[: self._max_suggestions]

    def _correct_morpheme(self, morpheme: str) -> list[tuple[str, int, int]]:
        """Get corrections for a single morpheme via SymSpell.

        Args:
            morpheme: The invalid morpheme to correct.

        Returns:
            List of (corrected_term, edit_distance, frequency) tuples.
        """
        results = self._symspell.lookup(morpheme, level="word", max_suggestions=3)
        return [(r.term, r.edit_distance, r.frequency) for r in results]
