"""Medial swap candidate generator for Myanmar text.

SymSpell's delete-distance model cannot reliably find medial swaps
(ျ↔ြ) as edit-distance-1 candidates. This strategy directly generates
orthographic variants by swapping, inserting, or deleting Myanmar medials
and validating against the dictionary.

The #1 error type in Myanmar text is ya-pin/ya-yit confusion (ျ↔ြ),
accounting for the majority of medial errors. This strategy complements SymSpell
by generating candidates that the delete-distance model misses.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import yaml

from myspellchecker.algorithms.ranker import SuggestionData
from myspellchecker.algorithms.suggestion_strategy import (
    BaseSuggestionStrategy,
    SuggestionContext,
    SuggestionResult,
)
from myspellchecker.core.constants import (
    COMPATIBLE_HA,
    COMPATIBLE_RA,
    COMPATIBLE_WA,
    COMPATIBLE_YA,
    CONSONANTS,
)
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Myanmar medial characters
MEDIAL_YA_PIN = "\u103b"  # ျ
MEDIAL_YA_YIT = "\u103c"  # ြ
MEDIAL_WA = "\u103d"  # ွ
MEDIAL_HA = "\u103e"  # ှ
ALL_MEDIALS = frozenset({MEDIAL_YA_PIN, MEDIAL_YA_YIT, MEDIAL_WA, MEDIAL_HA})

# Anusvara
ANUSVARA = "\u1036"  # ံ

# Consonant compatibility map: medial → set of consonants that can take it
_MEDIAL_COMPAT: dict[str, frozenset[str]] = {
    MEDIAL_YA_PIN: frozenset(COMPATIBLE_YA),
    MEDIAL_YA_YIT: frozenset(COMPATIBLE_RA),
    MEDIAL_WA: frozenset(COMPATIBLE_WA),
    MEDIAL_HA: frozenset(COMPATIBLE_HA),
}

# Default swap pairs (bidirectional)
_DEFAULT_SWAP_PAIRS: list[tuple[str, str, float]] = [
    (MEDIAL_YA_PIN, MEDIAL_YA_YIT, 1.0),  # ျ ↔ ြ (dominant: 30 cases)
    (MEDIAL_WA, MEDIAL_HA, 0.8),  # ွ ↔ ှ
]

# Default insertion targets (medial, weight)
_DEFAULT_INSERTIONS: list[tuple[str, float]] = [
    (MEDIAL_HA, 0.9),  # ှ insertion (21 cases: မာ→မှာ, နာ→နှာ)
    (MEDIAL_WA, 0.7),  # ွ insertion
]

# Default deletion targets (medial, weight)
_DEFAULT_DELETIONS: list[tuple[str, float]] = [
    (MEDIAL_HA, 0.8),  # ှ deletion
    (MEDIAL_WA, 0.7),  # ွ deletion
]

_RULES_DIR = Path(__file__).parent.parent / "rules"


def _load_medial_swap_rules(
    path: Path | None = None,
) -> tuple[
    list[tuple[str, str, float]],
    list[tuple[str, float]],
    list[tuple[str, float]],
    bool,
    float,
    float,
]:
    """Load medial swap rules from YAML.

    Returns:
        Tuple of (swap_pairs, insertions, deletions, include_anusvara,
        anusvara_insert_weight, anusvara_delete_weight).
        Falls back to defaults if YAML not found.
    """
    yaml_path = path or (_RULES_DIR / "medial_swap_pairs.yaml")
    if not yaml_path.exists():
        return _DEFAULT_SWAP_PAIRS, _DEFAULT_INSERTIONS, _DEFAULT_DELETIONS, True, 0.85, 0.85

    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        logger.warning("Failed to load medial swap rules from %s: %s", yaml_path, e)
        return _DEFAULT_SWAP_PAIRS, _DEFAULT_INSERTIONS, _DEFAULT_DELETIONS, True, 0.85, 0.85

    if not isinstance(data, dict):
        return _DEFAULT_SWAP_PAIRS, _DEFAULT_INSERTIONS, _DEFAULT_DELETIONS, True, 0.85, 0.85

    # Parse swap pairs
    swap_pairs: list[tuple[str, str, float]] = []
    for entry in data.get("swap_pairs", []):
        from_char = entry.get("from", "")
        to_char = entry.get("to", "")
        weight = float(entry.get("weight", 1.0))
        if from_char and to_char:
            swap_pairs.append((from_char, to_char, weight))

    # Parse insertions
    insertions: list[tuple[str, float]] = []
    for entry in data.get("insertions", []):
        medial = entry.get("medial", "")
        weight = float(entry.get("weight", 0.8))
        if medial:
            insertions.append((medial, weight))

    # Parse deletions
    deletions: list[tuple[str, float]] = []
    for entry in data.get("deletions", []):
        medial = entry.get("medial", "")
        weight = float(entry.get("weight", 0.8))
        if medial:
            deletions.append((medial, weight))

    # Anusvara config
    anusvara_section = data.get("anusvara", {})
    if isinstance(anusvara_section, dict):
        include_anusvara = bool(anusvara_section)
        anusvara_insert_weight = float(anusvara_section.get("insert_weight", 0.85))
        anusvara_delete_weight = float(anusvara_section.get("delete_weight", 0.85))
    elif isinstance(anusvara_section, bool):
        include_anusvara = anusvara_section
        anusvara_insert_weight = 0.85
        anusvara_delete_weight = 0.85
    else:
        include_anusvara = True
        anusvara_insert_weight = 0.85
        anusvara_delete_weight = 0.85

    return (
        swap_pairs or _DEFAULT_SWAP_PAIRS,
        insertions or _DEFAULT_INSERTIONS,
        deletions or _DEFAULT_DELETIONS,
        include_anusvara,
        anusvara_insert_weight,
        anusvara_delete_weight,
    )


class MedialSwapSuggestionStrategy(BaseSuggestionStrategy):
    """Generate candidates by swapping, inserting, or deleting Myanmar medials.

    Complements SymSpell's delete-distance model which cannot efficiently
    find medial swaps (ျ↔ြ) — the #1 error type in Myanmar text.

    Algorithm:
        1. Scan word for medial characters (U+103B-U+103E)
        2. Generate swap variants using configured pairs (ျ↔ြ, ွ↔ှ)
        3. Generate insertion variants (add missing medial after consonant)
        4. Generate deletion variants (remove extra medial)
        5. Optionally generate anusvara (ံ) insert/delete variants
        6. Validate each variant against dictionary
        7. Return valid variants as SuggestionData

    Performance: 3-8 variants per word (O(1) dictionary lookup each).
    """

    def __init__(
        self,
        dictionary_check: Callable[[str], bool],
        get_frequency: Callable[[str], int],
        max_suggestions: int = 5,
        max_variants_per_word: int = 8,
        confidence: float = 0.90,
        include_insertions: bool = True,
        include_deletions: bool = True,
        include_anusvara: bool = True,
        rules_path: Path | str | None = None,
    ):
        super().__init__(max_suggestions=max_suggestions)
        self._dictionary_check = dictionary_check
        self._get_frequency = get_frequency
        self._max_variants = max_variants_per_word
        self._confidence = confidence
        self._include_insertions = include_insertions
        self._include_deletions = include_deletions

        # Load rules from YAML
        rules_p = Path(rules_path) if isinstance(rules_path, str) else rules_path
        (
            self._swap_pairs,
            self._insertions,
            self._deletions,
            yaml_anusvara,
            self._anusvara_insert_weight,
            self._anusvara_delete_weight,
        ) = _load_medial_swap_rules(rules_p)
        self._include_anusvara = include_anusvara and yaml_anusvara

    @property
    def name(self) -> str:
        return "medial_swap"

    def supports_context(self) -> bool:
        return False

    def suggest(
        self,
        term: str,
        context: SuggestionContext | None = None,
    ) -> SuggestionResult:
        """Generate medial-swapped candidates for the given term."""
        candidates: list[SuggestionData] = []
        seen: set[str] = {term}  # Exclude the input itself

        # 1. Medial swap variants (ျ↔ြ, ွ↔ှ)
        for variant, weight in self._generate_swap_variants(term):
            if variant not in seen:
                self._try_add(variant, weight, candidates, seen)
                if len(candidates) >= self._max_variants:
                    break

        # 2. Medial insertion variants (add missing ှ, ွ, etc.)
        if self._include_insertions and len(candidates) < self._max_variants:
            for variant, weight in self._generate_insertion_variants(term):
                if variant not in seen:
                    self._try_add(variant, weight, candidates, seen)
                    if len(candidates) >= self._max_variants:
                        break

        # 3. Medial deletion variants (remove extra medial)
        if self._include_deletions and len(candidates) < self._max_variants:
            for variant, weight in self._generate_deletion_variants(term):
                if variant not in seen:
                    self._try_add(variant, weight, candidates, seen)
                    if len(candidates) >= self._max_variants:
                        break

        # 4. Anusvara insert/delete
        if self._include_anusvara and len(candidates) < self._max_variants:
            for variant, weight in self._generate_anusvara_variants(term):
                if variant not in seen:
                    self._try_add(variant, weight, candidates, seen)
                    if len(candidates) >= self._max_variants:
                        break

        # Sort by confidence then frequency
        candidates.sort(key=lambda s: (-s.confidence, -s.frequency))

        return self._create_result(
            candidates[: self._max_variants],
            metadata={"strategy": "medial_swap", "variants_checked": len(seen) - 1},
        )

    def _generate_swap_variants(self, term: str) -> Iterator[tuple[str, float]]:
        """Swap each medial with its confusable partner."""
        chars = list(term)
        for i, ch in enumerate(chars):
            if ch not in ALL_MEDIALS:
                continue
            # Find the preceding consonant for compatibility check
            consonant = self._find_preceding_consonant(chars, i)
            for from_char, to_char, weight in self._swap_pairs:
                target: str | None = None
                if ch == from_char:
                    target = to_char
                elif ch == to_char:
                    target = from_char  # Bidirectional
                if target is not None and self._is_compatible(consonant, target):
                    new_chars = chars.copy()
                    new_chars[i] = target
                    yield "".join(new_chars), weight

    def _generate_insertion_variants(self, term: str) -> Iterator[tuple[str, float]]:
        """Insert a medial after each consonant that doesn't already have one."""
        chars = list(term)
        for i, ch in enumerate(chars):
            if ch not in CONSONANTS:
                continue
            # Skip if next char is already a medial
            if i + 1 < len(chars) and chars[i + 1] in ALL_MEDIALS:
                continue
            for medial, weight in self._insertions:
                if self._is_compatible(ch, medial):
                    new_chars = chars[: i + 1] + [medial] + chars[i + 1 :]
                    yield "".join(new_chars), weight

    def _generate_deletion_variants(self, term: str) -> Iterator[tuple[str, float]]:
        """Remove each medial from the term."""
        chars = list(term)
        for i, ch in enumerate(chars):
            if ch not in ALL_MEDIALS:
                continue
            for medial, weight in self._deletions:
                if ch == medial:
                    new_chars = chars[:i] + chars[i + 1 :]
                    yield "".join(new_chars), weight

    def _generate_anusvara_variants(self, term: str) -> Iterator[tuple[str, float]]:
        """Insert or delete anusvara (ံ U+1036)."""
        chars = list(term)
        if ANUSVARA in term:
            # Delete anusvara
            idx = term.index(ANUSVARA)
            new_chars = chars[:idx] + chars[idx + 1 :]
            yield "".join(new_chars), self._anusvara_delete_weight
        else:
            # Insert anusvara before asat/tone marks at end of each syllable
            for i, ch in enumerate(chars):
                if ch not in CONSONANTS:
                    continue
                # Insert after consonant + optional vowels, before asat/tone
                insert_pos = i + 1
                while insert_pos < len(chars) and chars[insert_pos] in ALL_MEDIALS:
                    insert_pos += 1
                # Skip vowel signs
                while insert_pos < len(chars) and 0x102B <= ord(chars[insert_pos]) <= 0x1032:
                    insert_pos += 1
                if insert_pos <= len(chars):
                    new_chars = chars[:insert_pos] + [ANUSVARA] + chars[insert_pos:]
                    yield "".join(new_chars), self._anusvara_insert_weight

    def _try_add(
        self,
        variant: str,
        weight: float,
        candidates: list[SuggestionData],
        seen: set[str],
    ) -> None:
        """Validate variant and add to candidates if it's a valid word."""
        seen.add(variant)
        if not self._dictionary_check(variant):
            return
        freq = self._get_frequency(variant)
        candidates.append(
            SuggestionData(
                term=variant,
                edit_distance=1,
                frequency=freq,
                source="medial_swap",
                confidence=self._confidence * weight,
            )
        )

    @staticmethod
    def _find_preceding_consonant(chars: list[str], medial_idx: int) -> str | None:
        """Find the consonant that the medial at medial_idx modifies."""
        for j in range(medial_idx - 1, -1, -1):
            if chars[j] in CONSONANTS:
                return chars[j]
            if chars[j] not in ALL_MEDIALS:
                break
        return None

    @staticmethod
    def _is_compatible(consonant: str | None, medial: str) -> bool:
        """Check if a consonant can take the given medial."""
        if consonant is None:
            return False
        compat_set = _MEDIAL_COMPAT.get(medial)
        if compat_set is None:
            return True  # Unknown medial — allow
        return consonant in compat_set
