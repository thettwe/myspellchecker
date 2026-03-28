"""
Edit distance algorithms for string comparison.

This module provides efficient implementations of Levenshtein and
Damerau-Levenshtein distance algorithms, used for spell checking
and phonetic similarity matching.

Includes Myanmar-specific syllable-aware edit distance that treats
medial character clusters as single units.
"""

from __future__ import annotations

import functools

from myspellchecker.algorithms.distance.keyboard import is_keyboard_adjacent
from myspellchecker.core.config.algorithm_configs import SymSpellConfig
from myspellchecker.core.constants import (
    ASAT,
    CONSONANTS,
    DAMERAU_CACHE_SIZE,
    MEDIALS,
    NGA,
    VIRAMA,
)
from myspellchecker.text.phonetic_data import MYANMAR_SUBSTITUTION_COSTS, VISUAL_SIMILAR

# Resolve default values from SymSpellConfig to keep constants in sync
_SYMSPELL_DEFAULTS = SymSpellConfig()

# Merge data-driven confusion matrix (from YAML) with hardcoded costs.
# This adds corpus-derived pairs (asat↔dot_below, ka↔ta, etc.) and
# refines costs for high-frequency confusions (ျ↔ြ: 0.3→0.2).
try:
    from myspellchecker.core.detection_rules import load_confusion_matrix

    _MERGED_SUBSTITUTION_COSTS = load_confusion_matrix()
except (ImportError, RuntimeError):
    _MERGED_SUBSTITUTION_COSTS = MYANMAR_SUBSTITUTION_COSTS

# Try to import Cythonized edit distance
try:
    from myspellchecker.algorithms.distance import edit_distance_c  # type: ignore[attr-defined]

    _HAS_CYTHON_EDIT_DISTANCE = True
    # Initialize Cython data maps
    from myspellchecker.algorithms.distance.keyboard import KEY_ADJACENCY

    edit_distance_c.set_weighted_data(VISUAL_SIMILAR, KEY_ADJACENCY)
    # Initialize Myanmar-specific substitution costs (merged with YAML data)
    edit_distance_c.set_myanmar_substitution_costs(_MERGED_SUBSTITUTION_COSTS)
except ImportError:
    _HAS_CYTHON_EDIT_DISTANCE = False

# Myanmar medial characters that commonly form clusters with consonants
MYANMAR_MEDIALS = MEDIALS  # {'\u103b', '\u103c', '\u103d', '\u103e'}

# Myanmar diacritical marks: insertion/deletion cost is reduced to 0.5
# because diacritic omission/addition errors are far more common than
# consonant insertion/deletion errors.
# NOTE: Asat (U+103A) is intentionally EXCLUDED — it changes word identity
# (e.g., ဝယ်→ဝ are entirely different words), unlike tone marks and medials
# which are more commonly omitted/added as typos.
MYANMAR_DIACRITICS = frozenset(
    {
        "\u1036",  # ံ Anusvara
        "\u1037",  # ့ Dot Below (Auk-myit)
        "\u1038",  # း Visarga (Wit-sa-pauk)
        "\u1039",  # ္ Virama (stacking marker)
        "\u103b",  # ျ Ya-pin medial
        "\u103c",  # ြ Ya-yit medial
        "\u103d",  # ွ Wa-hswe medial
        "\u103e",  # ှ Ha-htoe medial
    }
)
DIACRITIC_INDEL_COST = _SYMSPELL_DEFAULTS.diacritic_indel_cost

# Medial confusion pairs (commonly swapped in typing errors)
MEDIAL_CONFUSIONS = {
    ("\u103b", "\u103c"),  # Ya-pin vs Ya-yit (ျ vs ြ)
    ("\u103d", "\u103e"),  # Wa-hswe vs Ha-htoe (ွ vs ှ)
}

# Myanmar character categories for syllable tokenization
MYANMAR_CONSONANTS = CONSONANTS

__all__ = [
    "damerau_levenshtein_distance",
    "levenshtein_distance",
    "myanmar_syllable_edit_distance",
    "tokenize_myanmar_syllable_units",
    "weighted_damerau_levenshtein_distance",
]

# Kinzi sequence: NGA + ASAT + VIRAMA (င်္)
# Kinzi appears visually above/before its host consonant and is encoded before it
# in the text stream. Example: "အင်္ဂလိပ်" (English) has Kinzi (င်္) encoded before ဂ
KINZI_SEQUENCE = NGA + ASAT + VIRAMA


@functools.lru_cache(maxsize=DAMERAU_CACHE_SIZE)
def weighted_damerau_levenshtein_distance(
    s1: str,
    s2: str,
    keyboard_weight: float = _SYMSPELL_DEFAULTS.keyboard_weight,
    visual_weight: float = _SYMSPELL_DEFAULTS.visual_weight,
    transposition_weight: float = _SYMSPELL_DEFAULTS.transposition_weight,
) -> float:
    """
    Calculate Weighted Damerau-Levenshtein distance.

    Args:
        s1: First string
        s2: Second string
        keyboard_weight: Cost for keyboard-adjacent character substitutions (0-1)
        visual_weight: Cost for visually similar character substitutions (0-1)
        transposition_weight: Cost for transposing adjacent characters (0-1).
            Lower values make transposition errors less costly, which is useful
            for common Myanmar vowel ordering mistakes like "ာေ" vs "ော".

    Returns:
        Weighted edit distance as float
    """
    if _HAS_CYTHON_EDIT_DISTANCE:
        result: float = edit_distance_c.weighted_damerau_levenshtein_distance(
            s1, s2, keyboard_weight, visual_weight, transposition_weight
        )
        return result

    len1, len2 = len(s1), len(s2)

    if len1 == 0:
        return float(len2)
    if len2 == 0:
        return float(len1)

    # d[i][j]
    d = [[0.0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        d[i][0] = float(i)
    for j in range(len2 + 1):
        d[0][j] = float(j)

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            c1 = s1[i - 1]
            c2 = s2[j - 1]

            if c1 == c2:
                cost = 0.0
            else:
                # Calculate substitution cost with Myanmar-specific weights
                cost = 1.0

                # 1. Check Myanmar-specific substitution costs first (most precise)
                # Uses merged costs (hardcoded + YAML confusion matrix)
                if c1 in _MERGED_SUBSTITUTION_COSTS and c2 in _MERGED_SUBSTITUTION_COSTS[c1]:
                    cost = min(cost, _MERGED_SUBSTITUTION_COSTS[c1][c2])
                # 2. Check visual similarity (bidirectional lookup)
                elif c1 in VISUAL_SIMILAR and c2 in VISUAL_SIMILAR[c1]:
                    cost = min(cost, visual_weight)
                # 3. Check keyboard adjacency
                elif is_keyboard_adjacent(c1, c2):
                    cost = min(cost, keyboard_weight)

            # Diacritical marks use reduced insertion/deletion cost
            del_cost = DIACRITIC_INDEL_COST if c1 in MYANMAR_DIACRITICS else 1.0
            ins_cost = DIACRITIC_INDEL_COST if c2 in MYANMAR_DIACRITICS else 1.0
            d[i][j] = min(
                d[i - 1][j] + del_cost,  # Deletion
                d[i][j - 1] + ins_cost,  # Insertion
                d[i - 1][j - 1] + cost,  # Substitution
            )

            # Transposition - weighted for common ordering errors
            # Common in Myanmar: "ာေ" vs "ော" (vowel ordering confusion)
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + transposition_weight)

    return d[len1][len2]


@functools.lru_cache(maxsize=DAMERAU_CACHE_SIZE)
def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Damerau-Levenshtein distance between two strings.
    """
    if _HAS_CYTHON_EDIT_DISTANCE:
        result: int = edit_distance_c.damerau_levenshtein_distance(s1, s2)
        return result

    len1, len2 = len(s1), len(s2)

    # Early exit for empty strings
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Create distance matrix
    # d[i][j] = distance between s1[0..i-1] and s2[0..j-1]
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Initialize first row and column
    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j

    # Fill matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1

            d[i][j] = min(
                d[i - 1][j] + 1,  # Deletion
                d[i][j - 1] + 1,  # Insertion
                d[i - 1][j - 1] + cost,  # Substitution
            )

            # Transposition cost is always 1 (unlike substitution which uses character weights)
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)

    return d[len1][len2]


@functools.lru_cache(maxsize=DAMERAU_CACHE_SIZE)
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    """
    if _HAS_CYTHON_EDIT_DISTANCE:
        result: int = edit_distance_c.levenshtein_distance(s1, s2)
        return result

    len1, len2 = len(s1), len(s2)

    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Use single array optimization (space efficient)
    prev_row = list(range(len2 + 1))

    for i in range(1, len1 + 1):
        current_row = [i]

        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1

            current_row.append(
                min(
                    prev_row[j] + 1,  # Deletion
                    current_row[j - 1] + 1,  # Insertion
                    prev_row[j - 1] + cost,  # Substitution
                )
            )

        prev_row = current_row

    return prev_row[len2]


def tokenize_myanmar_syllable_units(text: str) -> list[str]:
    """
    Tokenize Myanmar text into syllable-aware units.

    Groups consonants with their medial characters so that medial
    swaps (like ျ vs ြ) are treated as single-unit operations.

    A syllable unit is:
    - Kinzi (င်္) + Consonant + optional medials grouped together
    - Consonant + optional medials (ျ, ြ, ွ, ှ) grouped together
    - Vowels, tones, and other characters as separate units

    Args:
        text: Myanmar text string

    Returns:
        List of syllable units (strings)

    Example:
        >>> tokenize_myanmar_syllable_units("မြန်")
        ['မြ', 'န', '်']
        >>> tokenize_myanmar_syllable_units("မျန်")
        ['မျ', 'န', '်']
        >>> tokenize_myanmar_syllable_units("အင်္ဂလိပ်")
        ['အ', 'င်္ဂ', 'လ', 'ိ', 'ပ', '်']
    """
    if not text:
        return []

    units: list[str] = []
    i = 0
    kinzi_len = len(KINZI_SEQUENCE)

    while i < len(text):
        char = text[i]

        # Check for Kinzi sequence (င်္) followed by a consonant
        # Kinzi modifies the following consonant, so group them together
        if text[i : i + kinzi_len] == KINZI_SEQUENCE:
            # Check if there's a consonant after Kinzi
            if i + kinzi_len < len(text) and text[i + kinzi_len] in MYANMAR_CONSONANTS:
                unit = KINZI_SEQUENCE + text[i + kinzi_len]
                i += kinzi_len + 1

                # Collect any following medials as part of this unit
                while i < len(text) and text[i] in MYANMAR_MEDIALS:
                    unit += text[i]
                    i += 1

                units.append(unit)
                continue
            # If no consonant follows, treat Kinzi characters separately
            # (this would be invalid Myanmar, but handle gracefully)

        # Check if this is a consonant that may have medials
        if char in MYANMAR_CONSONANTS:
            unit = char
            i += 1

            # Collect any following medials as part of this unit
            while i < len(text) and text[i] in MYANMAR_MEDIALS:
                unit += text[i]
                i += 1

            units.append(unit)
        else:
            # Other characters (vowels, tones, etc.) are separate units
            units.append(char)
            i += 1

    return units


def _are_medial_confusions(unit1: str, unit2: str) -> bool:
    """
    Check if two syllable units differ only by a commonly confused medial.

    Args:
        unit1: First syllable unit
        unit2: Second syllable unit

    Returns:
        True if the units are medial confusion pairs (e.g., မျ vs မြ)
    """
    if len(unit1) < 2 or len(unit2) < 2:
        return False

    # Must have same base consonant
    if unit1[0] != unit2[0]:
        return False

    # Extract medials from each unit
    medials1 = set(unit1[1:])
    medials2 = set(unit2[1:])

    # If they're the same, no confusion
    if medials1 == medials2:
        return False

    # Check if the difference is a known confusion pair
    diff1 = medials1 - medials2
    diff2 = medials2 - medials1

    if len(diff1) == 1 and len(diff2) == 1:
        m1 = next(iter(diff1))
        m2 = next(iter(diff2))
        return (m1, m2) in MEDIAL_CONFUSIONS or (m2, m1) in MEDIAL_CONFUSIONS

    return False


@functools.lru_cache(maxsize=DAMERAU_CACHE_SIZE)
def myanmar_syllable_edit_distance(s1: str, s2: str) -> tuple[int, float]:
    """
    Calculate syllable-aware edit distance for Myanmar text.
    """
    if _HAS_CYTHON_EDIT_DISTANCE:
        result: tuple[int, float] = edit_distance_c.myanmar_syllable_edit_distance(s1, s2)
        return result

    units1 = tokenize_myanmar_syllable_units(s1)
    units2 = tokenize_myanmar_syllable_units(s2)

    len1, len2 = len(units1), len(units2)

    if len1 == 0:
        return (len2, float(len2))
    if len2 == 0:
        return (len1, float(len1))

    # Distance matrices for both integer and weighted calculations
    d_int = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    d_weighted = [[0.0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        d_int[i][0] = i
        d_weighted[i][0] = float(i)
    for j in range(len2 + 1):
        d_int[0][j] = j
        d_weighted[0][j] = float(j)

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            unit1 = units1[i - 1]
            unit2 = units2[j - 1]

            if unit1 == unit2:
                cost_int = 0
                cost_weighted = 0.0
            elif _are_medial_confusions(unit1, unit2):
                # Medial confusion: full integer cost, reduced weighted cost
                cost_int = 1
                cost_weighted = 0.5
            else:
                cost_int = 1
                cost_weighted = 1.0

            d_int[i][j] = min(
                d_int[i - 1][j] + 1,  # Deletion
                d_int[i][j - 1] + 1,  # Insertion
                d_int[i - 1][j - 1] + cost_int,  # Substitution
            )

            d_weighted[i][j] = min(
                d_weighted[i - 1][j] + 1.0,  # Deletion
                d_weighted[i][j - 1] + 1.0,  # Insertion
                d_weighted[i - 1][j - 1] + cost_weighted,  # Substitution
            )

            # Transposition cost is always 1 (unlike substitution which uses syllable class weights)
            if (
                i > 1
                and j > 1
                and units1[i - 1] == units2[j - 2]
                and units1[i - 2] == units2[j - 1]
            ):
                d_int[i][j] = min(d_int[i][j], d_int[i - 2][j - 2] + 1)
                d_weighted[i][j] = min(d_weighted[i][j], d_weighted[i - 2][j - 2] + 1.0)

    return (d_int[len1][len2], d_weighted[len1][len2])
