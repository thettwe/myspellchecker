"""
Centralized Myanmar character confusion pair definitions.

All Myanmar-specific character confusion relationships are defined here:
medial swaps, aspiration pairs, nasal interchanges, stop-coda mergers,
tone marks, and vowel lengths.  This is the single source of truth --
all consumers import from here.

Data structures are immutable (tuple-of-tuples / frozenset) to prevent
accidental mutation at runtime.
"""

from __future__ import annotations

import unicodedata

from myspellchecker.text.phonetic import PhoneticHasher


def _replace_per_position(text: str, old: str, new: str, variants: set[str]) -> None:
    """Replace each occurrence of ``old`` one at a time, adding to ``variants``.

    Unlike ``str.replace()`` which swaps ALL occurrences simultaneously,
    this produces per-position variants.
    """
    start = 0
    while True:
        idx = text.find(old, start)
        if idx < 0:
            break
        variants.add(text[:idx] + new + text[idx + len(old) :])
        start = idx + 1


# ---------------------------------------------------------------------------
# Myanmar character constants (used internally by variant generators)
# ---------------------------------------------------------------------------

_MEDIAL_YA_PIN = "\u103b"  # ျ
_MEDIAL_YA_YIT = "\u103c"  # ြ
_MEDIAL_WA_SWE = "\u103d"  # ွ
_MEDIAL_HA_HTOE = "\u103e"  # ှ

ALL_MEDIALS: tuple[str, ...] = (_MEDIAL_YA_PIN, _MEDIAL_YA_YIT, _MEDIAL_WA_SWE, _MEDIAL_HA_HTOE)
"""All four Myanmar medial characters (ျ ြ ွ ှ)."""

# Medials that support full insert+delete (aspiration/labialization confusion)
_MEDIALS_INSERT_DELETE: tuple[str, ...] = (_MEDIAL_HA_HTOE, _MEDIAL_WA_SWE)

# Medials that support insert-only (deleting ya-yit/ya-pin creates FPs
# on correct words like မနက်ဖြန်→မနက်ဖန်)
_MEDIALS_INSERT_ONLY: tuple[str, ...] = (_MEDIAL_YA_YIT, _MEDIAL_YA_PIN)

# Myanmar consonant codepoint range (for medial insertion positions)
_CONSONANT_START = 0x1000
_CONSONANT_END = 0x1021

# Virama (U+1039) and Asat (U+103A) for stacking repair variants
_VIRAMA = "\u1039"  # ္ (stacker, creates consonant clusters)
_ASAT = "\u103a"  # ် (killed consonant, visible in standalone form)

# Nga consonant (U+1004) for Kinzi detection
_NGA = "\u1004"  # င

# Kinzi sequence: nga + asat + virama (င်္)
_KINZI_SEQ = _NGA + _ASAT + _VIRAMA

# ---------------------------------------------------------------------------
# Confusion pair definitions (single source of truth)
# ---------------------------------------------------------------------------

# Medial swap pairs (commonly confused)
MEDIAL_SWAP_PAIRS: tuple[tuple[str, str], ...] = (
    (_MEDIAL_YA_YIT, _MEDIAL_YA_PIN),  # ျ ↔ ြ
    (_MEDIAL_WA_SWE, _MEDIAL_HA_HTOE),  # ွ ↔ ှ
)
"""Medial character pairs that are commonly swapped in typing errors."""

# Aspirated consonant pairs (unaspirated ↔ aspirated).
# These are the most common consonant confusions in Myanmar typing.
ASPIRATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("\u1000", "\u1001"),  # က ↔ ခ
    ("\u1002", "\u1003"),  # ဂ ↔ ဃ
    ("\u1005", "\u1006"),  # စ ↔ ဆ
    ("\u1010", "\u1011"),  # တ ↔ ထ
    ("\u1012", "\u1013"),  # ဒ ↔ ဓ
    ("\u1015", "\u1016"),  # ပ ↔ ဖ
    ("\u1015", "\u1017"),  # ပ ↔ ဗ
    ("\u1016", "\u1018"),  # ဖ ↔ ဘ
)
"""Aspirated consonant pairs.  The full set used by confusable variant generation."""

# Subset used by rerank candidate generation (omits ဂ↔ဃ, ဒ↔ဓ, ပ↔ဗ).
ASPIRATION_PAIRS_RERANK: tuple[tuple[str, str], ...] = (
    ("\u1000", "\u1001"),  # က ↔ ခ
    ("\u1005", "\u1006"),  # စ ↔ ဆ
    ("\u1010", "\u1011"),  # တ ↔ ထ
    ("\u1015", "\u1016"),  # ပ ↔ ဖ
    ("\u1017", "\u1018"),  # ဗ ↔ ဘ
)
"""Aspirated pairs used by generalized rerank candidate generation."""

# Bidirectional swap map built from ASPIRATION_PAIRS_RERANK
# (per-position aspirated consonant replacement in rerank).
ASPIRATION_SWAP_MAP: dict[str, str] = {}
for _a, _b in ASPIRATION_PAIRS_RERANK:
    ASPIRATION_SWAP_MAP[_a] = _b
    ASPIRATION_SWAP_MAP[_b] = _a

# Nasal ending confusion pairs (coda consonants often confused).
NASAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("\u1014\u103a", "\u1019\u103a"),  # န် ↔ မ် (na-that ↔ ma-that)
    ("\u1014\u103a", "\u1036"),  # န် ↔ ံ (na-that ↔ anusvara)
    ("\u1019\u103a", "\u1036"),  # မ် ↔ ံ (ma-that ↔ anusvara)
)
"""Nasal ending interchange pairs."""

# Stop-coda confusion pairs (voiceless stops merge to glottal stop [ʔ]).
# In spoken Myanmar, final -က်, -တ်, -ပ် are all realized as [ʔ],
# making them homophones in coda position.
STOP_CODA_PAIRS: tuple[tuple[str, str], ...] = (
    ("\u1000\u103a", "\u1010\u103a"),  # က် ↔ တ် (ka-that ↔ ta-that)
    ("\u1000\u103a", "\u1015\u103a"),  # က် ↔ ပ် (ka-that ↔ pa-that)
    ("\u1010\u103a", "\u1015\u103a"),  # တ် ↔ ပ် (ta-that ↔ pa-that)
)
"""Stop-coda pairs that merge to glottal stop in speech."""

# Tone mark interchanges (့ ↔ း)
TONE_MARK_PAIRS: tuple[tuple[str, str], ...] = (
    ("\u1037", "\u1038"),  # ့ ↔ း (dot-below ↔ visarga)
)
"""Tone mark pairs."""

# Vowel length pairs (short ↔ long)
VOWEL_LENGTH_PAIRS: tuple[tuple[str, str], ...] = (
    ("\u102d", "\u102e"),  # ိ ↔ ီ
    ("\u102f", "\u1030"),  # ု ↔ ူ
)
"""Vowel length pairs (short vs long)."""


# ---------------------------------------------------------------------------
# Helper functions (internal to variant generation)
# ---------------------------------------------------------------------------


def _insert_medial_after_consonants(normalized: str, medial: str, variants: set[str]) -> None:
    """Insert a medial after each consonant in the word, adding results to variants."""
    for i, char in enumerate(normalized):
        cp = ord(char)
        if _CONSONANT_START <= cp <= _CONSONANT_END:
            # Find the right insertion point (after consonant and existing medials)
            insert_pos = i + 1
            while insert_pos < len(normalized) and ord(normalized[insert_pos]) in range(
                0x103B, 0x103F
            ):
                # Skip existing medials (ျြွှ = 103B-103E)
                insert_pos += 1
            variant = normalized[:insert_pos] + medial + normalized[insert_pos:]
            variants.add(variant)


def _get_medial_variants(word: str) -> set[str]:
    """
    Generate variants by inserting, removing, or swapping medials.

    For Myanmar, medials appear after the consonant:
    - consonant + medial_ya(ျ) + medial_ra(ြ) + medial_wa(ွ) + medial_ha(ှ)

    This handles cases like:
    - မာ → မှာ (insert ha-htoe after consonant)
    - မှာ → မာ (remove ha-htoe)
    - ပြော် → ပျော် (swap ya-yit for ya-pin)
    """
    variants: set[str] = set()
    normalized = unicodedata.normalize("NFC", word)

    # 1. Insertion/deletion of ha-htoe and wa-hswe
    for medial in _MEDIALS_INSERT_DELETE:
        # Deletion: remove this medial per-position
        start = 0
        while True:
            idx = normalized.find(medial, start)
            if idx < 0:
                break
            deleted = normalized[:idx] + normalized[idx + 1 :]
            # Don't generate single-consonant variants from medial deletion
            # (e.g., မှ → မ: these are different morphemes, not confusables)
            if len(deleted) > 1:
                variants.add(deleted)
            start = idx + 1

        # Insertion: add medial after each consonant
        _insert_medial_after_consonants(normalized, medial, variants)

    # 1b. Insert-only for ya-yit/ya-pin (no deletion -- removing an existing
    # ya-yit/ya-pin creates FPs on correct words like မနက်ဖြန်→မနက်ဖန်)
    for medial in _MEDIALS_INSERT_ONLY:
        _insert_medial_after_consonants(normalized, medial, variants)

    # 2. Medial swaps (ျ↔ြ, ွ↔ှ) — per-position
    for medial_a, medial_b in MEDIAL_SWAP_PAIRS:
        _replace_per_position(normalized, medial_a, medial_b, variants)
        _replace_per_position(normalized, medial_b, medial_a, variants)

    return variants


def _get_nasal_variants(word: str) -> set[str]:
    """
    Generate variants by swapping nasal endings.

    Handles confusion between:
    - န် (na-that) ↔ မ် (ma-that)
    - န် (na-that) ↔ ံ (anusvara)
    - မ် (ma-that) ↔ ံ (anusvara)
    """
    variants: set[str] = set()
    normalized = unicodedata.normalize("NFC", word)

    for nasal_a, nasal_b in NASAL_PAIRS:
        _replace_per_position(normalized, nasal_a, nasal_b, variants)
        _replace_per_position(normalized, nasal_b, nasal_a, variants)

    return variants


def _get_stop_coda_variants(word: str) -> set[str]:
    """
    Generate variants by swapping stop-coda endings.

    In spoken Myanmar, final voiceless stops -က်, -တ်, -ပ် all merge
    to a glottal stop [ʔ], making them homophones in coda position.

    Handles confusion between:
    - က် (ka-that) ↔ တ် (ta-that)
    - က် (ka-that) ↔ ပ် (pa-that)
    - တ် (ta-that) ↔ ပ် (pa-that)
    """
    variants: set[str] = set()
    normalized = unicodedata.normalize("NFC", word)

    for stop_a, stop_b in STOP_CODA_PAIRS:
        start = 0
        while True:
            idx = normalized.find(stop_a, start)
            if idx < 0:
                break
            variants.add(normalized[:idx] + stop_b + normalized[idx + len(stop_a) :])
            start = idx + 1

        start = 0
        while True:
            idx = normalized.find(stop_b, start)
            if idx < 0:
                break
            variants.add(normalized[:idx] + stop_a + normalized[idx + len(stop_b) :])
            start = idx + 1

    return variants


# ---------------------------------------------------------------------------
# Confusable-check predicates
# ---------------------------------------------------------------------------


def is_aspirated_confusable(word: str, variant: str) -> bool:
    """Check if the difference between word and variant is an aspiration change.

    Covers pairs like က↔ခ, တ↔ထ, ပ↔ဖ at ANY position (not just initial).
    Returns True when exactly one character differs and that diff is an
    aspiration pair.
    """
    if not word or not variant or len(word) != len(variant):
        return False
    # Find all differing positions
    diff_pos = [(i, word[i], variant[i]) for i in range(len(word)) if word[i] != variant[i]]
    if len(diff_pos) != 1:
        return False
    _, a_char, b_char = diff_pos[0]
    # Check if the single-character swap is an aspiration pair
    for a, b in ASPIRATION_PAIRS:
        if (a_char == a and b_char == b) or (a_char == b and b_char == a):
            return True
    return False


def is_medial_confusable(word: str, variant: str) -> bool:
    """Check if the difference between word and variant is a medial change.

    Covers ျ↔ြ swap and single medial insertion/deletion.
    """
    # Medial swap check (ျ↔ြ)
    for a, b in MEDIAL_SWAP_PAIRS:
        if a in word and b not in word and b in variant and a not in variant:
            if word.replace(a, b) == variant:
                return True
        if b in word and a not in word and a in variant and b not in variant:
            if word.replace(b, a) == variant:
                return True
    # Medial insertion/deletion
    for m in ALL_MEDIALS:
        if m in variant and m not in word and variant.replace(m, "", 1) == word:
            return True
        if m in word and m not in variant and word.replace(m, "", 1) == variant:
            return True
    return False


# ---------------------------------------------------------------------------
# Public variant generation
# ---------------------------------------------------------------------------


def _get_aspiration_variants(word: str) -> set[str]:
    """
    Generate variants by swapping aspirated consonant pairs.

    Handles confusion between unaspirated and aspirated consonants:
    က↔ခ, ဂ↔ဃ, စ↔ဆ, တ↔ထ, ဒ↔ဓ, ပ↔ဖ, ပ↔ဗ, ဖ↔ဘ
    """
    variants: set[str] = set()
    normalized = unicodedata.normalize("NFC", word)

    for a, b in ASPIRATION_PAIRS:
        for i, char in enumerate(normalized):
            if char == a:
                variants.add(normalized[:i] + b + normalized[i + 1 :])
            elif char == b:
                variants.add(normalized[:i] + a + normalized[i + 1 :])

    return variants


def _get_tone_variants(word: str) -> set[str]:
    """
    Generate variants by swapping tone marks (့ ↔ း).
    """
    variants: set[str] = set()
    normalized = unicodedata.normalize("NFC", word)

    for a, b in TONE_MARK_PAIRS:
        _replace_per_position(normalized, a, b, variants)
        _replace_per_position(normalized, b, a, variants)

    return variants


def _get_vowel_length_variants(word: str) -> set[str]:
    """
    Generate variants by swapping vowel length (short ↔ long: ိ↔ီ, ု↔ူ).
    """
    variants: set[str] = set()
    normalized = unicodedata.normalize("NFC", word)

    for a, b in VOWEL_LENGTH_PAIRS:
        _replace_per_position(normalized, a, b, variants)
        _replace_per_position(normalized, b, a, variants)

    return variants


def _get_stacking_variants(word: str) -> set[str]:
    """
    Generate variants by swapping between asat+consonant and virama stacking.

    Handles the common Pali/Sanskrit loanword confusion where users type
    the unstacked form (consonant + asat + consonant) instead of the
    stacked form (consonant + virama + consonant) or vice versa.

    Examples:
        ဗုဒ်ဓ → ဗုဒ္ဓ  (asat → virama: unstacked → stacked)
        ဗုဒ္ဓ → ဗုဒ်ဓ  (virama → asat: stacked → unstacked)
        သဒ်ဒါ → သဒ္ဒါ  (asat → virama)
        ဝိဇ်ဇာ → ဝိဇ္ဇာ  (asat → virama)
    """
    variants: set[str] = set()
    normalized = unicodedata.normalize("NFC", word)
    n = len(normalized)

    for i in range(n - 1):
        cp_i = ord(normalized[i])
        # Must be a consonant (U+1000-U+1021)
        if not (_CONSONANT_START <= cp_i <= _CONSONANT_END):
            continue

        next_char = normalized[i + 1]

        # Pattern 1: consonant + asat (်) + consonant → consonant + virama (္) + consonant
        if next_char == _ASAT and i + 2 < n:
            cp_after = ord(normalized[i + 2])
            if _CONSONANT_START <= cp_after <= _CONSONANT_END:
                variant = normalized[: i + 1] + _VIRAMA + normalized[i + 2 :]
                variants.add(variant)

        # Pattern 2: consonant + virama (္) + consonant → consonant + asat (်) + consonant
        if next_char == _VIRAMA and i + 2 < n:
            cp_after = ord(normalized[i + 2])
            if _CONSONANT_START <= cp_after <= _CONSONANT_END:
                variant = normalized[: i + 1] + _ASAT + normalized[i + 2 :]
                variants.add(variant)

    return variants


def _get_kinzi_variants(word: str) -> set[str]:
    """
    Generate variants by adding or removing Kinzi (င်္).

    Kinzi is a Myanmar orthographic prefix where nga (င) stacks above
    the following consonant via virama.  It is encoded as:
        nga(U+1004) + asat(U+103A) + virama(U+1039) + consonant

    The most common Myanmar spelling error is omitting the Kinzi
    (dropping the virama), writing e.g. အင်ဂလိပ် instead of အင်္ဂလိပ်.

    Handles two patterns per-position:
    - Kinzi removal: nga+asat+virama+C → nga+asat+C  (correct → error)
    - Kinzi insertion: nga+asat+C → nga+asat+virama+C  (error → correct)
    """
    variants: set[str] = set()
    normalized = unicodedata.normalize("NFC", word)
    n = len(normalized)

    for i in range(n):
        if normalized[i] != _NGA:
            continue

        # Pattern 1: Kinzi removal — find nga+asat+virama+consonant,
        # replace with nga+asat+consonant (drop the virama).
        if (
            i + 3 < n
            and normalized[i + 1] == _ASAT
            and normalized[i + 2] == _VIRAMA
            and _CONSONANT_START <= ord(normalized[i + 3]) <= _CONSONANT_END
        ):
            variant = normalized[: i + 2] + normalized[i + 3 :]
            variants.add(variant)

        # Pattern 2: Kinzi insertion — find nga+asat+consonant (not
        # already a Kinzi).  Pattern 1 requires virama at i+2; here we
        # require a consonant at i+2, so the two patterns are mutually
        # exclusive.  Insert virama to create Kinzi.
        if (
            i + 2 < n
            and normalized[i + 1] == _ASAT
            and _CONSONANT_START <= ord(normalized[i + 2]) <= _CONSONANT_END
        ):
            # Skip if the consonant at i+2 already stacks with something
            # (i.e., i+3 is virama) — that's a stacking pattern, not a
            # Kinzi insertion candidate.
            if i + 3 < n and normalized[i + 3] == _VIRAMA:
                continue
            variant = normalized[: i + 2] + _VIRAMA + normalized[i + 2 :]
            variants.add(variant)

    return variants


def generate_myanmar_variants(word: str) -> set[str]:
    """
    Generate Myanmar-specific variant candidates without requiring a PhoneticHasher.

    Combines structural confusion patterns:
    1. Aspiration swaps (က↔ခ, တ↔ထ, etc.)
    2. Medial insertion/deletion/swap (ျ↔ြ, ွ↔ှ, insert/delete ှ/ွ)
    3. Nasal ending confusion (န်↔မ်↔ံ)
    4. Stop-coda confusion (က်↔တ်↔ပ်)
    5. Tone mark swaps (့↔း)
    6. Vowel length swaps (ိ↔ီ, ု↔ူ)
    7. Stacking repair (consonant+asat ↔ consonant+virama)

    Args:
        word: The word to generate variants for.

    Returns:
        Set of variant strings (excluding the original word).
    """
    from myspellchecker.text.normalize import normalize

    variants: set[str] = set()

    # 1. Aspiration swaps
    variants.update(_get_aspiration_variants(word))

    # 2. Medial insertion/deletion/swap
    variants.update(_get_medial_variants(word))

    # 3. Nasal ending confusion
    variants.update(_get_nasal_variants(word))

    # 4. Stop-coda confusion
    variants.update(_get_stop_coda_variants(word))

    # 5. Tone mark swaps
    variants.update(_get_tone_variants(word))

    # 6. Vowel length swaps
    variants.update(_get_vowel_length_variants(word))

    # 7. Stacking repair (asat ↔ virama)
    variants.update(_get_stacking_variants(word))

    # 8. Kinzi variants (င်္ add/remove)
    variants.update(_get_kinzi_variants(word))

    # Normalize all variants to canonical form
    variants = {normalize(v) for v in variants}

    variants.discard(word)

    return variants


def generate_confusable_variants(word: str, hasher: PhoneticHasher) -> set[str]:
    """
    Generate confusable variants of a word.

    Combines:
    1. Phonetic variants (aspiration swaps, medial swaps, visual confusables)
    2. Tonal variants (visarga add/remove, tone mark swaps)
    3. Medial insertion/deletion (ha-htoe, wa-hswe)
    4. Nasal ending confusion (န်↔မ်↔ံ)
    5. Stop-coda confusion (က်↔တ်↔ပ်)
    6. Stacking repair (asat ↔ virama)

    Args:
        word: The word to generate variants for.
        hasher: PhoneticHasher instance for phonetic/tonal variants.

    Returns:
        Set of variant strings (excluding the original word).
    """
    from myspellchecker.text.normalize import normalize

    variants: set[str] = set()

    # 1. Phonetic variants from PhoneticHasher
    variants.update(hasher.get_phonetic_variants(word))

    # 2. Tonal variants from PhoneticHasher
    variants.update(hasher.get_tonal_variants(word))

    # 3. Medial insertion/deletion
    variants.update(_get_medial_variants(word))

    # 4. Nasal ending confusion
    variants.update(_get_nasal_variants(word))

    # 5. Stop-coda confusion (က်↔တ်↔ပ် merge to [ʔ])
    variants.update(_get_stop_coda_variants(word))

    # 6. Stacking repair (asat ↔ virama)
    variants.update(_get_stacking_variants(word))

    # 7. Kinzi variants (င်္ add/remove)
    variants.update(_get_kinzi_variants(word))

    # Normalize all variants -- medial/tone insertions can produce
    # non-canonical character sequences (e.g., dot-below before asat)
    # that mismatch the normalized forms in n-gram DB lookups.
    variants = {normalize(v) for v in variants}

    variants.discard(word)

    return variants
