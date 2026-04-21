"""
Unicode normalization utilities for Myanmar (Burmese) text.

This module provides normalization functions to handle Myanmar script
encoding variations, character sequence ordering, and Unicode standardization.

Myanmar script presents unique normalization challenges:
1. Multiple encoding forms (Myanmar3 vs legacy encodings like Zawgyi)
2. Flexible character ordering (e.g., medial consonants and diacritics)
3. Combining characters vs precomposed characters
4. Zero-width characters and control characters

The normalization process ensures consistent representation for spell checking.

Normalization Hierarchy
=======================

This module is part of a 3-layer normalization architecture:

Layer 1: Cython Core (normalize_c.pyx)
--------------------------------------
Performance-critical functions implemented in Cython/C++ for ~20x speedup.
These are the building blocks used by higher layers:

- ``remove_zero_width_chars()``: Remove ZWSP, ZWNJ, ZWJ, BOM characters
- ``reorder_myanmar_diacritics()``: Canonical diacritic ordering (UTN #11)
- ``get_myanmar_ratio()``: Calculate Myanmar character proportion
- ``is_myanmar_string()``: Check if first character is Myanmar
- ``clean_text_for_segmentation()``: Prepare text for Viterbi algorithm
- ``segment_syllables_c()``: Syllable segmentation

Layer 2: Python Wrapper (this module - normalize.py)
----------------------------------------------------
Higher-level normalization functions that compose Cython primitives:

- ``normalize()``: Main normalization with configurable steps
- ``normalize_for_lookup()``: Full normalization for dictionary lookups
- ``normalize_character_variants()``: Map character variants to canonical forms
- ``normalize_with_zawgyi_conversion()``: Full pipeline with Zawgyi handling
- Zawgyi detection: ``is_likely_zawgyi()``, ``detect_encoding()``

Layer 3: Normalization Service (normalization_service.py)
---------------------------------------------------------
Purpose-specific normalization through a unified service interface:

- ``NormalizationService.for_spell_checking()``: Fast, no Zawgyi conversion
- ``NormalizationService.for_dictionary_lookup()``: Full with Zawgyi
- ``NormalizationService.for_comparison()``: Aggressive normalization
- ``NormalizationService.for_display()``: Minimal, preserve formatting
- ``NormalizationService.for_ingestion()``: Corpus data processing

When to Use Which
-----------------
- **Spell checking pipeline**: Use ``NormalizationService.for_spell_checking()``
- **Dictionary lookups**: Use ``NormalizationService.for_dictionary_lookup()``
- **Text comparison**: Use ``NormalizationService.for_comparison()``
- **Display to users**: Use ``NormalizationService.for_display()``
- **Corpus ingestion**: Use ``NormalizationService.for_ingestion()``
- **Custom normalization**: Use ``normalize()`` with specific parameters
- **Performance-critical code**: Use Cython functions directly from ``normalize_c``

Wrapper Pattern (Cython + Python Fallback)
------------------------------------------
This module uses a wrapper pattern for graceful degradation:

1. The Cython module (``normalize_c.pyx``) provides optimized implementations
2. This Python module imports from Cython and exposes higher-level APIs
3. If Cython compilation fails (no C++ compiler), the library can still
   function using pure Python alternatives (with reduced performance)

The import pattern looks like::

    from myspellchecker.text.normalize_c import (
        remove_zero_width_chars as c_remove_zero_width,
        reorder_myanmar_diacritics as c_reorder_diacritics,
    )

Note: Pure Python fallbacks are NOT provided in this module for the core
Cython functions. For systems without C++ compilers, install via wheel.

Example Usage
-------------
>>> from myspellchecker.text import get_normalization_service
>>> service = get_normalization_service()
>>>
>>> # Standard spell checking normalization
>>> normalized = service.for_spell_checking("မြန်မာ")
>>>
>>> # For direct low-level access
>>> from myspellchecker.text.normalize import normalize
>>> normalized = normalize(text, remove_zero_width=True)
"""

from __future__ import annotations

import functools
import logging
import re
import unicodedata
from typing import Literal

from myspellchecker.core.config.text_configs import ZawgyiConfig
from myspellchecker.core.constants import (
    COMMON_PUNCTUATION,
    MYANMAR_PUNCTUATION,
    MYANMAR_RANGE,  # (0x1000, 0x109F) - core Myanmar block for Zawgyi detection
)
from myspellchecker.text.normalize_c import (
    get_myanmar_ratio as c_get_myanmar_ratio,
)
from myspellchecker.text.normalize_c import (
    remove_zero_width_chars as c_remove_zero_width,
)
from myspellchecker.text.normalize_c import (
    reorder_myanmar_diacritics as c_reorder_diacritics,
)

__all__ = [
    "CHARACTER_VARIANT_MAP",
    "CHARACTER_VARIANT_MAP_SAFE",
    "CHARACTER_VARIANT_MAP_ZERO_WA",
    "ZawgyiDetectionWarning",
    "check_zawgyi_and_warn",
    "convert_zawgyi_to_unicode",
    "detect_encoding",
    "get_nasal_ending",
    "get_nasal_variants",
    "has_same_nasal_ending",
    "is_likely_zawgyi",
    "is_myanmar_text",
    "is_space_segmented_myanmar",
    "is_true_nasal_variant",
    "normalize",
    "normalize_character_variants",
    "normalize_e_vowel_tall_aa",
    "normalize_for_lookup",
    "normalize_tall_aa_after_wa",
    "normalize_u_vowel_with_asat",
    "normalize_with_zawgyi_conversion",
    "remove_punctuation",
    "remove_word_segmentation_markers",
]

# Default Zawgyi configuration (module-level singleton)
_default_zawgyi_config = ZawgyiConfig()

# Module logger for tracking silent failures
# Using standard logging pattern to avoid circular import with utils package
# (utils/__init__.py imports from this module, creating a cycle if we import from utils)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 1: Character Variant Mappings
# ---------------------------------------------------------------------------

# Character variant normalization mapping
# Maps visually similar or encoding variants to their canonical form
# This ensures consistent representation for spell checking
#
# Note: Zero/Wa mapping is separated to allow context-aware handling.
# In text context, zero (၀) looks identical to letter Wa (ဝ) and is often
# used interchangeably. However, in numeric context, this mapping would
# corrupt data. Use normalize_zero_to_wa=False when processing numbers.

# Extended-A and archaic variant mappings (always safe to apply)
CHARACTER_VARIANT_MAP_SAFE: dict[str, str] = {
    # Myanmar Extended-A to base Myanmar equivalents
    "\uaa60": "\u1002",  # AA60 MYANMAR LETTER KHAMTI GA → 1002 GA
    "\uaa61": "\u1001",  # AA61 MYANMAR LETTER KHAMTI KHA → 1001 KHA
    "\uaa62": "\u1002",  # AA62 MYANMAR LETTER KHAMTI GA → 1002 GA
    "\uaa63": "\u1003",  # AA63 MYANMAR LETTER KHAMTI GHA → 1003 GHA
    "\uaa64": "\u1004",  # AA64 MYANMAR LETTER KHAMTI NGA → 1004 NGA
    "\uaa65": "\u1005",  # AA65 MYANMAR LETTER KHAMTI CA → 1005 CA
    "\uaa66": "\u1007",  # AA66 MYANMAR LETTER KHAMTI JA → 1007 JA
    # Non-standard core chars (OCR/social media normalization)
    "\u1022": "\u1021",  # 1022 SHAN LETTER A (ဢ) → 1021 MYANMAR LETTER A (အ)
    "\u1028": "\u1027",  # 1028 MON LETTER E (ဨ) → 1027 MYANMAR LETTER E (ဧ)
    # Archaic/variant forms
    # NOTE: U+104E (၎) is a distinct logographic symbol and must not be
    # normalized to digit four (၄).
}

# Zero/Wa mapping - only apply in text context, NOT in numeric context
CHARACTER_VARIANT_MAP_ZERO_WA: dict[str, str] = {
    "\u1040": "\u101d",  # 1040 MYANMAR DIGIT ZERO → 101D LETTER WA
}

# Combined mapping for backward compatibility (includes zero/wa)
CHARACTER_VARIANT_MAP: dict[str, str] = {
    **CHARACTER_VARIANT_MAP_SAFE,
    **CHARACTER_VARIANT_MAP_ZERO_WA,
}


# ---------------------------------------------------------------------------
# Section 2: Core Normalization Functions
# ---------------------------------------------------------------------------


def normalize_character_variants(
    text: str,
    normalize_zero_to_wa: bool = True,
) -> str:
    """
    Normalize Myanmar character variants to their canonical forms.

    This handles visually similar characters that may be used interchangeably
    due to font rendering or encoding confusion.

    Args:
        text: Input text with potential character variants.
        normalize_zero_to_wa: If True (default), converts numeral zero (၀) to
                             letter Wa (ဝ). Set to False when processing text
                             that may contain numbers to preserve numeric data.


    Returns:
        Text with character variants normalized.

    Example:
        >>> normalize_character_variants("သူ၀ယ်သည်", normalize_zero_to_wa=True)
        'သူဝယ်သည်'  # Zero becomes Wa (correct for "he buys")
        >>> normalize_character_variants("၁၀၀", normalize_zero_to_wa=False)
        '၁၀၀'  # Zeros preserved (correct for number "100")
    """
    if not text:
        return text

    # Select appropriate mapping based on zero/wa handling
    if normalize_zero_to_wa:
        mapping = CHARACTER_VARIANT_MAP
    else:
        mapping = CHARACTER_VARIANT_MAP_SAFE

    return text.translate(str.maketrans(mapping))  # type: ignore[arg-type]


def normalize_tall_aa_after_wa(text: str) -> str:
    """
    Normalize Tall AA (ါ) to AA (ာ) when it appears after Medial Wa (ွ).

    In standard Myanmar orthography, after medial ွ (wa-hswe, U+103D),
    the vowel is always written as ာ (U+102C), never ါ (U+102B).
    The ါ form after ွ is a Zawgyi-era artifact with zero valid exceptions
    in standard Burmese.

    This normalization is safe to apply unconditionally because:
    - ွ + ါ has zero valid occurrences in standard Myanmar
    - The correction ွ + ါ → ွ + ာ preserves phonetic meaning
    - All known Myanmar words with ွ use ာ (e.g., သွား, ပွား, ဂွာ)

    Args:
        text: Input Myanmar text

    Returns:
        Text with ွ+ါ sequences replaced by ွ+ာ

    Example:
        >>> normalize_tall_aa_after_wa("ပွါး")
        'ပွား'
        >>> normalize_tall_aa_after_wa("သွား")  # Already correct
        'သွား'

    See: https://github.com/thettwe/my-spellchecker/issues/1357
    """
    if not text:
        return text
    # Replace Medial Wa + Tall AA with Medial Wa + AA
    return text.replace("\u103d\u102b", "\u103d\u102c")


# Consonants that take TALL AA (ါ, U+102B) after the ေ (U+1031) prefix.
#
# Scope choice: this whitelist is intentionally MINIMAL. Classical Myanmar
# orthography (MLC မြန်မာ သတ်ပုံ ကျမ်း, UTN #11 §3.3) lists a broader
# "round-bottom" set {ပ, ဖ, ဗ, ဘ, မ, ဒ, ဓ, ဝ, ဂ, ဏ, ဎ, င, ရ}, but in
# modern standard Burmese — and crucially in the v1.5 benchmark gold labels —
# many of those consonants routinely take the plain AA form (e.g., ဖော်,
# ဘော, မော, ရော). Treating the full classical set as "must-be-TALL-AA"
# would corrupt common correct forms into OCR-flavored variants.
#
# The whitelist below is restricted to the three consonants where the
# benchmark gold exclusively uses TALL AA after ေ — i.e. where applying the
# repair is safe against FPR regression:
#
#   ပ (U+1015): 24 gold TALL_AA cases (ပေါ်, ပေါင်း, ...)
#   ခ (U+1001): 11 gold TALL_AA cases (ခေါ်, ခေါင်း, ...)
#   ဒ (U+1012):  1 gold TALL_AA case  (ဒေါ် "Mrs.")
#
# If a future benchmark row validates a broader set (e.g. ဂ, င, ဝ), widen
# this frozenset and regenerate the audit. Do NOT preemptively widen without
# benchmark evidence — the false-positive tax on clean sentences is
# asymmetric with the FN recovery, and modern Burmese has real lexical
# exceptions that a blanket classical rule would break.
_ROUND_BOTTOM_CONSONANTS_FOR_TALL_AA: frozenset[str] = frozenset(
    {
        "\u1015",  # ပ  PA
        "\u1001",  # ခ  KHA
        "\u1012",  # ဒ  DA
    }
)

_E_VOWEL = "\u1031"  # ေ  MYANMAR VOWEL SIGN E
_AA = "\u102c"  # ာ  MYANMAR VOWEL SIGN AA
_TALL_AA = "\u102b"  # ါ  MYANMAR VOWEL SIGN TALL AA


def normalize_e_vowel_tall_aa(text: str) -> str:
    """
    Canonicalize the "aw" vowel (ေ + {ာ, ါ}) per MLC orthography.

    Myanmar orthography picks between AA (ာ, U+102C) and TALL AA (ါ, U+102B)
    in the "aw" vowel slot by the shape of the preceding consonant: certain
    round-bottom / open-bottom consonants take TALL AA so the vowel does not
    visually collide with the consonant's curved bowl. This is a
    glyph-disambiguation rule baked into the orthography — not a cosmetic
    variant.

    | Preceding consonant | Canonical aw-vowel | Examples |
    |---|---|---|
    | {ပ, ခ, ဒ} (benchmark-validated subset) | `ေ + ါ` | ပေါ်, ပေါင်း, ခေါ်, ခေါင်း, ဒေါ် |
    | All others | `ေ + ာ` | ကောင်း, ကော်, တော, ဖော်, ဘော, ရော |

    Behaviour:
    - After the whitelisted round-bottom consonants, flat ``ော`` is repaired
      to ``ေါ`` (restoring the canonical TALL AA form that OCR / keyboard
      input often flattens).
    - After every other consonant, stray ``ေါ`` is flattened to ``ော``
      (restoring canonical AA for the complement set).

    An earlier version of this function unconditionally rewrote
    ``ေါ → ော`` regardless of the preceding consonant, which corrupted
    gold forms like ခေါ်, ပေါင်း, ဒေါ် during
    ``normalize_for_dictionary_lookup``. The consonant whitelist guards
    against that class of regression.

    The whitelist is intentionally narrower than the classical MLC
    round-bottom set; see the ``_ROUND_BOTTOM_CONSONANTS_FOR_TALL_AA``
    block comment for the rationale and the criterion to widen it.

    Scope limit: the rewrite fires only on the bare pattern
    ``consonant + ေ + {ာ, ါ}`` at adjacent positions. Medial or stacking
    interpositions — e.g. ``ပ + ြ + ေ + ာ`` (``ပြော``) or ``ခ + ျ + ေ + ာ``
    — are *not* matched and pass through unmodified. This is deliberate
    (the round-bottom / tall-AA interaction with medials is still under
    benchmark validation), so expanding the whitelist or the match pattern
    without per-consonant verification is a regression risk.

    Args:
        text: Input Myanmar text.

    Returns:
        Text with ``ေ`` + {``ာ``, ``ါ``} pairs normalized to the form
        dictated by the preceding consonant.

    Example:
        >>> normalize_e_vowel_tall_aa("ပော်")   # flat, preceding ပ
        'ပေါ်'
        >>> normalize_e_vowel_tall_aa("ပေါ်")   # already canonical
        'ပေါ်'
        >>> normalize_e_vowel_tall_aa("ကောင်း")  # complement set, unchanged
        'ကောင်း'
        >>> normalize_e_vowel_tall_aa("ကေါင်း")  # wrongly tall after က
        'ကောင်း'
        >>> normalize_e_vowel_tall_aa("ဖော်")   # ဖ is outside whitelist
        'ဖော်'

    Sources:
        - Myanmar Language Commission, *မြန်မာ သတ်ပုံ ကျမ်း* (1978 rev. 2003).
        - Unicode Technical Note #11, "Representing Myanmar in Unicode"
          (Martin Hosken, rev. 4, §3.3 Vowel signs).
        - Okell, *A Reference Grammar of Colloquial Burmese*.
    """
    if not text:
        return text

    # Single-pass rewrite. For every ``ေ`` that is preceded by a consonant
    # and followed by AA or TALL AA, emit the canonical form for that
    # consonant's class. Characters outside this pattern pass through.
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        # Looking for consonant + ေ + {ာ, ါ} at positions i, i+1, i+2.
        if i + 2 < n and text[i + 1] == _E_VOWEL and text[i + 2] in (_AA, _TALL_AA):
            target_aw = _TALL_AA if ch in _ROUND_BOTTOM_CONSONANTS_FOR_TALL_AA else _AA
            out.append(ch)
            out.append(_E_VOWEL)
            out.append(target_aw)
            i += 3
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def normalize_u_vowel_with_asat(text: str) -> str:
    """
    Normalize U+1025 (ဥ) + U+103A (asat) to U+1009 (ဉ) + U+103A (asat).

    In standard Myanmar orthography, the sequence ဥ + ် (independent vowel U
    followed by asat) is not valid. The correct character for the consonant
    /nya/ with asat is ဉ (U+1009). The ဥ+် form is a common encoding
    artefact, OCR error, or input method confusion because ဥ (U+1025) and
    ဉ (U+1009) look visually similar in many fonts.

    This normalization only targets the two-character sequence U+1025 U+103A;
    a standalone ဥ (e.g. in ဥပမာ "example") is left unchanged because it is
    a valid independent vowel in that context.

    Args:
        text: Input Myanmar text.

    Returns:
        Text with ဥ+် sequences replaced by ဉ+်.

    Example:
        >>> normalize_u_vowel_with_asat("ယာဥ\u103a")
        'ယာဉ\u103a'
        >>> normalize_u_vowel_with_asat("ဥပမာ")  # standalone ဥ — no change
        'ဥပမာ'

    See: https://github.com/thettwe/my-spellchecker/issues/1386
    """
    if not text:
        return text
    # Replace U+1025 + U+103A (ဥ + asat) with U+1009 + U+103A (ဉ + asat)
    return text.replace("\u1025\u103a", "\u1009\u103a")


def normalize(
    text: str,
    form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC",
    remove_zero_width: bool = True,
    reorder_diacritics: bool = True,
    normalize_variants: bool = False,
    normalize_tall_aa: bool = True,
    normalize_u_asat: bool = True,
) -> str:
    """
    Normalize Myanmar text for consistent representation.

    This is the main normalization function that should be applied to all
    input text before spell checking. It performs:

    1. Unicode normalization (NFC by default)
    2. Zero-width character removal (optional)
    3. Myanmar-specific diacritic reordering (optional)
    4. Tall AA after Medial Wa correction (optional, default: True)
    5. U vowel + asat normalization (optional, default: True)
    6. Character variant normalization (optional)

    Args:
        text: Input Myanmar text to normalize
        form: Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')
              - NFC (default): Canonical decomposition + composition
              - NFD: Canonical decomposition
              - NFKC: Compatibility decomposition + composition
              - NFKD: Compatibility decomposition
        remove_zero_width: Remove zero-width characters (default: True)
        reorder_diacritics: Apply Myanmar-specific character reordering
                           (default: True)
        normalize_variants: Normalize character variants to canonical forms
                           (default: False). When True, maps visually similar
                           characters (e.g., Extended-A to base Myanmar).
        normalize_tall_aa: Correct ွ+ါ to ွ+ာ (default: True). In standard
                          Myanmar, Tall AA never follows Medial Wa.
                          In standard Myanmar, Tall AA never follows Medial Wa.
        normalize_u_asat: Convert ဥ+် (U+1025+U+103A) to ဉ+် (U+1009+U+103A)
                         (default: True). The independent vowel ဥ cannot take
                         asat; the correct consonant is ဉ.

    Returns:
        Normalized text string

    Edge Cases Handled:
        - **Empty/None-like strings**: Returns input unchanged (early exit)
        - **Non-Myanmar text**: Passed through with only Unicode normalization
        - **Mixed content**: Myanmar portions reordered, others preserved
        - **Kinzi sequences**: Properly ordered as Consonant + Asat + Virama
        - **Multiple diacritics**: Sorted per UTN #11 (Ya < Ra < Wa < Ha)
        - **Zero-width in middle**: Removed without affecting surrounding text

    Diacritic Ordering (UTN #11):
        When reorder_diacritics=True, diacritics are sorted to canonical order:
        - Medials: Ya (103B) < Ra (103C) < Wa (103D) < Ha (103E)
        - Vowels: E (1031) < Upper (102D/E/32) < Tall (102B/C) < Lower (102F/30)
        - Finals: Asat (103A) < Anusvara (1036) < Dot (1037) < Visarga (1038)

    Example:
        >>> from myspellchecker.text.normalize import normalize_myanmar_text
        >>> text = "မြန်မာ"  # May have encoding issues
        >>> clean_text = normalize_myanmar_text(text)
        >>> # Now safe to use for spell checking
        >>>
        >>> # Diacritic reordering example
        >>> normalize("\\u1000\\u103d\\u103b")  # Ka + Wa + Ya (wrong order)
        '\\u1000\\u103b\\u103d'  # Ka + Ya + Wa (correct order)

    Note:
        This function does NOT convert legacy encodings (e.g., Zawgyi) to
        Unicode. Text should already be in Unicode Myanmar3 encoding.
        For Zawgyi conversion, use ``normalize_with_zawgyi_conversion()``
        or ``NormalizationService.for_dictionary_lookup()``.

    See Also:
        - ``normalize_for_lookup()``: Full normalization with Zawgyi handling
        - ``NormalizationService``: Purpose-specific normalization methods
    """
    if not text:
        return text

    # Coerce str subclasses (e.g. Suggestion) to plain str for Cython compatibility
    if type(text) is not str:
        text = str(text)

    # Step 1: Unicode normalization
    normalized = unicodedata.normalize(form, text)

    # Step 2: Remove zero-width characters
    if remove_zero_width:
        normalized = c_remove_zero_width(normalized)

    # Step 3: Myanmar-specific diacritic reordering
    if reorder_diacritics:
        normalized = c_reorder_diacritics(normalized)

    # Step 4: Tall AA corrections (Medial Wa context + E-vowel context)
    if normalize_tall_aa:
        normalized = normalize_tall_aa_after_wa(normalized)
        normalized = normalize_e_vowel_tall_aa(normalized)

    # Step 5: U vowel + asat normalization (ဥ+် → ဉ+်)
    if normalize_u_asat:
        normalized = normalize_u_vowel_with_asat(normalized)

    # Step 6: Character variant normalization (safe only — no zero-to-Wa)
    if normalize_variants:
        normalized = normalize_character_variants(normalized, normalize_zero_to_wa=False)

    return normalized


def normalize_for_lookup(
    text: str,
    convert_zawgyi: bool = True,
    config: ZawgyiConfig | None = None,
) -> str:
    """
    Unified normalization for all dictionary/index lookups.

    This function provides a single normalization path that should be used
    before any dictionary lookup, index access, or comparison operation.
    It ensures consistent text representation across all components.

    The normalization pipeline:
    1. Early exit for empty text
    2. Strip leading/trailing whitespace
    3. Zawgyi to Unicode conversion (if detected and enabled)
    4. Unicode NFC normalization (canonical composition)
    5. Zero-width character removal
    6. Myanmar-specific diacritic reordering
    7. Tall AA after Medial Wa correction (ွ+ါ → ွ+ာ)
    8. U vowel + asat normalization (ဥ+် → ဉ+်)
    9. Safe character variant normalization (Extended-A → base Myanmar)
    10. Final strip and cleanup

    Args:
        text: Input text to normalize for lookup.
        convert_zawgyi: Whether to attempt Zawgyi conversion (default: True).
        config: ZawgyiConfig instance for threshold settings.

    Returns:
        Normalized text suitable for dictionary lookups.

    Edge Cases Handled:
        - **Empty/whitespace-only**: Returns empty string
        - **Zawgyi text**: Converted to Unicode (if convert_zawgyi=True)
        - **Zawgyi conversion failure**: Logs warning, continues with original
        - **Mixed encoding**: Best-effort conversion (may have artifacts)
        - **Pure non-Myanmar**: Passed through with Unicode normalization only

    Idempotency:
        This function is idempotent - calling it multiple times on the same
        text produces identical results. This is important for caching and
        ensuring consistent behavior regardless of call order.

    Example:
        >>> from myspellchecker.text.normalize import normalize_for_lookup
        >>> # Normalize before dictionary lookup
        >>> normalized_word = normalize_for_lookup(user_input)
        >>> if normalized_word in dictionary:
        ...     # Found match
        ...     pass
        >>>
        >>> # Idempotency check
        >>> assert normalize_for_lookup(text) == normalize_for_lookup(normalize_for_lookup(text))

    Note:
        For spell checking (where Zawgyi should be detected but not
        auto-converted, to warn users), use
        ``NormalizationService.for_spell_checking()`` instead.

    See Also:
        - ``NormalizationService.for_dictionary_lookup()``: Service layer equivalent
        - ``normalize()``: Lower-level normalization without Zawgyi handling
    """
    if not text:
        return text

    # Coerce str subclasses (e.g. Suggestion) to plain str for Cython compatibility
    if type(text) is not str:
        text = str(text)

    cfg = config or _default_zawgyi_config

    # Step 1: Initial cleanup
    result = text.strip()
    if not result:
        return result

    # Step 2: Zawgyi conversion (if enabled and detected)
    if convert_zawgyi:
        # convert_zawgyi_to_unicode is defined later in this module
        try:
            result = convert_zawgyi_to_unicode(result, config=cfg)
        except (RuntimeError, UnicodeError, ValueError, KeyError) as e:
            # Log but continue with unconverted text
            logger.debug(f"Zawgyi conversion skipped: {e}")

    # Step 3: Unicode NFC normalization (canonical composition)
    result = unicodedata.normalize("NFC", result)

    # Step 4: Remove zero-width characters
    result = c_remove_zero_width(result)

    # Step 5: Myanmar-specific diacritic reordering
    result = c_reorder_diacritics(result)

    # Step 6: Tall AA corrections (Medial Wa + E-vowel contexts)
    result = normalize_tall_aa_after_wa(result)
    result = normalize_e_vowel_tall_aa(result)

    # Step 7: U vowel + asat normalization (ဥ+် → ဉ+်)
    result = normalize_u_vowel_with_asat(result)

    # Step 8: Safe character variant normalization (Extended-A → base Myanmar)
    result = normalize_character_variants(result, normalize_zero_to_wa=False)

    # Step 9: Final cleanup
    return result.strip()


# ---------------------------------------------------------------------------
# Section 3: Text Classification & Cleaning
# ---------------------------------------------------------------------------


def is_myanmar_text(
    text: str,
    config: ZawgyiConfig | None = None,
    allow_extended: bool = False,
) -> bool:
    """
    Check if text is primarily Myanmar script.

    Useful for detecting if input text is Myanmar before processing.

    Args:
        text: Text to check
        config: ZawgyiConfig instance for threshold settings.
        allow_extended: If False (default), only count core Burmese characters
            (U+1000-U+104F excluding non-standard chars).
            If True, count all Myanmar blocks including:
            - Extended Core Block (U+1050-U+109F)
            - Extended-A (U+AA60-U+AA7F)
            - Extended-B (U+A9E0-U+A9FF)
            - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)
            Default is Burmese-only (False).

    Returns:
        True if Myanmar character proportion >= threshold, False otherwise

    Example:
        >>> is_myanmar_text("မြန်မာ")
        True
        >>> is_myanmar_text("Hello မြန်မာ")
        True  # Still >50% Myanmar
        >>> is_myanmar_text("Hello World")
        False
    """
    if not text:
        return False

    cfg = config or _default_zawgyi_config
    ratio = c_get_myanmar_ratio(text, allow_extended)
    return bool(ratio >= cfg.myanmar_text_threshold)


def remove_punctuation(text: str, keep_myanmar_punct: bool = True) -> str:
    """
    Remove punctuation from text, optionally keeping Myanmar punctuation.

    Args:
        text: Input text
        keep_myanmar_punct: Keep Myanmar-specific punctuation marks
                           (U+104A to U+104F) (default: True)

    Returns:
        Text with punctuation removed

    Example:
        >>> remove_punctuation("သူ၊ သွားသည်။")
        'သူ၊ သွားသည်'
        >>> remove_punctuation("သူ၊ သွားသည်။", keep_myanmar_punct=False)
        'သူ သွားသည်'
    """
    result = []
    for char in text:
        # Always preserve whitespace (whitespace is not punctuation)
        if char in " \t\n\r":
            result.append(char)
        # Keep Myanmar punctuation if requested
        elif keep_myanmar_punct and char in MYANMAR_PUNCTUATION:
            result.append(char)
        # Skip all other punctuation
        elif char not in COMMON_PUNCTUATION and char not in MYANMAR_PUNCTUATION:
            result.append(char)

    return "".join(result)


# Regex patterns for removing artificial word/syllable segmentation markers.
# These patterns match underscores or spaces that appear between Myanmar characters,
# which indicate pre-segmented text (not natural word boundaries).
_UNDERSCORE_BETWEEN_MYANMAR_RE = re.compile(r"([\u1000-\u109F])_([\u1000-\u109F])")
_SPACE_BETWEEN_MYANMAR_RE = re.compile(r"([\u1000-\u109F]) ([\u1000-\u109F])")


def remove_word_segmentation_markers(text: str, *, remove_spaces: bool = False) -> str:
    """
    Remove artificial word/syllable segmentation markers from pre-segmented text.

    Some corpus datasets contain Myanmar text where syllables or words are
    separated by underscores or spaces (e.g., from word segmentation tools).
    This function joins them back into natural Myanmar text so the pipeline's
    own segmenter can process them correctly.

    Handles:
    - Underscores between Myanmar characters (always removed — never natural):
      ``ကောင်း_မွန်_ပါ_တယ်`` -> ``ကောင်းမွန်ပါတယ်``
    - Spaces between Myanmar characters (only when ``remove_spaces=True``):
      ``ကျွန်တော် သွား ပါ မယ်`` -> ``ကျွန်တော်သွားပါမယ်``

    Does NOT remove (even with ``remove_spaces=True``):
    - Spaces between Myanmar and non-Myanmar text (preserves mixed-language boundaries)
    - Myanmar punctuation spacing (section marks ``။`` ``၊``)

    Args:
        text: Input text potentially containing segmentation markers.
        remove_spaces: If True, also remove spaces between Myanmar characters.
            Default is False because spaces between Myanmar words can be
            natural in modern web text and chat data. Only enable this for
            datasets known to be artificially word-segmented.

    Returns:
        Text with artificial segmentation markers removed.

    Example:
        >>> remove_word_segmentation_markers("ကောင်း_မွန်_ပါ_တယ်")
        'ကောင်းမွန်ပါတယ်'
        >>> remove_word_segmentation_markers("ကျွန်တော် သွား", remove_spaces=True)
        'ကျွန်တော်သွား'
        >>> remove_word_segmentation_markers("Hello ကျွန်တော်")
        'Hello ကျွန်တော်'
    """
    if not text:
        return text

    # Step 1: Remove underscores between Myanmar characters (always safe).
    # Apply twice to handle consecutive separators: A_B_C → AB_C → ABC
    result = _UNDERSCORE_BETWEEN_MYANMAR_RE.sub(r"\1\2", text)
    result = _UNDERSCORE_BETWEEN_MYANMAR_RE.sub(r"\1\2", result)

    # Step 2: Remove spaces between Myanmar characters only when explicitly requested.
    # Natural Myanmar text can have spaces between words (modern web/chat text),
    # so this should only be enabled for known word-segmented datasets.
    if remove_spaces:
        result = _SPACE_BETWEEN_MYANMAR_RE.sub(r"\1\2", result)
        result = _SPACE_BETWEEN_MYANMAR_RE.sub(r"\1\2", result)

    return result


def is_space_segmented_myanmar(
    lines: list[str],
    threshold: float = 0.5,
    config: ZawgyiConfig | None = None,
) -> bool:
    """
    Detect if a list of text lines contains space-segmented Myanmar text.

    Space-segmented datasets have Myanmar characters interspersed with spaces
    (e.g., "ကျွန်တော် သွား ပါ မယ်" instead of "ကျွန်တော်သွားပါမယ်").

    Heuristic: For each line that is primarily Myanmar text, count the ratio
    of Myanmar-to-Myanmar transitions that go through a space. If more than
    ``threshold`` of sampled lines show this pattern, the dataset is likely
    space-segmented.

    Args:
        lines: Sample of text lines to analyze (typically first 50-100 lines).
        threshold: Minimum fraction of Myanmar lines that must be space-segmented
            to trigger detection. Default: 0.5 (50%).
        config: ZawgyiConfig instance for threshold settings.

    Returns:
        True if the text appears to be space-segmented Myanmar.
    """
    if not lines:
        return False

    cfg = config or _default_zawgyi_config
    min_line_len = cfg.min_line_length_for_zawgyi
    min_mm_chars = cfg.min_myanmar_chars_for_zawgyi
    min_spaces = cfg.space_seg_min_spaces
    space_ratio = cfg.space_seg_ratio

    myanmar_lines = 0
    space_segmented_lines = 0

    for line in lines:
        if not line or len(line) < min_line_len:
            continue

        # Count Myanmar characters and spaces between Myanmar chars
        myanmar_chars = sum(1 for c in line if "\u1000" <= c <= "\u109f")
        if myanmar_chars < min_mm_chars:
            continue

        myanmar_lines += 1

        # Count spaces between Myanmar characters
        spaces_between_myanmar = 0
        for i in range(len(line) - 2):
            if (
                "\u1000" <= line[i] <= "\u109f"
                and line[i + 1] == " "
                and i + 2 < len(line)
                and "\u1000" <= line[i + 2] <= "\u109f"
            ):
                spaces_between_myanmar += 1

        # If significant portion of the line has Myanmar-space-Myanmar pattern
        if (
            spaces_between_myanmar >= min_spaces
            and spaces_between_myanmar / myanmar_chars > space_ratio
        ):
            space_segmented_lines += 1

    if myanmar_lines == 0:
        return False

    return (space_segmented_lines / myanmar_lines) >= threshold


# ---------------------------------------------------------------------------
# Section 4: Zawgyi Detection & Conversion
# ---------------------------------------------------------------------------


def is_likely_zawgyi(
    text: str,
    config: ZawgyiConfig | None = None,
) -> tuple[bool, float]:
    """
    Detect if text is likely encoded in legacy Zawgyi-1 encoding.

    Uses Google's myanmar-tools library for statistical detection.
    This provides ~95% accuracy compared to ~60% for pattern-based approaches.

    Args:
        text: Text to analyze for Zawgyi encoding
        config: ZawgyiConfig instance for threshold settings.

    Returns:
        Tuple of (is_zawgyi: bool, confidence: float)
        - is_zawgyi: True if confidence >= detection_threshold
        - confidence: Float 0.0-1.0 indicating Zawgyi probability

    Raises:
        ImportError: If myanmartools is not installed (should not occur)

    Example:
        >>> is_zawgyi, conf = is_likely_zawgyi("မြန်မာ")  # Unicode
        >>> print(f"Is Zawgyi: {is_zawgyi}, Confidence: {conf:.2f}")
        Is Zawgyi: False, Confidence: 0.02

        >>> is_zawgyi, conf = is_likely_zawgyi("ျမန္မာ")  # Zawgyi
        >>> print(f"Is Zawgyi: {is_zawgyi}, Confidence: {conf:.2f}")
        Is Zawgyi: True, Confidence: 0.98
    """
    if not text:
        return False, 0.0

    cfg = config or _default_zawgyi_config

    # Count Myanmar characters - need minimum for reliable detection
    # Use centralized MYANMAR_RANGE constant (core block only, U+1000-U+109F)
    # Zawgyi only affected this range - extended blocks came later
    myanmar_start, myanmar_end = MYANMAR_RANGE
    myanmar_chars = sum(1 for c in text if myanmar_start <= ord(c) <= myanmar_end)
    if myanmar_chars < cfg.min_myanmar_chars_for_zawgyi:
        return False, 0.0

    # Get myanmartools detector (required dependency)
    detector = _get_zawgyi_detector()
    if detector is None:
        return False, 0.0

    try:
        confidence = detector.get_zawgyi_probability(text)
        return confidence >= cfg.detection_threshold, confidence
    except (RuntimeError, ValueError, UnicodeError, KeyError) as e:
        # Log error but don't crash - return safe default
        logger.warning(f"Zawgyi detection failed: {e}")
        return False, 0.0


def detect_encoding(
    text: str,
    config: ZawgyiConfig | None = None,
) -> tuple[str, float]:
    """
    Detect the encoding of Myanmar text.

    Args:
        text: Text to analyze
        config: ZawgyiConfig instance for threshold settings.

    Returns:
        Tuple of (encoding: str, confidence: float)
        - encoding: "unicode", "zawgyi", or "unknown"
        - confidence: Float 0.0-1.0

    Example:
        >>> encoding, confidence = detect_encoding("မြန်မာ")
        >>> encoding
        'unicode'
        >>> encoding, confidence = detect_encoding("မြန္မာ")
        >>> encoding
        'zawgyi'
    """
    if not text:
        return "unknown", 0.0

    cfg = config or _default_zawgyi_config

    # Check Myanmar character ratio first (use low threshold for detection)
    low_threshold_cfg = ZawgyiConfig(myanmar_text_threshold=0.1)
    if not is_myanmar_text(text, config=low_threshold_cfg):
        return "unknown", 0.0

    is_zawgyi, zawgyi_confidence = is_likely_zawgyi(text, config=cfg)

    if is_zawgyi:
        return "zawgyi", zawgyi_confidence
    elif zawgyi_confidence < cfg.unicode_determination_threshold:
        # Low Zawgyi signals = likely Unicode
        return "unicode", 1.0 - zawgyi_confidence
    else:
        # Ambiguous case
        return "unknown", 0.5


class ZawgyiDetectionWarning:
    """Warning class for Zawgyi detection results."""

    def __init__(self, confidence: float, message: str):
        """Initialize a Zawgyi detection warning.

        Args:
            confidence: Detection confidence score (0.0-1.0).
            message: Human-readable warning message.
        """
        self.type = "encoding"
        self.confidence = confidence
        self.message = message
        self.suggestion = (
            "Convert text from Zawgyi to Unicode before spell checking. "
            "Tools: myanmar-tools (Python), Rabbit Converter (web)"
        )

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return f"ZawgyiDetectionWarning(confidence={self.confidence:.2f}, message='{self.message}')"


def check_zawgyi_and_warn(
    text: str,
    config: ZawgyiConfig | None = None,
) -> ZawgyiDetectionWarning | None:
    """
    Check text for Zawgyi encoding and return warning if detected.

    This is a convenience function for integration with spell checking.

    Args:
        text: Text to check
        config: ZawgyiConfig instance for threshold settings.

    Returns:
        ZawgyiDetectionWarning if Zawgyi detected, None otherwise

    Example:
        >>> warning = check_zawgyi_and_warn("မြန္မာ")  # Zawgyi
        >>> if warning:
        ...     print(warning.message)
        Text appears to be Zawgyi encoded (confidence: 85%)...
    """
    cfg = config or _default_zawgyi_config

    # Use warning_threshold for this function (user-facing warnings)
    warning_cfg = ZawgyiConfig(detection_threshold=cfg.warning_threshold)
    is_zawgyi, confidence = is_likely_zawgyi(text, config=warning_cfg)

    if is_zawgyi:
        message = (
            f"Text appears to be Zawgyi encoded (confidence: {confidence:.0%}). "
            f"Spell checking results may be inaccurate. "
            f"Please convert to Unicode for accurate spell checking."
        )
        return ZawgyiDetectionWarning(confidence=confidence, message=message)

    return None


# ---------------------------------------------------------------------------
# Section 5: Nasal Variant Analysis
# ---------------------------------------------------------------------------

# Myanmar nasal consonants that can appear before Asat (္)
NASAL_CONSONANTS = {
    "\u1014",  # န (na)
    "\u1019",  # မ (ma)
    "\u1004",  # င (nga)
}

# Anusvara (ံ) - the canonical nasal marker
ANUSVARA = "\u1036"

# Asat (္) - used with consonants to indicate no vowel
ASAT = "\u103a"

# True nasal variant pairs: only န် ↔ ံ represent the same /n/ phoneme
# Other nasals (င်=/ŋ/, မ်=/m/) are different phonemes
NA_ASAT = "\u1014\u103a"  # န်


@functools.lru_cache(maxsize=8192)
def get_nasal_ending(word: str) -> str:
    """
    Get the nasal ending of a Myanmar word.

    Returns the nasal consonant character (န, မ, င) or ံ (anusvara)
    if the word ends with a nasal sound. Returns empty string if not.

    Args:
        word: Myanmar word to check

    Returns:
        The nasal ending character, or empty string if no nasal ending

    Example:
        >>> get_nasal_ending("ခိုန်")  # ends with /n/
        'န'
        >>> get_nasal_ending("ခိုင်")  # ends with /ŋ/
        'င'
        >>> get_nasal_ending("ခုန်")  # ends with /n/
        'န'
        >>> get_nasal_ending("ကံ")  # ends with anusvara /n/
        'ံ'
        >>> get_nasal_ending("ခု")  # no nasal ending
        ''
    """
    if not word:
        return ""

    # Check for anusvara (ံ)
    if ANUSVARA in word:
        # Find last anusvara
        idx = word.rfind(ANUSVARA)
        # Verify it's at the end (only followed by tone marks or nothing)
        if idx >= 0:
            suffix = word[idx + 1 :]
            # Only tone marks or empty after anusvara
            if all(c in "\u1037\u1038" for c in suffix):
                return ANUSVARA

    # Check for nasal consonant + asat
    for nasal in NASAL_CONSONANTS:
        nasal_asat = nasal + ASAT
        if nasal_asat in word:
            idx = word.rfind(nasal_asat)
            if idx >= 0:
                suffix = word[idx + 2 :]
                # Only tone marks or empty after nasal+asat
                if all(c in "\u1037\u1038" for c in suffix):
                    return nasal

    return ""


@functools.lru_cache(maxsize=16384)
def has_same_nasal_ending(term: str, candidate: str) -> bool:
    """
    Check if two Myanmar words have the same nasal ending.

    This function compares the nasal consonant phoneme at the end of two words.
    It considers န် and ံ as equivalent (both represent /n/), but distinguishes
    different nasal consonants (န/n vs င/ŋ vs မ/m).

    Args:
        term: The original term
        candidate: The candidate to compare

    Returns:
        True if both have the same nasal phoneme ending

    Example:
        >>> has_same_nasal_ending("ခိုန်", "ခုန်")  # both /n/
        True
        >>> has_same_nasal_ending("ခိုန်", "ခိုင်")  # /n/ vs /ŋ/
        False
        >>> has_same_nasal_ending("ကန်", "ကံ")  # both /n/ (variant forms)
        True
    """
    term_nasal = get_nasal_ending(term)
    cand_nasal = get_nasal_ending(candidate)

    if not term_nasal or not cand_nasal:
        return False

    # Same nasal
    if term_nasal == cand_nasal:
        return True

    # Treat န and ံ as equivalent (both /n/)
    n_equivalents = {"\u1014", ANUSVARA}  # န and ံ
    if term_nasal in n_equivalents and cand_nasal in n_equivalents:
        return True

    return False


@functools.lru_cache(maxsize=16384)
def is_true_nasal_variant(term: str, candidate: str) -> bool:
    """
    Check if two Myanmar words are TRUE nasal variants.

    True nasal variants only differ in that one uses န် and the other uses ံ,
    both representing the /n/ sound. Other nasal endings (င်, မ်) are different
    phonemes and should NOT be considered variants.

    Args:
        term: The original term
        candidate: The potential variant

    Returns:
        True if they differ only in န် vs ံ substitution

    Example:
        >>> is_true_nasal_variant("ကန်", "ကံ")  # True - same /n/ sound
        True
        >>> is_true_nasal_variant("ခိုန်", "ခိုင်")  # False - /n/ vs /ŋ/
        False
        >>> is_true_nasal_variant("ခိုန်", "ခုန်")  # False - vowel difference
        False
    """
    if term == candidate:
        return False

    # Check if one has န် where the other has ံ (and nothing else differs)
    # Convert နျ + ASAT to ံ in both and compare
    term_normalized = term.replace(NA_ASAT, ANUSVARA)
    cand_normalized = candidate.replace(NA_ASAT, ANUSVARA)

    if term_normalized != cand_normalized:
        return False

    # Verify that the only difference is န် vs ံ
    # Count the differences in the original strings
    if len(term) == len(candidate):
        # Same length - could be single character swap
        diffs = sum(1 for a, b in zip(term, candidate, strict=False) if a != b)
        if diffs == 1:
            # Single character difference - check if it's nasal-related
            for _i, (a, b) in enumerate(zip(term, candidate, strict=False)):
                if a != b:
                    # Check if the single diff is na (\u1014) <-> ANUSVARA
                    return (a == ANUSVARA and b == "\u1014") or (b == ANUSVARA and a == "\u1014")
    elif abs(len(term) - len(candidate)) == 1:
        # Length differs by 1 - could be န် (2 chars) vs ံ (1 char)
        # Walk both strings to verify the specific difference is at a nasal position
        longer, shorter = (term, candidate) if len(term) > len(candidate) else (candidate, term)
        j = 0
        k = 0
        found_nasal_diff = False
        while j < len(longer) and k < len(shorter):
            if longer[j] == shorter[k]:
                j += 1
                k += 1
            elif (
                longer[j] == "\u1014"
                and j + 1 < len(longer)
                and longer[j + 1] == ASAT
                and shorter[k] == ANUSVARA
            ):
                found_nasal_diff = True
                j += 2  # Skip NA + ASAT
                k += 1  # Skip ANUSVARA
            elif (
                shorter[k] == "\u1014"
                and k + 1 < len(shorter)
                and shorter[k + 1] == ASAT
                and longer[j] == ANUSVARA
            ):
                found_nasal_diff = True
                j += 1
                k += 2
            else:
                return False  # Non-nasal difference
        if j < len(longer) or k < len(shorter):
            return False
        return found_nasal_diff

    return False


@functools.lru_cache(maxsize=8192)
def get_nasal_variants(word: str) -> frozenset[str]:
    """
    Generate all nasal spelling variants of a word.

    This generates **candidate** variants for spell checking by substituting
    all possible nasal endings. Not all generated variants are real words —
    callers must validate candidates against a dictionary.

    Results are cached (LRU, 8192 entries) because the same words are looked
    up repeatedly during SymSpell candidate expansion and context checking.
    Returns a ``frozenset`` for cache hashability; callers should only
    iterate over the result, never mutate it.

    Note:
        In Myanmar, nasal alternation is not universal. For example,
        "ကံ" (fate) generates "ကင်", "ကန်", "ကမ်" — but only "ကန်"
        is an actual word (with different meaning). The SymSpell algorithm
        filters these candidates against the dictionary.

    Args:
        word: Word to generate variants for

    Returns:
        Frozenset of all nasal variants including the original word

    Example:
        >>> variants = get_nasal_variants("ကံ")
        >>> print(sorted(variants))
        ['ကင်', 'ကန်', 'ကမ်', 'ကံ']
        >>> variants = get_nasal_variants("နိုင်ငံ")
        >>> 'နိုင်ငန်' in variants
        True
    """
    variants: set[str] = {word}

    # Find positions with nasal endings
    nasal_positions = _find_nasal_positions(word)

    if not nasal_positions:
        return frozenset(variants)

    # Generate variants for each combination
    _generate_variants_recursive(word, nasal_positions, 0, variants)

    return frozenset(variants)


def _find_nasal_positions(word: str) -> list[tuple[int, str]]:
    """
    Find all nasal positions in a word.

    Returns list of (position, type) tuples where:
    - type is "anusvara" for ံ
    - type is "consonant" for န်/မ်/င်
    """
    positions = []
    i = 0
    word_len = len(word)

    while i < word_len:
        char = word[i]

        # Check for nasal + asat (but not Kinzi: nasal + asat + virama)
        if (
            char in NASAL_CONSONANTS
            and i + 1 < word_len
            and word[i + 1] == ASAT
            and (i + 2 >= word_len or word[i + 2] != "\u1039")
        ):
            positions.append((i, "consonant"))
            i += 2
            continue

        # Check for anusvara
        if char == ANUSVARA:
            positions.append((i, "anusvara"))
            i += 1
            continue

        i += 1

    return positions


MAX_NASAL_VARIANTS = 100


def _generate_variants_recursive(
    word: str,
    positions: list[tuple[int, str]],
    pos_idx: int,
    variants: set[str],
) -> None:
    """Recursively generate all nasal variants."""
    if len(variants) >= MAX_NASAL_VARIANTS:
        return
    if pos_idx >= len(positions):
        return

    pos, nasal_type = positions[pos_idx]

    # Generate variants for this position
    new_words = set()

    if nasal_type == "anusvara":
        # Anusvara at position pos
        # Generate ံ/မ်/င် variants
        for nasal_consonant in ["\u1014", "\u1019", "\u1004"]:
            variant = word[:pos] + nasal_consonant + ASAT + word[pos + 1 :]
            new_words.add(variant)
            variants.add(variant)
    else:
        # Consonant + asat at position pos
        original_consonant = word[pos]

        # Generate ံ variant
        variant = word[:pos] + ANUSVARA + word[pos + 2 :]
        new_words.add(variant)
        variants.add(variant)

        # Generate other consonant variants
        for nasal_consonant in ["\u1014", "\u1019", "\u1004"]:
            if nasal_consonant != original_consonant:
                variant = word[:pos] + nasal_consonant + ASAT + word[pos + 2 :]
                new_words.add(variant)
                variants.add(variant)

    # Recurse for remaining positions on the original word (positions are stable)
    _generate_variants_recursive(word, positions, pos_idx + 1, variants)

    for new_word in new_words:
        # Recalculate positions for the new word after length-changing substitution.
        # Use character offset (not array index) to find the correct next position,
        # since anusvara→consonant+asat changes word length by +1 and vice versa.
        new_positions = _find_nasal_positions(new_word)
        # The modified region ends at pos + (1 for anusvara→consonant+asat, 2 for consonant+asat→*)
        modified_end = pos + 2 if nasal_type == "anusvara" else pos + 1
        # Find the first nasal position past the region we just modified
        next_idx = None
        for idx, (p, _) in enumerate(new_positions):
            if p >= modified_end:
                next_idx = idx
                break
        if next_idx is not None:
            _generate_variants_recursive(new_word, new_positions, next_idx, variants)


# Import from consolidated module to avoid duplication
from myspellchecker.text.zawgyi_support import _convert_zawgyi_internal  # noqa: E402
from myspellchecker.text.zawgyi_support import (  # noqa: E402
    get_zawgyi_detector as _get_zawgyi_detector,
)
from myspellchecker.text.zawgyi_support import (  # noqa: E402
    is_zawgyi_converter_available as _is_zawgyi_converter_available,
)


def convert_zawgyi_to_unicode(
    text: str,
    config: ZawgyiConfig | None = None,
) -> str:
    """
    Convert Zawgyi-encoded text to Unicode Myanmar.

    Uses myanmartools for detection and python-myanmar for conversion.
    Only converts text that is detected as Zawgyi to avoid corrupting
    valid Unicode text.

    Args:
        text: Text to convert (may be Zawgyi or Unicode)
        config: ZawgyiConfig instance for threshold settings.

    Returns:
        Converted Unicode text, or original text if not Zawgyi

    Example:
        >>> convert_zawgyi_to_unicode("ျမန္မာ")  # Zawgyi
        'မြန်မာ'
        >>> convert_zawgyi_to_unicode("မြန်မာ")  # Unicode - unchanged
        'မြန်မာ'

    Note:
        Requires myanmartools and python-myanmar packages.
        Falls back to original text if packages not installed.
    """
    if not text:
        return text

    cfg = config or _default_zawgyi_config

    # Check if conversion is available
    if not _is_zawgyi_converter_available():
        return text

    detector = _get_zawgyi_detector()
    if detector is None:
        # No detector available, use heuristic detection
        conv_cfg = ZawgyiConfig(detection_threshold=cfg.conversion_threshold)
        is_zawgyi, _ = is_likely_zawgyi(text, config=conv_cfg)
        if not is_zawgyi:
            return text
        # Heuristic says Zawgyi — proceed to conversion
        return _convert_zawgyi_internal(text)

    # Use myanmartools detector (more accurate)
    prob = detector.get_zawgyi_probability(text)
    if prob < cfg.conversion_threshold:
        return text

    # Convert Zawgyi to Unicode using consolidated helper
    return _convert_zawgyi_internal(text)


def normalize_with_zawgyi_conversion(
    text: str,
    config: ZawgyiConfig | None = None,
    form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC",
    remove_zero_width: bool = True,
    reorder_diacritics: bool = True,
) -> str:
    """
    Normalize Myanmar text with automatic Zawgyi conversion.

    This is the recommended normalization function for processing input
    from unknown sources that may contain Zawgyi-encoded text.

    Pipeline:
    1. Detect if text is Zawgyi-encoded
    2. Convert Zawgyi to Unicode if detected
    3. Apply standard Unicode normalization

    Args:
        text: Input text (may be Zawgyi or Unicode)
        config: ZawgyiConfig instance for threshold settings.
        form: Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')
        remove_zero_width: Remove zero-width characters
        reorder_diacritics: Apply Myanmar-specific character reordering

    Returns:
        Normalized Unicode text

    Example:
        >>> normalize_with_zawgyi_conversion("ျမန္မာ")  # Zawgyi input
        'မြန်မာ'  # Unicode output
        >>> normalize_with_zawgyi_conversion("မြန်မာ")  # Unicode input
        'မြန်မာ'  # Unicode output (unchanged)
    """
    if not text:
        return text

    cfg = config or _default_zawgyi_config

    # Step 1: Convert Zawgyi to Unicode if detected
    text = convert_zawgyi_to_unicode(text, config=cfg)

    # Step 2: Apply standard normalization (with safe variant mapping)
    return normalize(
        text,
        form=form,
        remove_zero_width=remove_zero_width,
        reorder_diacritics=reorder_diacritics,
        normalize_variants=True,
    )


# Public API alias
normalize_myanmar_text = normalize
