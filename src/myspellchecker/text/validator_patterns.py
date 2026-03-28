"""Compiled regex patterns and character-set constants for Myanmar text validation.

All patterns are pre-compiled at import time for performance.  The module is
intentionally data-only (no functions) so that it can be imported cheaply by
any sub-module that needs a specific pattern.
"""

from __future__ import annotations

import re

from myspellchecker.core.constants import (
    CONSONANTS,
    INDEPENDENT_VOWELS,
    MEDIALS,
    MYANMAR_NUMERALS,
    NON_STANDARD_CHARS,
    TONE_MARKS,
    VOWEL_CARRIER,
    VOWEL_SIGNS,
)

__all__ = [
    "ASAT_BEFORE_VOWEL_PATTERN",
    "ASAT_INITIAL_PATTERN",
    "BROKEN_VIRAMA_PATTERN",
    "COMPOUND_TRUNCATED_ENDING_PATTERN",
    "CONSECUTIVE_ASAT_PATTERN",
    "DIGIT_TONE_PATTERN",
    "DOUBLE_ENDING_PATTERN",
    "DOUBLED_CONSONANT_PATTERN",
    "DOUBLED_INITIAL_CONSONANT_PATTERN",
    "DOUBLED_MEDIAL_PATTERN",
    "DOUBLED_VOWEL_PATTERN",
    "EXTENDED_MYANMAR_PATTERN",
    "FRAGMENT_CONSONANT_ASAT_PATTERN",
    "FRAGMENT_CONSONANT_TONE_ASAT_PATTERN",
    "FRAGMENT_CONSONANT_TONE_PATTERN",
    "INCOMPLETE_CONSONANT_MEDIAL_PATTERN",
    "INCOMPLETE_MEDIAL_CONSONANT_END_PATTERN",
    "INCOMPLETE_MEDIAL_END_PATTERN",
    "INCOMPLETE_O_VOWEL_PATTERN",
    "INCOMPLETE_STACKING_PATTERN",
    "INVALID_TONE_SEQUENCE",
    "INVALID_VOWEL_SEQUENCE",
    "MISSING_E_CONSONANT_AA_NG_PATTERN",
    "MISSING_E_MEDIAL_AA_NG_PATTERN",
    "MIXED_LETTER_NUMERAL_PATTERN",
    "MYANMAR_CONSONANTS",
    "MYANMAR_DIGITS",
    "MYANMAR_MEDIALS",
    "MYANMAR_TONES",
    "MYANMAR_VOWELS",
    "NON_STANDARD_PATTERN",
    "PURE_NUMERAL_PATTERN",
    "SCRAMBLED_ASAT_PATTERN",
    "VALID_STARTERS",
    "VIRAMA_AT_END_PATTERN",
    "VOWEL_BEFORE_ASAT_PATTERN",
    "ZAWGYI_YA_ASAT_PATTERN",
    "ZAWGYI_YA_TERMINAL_PATTERN",
]

# ============================================================================
# CHARACTER SETS (imported from core/constants for consistency)
# ============================================================================

# Aliases for backward compatibility - all point to canonical definitions
MYANMAR_CONSONANTS: set[str] = CONSONANTS | {VOWEL_CARRIER}  # Includes vowel carrier အ
MYANMAR_VOWELS: set[str] = VOWEL_SIGNS  # Dependent vowel signs
MYANMAR_MEDIALS: set[str] = MEDIALS  # The four medial consonant forms
MYANMAR_TONES: set[str] = TONE_MARKS  # Anusvara, Dot Below, Visarga
MYANMAR_DIGITS: set[str] = MYANMAR_NUMERALS  # Myanmar digits ၀-၉

# Valid word starters (consonants and independent vowels)
VALID_STARTERS: set[str] = MYANMAR_CONSONANTS | set(INDEPENDENT_VOWELS) | MYANMAR_DIGITS

# ============================================================================
# DETECTION PATTERNS
# ============================================================================

# Extended Myanmar character detection (Shan, Mon, Karen languages)
# These should NOT appear in standard Myanmar/Burmese text when allow_extended_myanmar=False
# Extended Myanmar includes:
# - U+1050-U+109F: Extended Core Block (Shan, Mon, Karen in main block)
# - U+AA60-U+AA7F: Extended-A (Shan, Khamti Shan, Aiton, Phake)
# - U+A9E0-U+A9FF: Extended-B (Shan, Pa'O)
# - Non-standard core chars (U+1022, U+1028, U+1033-U+1035): Mon/Shan (separate pattern below)
EXTENDED_MYANMAR_PATTERN = re.compile(r"[\u1050-\u109f\uAA60-\uAA7F\uA9E0-\uA9FF]")

# Non-Standard Characters (Mon/Shan specific chars in Core Block)
# U+1022, U+1028, U+1033, U+1034, U+1035 - part of extended Myanmar scope
NON_STANDARD_PATTERN = re.compile(r"[" + "".join(sorted(NON_STANDARD_CHARS)) + r"]")

# Zawgyi ျ used as pseudo-asat pattern
# In Zawgyi: ငျး was used instead of င်း (ya-medial as asat substitute)
# Valid: ကျ, ချ, မျက် - ya-medial followed by consonant
# Invalid: ငျး, မျ့ - ya-medial followed by tone (acting as asat)
ZAWGYI_YA_ASAT_PATTERN = re.compile(r"[ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအ]ျ[့း]")

# Zawgyi ျ at word boundary (terminal position acting as asat)
# Only invalid when at END of word (not followed by anything)
ZAWGYI_YA_TERMINAL_PATTERN = re.compile(r"[ငညတနပမသ]ျ$")

# Invalid asat ordering: asat (်) before vowels
# Correct order: consonant + vowels + asat
# Invalid: consonant + asat + vowel
ASAT_BEFORE_VOWEL_PATTERN = re.compile(r"်[ိီုူေဲာါ]")

# Asat after dependent vowels (except Aa/Tall-A which form Aw)
# Invalid: i/ii/u/uu/e/ai + asat
# Example: မို်း (u + asat)
VOWEL_BEFORE_ASAT_PATTERN = re.compile(r"[ိီုူေဲ]်")

# Digit followed by tone marks (invalid for dictionary words)
DIGIT_TONE_PATTERN = re.compile(r"[၀-၉][့း]")

# Scrambled character sequences
# Asat appearing between vowels
SCRAMBLED_ASAT_PATTERN = re.compile(r"[ိီုူေဲာါ]်[ိီုူေဲာါ]")

# Incomplete O-vowel pattern: i-vowel + tone without u-vowel
# Valid: ဖို့ (for) = ိ + ု + ့
# Invalid: ဖိ့ = ိ + ့ (missing ု)
INCOMPLETE_O_VOWEL_PATTERN = re.compile(r"ိ့(?!ု)")

# Doubled vowels (invalid)
DOUBLED_VOWEL_PATTERN = re.compile(r"([ါာိီုူေဲဳဴ])\1+")

# Doubled medials (invalid)
DOUBLED_MEDIAL_PATTERN = re.compile(r"([ျြွှ])\1+")

# Invalid tone sequence (Dot/Visarga followed by Dot/Visarga)
# Example: ့း, း့, ့့, းး
INVALID_TONE_SEQUENCE = re.compile(r"[့း][့း]")

# Virama at end of word (incomplete stacking)
VIRAMA_AT_END_PATTERN = re.compile(r"္$")

# Invalid vowel sequences
# Includes: doubled i-vowels, doubled u-vowels, aa combinations, aa+u (ာု) which is never valid,
# and ေါ (e + short-aa U+102B) which is invalid (valid form is ော with tall-aa U+102C)
INVALID_VOWEL_SEQUENCE = re.compile(r"[ိီ][ိီ]|[ုူ][ုူ]|ါာ|ာါ|ာု|ါု|ေါ")

# ============================================================================
# WORD QUALITY PATTERNS (for filtering segmentation artifacts)
# ============================================================================

# Fragment patterns - single consonant + closing char (not valid standalone words)
# Examples: ဉ်, ည်, န့်, မး
# These are segmentation artifacts that shouldn't be in the dictionary
FRAGMENT_CONSONANT_ASAT_PATTERN = re.compile(r"^[ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဉ]်$")
# Consonant + tone only (e.g., မး, က့)
FRAGMENT_CONSONANT_TONE_PATTERN = re.compile(r"^[ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဉ][့း]$")
# Consonant + tone + asat (e.g., န့်) - also a fragment
FRAGMENT_CONSONANT_TONE_ASAT_PATTERN = re.compile(r"^[ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဉ][့း]်$")

# Double-ending pattern - valid word + fragment ending
# Examples: တွင်င်း (တွင် + င်း), သော်င်း (သော် + င်း)
# These are incorrectly merged segmentation artifacts
DOUBLE_ENDING_PATTERN = re.compile(r"[်းံ့]င်း$")

# Incomplete word patterns
# Ends with medial only (no vowel or asat following)
# Examples: ကြ (consonant + medial only, no following)
INCOMPLETE_MEDIAL_END_PATTERN = re.compile(r"[ျြွှ]$")

# Ends with stacking marker + consonant but no closing (incomplete stacking)
# Examples: န္တ (should have vowel/asat after stacked consonant)
# This is different from VIRAMA_AT_END - here virama is followed by consonant
INCOMPLETE_STACKING_PATTERN = re.compile(r"္[ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဉ]$")

# Ends with consonant + medial but no vowel/asat (incomplete syllable)
# Examples: ကျ (consonant + medial at end, no vowel/asat)
# Note: This is rarely a complete word
INCOMPLETE_CONSONANT_MEDIAL_PATTERN = re.compile(r"[ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဉ][ျြွှ]+$")

# Ends with medial + bare consonant (no asat/vowel/tone on final consonant)
# Examples: မြန (should be မြန်), အခွင (should be အခွင့်)
# The pattern matches: ...medial + consonant at end
INCOMPLETE_MEDIAL_CONSONANT_END_PATTERN = re.compile(r"[ျြွှ][ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဉ]$")

# Mixed letter + numeral pattern (safety net)
# Examples: လ၉, ခ၂၅, အ၁
# These should be split by split_word_numeral_tokens() but catch here as fallback
MIXED_LETTER_NUMERAL_PATTERN = re.compile(
    r"[ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဉ][၀-၉]"
    r"|[၀-၉][ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဉ]"
)

# Asat-initial fragment pattern (segmentation errors)
# Examples: က်ဆံ, က်ခံ, က်အ - these are fragments from incorrectly split words
# Words should NEVER start with consonant+asat (that's a closed syllable mid-word)
ASAT_INITIAL_PATTERN = re.compile(r"^[ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဉ]်")

# Compound word with truncated ending pattern
# Pattern: has asat mid-word AND ends with common syllable that usually needs asat
# Examples: ထုတ်ကုန (should be ထုတ်ကုန်), ရန်ကုန (should be ရန်ကုန်)
# ောင, ုန, ိန are syllables that almost always need asat in native Myanmar
COMPOUND_TRUNCATED_ENDING_PATTERN = re.compile(r"်.+(ောင|ုန|ိန)$")

# Missing ေ in ောင pattern (common typo)
# Pattern: medial + ာင + closing char(s) (should be medial + ေ + ာင + closing)
# Examples: ကြာင့် (should be ကြောင့်), ကျာင်း (should be ကျောင်း)
# The ောင sequence is extremely common in Myanmar; ာင (without ေ) is usually a typo
# Note: [့်း]+ allows for combinations like ့် (tone+asat) or ်း (asat+visarga)
MISSING_E_MEDIAL_AA_NG_PATTERN = re.compile(r"[ျြွှ]ာင[့်း]+$")

# Similar pattern but for consonant directly before ာင (no medial)
# Examples: အာင် (should be အောင်), ဆာင် (should be ဆောင်)
# Only match at end of word to avoid false positives
MISSING_E_CONSONANT_AA_NG_PATTERN = re.compile(r"^[ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဉ]ာင[့်း]*$")

# ============================================================================
# PHASE 1 QUALITY FILTERS
# ============================================================================

# Pure numeral pattern - sequences of only Myanmar digits
# Examples: ၆၉၀၀, ၁၆၄၂, ၅၀၀၀၀၀
# These are dates, quantities, or phone numbers that leaked through ingestion
PURE_NUMERAL_PATTERN = re.compile(r"^[၀-၉]+$")

# Doubled consonant pattern - two identical consonants only
# Examples: ဆဆ, အအ, တတ, ညည
# These are segmentation artifacts, not valid words
DOUBLED_CONSONANT_PATTERN = re.compile(r"^([ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဉ])\1$")

# Doubled initial consonant pattern - same consonant twice at word start
# without stacking (U+1039), followed by a vowel sign or medial.
# Examples: ကကာ, ကကု -> invalid segmentation artifact
# NOTE: Excludes asat (U+103A) and virama (U+1039) from the follow set
# because patterns like စစ်ပွဲ (military battle), နန်း (palace) are valid
# words where the same consonant + asat is a legitimate syllable structure.
DOUBLED_INITIAL_CONSONANT_PATTERN = re.compile(
    r"^([\u1000-\u1021])\1[\u102b-\u1032\u1036\u103b-\u103e]"
)

# Consecutive asat pattern - doubled ် (U+103A U+103A)
# Examples: ဖြစ်် (should be ဖြစ်), နှင့်် (should be နှင့်)
# Always an encoding error where the asat mark is duplicated.
CONSECUTIVE_ASAT_PATTERN = re.compile(r"\u103a\u103a")

# Broken virama stacking pattern - U+1039 followed by non-consonant
# Valid stacking is always consonant + U+1039 + consonant (U+1000-U+1021).
# When U+1039 is followed by a vowel or other non-consonant, it means the
# stacked consonant was dropped (e.g., ဘဏ္ာ should be ဘဏ္ဍာ).
BROKEN_VIRAMA_PATTERN = re.compile(r"\u1039[^\u1000-\u1021]")
