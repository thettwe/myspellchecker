"""
Myanmar Language Constants.

This module contains Myanmar (Burmese) language-specific constants including:
- Unicode ranges and character categories
- Punctuation and separators
- Named Entity Recognition data (honorifics)
- Common words and particles
- Character compatibility sets
- Numeral systems
- Negation patterns
- Syllable validation rules
"""

from __future__ import annotations

import logging
from pathlib import Path

# =============================================================================
# Myanmar Unicode Ranges
# =============================================================================

# Main Myanmar block
MYANMAR_RANGE = (0x1000, 0x109F)

# Extended-A: Used by Shan, Khamti Shan, Aiton, Phake, and Pao Karen
# Reference: https://www.unicode.org/charts/PDF/UAA60.pdf
MYANMAR_EXTENDED_A_RANGE = (0xAA60, 0xAA7F)

# Extended-B: Used by Shan, Pa'O, etc.
# Reference: https://www.unicode.org/charts/PDF/UA9E0.pdf
MYANMAR_EXTENDED_B_RANGE = (0xA9E0, 0xA9FF)

# Regex pattern for all Myanmar ranges
MYANMAR_RANGE_REGEX_STR = r"[\u1000-\u109F\uA9E0-\uA9FF\uAA60-\uAA7F]"

# =============================================================================
# Character Sets
# =============================================================================

# Non-Standard Characters (Mon/Shan specific chars in Core Block)
# These are technically in the U+1000-U+104F block but not used in Standard Burmese
NON_STANDARD_CHARS = {
    "\u1022",  # SHAN LETTER A (ဢ)
    "\u1028",  # MYANMAR LETTER MON E (ဨ)
    "\u1033",  # MYANMAR VOWEL SIGN MON II
    "\u1034",  # MYANMAR VOWEL SIGN MON O
    "\u1035",  # MYANMAR VOWEL SIGN E ABOVE
}

# Extended-A characters
MYANMAR_EXTENDED_A_CHARS = set(chr(c) for c in range(0xAA60, 0xAA80))

# Extended-B characters
MYANMAR_EXTENDED_B_CHARS = set(chr(c) for c in range(0xA9E0, 0xAA00))

# All Myanmar characters (combined)
ALL_MYANMAR_CHARS = (
    set(chr(c) for c in range(0x1000, 0x10A0))  # Main Myanmar block
    | MYANMAR_EXTENDED_A_CHARS  # Extended-A
    | MYANMAR_EXTENDED_B_CHARS  # Extended-B
)

# Burmese-only character set - U+1000-U+104F only
# This excludes:
# - Extended Core Block (U+1050-U+109F) - Shan, Mon, Karen additions to main block
# - Extended-A (U+AA60-U+AA7F) - Shan, Khamti Shan, Aiton, Phake, Pao Karen
# - Extended-B (U+A9E0-U+A9FF) - Shan, Pa'O
MYANMAR_CORE_CHARS: set[str] = set(chr(c) for c in range(0x1000, 0x1050)) - NON_STANDARD_CHARS

# Extended Core Block (U+1050-U+109F) - in main Myanmar block but for Shan/Mon/Karen
# These characters are technically in the "main" Myanmar Unicode block but are used
# by other languages (Shan, Mon, Karen, etc.) and are out of scope for Burmese.
MYANMAR_EXTENDED_CORE_BLOCK: set[str] = set(chr(c) for c in range(0x1050, 0x10A0))

# Extended blocks combined (Shan, Mon, Karen - out of scope for Burmese)
# Includes Extended Core Block + Extended-A + Extended-B
EXTENDED_MYANMAR_CHARS: set[str] = (
    MYANMAR_EXTENDED_CORE_BLOCK | MYANMAR_EXTENDED_A_CHARS | MYANMAR_EXTENDED_B_CHARS
)


def get_myanmar_char_set(allow_extended: bool = False) -> set[str]:
    """
    Get Myanmar character set based on scope.

    Args:
        allow_extended: If True, include all Myanmar characters including:
                       - Extended Core Block (U+1050-U+109F)
                       - Extended-A (U+AA60-U+AA7F)
                       - Extended-B (U+A9E0-U+A9FF)
                       - Non-standard core chars (U+1022, U+1028, U+1033-U+1035)
                       If False (default), return only core Burmese characters.

    Returns:
        Set of Myanmar characters based on the requested scope.
    """
    if allow_extended:
        # Include ALL Myanmar chars for extended support (non-Burmese Myanmar scripts)
        return ALL_MYANMAR_CHARS
    return MYANMAR_CORE_CHARS


def has_extended_myanmar_chars(text: str) -> bool:
    """
    Check if text contains Extended Myanmar characters.

    These characters are used by Shan, Mon, Karen languages and are
    out of scope for Burmese-only validation.

    Checks for:
    - Extended Core Block (U+1050-U+109F): Shan/Mon/Karen additions in main block
    - Extended-A (U+AA60-U+AA7F): Shan, Khamti Shan, Aiton, Phake, Pao Karen
    - Extended-B (U+A9E0-U+A9FF): Shan, Pa'O

    Args:
        text: Text to check.

    Returns:
        True if text contains any Extended Myanmar characters.
    """
    for c in text:
        code = ord(c)
        # Extended Core Block (U+1050-U+109F)
        if 0x1050 <= code <= 0x109F:
            return True
        # Extended-A (U+AA60-U+AA7F)
        if 0xAA60 <= code <= 0xAA7F:
            return True
        # Extended-B (U+A9E0-U+A9FF)
        if 0xA9E0 <= code <= 0xA9FF:
            return True
    return False


def is_myanmar_text(text: str, allow_extended: bool = False) -> bool:
    """
    Check if text contains any Myanmar characters.

    This is a shared helper for consistent Myanmar text detection across
    validators, semantic checkers, and suggestion strategies.

    Args:
        text: Text to check.
        allow_extended: If True, Extended-A/B blocks are also considered Myanmar.
                       If False (default), only core Burmese characters count.

    Returns:
        True if text contains at least one Myanmar character.
    """
    char_set = get_myanmar_char_set(allow_extended)
    return any(c in char_set for c in text)


# =============================================================================
# Character Categories
# =============================================================================

# Consonants (U+1000 - U+1021)
# U+1021 (အ) is the vowel carrier but functions like a consonant syllable-initially:
# it can take medials (e.g., အွန်လိုင်း "online"), vowels, and tone marks.
# Both COMPATIBLE_WA and _CONSONANTS include it.
CONSONANTS = set(chr(i) for i in range(0x1000, 0x1022))

# Independent vowels and the vowel carrier (U+1021 - U+102A)
# This set includes:
# - U+1021 (အ): Vowel carrier - functions like a consonant but carries inherent vowel
# - U+1022 (ဢ): Shan Letter A - NON-STANDARD, filtered by NON_STANDARD_CHARS
# - U+1023 (ဣ): Independent I
# - U+1024 (ဤ): Independent II
# - U+1025 (ဥ): Independent U
# - U+1026 (ဦ): Independent UU
# - U+1027 (ဧ): Independent E
# - U+1028 (ဨ): Mon E - NON-STANDARD, filtered by NON_STANDARD_CHARS
# - U+1029 (ဩ): Independent O
# - U+102A (ဪ): Independent AU
# Note: U+1021 is handled specially in syllable_rules.py (treated as consonant carrier)
INDEPENDENT_VOWELS = set(chr(i) for i in range(0x1021, 0x102B))

# True independent vowels (standard Burmese only, excluding carrier and non-standard)
# Use this for strict validation where only true independent vowels are allowed
INDEPENDENT_VOWELS_STRICT = {
    "\u1023",  # ဣ - I
    "\u1024",  # ဤ - II
    "\u1025",  # ဥ - U
    "\u1026",  # ဦ - UU
    "\u1027",  # ဧ - E
    "\u1029",  # ဩ - O
    "\u102a",  # ဪ - AU
}

# Vowel carrier (အ) - special character that carries inherent vowel
# Behaves like a consonant in syllable structure but is technically in the vowel range
VOWEL_CARRIER = "\u1021"

# Medials
# Myanmar has four medial consonant forms that modify the base consonant:
# - Ya-pin (ျ U+103B): Subscript ya, creates /j/ glide (palatal)
# - Ya-yit (ြ U+103C): Left-side ra/ya, creates /r/ glide (rhotic)
# - Wa-hswe (ွ U+103D): Subscript wa, creates /w/ glide (labial-velar)
# - Ha-htoe (ှ U+103E): Subscript ha, creates aspiration
MEDIALS = {"\u103b", "\u103c", "\u103d", "\u103e"}
MEDIAL_YA = "\u103b"  # Ya-pin (ယပင့်, subscript ya) — palatal glide /j/
MEDIAL_RA = "\u103c"  # Ya-yit (ရရစ်, left-side ra) — rhotic glide /r/
MEDIAL_WA = "\u103d"  # Wa-hswe
MEDIAL_HA = "\u103e"  # Ha-htoe

# Aliases for clearer phonetic naming
MEDIAL_YA_PIN = MEDIAL_YA  # ျ - /j/ glide (palatal approximant)
MEDIAL_YA_YIT = MEDIAL_RA  # ြ - /r/ glide (rhotic approximant)

# Vowel signs (U+102B - U+1032)
VOWEL_SIGNS = set(chr(i) for i in range(0x102B, 0x1033))  # Include U+1032 (ဲ ai vowel)

# Tone marks
TONE_MARKS = {
    "\u1036",  # Anusvara (Thay-thay-tin)
    "\u1037",  # Dot Below (Auk-myit)
    "\u1038",  # Visarga (Wit-sa-pauk)
}

ANUSVARA = "\u1036"  # ံ - Nasalization marker

# Dependent various signs (U+1032 - U+103E)
DEPENDENT_VARIOUS_SIGNS = set(chr(i) for i in range(0x1032, 0x103F))

# =============================================================================
# Specific Characters
# =============================================================================

VIRAMA = "\u1039"  # Pat Sint / Stacking
ASAT = "\u103a"  # Killer
NGA = "\u1004"  # Nga
DOT_BELOW = "\u1037"  # Auk-myit
# Visarga (U+1038, း) has dual roles in Myanmar:
# 1. Tone marker (wit-sa-pauk/ဝစ်ဆပေါက်): Indicates creaky tone in syllables
#    like "ကား" (car), "လား" (question particle), "ပေးပါ" + "း" = emphasis
# 2. Sentence-final particle: Standalone "း" can mark emphasis or boundaries
#    in informal writing. This usage is at word/sentence level, not syllable.
# For syllable validation, treat as tone marker requiring a valid syllable base.
VISARGA = "\u1038"  # Wit-sa-pauk (Creaky tone marker)
GREAT_SA = "\u103f"  # Great Sa

# Add Great Sa to consonants
CONSONANTS.add(GREAT_SA)

# Specific vowel combinations
VOWEL_E = "\u1031"  # E vowel (pre-consonant)
VOWEL_AI = "\u1032"  # AI vowel
VOWEL_U = "\u102f"  # U vowel (lower slot)
VOWEL_UU = "\u1030"  # UU vowel (lower slot)

# English token placeholder
ENG_TOKEN = "<ENG>"

# =============================================================================
# Punctuation and Separators
# =============================================================================

# Myanmar punctuation (U+104A - U+104F)
MYANMAR_PUNCTUATION = set(chr(i) for i in range(0x104A, 0x1050))

# Common punctuation
COMMON_PUNCTUATION = set(
    ".,!?;:()[]{}'\u2013\u2014"  # Punctuation characters (en-dash, em-dash)
    " \t\n\r"  # Whitespace characters
)

# Myanmar-specific separators
SENTENCE_SEPARATOR = "။"  # Myanmar full stop
PHRASE_SEPARATOR = "၊"  # Myanmar comma

# =============================================================================
# Named Entity Recognition Data
# =============================================================================

HONORIFICS = {
    # Standard Titles
    "ဦး",  # U (Mr.)
    "ဒေါ်",  # Daw (Ms./Mrs.)
    "ကို",  # Ko (Brother/Mr.)
    "မောင်",  # Maung (Younger brother/Mr.)
    "မ",  # Ma (Sister/Ms.)
    # Family/Informal Titles
    "ကိုကို",  # Ko Ko (Older Brother)
    "မမ",  # Ma Ma (Older Sister)
    "ညီ",  # Nyi (Younger Brother)
    "ညီမ",  # Nyi Ma (Younger Sister)
    "ဦးဦး",  # U U (Uncle)
    "ဒေါ်ဒေါ်",  # Daw Daw (Aunt)
    "အန်တီ",  # Aunty
    "အစ်ကို",  # Ako (Older Brother)
    "အစ်မ",  # Ama (Older Sister)
    "ဖေဖေ",  # Phay Phay (Father)
    "မေမေ",  # May May (Mother)
    "ဖိုးဖိုး",  # Phoe Phoe (Grandfather)
    "ဖွားဖွား",  # Phwar Phwar (Grandmother)
    # Professional/Academic
    "ဆရာ",  # Saya (Teacher/Master - Male)
    "ဆရာမ",  # Sayama (Teacher/Master - Female)
    "ဒေါက်တာ",  # Dr. (Doctor)
    "ပါမောက္ခ",  # Professor
    "ဆရာဝန်",  # Doctor (Medical)
    "သူနာပြု",  # Nurse
    "အင်ဂျင်နီယာ",  # Engineer
    "ရှေ့နေ",  # Lawyer
    # Religious
    "ရှင်",  # Shin (Novice Monk)
    "အရှင်",  # Ashin (Lord/Monk)
    "ဦးဇင်း",  # U Zin (Monk)
    "ဘုန်းကြီး",  # Phone Gyi (Monk)
    "ဆရာတော်",  # Sayadaw (Abbot/Senior Monk)
    "ရဟန်း",  # Yahan (Monk)
    "ရှင်လူ",  # Shin Lu
    "သီလရှင်",  # Thilashin (Nun)
    # Official/Military/Historical
    "သခင်",  # Thakin (Master/Lord - Historical)
    "ဗိုလ်",  # Bo (Officer)
    "ဗိုလ်ချုပ်",  # Bogyoke (General)
    "ဗိုလ်မှူး",  # Bo Hmu (Major)
    "တပ်ကြပ်",  # Tat Kyat (Sergeant)
    "တပ်မှူး",  # Tat Hmu (Commander)
    "ရဲအုပ်",  # Ye Oke (Police Officer)
    "ရဲမှူး",  # Ye Hmu (Police Chief)
    "သမ္မတ",  # President
    "ဝန်ကြီး",  # Minister
}

# Core respectful title prefixes shared across detector modules.
# This is the intersection of titles used for BOTH politeness-context
# detection (sentence_detectors._HONORIFIC_TERMS) AND name-prefix
# detection (particle_detection_mixin._HONORIFIC_PREFIXES).
# Detectors extend this with domain-specific entries (e.g., compound titles
# like ဆရာကြီး, or transliterated forms like ပရော်ဖက်ဆာ).
CORE_RESPECTFUL_TITLES: frozenset[str] = frozenset(
    {
        "ဒေါ်",  # Daw (Mrs/Ms)
        "ဦး",  # U (Mr/Uncle)
        "ဆရာ",  # Teacher/sir
        "ဆရာမ",  # Teacher (female)
    }
)

# =============================================================================
# Invalid and Skipped Words
# =============================================================================

# Myanmar consonants (for generating invalid patterns)
_CONSONANTS = "ကခဂဃငစဆဇဈဉညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအ"

# Consonant + Asat patterns (invalid as standalone words)
# These are syllable-closing patterns that should not exist independently.
# They are typically segmentation artifacts from noisy training data in the
# word segmentation dictionary (segmentation.mmap).
_CONSONANT_ASAT_FRAGMENTS = {f"{c}်" for c in _CONSONANTS}

# Consonant + Tone + Asat patterns (also invalid)
# Examples: င့်, ည့်, က့်, မ့်
_CONSONANT_TONE_ASAT_FRAGMENTS = {f"{c}့်" for c in _CONSONANTS} | {f"{c}း်" for c in _CONSONANTS}

# Invalid words (segmentation artifacts to filter out)
INVALID_WORDS = (
    {
        "င်း",  # Common artifact from incorrect segmentation of Kinzi/medials
        "်",  # Floating Asat
        "့",  # Floating Dot Below
        "း",  # Floating Visarga
    }
    | _CONSONANT_ASAT_FRAGMENTS
    | _CONSONANT_TONE_ASAT_FRAGMENTS
)

# Words skipped by Context Validator (common particles that appear everywhere)
# These high-frequency particles should not trigger n-gram context errors
# as they appear in virtually all contexts and have low discriminative power
SKIPPED_CONTEXT_WORDS = {
    # Interjections and emphasis particles (original)
    "ကွာ",  # emphasis/exclamation
    "ဗျာ",  # emphasis
    "နော်",  # tag question/emphasis
    "ဟေ့",  # hey/attention
    "ကွ",  # emphasis (colloquial)
    "လေ",  # emphasis/evidential
    "ပါ",  # politeness marker (very high frequency)
    "ပဲ",  # only/just/emphasis
    "ပေါ့",  # of course/naturally
    # Subject/object markers (highest frequency)
    "က",  # subject marker (most common particle)
    "ကို",  # object/dative marker
    "သည်",  # topic/statement marker (formal)
    "တယ်",  # statement ending (colloquial)
    # Locative particles
    "မှာ",  # locative ("at/in")
    "မှ",  # locative/ablative ("from")
    "တွင်",  # locative (formal)
    # Comitative and conjunctive
    "နဲ့",  # comitative ("with", colloquial)
    "နှင့်",  # comitative ("with", formal)
    "နှင်",  # comitative variant
    # Genitive and possessive
    "ရဲ့",  # genitive (colloquial)
    "၏",  # genitive (formal)
    # Other common particles
    "များ",  # plural marker
    "လည်း",  # also/too
    "တော့",  # emphasis/contrast/then
    "ပြီး",  # after/and then
    "ဖို့",  # purpose marker
    "အတွက်",  # for/because of
    # Question and ending particles
    "လား",  # yes/no question
    "လဲ",  # wh-question
    "လို့",  # causal particle ("because")
    # Common verb complements (segmenter splits these from compounds)
    # These are too short for meaningful n-gram validation and produce FPs
    # when separated from their host verb (e.g., ပျက်ကျ → ပျက် + ကျ)
    "ကျ",  # fall/down complement (ပျက်ကျ, ကျဆင်း)
    "ပြ",  # show complement (ဆိုပြ, ပြောပြ)
    "ချ",  # place/put-down complement (တင်ချ, ချရေး)
    # Quotative and evidential markers
    "ဟု",  # quotative ("that", reported speech marker)
    # Loanword-derived particles
    "ဖေး",  # café (from ကဖေး, loanword — low bigram but valid)
    # Common adjective/adverb particles
    "လှ",  # beautiful/very (adjective complement)
}

# Particle confusable pairs for MLM-based detection.
# These are grammatically distinct particles that share similar
# syntactic slots but are NOT phonetic/visual confusables.
# Detection requires semantic context (MLM), not phonetic rules.
PARTICLE_CONFUSABLES: dict[str, list[str]] = {
    "က": ["ကို"],  # subject marker ↔ object marker
    "ကို": ["က"],  # object marker ↔ subject marker
    "မှာ": ["မှ"],  # locative ("at") ↔ ablative ("from")
    "မှ": ["မှာ", "မ"],  # ablative ("from") ↔ locative ("at") / bare consonant
    "မ": ["မှ"],  # negation/bare ↔ ablative ("from") — ha-htoe confusable
    "လို": ["လို့"],  # complement ("want/like") ↔ causal ("because")
    "လို့": ["လို"],  # causal ("because") ↔ complement ("want/like")
    "လဲ": ["လား"],  # wh-question ↔ yes/no question
    "လား": ["လဲ"],  # yes/no question ↔ wh-question
    "ဖူး": ["ဘူး"],  # experiential ("ever") ↔ negation ("not")
    "ဘူး": ["ဖူး"],  # negation ("not") ↔ experiential ("ever")
    "လ": ["လှ"],  # bare consonant ↔ "beautiful" — ha-htoe confusable
    "လှ": ["လ"],  # "beautiful" ↔ bare consonant
    "မဲ": ["မယ်"],  # vote/dark ↔ future particle
    "မယ်": ["မဲ"],  # future particle ↔ vote/dark
}

# Known DB-contaminated variants that should never be suggested.
# These are invalid forms that exist in the dictionary due to
# corpus contamination (Zawgyi artifacts, encoding errors).
VARIANT_BLOCKLIST: frozenset[str] = frozenset(
    {
        "ခှဲဝေ",  # Zawgyi artifact of ခွဲဝေ
        "စာမေးပှဲ",  # Zawgyi artifact of စာမေးပွဲ
        "အီးမေလ်",  # wrong form of အီးမေးလ်
    }
)

# Register variant pairs that should NOT be treated as confusables.
# These are semantically equivalent words from different registers
# (formal vs informal) — suggesting one over the other is unhelpful.
# Stored as frozenset of (word, variant) pairs in both directions.
# Confusable exempt pairs: word-variant pairs that should NOT be
# flagged as confusable errors. These are valid word pairs where the
# variant generation produces a plausible but incorrect suggestion.
# Stored as frozenset of (word, variant) tuples in both directions.
_DEFAULT_CONFUSABLE_EXEMPT_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        # ဖေး (café, loanword from ကဖေး) ↔ ဘေး (beside/side)
        # Aspiration swap ဖ↔ဘ generates this pair, but ဖေး is a valid
        # loanword component (ကဖေး = café) and not a misspelling of ဘေး.
        ("ဖေး", "ဘေး"),
        ("ဘေး", "ဖေး"),
        # ထား (place/put) ↔ တား (block/prevent)
        # Aspiration swap ထ↔တ. Both are common verbs in similar syntactic
        # slots — MLM frequency bias dominates over contextual signal.
        ("ထား", "တား"),
        ("တား", "ထား"),
    }
)

# Confusable exempt suffix pairs: suffix pairs where the word ending
# should not trigger confusable detection. The variant generator
# produces these via tone mark manipulation, but the distinction is
# purely syntactic and the MLM cannot reliably distinguish them.
# Each entry is (suffix_a, suffix_b) — checked in both directions.
_DEFAULT_CONFUSABLE_EXEMPT_SUFFIX_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        # သည် (sentence-final declarative) ↔ သည့် (attributive/relative clause)
        # The dot-below (့) distinguishes syntactic function, not spelling.
        # Use explicit codepoints to ensure normalized order (asat U+103A before dot-below U+1037).
        ("\u101e\u100a\u103a", "\u101e\u100a\u103a\u1037"),
        ("\u101e\u100a\u103a\u1037", "\u101e\u100a\u103a"),
    }
)

_logger = logging.getLogger(__name__)

_CONFUSABLE_PAIRS_YAML_PATH = (
    Path(__file__).resolve().parent.parent.parent / "rules" / "confusable_pairs.yaml"
)


def _load_confusable_exemptions() -> tuple[frozenset[tuple[str, str]], frozenset[tuple[str, str]]]:
    """Load confusable exempt pairs from confusable_pairs.yaml with fallback.

    Reads the ``exempt_pairs`` and ``exempt_suffix_pairs`` sections from
    confusable_pairs.yaml and returns normalized frozensets. Falls back
    to hardcoded defaults if the YAML file is missing, invalid, or the
    sections are absent.

    Returns:
        Tuple of (exempt_pairs, exempt_suffix_pairs) as frozensets of
        (str, str) tuples.
    """
    if not _CONFUSABLE_PAIRS_YAML_PATH.exists():
        _logger.debug(
            "Confusable pairs YAML not found at %s, using default exemptions",
            _CONFUSABLE_PAIRS_YAML_PATH,
        )
        return _DEFAULT_CONFUSABLE_EXEMPT_PAIRS, _DEFAULT_CONFUSABLE_EXEMPT_SUFFIX_PAIRS

    try:
        import yaml  # type: ignore[import-untyped]

        with open(_CONFUSABLE_PAIRS_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            _logger.warning("Confusable pairs YAML empty or invalid, using default exemptions")
            return (
                _DEFAULT_CONFUSABLE_EXEMPT_PAIRS,
                _DEFAULT_CONFUSABLE_EXEMPT_SUFFIX_PAIRS,
            )

        # Parse exempt_pairs
        raw_exempt = data.get("exempt_pairs", [])
        if isinstance(raw_exempt, list) and raw_exempt:
            pairs: set[tuple[str, str]] = set()
            for entry in raw_exempt:
                if isinstance(entry, dict):
                    word = entry.get("word", "")
                    variant = entry.get("variant", "")
                    if word and variant:
                        pairs.add((word, variant))
                        pairs.add((variant, word))  # Bidirectional
            exempt_pairs = frozenset(pairs) if pairs else _DEFAULT_CONFUSABLE_EXEMPT_PAIRS
        else:
            exempt_pairs = _DEFAULT_CONFUSABLE_EXEMPT_PAIRS

        # Parse exempt_suffix_pairs
        raw_suffix = data.get("exempt_suffix_pairs", [])
        if isinstance(raw_suffix, list) and raw_suffix:
            suffix_pairs: set[tuple[str, str]] = set()
            for entry in raw_suffix:
                if isinstance(entry, dict):
                    suffix_a = entry.get("suffix_a", "")
                    suffix_b = entry.get("suffix_b", "")
                    if suffix_a and suffix_b:
                        # Normalize to ensure codepoint order matches normalized text.
                        # Use lazy import to avoid circular dependency at early load.
                        try:
                            from myspellchecker.text.normalize import normalize as _norm

                            na, nb = _norm(suffix_a), _norm(suffix_b)
                        except ImportError:
                            na, nb = suffix_a, suffix_b
                        suffix_pairs.add((na, nb))
                        suffix_pairs.add((nb, na))  # Bidirectional
            exempt_suffix = (
                frozenset(suffix_pairs) if suffix_pairs else _DEFAULT_CONFUSABLE_EXEMPT_SUFFIX_PAIRS
            )
        else:
            exempt_suffix = _DEFAULT_CONFUSABLE_EXEMPT_SUFFIX_PAIRS

        _logger.debug(
            "Loaded confusable exemptions from YAML: %d exempt pairs, %d exempt suffix pairs",
            len(exempt_pairs),
            len(exempt_suffix),
        )
        return exempt_pairs, exempt_suffix

    except Exception:
        _logger.warning(
            "Failed to load confusable exemptions from YAML, using defaults",
            exc_info=True,
        )
        return _DEFAULT_CONFUSABLE_EXEMPT_PAIRS, _DEFAULT_CONFUSABLE_EXEMPT_SUFFIX_PAIRS


# Load at module level (once, at import time)
CONFUSABLE_EXEMPT_PAIRS: frozenset[tuple[str, str]]
CONFUSABLE_EXEMPT_SUFFIX_PAIRS: frozenset[tuple[str, str]]
CONFUSABLE_EXEMPT_PAIRS, CONFUSABLE_EXEMPT_SUFFIX_PAIRS = _load_confusable_exemptions()

REGISTER_VARIANT_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("ငါ", "ကျွန်တော်"),
        ("ကျွန်တော်", "ငါ"),
        ("ငါ", "ကျွန်မ"),
        ("ကျွန်မ", "ငါ"),
        ("မင်း", "ခင်ဗျား"),
        ("ခင်ဗျား", "မင်း"),
        ("သူ", "သူသည်"),
        ("သူသည်", "သူ"),
        ("ငါတို့", "ကျွန်တော်တို့"),
        ("ကျွန်တော်တို့", "ငါတို့"),
        ("ငါတို့", "ကျွန်မတို့"),
        ("ကျွန်မတို့", "ငါတို့"),
    }
)

# =============================================================================
# Medial Compatibility Sets
# =============================================================================
#
# Myanmar medials (ျ Ya-pin, ြ Ya-yit, ွ Wa-hswe, ှ Ha-htoe) can only attach
# to specific consonants based on phonotactic constraints. These sets define
# which consonants can validly combine with each medial.
#
# Linguistic basis:
# - Ya (ျ) and Ra (ြ) typically combine with stops and some fricatives
# - Wa (ွ) has broader compatibility, including approximants
# - Ha (ှ) combines with sonorants to create aspirated sonorants
#
# Reference: Myanmar Language Commission orthography guidelines

#: Consonants that can take Medial Ya-pin (ျ U+103B).
#: Ya-pin creates palatal glide /Cj/ clusters (palatalization).
#: Primarily Ka-group (velar) and Pa-group (labial) consonants.
#: Retroflex consonants (Tta-series) added for Pali/Sanskrit loanwords.
#: Includes Tha (သ) which takes Ya-pin but NOT Ya-yit.
#: La (လ) added - common in native Burmese (e.g., လျာ "tongue").
COMPATIBLE_YA = {
    "\u1000",
    "\u1001",
    "\u1002",
    "\u1003",
    "\u1004",  # Ka group (ကခဂဃင)
    "\u1005",
    "\u1006",
    "\u1007",
    "\u1008",  # Ca group (စဆဇဈ) - e.g., စျေး (market/price)
    "\u100b",
    "\u100c",
    "\u100d",
    "\u100e",
    "\u100f",  # Tta group retroflex (ဋဌဍဎဏ) - Pali/Sanskrit
    "\u1015",
    "\u1016",
    "\u1017",
    "\u1018",
    "\u1019",  # Pa group (ပဖဗဘမ)
    "\u101c",  # La (လ) - common with Ya-pin (e.g., လျာ, လျင်မြန်)
    "\u101e",  # Tha (သ) - Ya-pin only, NOT Ya-yit
}

#: Alias for clear phonetic naming
COMPATIBLE_YA_PIN = COMPATIBLE_YA

#: Consonants that can take Medial Ya-yit (ြ U+103C).
#: Ya-yit creates rhotic glide /Cr/ clusters (rhotacization).
#: Similar to Ya-pin but EXCLUDES Tha (သ).
#: Retroflex consonants (Tta-series) added for Pali/Sanskrit loanwords.
#: Key distinction: Tha + Ya-pin (သျှ) is valid, Tha + Ya-yit is NOT.
#: La (လ) added - valid though less common than with Ya-pin.
COMPATIBLE_RA = {
    "\u1000",
    "\u1001",
    "\u1002",
    "\u1003",
    "\u1004",  # Ka group (ကခဂဃင)
    "\u100b",
    "\u100c",
    "\u100d",
    "\u100e",
    "\u100f",  # Tta group retroflex (ဋဌဍဎဏ) - Pali/Sanskrit
    "\u1015",
    "\u1016",
    "\u1017",
    "\u1018",
    "\u1019",  # Pa group (ပဖဗဘမ)
    "\u101c",  # La (လ) - valid with Ya-yit, though less common
    # NOTE: Tha (သ U+101E) is intentionally EXCLUDED - it cannot take Ya-yit
}

#: Alias for clear phonetic naming
COMPATIBLE_YA_YIT = COMPATIBLE_RA

#: Consonants that can take Medial Wa (ွ Wa-hswe).
#: Wa has the broadest compatibility, creating /w/ labialized clusters.
#: Includes all consonant groups plus approximants (Ya, Ra, La, Wa, Ha).
#: Retroflex consonants (Tta-series) included for Pali/Sanskrit loanwords.
#: Updated: Added Lla (U+1020) and archaic Nya (U+1009) for Pali/Sanskrit.
COMPATIBLE_WA = {
    "\u1000",
    "\u1001",
    "\u1002",
    "\u1003",
    "\u1004",  # Ka group (incl Nga)
    "\u1005",
    "\u1006",
    "\u1007",
    "\u1008",
    "\u1009",  # Nya archaic (ဉ) - Pali/Sanskrit
    "\u100a",  # Ca group (incl Nya)
    "\u100b",
    "\u100c",
    "\u100d",
    "\u100e",
    "\u100f",  # Tta group (retroflex, Pali/Sanskrit)
    "\u1010",
    "\u1011",
    "\u1012",
    "\u1013",
    "\u1014",  # Ta group (incl Na)
    "\u1015",
    "\u1016",
    "\u1017",
    "\u1018",
    "\u1019",  # Pa group (incl Ma)
    "\u101a",  # Ya
    "\u101b",  # Ra
    "\u101c",  # La
    "\u101d",  # Wa
    "\u101f",  # Ha
    "\u101e",  # Tha (e.g. Thwa - Tooth/Go)
    "\u1020",  # Lla (ဠ) - retroflex lateral, Pali/Sanskrit
    "\u1021",  # Vowel carrier (အ) - loanwords e.g., အွန်လိုင်း (online)
}

#: Consonants that can take Medial Ha (ှ Ha-htoe/Ha-sub).
#: Ha-htoe creates aspirated sonorant clusters and primarily combines
#: with sonorant consonants: nasals (Nga, Nya, Na, Ma), liquids (La, Lla),
#: and glides (Ya, Ra, Wa).
#:
#: Linguistic note: Medial Ha aspirates the preceding consonant, creating
#: voiceless or breathy-voiced sonorants. This phonological process is
#: only meaningful for sonorant consonants, not stops or fricatives.
#:
#: Valid examples: နှ (aspirated /n/), မှ (aspirated /m/), လှ (aspirated /l/), ဝှ (aspirated /w/)
#: Invalid: ကှ, စှ, တှ (stops cannot be aspirated via medial Ha)
COMPATIBLE_HA = {
    "\u1004",  # Nga (င) - velar nasal, creates /ŋ̊/ or breathy /ŋʱ/
    "\u100a",  # Nya (ည) - palatal nasal, creates aspirated /ɲ̊/ (modern Burmese)
    "\u1009",  # Nya archaic (ဉ) - Pali/archaic form, e.g., ဉာဏ် (wisdom)
    "\u1014",  # Na (န) - alveolar nasal, creates /n̥/ (voiceless n)
    "\u1019",  # Ma (မ) - bilabial nasal, creates /m̥/ (voiceless m)
    "\u101c",  # La (လ) - lateral, creates /l̥/ (voiceless l)
    "\u1020",  # Lla (ဠ) - retroflex lateral, creates /ɭ̊/ (voiceless retroflex l)
    "\u101a",  # Ya (ယ) - palatal glide, creates /j̊/ (voiceless y)
    "\u101b",  # Ra (ရ) - alveolar tap, creates /ɾ̥/ (voiceless r)
    "\u101d",  # Wa (ဝ) - labial-velar glide, creates /w̥/ (voiceless w)
}

# =============================================================================
# Phonetic Character Sets
# =============================================================================
#
# These sets classify Myanmar consonants by their phonetic properties,
# which affects syllable structure rules and valid consonant sequences.
#
# Phonetic classification based on manner of articulation:
# - Sonorants: Consonants produced with continuous airflow (nasals, liquids, glides)
# - Stops/Obstruents: Consonants produced with complete closure of vocal tract

#: Sonorant consonants - nasals (Nga, Nya, Na, Ma), liquids (La), and glides (Ya, Ra, Wa).
#: Sonorants can occur as syllable codas with special tone behavior.
#: IPA: /ŋ/, /ɲ/, /n/, /m/, /l/, /j/, /ɹ/, /w/
SONORANTS = {
    "\u1004",  # Nga (င) - velar nasal /ŋ/
    "\u100a",  # Nya (ည) - palatal nasal /ɲ/ (modern Burmese, e.g., ညီ "brother")
    "\u1009",  # Nya archaic (ဉ) - Pali/archaic palatal nasal (e.g., ဉာဏ် "wisdom")
    "\u1014",  # Na (န) - alveolar nasal /n/
    "\u1019",  # Ma (မ) - bilabial nasal /m/
    "\u101c",  # La (လ) - lateral approximant /l/
    "\u1020",  # Lla (ဠ) - retroflex lateral
    "\u101a",  # Ya (ယ) - palatal glide /j/
    "\u101b",  # Ra (ရ) - alveolar tap /ɾ/
    "\u101d",  # Wa (ဝ) - labial-velar glide /w/
}

#: Stop/obstruent consonants that can appear in syllable-final position with asat.
#: These create "checked" syllables with glottal stop release.
#: Includes velar (Ka), palatal (Ca), retroflex (Tta), dental (Ta), and labial (Pa) stops.
STOP_FINALS = {
    "\u1000",
    "\u1001",
    "\u1002",
    "\u1003",  # Ka group
    "\u1005",
    "\u1006",
    "\u1007",
    "\u1008",  # Ca group
    "\u100b",
    "\u100c",
    "\u100d",
    "\u100e",
    "\u100f",  # Tta group
    "\u1010",
    "\u1011",
    "\u1012",
    "\u1013",  # Ta group
    "\u1015",
    "\u1016",
    "\u1017",
    "\u1018",  # Pa group
    "\u101e",
    "\u101f",  # Tha, Ha
}

# =============================================================================
# Zero-width and Control Characters
# =============================================================================

ZERO_WIDTH_CHARS = {
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\ufeff",  # Zero-width no-break space (BOM)
}

# =============================================================================
# Normalization Order Weights
# =============================================================================

# UTN #11 canonical order for medials: Ya < Ra < Wa < Ha
# Reference: https://unicode.org/notes/tn11/UTN11_4.pdf
ORDER_WEIGHTS = {
    "\u103b": 10,  # Medial Ya (ျ) - slot 3
    "\u103c": 11,  # Medial Ra (ြ) - slot 4
    "\u103d": 12,  # Medial Wa (ွ) - slot 5
    "\u103e": 13,  # Medial Ha (ှ) - slot 6
    "\u1031": 20,  # Vowel E
    "\u102d": 21,  # Vowel I (Upper)
    "\u102e": 21,  # Vowel II (Upper)
    "\u1032": 21,  # Vowel AI (Upper)
    "\u102f": 22,  # Vowel U (Lower)
    "\u1030": 22,  # Vowel UU (Lower)
    "\u102b": 21.4,  # Vowel A (Tall)
    "\u102c": 21.4,  # Vowel AA
    "\u1036": 30,  # Anusvara
    "\u103a": 21.5,  # Asat
    "\u1037": 32,  # Dot Below
    "\u1038": 33,  # Visarga
    "\u1039": 40,  # Virama
}

# =============================================================================
# Vowel Classification and Validation
# =============================================================================

UPPER_VOWELS = {"\u102d", "\u102e", "\u1032"}  # I, II, AI
LOWER_VOWELS = {"\u102f", "\u1030"}  # U, UU

INVALID_E_COMBINATIONS = {"\u102d", "\u102e", "\u102f", "\u1030"}

# Valid Vowel Combinations (Digraphs)
VALID_VOWEL_COMBINATIONS = {
    frozenset({"\u1031", "\u102c"}),  # E + Aa (ေ + ာ = Aw)
    frozenset({"\u1031", "\u102b"}),  # E + Tall A (ေ + ါ = Aw)
    frozenset({"\u102d", "\u102f"}),  # I + U (ိ + ု = O)
}

# Anusvara (1036) Compatibility
ANUSVARA_ALLOWED_VOWELS = {
    "\u102d",  # I
    "\u102f",  # U
}

# =============================================================================
# Valid Medial Sequences
# =============================================================================

VALID_MEDIAL_SEQUENCES = {
    # All valid medial combinations in canonical order: Ya (ျ) → Ra (ြ) → Wa (ွ) → Ha (ှ)
    # Reference: Unicode Technical Note #11 (UTN#11) for Myanmar canonical ordering
    #
    # Four-medial combination (Ya+Ra+Wa+Ha)
    "ျြွှ",
    # Three-medial combinations
    "ျြွ",  # Ya+Ra+Wa
    "ျြှ",  # Ya+Ra+Ha
    "ျွှ",  # Ya+Wa+Ha
    "ြွှ",  # Ra+Wa+Ha
    # Two-medial combinations
    "ျြ",  # Ya+Ra (e.g., ကျြေး crane)
    "ျွ",  # Ya+Wa
    "ျှ",  # Ya+Ha
    "ြွ",  # Ra+Wa
    "ြှ",  # Ra+Ha
    "ွှ",  # Wa+Ha
    # Single medials
    "ျ",  # Ya (Ya-pin)
    "ြ",  # Ra (Ya-yit)
    "ွ",  # Wa (Wa-hswe)
    "ှ",  # Ha (Ha-htoe)
}

# =============================================================================
# Valid Particles
# =============================================================================

# Valid Symbol Particles (Myanmar punctuation marks that function as particles)
VALID_PARTICLES = frozenset(["\u104c", "\u104d", "\u104e", "\u104f"])  # ၌ ၍ ၎ ၏

# =============================================================================
# Section Signs and Abbreviation Marks
# =============================================================================
#
# Myanmar has special logographic symbols for common words/concepts.
# These are used as shorthand in formal and literary texts.
#
# Usage contexts:
# - ၌ (U+104C): Locative "at/in" - typically follows a noun
# - ၍ (U+104D): Conjunction/continuation "and then" - sentence connector
# - ၎ (U+104E): Aforementioned "the said" - references previous noun
# - ၏ (U+104F): Genitive/possessive "of" - marks possession
#
# These are standalone particles and should not be combined with vowels/medials.

#: Section marks that indicate locative, conjunctive relationships
SECTION_MARKS: frozenset = frozenset(
    {
        "\u104c",  # ၌ MYANMAR SYMBOL LOCATIVE - "at, in"
        "\u104d",  # ၍ MYANMAR SYMBOL COMPLETED - "and then, having done"
    }
)

#: Reference/possessive marks
REFERENCE_MARKS: frozenset = frozenset(
    {
        "\u104e",  # ၎ MYANMAR SYMBOL AFOREMENTIONED - "the said, aforementioned"
        "\u104f",  # ၏ MYANMAR SYMBOL GENITIVE - "of, 's"
    }
)

#: All Myanmar logographic particles combined
LOGOGRAPHIC_PARTICLES: frozenset = SECTION_MARKS | REFERENCE_MARKS

#: Combined set of all Myanmar special symbols
MYANMAR_SPECIAL_SYMBOLS: frozenset = LOGOGRAPHIC_PARTICLES | MYANMAR_PUNCTUATION


# =============================================================================
# Stacking Exceptions
# =============================================================================
#
# Myanmar uses virama (္ U+1039) to stack consonants, primarily for Pali/Sanskrit
# loanwords. Not all consonant pairs can be stacked - only specific combinations
# are valid based on Indic phonology and Myanmar orthographic conventions.
#
# Categories:
# 1. Gemination: Same consonant doubled (e.g., က္က kka, တ္တ tta)
# 2. Homorganic clusters: Consonants from same place of articulation (e.g., ဏ္ဍ ṇḍa)
# 3. Cross-row clusters: Common Pali/Sanskrit combinations (e.g., က္တ kta, န္ဒ nda)
# 4. Sonorant clusters: Consonant + Ya/Ra/Wa (e.g., က္ယ kya, မ္ရ mra)
#
# Reference: Myanmar Language Commission, "Standard Myanmar Orthography"

#: Valid consonant stacking pairs for Pali/Sanskrit loanwords.
#: Format: (upper_consonant, lower_consonant) - upper stacks over lower via virama.
STACKING_EXCEPTIONS = {
    # Same consonant gemination
    ("\u1000", "\u1000"),  # က္က - kka
    ("\u1002", "\u1002"),  # ဂ္ဂ - gga
    ("\u1003", "\u1003"),  # ဃ္ဃ - ggha
    ("\u1005", "\u1005"),  # စ္စ - cca
    ("\u1006", "\u1006"),  # ဆ္ဆ - chcha
    ("\u1007", "\u1007"),  # ဇ္ဇ - jja (e.g., မဇ္ဈိမ Majjhima)
    ("\u1010", "\u1010"),  # တ္တ - tta
    ("\u1012", "\u1012"),  # ဒ္ဒ - dda
    ("\u1014", "\u1014"),  # န္န - nna
    ("\u1015", "\u1015"),  # ပ္ပ - ppa
    ("\u1017", "\u1017"),  # ဗ္ဗ - bba
    ("\u1019", "\u1019"),  # မ္မ - mma
    ("\u101c", "\u101c"),  # လ္လ - lla
    ("\u101e", "\u101e"),  # သ္သ - ssa
    # Cross-consonant valid pairs
    ("\u100f", "\u100d"),  # ဏ္ဍ - ṇḍa
    ("\u100f", "\u100c"),  # ဏ္ဌ - ṇṭha
    ("\u100f", "\u100b"),  # ဏ္ဋ - ṇṭa (Pali retroflex)
    ("\u100b", "\u100c"),  # ဋ္ဌ - ṭṭha (604 words: ဥက္ကဋ္ဌ, ပြဋ္ဌာန်း)
    ("\u100d", "\u100e"),  # ဍ္ဎ - ḍḍha (retroflex cluster)
    # Palatal nasal clusters (ñ + palatal stop)
    ("\u1009", "\u1005"),  # ဉ္စ - ñca (96 words: ပဉ္စမ fifth)
    ("\u1009", "\u1007"),  # ဉ္ဇ - ñja (46 words: သဉ္ဇာ)
    ("\u1009", "\u1016"),  # ဉ္ဖ - ñpha (6 words: ပဉ္ဖင်း)
    ("\u1000", "\u1001"),  # က္ခ - kkha
    ("\u1002", "\u1003"),  # ဂ္ဃ - ggha
    ("\u1005", "\u1006"),  # စ္ဆ - ccha
    ("\u1010", "\u1011"),  # တ္ထ - ttha
    ("\u1012", "\u1013"),  # ဒ္ဓ - ddha
    ("\u1015", "\u1016"),  # ပ္ဖ - ppha
    ("\u1017", "\u1018"),  # ဗ္ဘ - bbha
    # Special cross-row combinations
    ("\u1000", "\u1010"),  # က္တ - kta
    ("\u1014", "\u1010"),  # န္တ - nta
    ("\u1019", "\u1015"),  # မ္ပ - mpa
    ("\u101e", "\u1010"),  # သ္တ - sta
    ("\u1000", "\u1014"),  # က္န - kna
    ("\u1010", "\u1019"),  # တ္မ - tma
    ("\u1014", "\u1012"),  # န္ဒ - nda
    ("\u1014", "\u1013"),  # န္ဓ - ndha
    ("\u1019", "\u1017"),  # မ္ဗ - mba
    ("\u1019", "\u1018"),  # မ္ဘ - mbha
    # La combinations
    ("\u101c", "\u1000"),  # လ္က - lka
    ("\u101c", "\u1010"),  # လ္တ - lta
    ("\u101c", "\u1015"),  # လ္ပ - lpa
    ("\u101c", "\u101c"),  # လ္လ - lla
    # Retroflex lateral (Lla) gemination
    ("\u1020", "\u1020"),  # ဠ္ဠ - ḷḷa (Pali retroflex lateral)
    # Sa combinations
    ("\u101e", "\u1014"),  # သ္န - sna
    ("\u101e", "\u1019"),  # သ္မ - sma
    ("\u101e", "\u1015"),  # သ္ပ - spa
    ("\u1014", "\u101e"),  # န္သ - nsa
    # Additional valid stacks
    ("\u1000", "\u1019"),  # က္မ - kma
    ("\u1002", "\u1014"),  # ဂ္န - gna
    ("\u1005", "\u1010"),  # စ္တ - cta
    ("\u1010", "\u1000"),  # တ္က - tka
    ("\u1012", "\u1019"),  # ဒ္မ - dma
    ("\u1015", "\u1010"),  # ပ္တ - pta
    ("\u1017", "\u1010"),  # ဗ္တ - bta
    # Ha combinations
    ("\u101f", "\u1019"),  # ဟ္မ - hma
    # Non-Vagga consonant stacking (Ya as operand)
    ("\u1019", "\u101a"),  # မ္ယ - mya
    ("\u1000", "\u101a"),  # က္ယ - kya
    ("\u101e", "\u101a"),  # သ္ယ - sya
    ("\u1010", "\u101a"),  # တ္ယ - tya
    ("\u1012", "\u101a"),  # ဒ္ယ - dya
    ("\u1014", "\u101a"),  # န္ယ - nya
    # Ra as operand
    ("\u1019", "\u101b"),  # မ္ရ - mra
    ("\u1000", "\u101b"),  # က္ရ - kra
    ("\u1010", "\u101b"),  # တ္ရ - tra
    ("\u1012", "\u101b"),  # ဒ္ရ - dra
    ("\u1014", "\u101b"),  # န္ရ - nra
    ("\u1015", "\u101b"),  # ပ္ရ - pra
    ("\u1017", "\u101b"),  # ဗ္ရ - bra
    ("\u101e", "\u101b"),  # သ္ရ - sra
    # Wa as operand
    ("\u1010", "\u101d"),  # တ္ဝ - twa
    ("\u1012", "\u101d"),  # ဒ္ဝ - dwa
    ("\u1014", "\u101d"),  # န္ဝ - nwa
    ("\u101e", "\u101d"),  # သ္ဝ - swa
    # Ha as lower operand
    ("\u1000", "\u101f"),  # က္ဟ - kha
    ("\u1010", "\u101f"),  # တ္ဟ - tha
    ("\u1014", "\u101f"),  # န္ဟ - nha
    # Additional Pali/Sanskrit combinations
    ("\u1005", "\u1014"),  # စ္န - cna (some Pali words)
    ("\u1010", "\u1014"),  # တ္န - tna (e.g., ရတ္န ratna/jewel)
    ("\u1019", "\u101f"),  # မ္ဟ - mha (Brahmi loanwords)
    # Additional Pali/Sanskrit stacking patterns
    # Retroflex gemination (ṭa-group)
    ("\u100b", "\u100b"),  # ဋ္ဋ - ṭṭa (retroflex gemination)
    ("\u100d", "\u100d"),  # ဍ္ဍ - ḍḍa (retroflex voiced gemination)
    ("\u100f", "\u100f"),  # ဏ္ဏ - ṇṇa (retroflex nasal gemination)
    # Ca-group cross-row patterns (Pali palatal + other)
    ("\u1005", "\u1007"),  # စ္ဇ - cja (rare Pali cluster)
    ("\u1007", "\u1008"),  # ဇ္ဈ - jjha (Pali cluster)
    ("\u100a", "\u100a"),  # ည္ည - ñña (Pali palatal nasal gemination, e.g., viññāṇa)
    ("\u1009", "\u1009"),  # ဉ္ဉ - ñña archaic form
    ("\u1007", "\u1014"),  # ဇ္န - jna (e.g., prajña/ပရဇ္ဉာ wisdom)
    # Cross-vagga clusters (valid in Pali but not in native Myanmar)
    ("\u1000", "\u1005"),  # က္စ - kca (Pali cluster)
    ("\u1002", "\u1019"),  # ဂ္မ - gma (Sanskrit cluster, e.g., in "agni" variants)
    ("\u1010", "\u1005"),  # တ္စ - tca (Pali cluster)
    ("\u1012", "\u1014"),  # ဒ္န - dna (Pali cluster)
    ("\u1015", "\u1014"),  # ပ္န - pna (Pali cluster)
    ("\u1017", "\u1014"),  # ဗ္န - bna (Pali cluster)
    # Nasal + aspirated combinations
    ("\u1014", "\u1011"),  # န္ထ - ntha (common Pali, e.g., ပန္ထ pantha/path)
    ("\u1019", "\u1016"),  # မ္ဖ - mpha (Pali cluster)
    ("\u1019", "\u101e"),  # မ္သ - msa (Pali cluster)
    # Additional cross-row nasal clusters
    ("\u1004", "\u1000"),  # င္က - ṅka (common in Pali, e.g., saṅkha)
    ("\u1004", "\u1001"),  # င္ခ - ṅkha (Pali cluster)
    ("\u1004", "\u1002"),  # င္ဂ - ṅga (common Pali, e.g., saṅgha/သင်္ဃ)
    ("\u1004", "\u1003"),  # င္ဃ - ṅgha (Pali cluster)
    # Additional Pali/Sanskrit geminations
    # Aspirated consonant geminations
    ("\u1008", "\u1008"),  # ဈ္ဈ - jjha (Pali palatal aspirated gemination)
    ("\u100e", "\u100e"),  # ဎ္ဎ - ḍḍha (Pali retroflex aspirated gemination)
    ("\u1011", "\u1011"),  # ထ္ထ - ttha (Pali dental aspirated gemination)
    ("\u1013", "\u1013"),  # ဓ္ဓ - ddha (Pali dental voiced aspirated gemination)
    ("\u1016", "\u1016"),  # ဖ္ဖ - ppha (Pali labial aspirated gemination)
    ("\u1018", "\u1018"),  # ဘ္ဘ - bbha (Pali labial voiced aspirated gemination)
    # Semi-vowel geminations
    ("\u101a", "\u101a"),  # ယ္ယ - yya (Pali palatal glide gemination)
    ("\u101b", "\u101b"),  # ရ္ရ - rra (Pali alveolar tap gemination)
    ("\u101d", "\u101d"),  # ဝ္ဝ - wwa (Pali labial-velar glide gemination)
    ("\u101f", "\u101f"),  # ဟ္ဟ - hha (Pali glottal gemination)
}

# =============================================================================
# Kinzi
# =============================================================================

# Kinzi sequence: Nga (င) + Asat (်) + Virama (္)
# Used to detect and merge kinzi coda fragments during syllable segmentation.
KINZI_SEQUENCE = "\u1004\u103a\u1039"

# Kinzi (င်္) pattern: Nga + Asat + Virama + following consonant
KINZI_VALID_FOLLOWERS = {
    # Ka-group (Velar consonants)
    "\u1000",  # က - ka
    "\u1001",  # ခ - kha
    "\u1002",  # ဂ - ga
    "\u1003",  # ဃ - gha
    "\u1004",  # င - nga
    # Ca-group (Palatal consonants)
    "\u1005",  # စ - ca
    "\u1006",  # ဆ - cha
    "\u1007",  # ဇ - ja
    "\u1008",  # ဈ - jha
    "\u1009",  # ဉ - nnya
    # Note: U+100A (ည - nnya big) intentionally excluded - rarely follows Kinzi
    # Ta-group (Retroflex consonants)
    "\u100b",  # ဋ - ṭa
    "\u100c",  # ဌ - ṭha
    "\u100d",  # ဍ - ḍa
    "\u100e",  # ဎ - ḍha
    "\u100f",  # ဏ - ṇa
    # Ta-group (Dental consonants)
    "\u1010",  # တ - ta
    "\u1011",  # ထ - tha
    "\u1012",  # ဒ - da
    "\u1013",  # ဓ - dha
    "\u1014",  # န - na
    # Pa-group (Labial consonants)
    "\u1015",  # ပ - pa
    "\u1016",  # ဖ - pha
    "\u1017",  # ဗ - ba
    "\u1018",  # ဘ - bha
    "\u1019",  # မ - ma
    # Semi-vowels and other consonants
    "\u101a",  # ယ - ya
    "\u101b",  # ရ - ra
    "\u101c",  # လ - la
    "\u101d",  # ဝ - wa
    "\u101e",  # သ - sa
    "\u101f",  # ဟ - ha
    "\u1020",  # ဠ - lla
    # Pali/Sanskrit formal consonant
    "\u103f",  # ဿ - Great Sa (U+103F, not U+104E ၎)
}

# =============================================================================
# Classifier System
# =============================================================================

# Myanmar numerals (၀-၉)
MYANMAR_NUMERALS = {
    "\u1040",  # ၀ - zero
    "\u1041",  # ၁ - one
    "\u1042",  # ၂ - two
    "\u1043",  # ၃ - three
    "\u1044",  # ၄ - four
    "\u1045",  # ၅ - five
    "\u1046",  # ၆ - six
    "\u1047",  # ၇ - seven
    "\u1048",  # ၈ - eight
    "\u1049",  # ၉ - nine
}

# Myanmar numeral words (written form)
MYANMAR_NUMERAL_WORDS: dict[str, int] = {
    "တစ်": 1,
    "နှစ်": 2,
    "သုံး": 3,
    "လေး": 4,
    "ငါး": 5,
    "ခြောက်": 6,
    "ခုနစ်": 7,
    "ရှစ်": 8,
    "ကိုး": 9,
    "ဆယ်": 10,
    "ဆယ့်": 10,
    "ရာ": 100,
    "ထောင်": 1000,
    "သောင်း": 10000,
    "သိန်း": 100000,
    "သန်း": 1000000,
}

# Myanmar Classifiers (Numerative Words / ရေတွက်ပုဒ်)
# Classifiers are words used to count nouns, similar to "pieces" in "three pieces of paper"
# In Myanmar: [Number] + [Classifier] pattern, e.g., ကလေး ၃ ယောက် (3 children)
#
# Categories based on semantic class:
CLASSIFIERS: dict[str, str] = {
    # People classifiers
    "ယောက်": "people (general)",  # ကလေး ၃ ယောက် - 3 children
    "ဦး": "people (respectful)",  # ဆရာ ၂ ဦး - 2 teachers
    "ပါး": "people (monks/royalty)",  # ဘုန်းကြီး ၃ ပါး - 3 monks
    # Animal classifiers
    "ကောင်": "animals (general)",  # ခွေး ၂ ကောင် - 2 dogs
    # Removed: "မြင်း" — this is a NOUN meaning "horse", not a classifier.
    # The correct classifier for animals (including horses) is "ကောင်" above.
    # Vehicle classifiers
    "စီး": "vehicles",  # ကား ၃ စီး - 3 cars
    "စင်း": "boats/planes",  # လှေ ၂ စင်း - 2 boats
    # Object classifiers by shape
    "ခု": "general objects",  # စာအုပ် ၂ ခု - 2 books
    "လုံး": "round objects/buildings",  # ပန်းသီး ၃ လုံး - 3 apples, အိမ် ၃ လုံး - 3 houses
    "ချပ်": "flat thin objects",  # စာရွက် ၅ ချပ် - 5 papers
    "ပြား": "flat objects",  # ပန်းကန် ၂ ပြား - 2 plates
    "တုံး": "block/chunk objects",  # ရေခဲ ၂ တုံး - 2 ice blocks
    "ခြောင်း": "long thin objects",  # ခဲတံ ၃ ခြောင်း - 3 pencils
    "စည်း": "bundles",  # ထင်း ၂ စည်း - 2 bundles of firewood
    "စုံ": "pairs/sets",  # ဖိနပ် ၂ စုံ - 2 pairs of shoes
    # Building/place classifiers (note: "လုံး" for buildings defined above with round objects)
    "ခန်း": "rooms",  # အခန်း ၂ ခန်း - 2 rooms
    "ဆိုင်": "shops",  # ဆိုင် ၃ ဆိုင် - 3 shops
    # Abstract/event classifiers
    "ကြိမ်": "times/occurrences",  # ၂ ကြိမ် - 2 times
    "ခါ": "times (colloquial)",  # ၃ ခါ - 3 times
    "ရက်": "days",  # ၇ ရက် - 7 days
    "လ": "months",  # ၃ လ - 3 months
    "နှစ်": "years",  # ၁၀ နှစ် - 10 years
    # Measurement classifiers
    "ပိဿာ": "viss (weight)",  # ၂ ပိဿာ - 2 viss
    "ပြည်": "pyi (volume)",  # ၃ ပြည် - 3 pyi
    "တောင်း": "basket",  # ဆန် ၂ တောင်း - 2 baskets of rice
    "အိတ်": "bags",  # ဆန် ၃ အိတ် - 3 bags of rice
    "ခွက်": "cups",  # လက်ဖက်ရည် ၂ ခွက် - 2 cups of tea
    "ပုလင်း": "bottles",  # ရေ ၃ ပုလင်း - 3 bottles of water
    # Text/document classifiers
    "ပုဒ်": "sentences/clauses",  # စာ ၃ ပုဒ် - 3 sentences
    "စောင်": "letters/newspapers",  # စာ ၂ စောင် - 2 letters
    "အုပ်": "books",  # စာအုပ် ၃ အုပ် - 3 books
    "ခွင်": "paintings/photos",  # ပန်းချီ ၂ ခွင် - 2 paintings
    # Organic/plant classifiers
    "ပင်": "plants/trees",  # သစ်ပင် ၃ ပင် - 3 trees
    "ပွင့်": "flowers",  # နှင်းဆီ ၅ ပွင့် - 5 roses
    "သီး": "fruits",  # ပန်းသီး ၃ သီး - 3 fruits
}

# Quick lookup set for classifier validation
CLASSIFIER_SET = frozenset(CLASSIFIERS.keys())

# =============================================================================
# Negation System
# =============================================================================

# Negation prefix
NEGATION_PREFIX = "မ"  # The universal negation marker

# Negative endings (particles that follow the verb in negation)
NEGATIVE_ENDINGS: dict[str, tuple[str, str]] = {
    # Standard negatives
    "ဘူး": ("standard_negative", "Colloquial negative ending"),
    "ပါဘူး": ("polite_negative", "Polite colloquial negative"),
    "ဘူ": ("typo_negative", "Missing visarga - common typo"),
    "ပါဘူ": ("typo_polite_negative", "Missing visarga in polite form"),
    # Formal negatives
    "ပါ": ("formal_negative", "Formal/written negative ending"),
    "မပါ": ("formal_polite_negative", "Formal polite negative"),
    # Conditional/hypothetical negatives
    "ရင်": ("conditional_negative", "If not - conditional"),
    "လျှင်": ("formal_conditional_negative", "If not - formal conditional"),
    # Imperative negatives (prohibitions)
    "နဲ့": ("prohibition", "Don't! - imperative prohibition"),
    "ပါနဲ့": ("polite_prohibition", "Please don't - polite prohibition"),
    "နှင့်": ("formal_prohibition", "Don't - formal prohibition"),
    # Question negatives
    "ဘူးလား": ("negative_question", "Isn't it? - colloquial question"),
    "ပါဘူးလား": ("polite_negative_question", "Isn't it? - polite question"),
    "သလား": ("formal_negative_question", "Isn't it? - formal question"),
}

# Valid negative endings set for quick lookup
VALID_NEGATIVE_ENDINGS = frozenset(NEGATIVE_ENDINGS.keys())

# Typo corrections for negative endings
NEGATIVE_TYPO_MAP: dict[str, str] = {
    # Missing visarga (း)
    "ဘူ": "ဘူး",
    "ပါဘူ": "ပါဘူး",
    # Wrong order
    "ူးဘ": "ဘူး",
    "ဘုူး": "ဘူး",
    # Missing asat (်)
    "ဘုး": "ဘူး",
    # Colloquial/slang
    "ဘူ:": "ဘူး",
}

# Auxiliary verbs that can appear between negation and ending
NEGATIVE_AUXILIARIES: dict[str, str] = {
    "တတ်": "habitual ability",
    "နိုင်": "ability/possibility",
    "ချင်": "want/desire",
    "ရ": "permitted/possible",
    "သေး": "yet (not yet)",
    "ခင်": "before",
    "လို့": "quotative/because",
}

# Words that commonly follow the negation prefix directly
COMMON_NEGATED_VERBS: dict[str, str] = {
    "သိ": "know",
    "ရှိ": "exist/have",
    "လာ": "come",
    "သွား": "go",
    "ပြော": "speak/say",
    "စား": "eat",
    "သောက်": "drink",
    "ကြည့်": "look/watch",
    "ကြား": "hear",
    "ဖတ်": "read",
    "ရေး": "write",
    "လုပ်": "do/work",
    "ပေး": "give",
    "ယူ": "take",
    "ထား": "put/place",
    "ဖြစ်": "be/happen",
    "ဝင်": "enter",
    "ထွက်": "exit",
    "တက်": "climb/attend",
    "ဆင်း": "descend",
}

# Register constants for Negation Map
REGISTER_FORMAL = "formal"
REGISTER_COLLOQUIAL = "colloquial"
REGISTER_NEUTRAL = "neutral"

# Register-aware negation markers
NEGATION_REGISTER_MAP: dict[str, str] = {
    # Formal register negations
    "မပါ": REGISTER_FORMAL,
    "မပါသည်": REGISTER_FORMAL,
    # Removed: "မသည်" — "သည်" is an affirmative/topic marker, never used in negation
    # Removed: "မမည်" — "မည်" is a future/intentional marker, does not participate in negation
    # Colloquial register negations
    "မဘူး": REGISTER_COLLOQUIAL,
    "မပါဘူး": REGISTER_COLLOQUIAL,
    # Removed: "မတယ်" — "တယ်" is affirmative, never used in negation
    # Removed: "မမယ်" — "မယ်" is a future marker, does not participate in negation
}

# =============================================================================
# Wet (Vagga) Mapping for Stacking Logic
# =============================================================================

# Mapping: Char -> (Row, Col)
WET_MAPPING: dict[str, tuple[int, int]] = {}
WET_MAPPING.update(
    {c: (0, i) for i, c in enumerate(["\u1000", "\u1001", "\u1002", "\u1003", "\u1004"], 1)}
)
WET_MAPPING.update(
    {c: (1, i) for i, c in enumerate(["\u1005", "\u1006", "\u1007", "\u1008", "\u100a"], 1)}
)
# Add Nya archaic (U+1009) to Row 1 (palatal/Ca vagga) - same phonemic class as Nya (U+100A)
WET_MAPPING["\u1009"] = (1, 5)  # ဉ - Nya archaic, palatal nasal
WET_MAPPING.update(
    {c: (2, i) for i, c in enumerate(["\u100b", "\u100c", "\u100d", "\u100e", "\u100f"], 1)}
)
WET_MAPPING.update(
    {c: (3, i) for i, c in enumerate(["\u1010", "\u1011", "\u1012", "\u1013", "\u1014"], 1)}
)
WET_MAPPING.update(
    {c: (4, i) for i, c in enumerate(["\u1015", "\u1016", "\u1017", "\u1018", "\u1019"], 1)}
)
