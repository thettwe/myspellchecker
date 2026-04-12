"""
Data constants for phonetic hashing and analysis.

This module contains static data structures used by the PhoneticHasher.
Separating data from logic improves readability and maintainability.
"""

from __future__ import annotations

from myspellchecker.core.constants import TONE_MARKS

__all__ = [
    "COLLOQUIAL_SUBSTITUTIONS",
    "G2P_HOMOPHONE_GROUPS",
    "MYANMAR_SUBSTITUTION_COSTS",
    "PHONETIC_GROUPS",
    "STANDARD_TO_COLLOQUIAL",
    "TONAL_GROUPS",
    "VISUAL_SIMILAR",
    "get_phonetic_equivalents",
    "get_standard_forms",
    "is_colloquial_variant",
]

# Phonetic Group Keys
KEY_LABIAL = "p"
KEY_ALVEOLAR = "t"
KEY_VELAR = "k"
KEY_PALATAL = "c"
KEY_RETROFLEX = "ṭ"
KEY_LIQUID_L = "l"
KEY_LIQUID_R = "r"
KEY_APPROX_Y = "y"
KEY_APPROX_W = "w"
KEY_SIBILANT = "s"
KEY_GLOTTAL = "h"
KEY_NASAL_M = "m"
KEY_NASAL_N = "n"
KEY_NASAL_NG = "ng"
KEY_NASAL_NY = "ny"
KEY_NASAL_N_RETRO = "n_retro"
KEY_MEDIAL_Y = "medial_y"
KEY_MEDIAL_R = "medial_r"
KEY_MEDIAL_W = "medial_w"
KEY_MEDIAL_H = "medial_h"
KEY_VOWEL_A = "vowel_a"
KEY_VOWEL_I = "vowel_i"
KEY_VOWEL_U = "vowel_u"
KEY_VOWEL_E = "vowel_e"
KEY_VOWEL_AI = "vowel_ai"  # Separated from E - ဲ is phonetically /ɛ/, not /e/
KEY_VOWEL_O = "vowel_o"
KEY_TONE = "tone"

# Phonetic similarity groups
# Characters in the same group sound similar and are often confused
PHONETIC_GROUPS: dict[str, list[str]] = {
    # Labial consonants (bilabial sounds)
    KEY_LABIAL: ["ပ", "ဖ", "ဗ", "ဘ"],
    KEY_NASAL_M: ["မ"],
    # Alveolar consonants (tongue-teeth sounds)
    KEY_ALVEOLAR: ["တ", "ထ", "ဒ", "ဓ"],
    KEY_NASAL_N: ["န"],
    # Velar consonants (back of throat sounds)
    KEY_VELAR: ["က", "ခ", "ဂ", "ဃ"],
    KEY_NASAL_NG: ["င"],
    # Palatal consonants
    KEY_PALATAL: ["စ", "ဆ", "ဇ", "ဈ"],
    KEY_NASAL_NY: ["ည", "\u1009"],
    # Retroflex consonants
    KEY_RETROFLEX: ["ဋ", "ဌ", "ဍ", "ဎ"],
    KEY_NASAL_N_RETRO: ["ဏ"],
    # Approximants and liquids
    KEY_LIQUID_L: ["လ", "ဠ"],
    KEY_LIQUID_R: ["ရ"],
    KEY_APPROX_Y: ["ယ"],
    KEY_APPROX_W: ["ဝ"],
    # Sibilants and fricatives
    KEY_SIBILANT: ["သ", "ဿ"],  # Shan SHA (U+1050) and SSA (U+1051) removed: Burmese only
    KEY_GLOTTAL: ["ဟ"],  # Ha
    "a": [],  # A/E moved to vowels
    # Medials (can be confused)
    KEY_MEDIAL_Y: ["ျ"],  # Ya-pin
    KEY_MEDIAL_R: ["ြ"],  # Ya-yit
    KEY_MEDIAL_W: ["ွ"],  # Wa-hswe
    KEY_MEDIAL_H: ["ှ"],  # Ha-htoe
    # Vowels
    KEY_VOWEL_A: ["ာ", "\u102b"],  # -aa (dependent vowels only)
    "vowel_carrier": ["\u1021"],  # အ glottal stop onset (not a vowel sign)
    KEY_VOWEL_I: ["ိ", "ီ", "\u1023", "\u1024"],  # -i, -ii, I, II
    KEY_VOWEL_U: ["ု", "ူ", "\u1025", "\u1026"],  # -u, -uu, U, UU
    # Separated ေ (/e/) and ဲ (/ɛ/) - phonetically distinct vowels
    # ေ (U+1031) = E vowel, prefix position, IPA /e/ (mid front unrounded)
    # ဲ (U+1032) = AI vowel, suffix position, IPA /ɛ/ (mid-low front + glide)
    KEY_VOWEL_E: ["ေ", "\u1027"],  # e-, E (Mon E U+1028 removed: Burmese only)
    KEY_VOWEL_AI: ["ဲ"],  # ai (suffix vowel, distinct phoneme)
    # Only single-codepoint independent vowels here; multi-codepoint "ော"/"ော်"
    # removed to prevent CHAR_TO_PHONETIC corruption.
    KEY_VOWEL_O: ["\u1029", "\u102a"],  # Independent O (ဩ), Independent AU (ဪ)
    # Tones (can be omitted or confused)
    KEY_TONE: list(TONE_MARKS),
}


# Visual confusability groups
# Characters that look similar and are often mistaken
VISUAL_SIMILAR: dict[str, set[str]] = {
    "ိ": {"ီ"},  # short i vs long ii
    "ီ": {"ိ"},  # long ii → short i
    "ု": {"ူ"},  # short u vs long uu
    "ူ": {"ု"},  # long uu → short u
    "ာ": {"\u102b"},  # different aa marks
    "\u102b": {"ာ"},  # tall AA → regular AA
    # Removed ေ/ဲ pairing - they are phonetically distinct
    # ေ (/e/) and ဲ (/ɛ/) should NOT be treated as interchangeable
    "ျ": {"ြ"},  # ya-pin vs ya-yit (sometimes confusing)
    "န": {"ည"},  # na vs nya
    # Note: "င" (\u1004) mapping defined below with NGA vs NYA confusion
    # Additional visual/typing confusions
    "\u101b": {"\u101a"},  # ရ <-> ယ (Ra and Ya - visually similar in some fonts)
    "\u101a": {"\u101b"},  # ယ <-> ရ (bidirectional)
    "\u101e": {"\u103f"},  # သ <-> ဿ (Sa and Great Sa - visually similar)
    "\u103f": {"\u101e"},  # ဿ <-> သ (bidirectional)
    "\u1015": {"\u1017"},  # ပ <-> ဗ (Pa and Ba - aspirated vs voiced confusion)
    "\u1017": {"\u1015"},  # ဗ <-> ပ (bidirectional)
    # Myanmar-specific substitution costs
    # NYA variants - ည (U+100A) and ဉ (U+1009) are phonetically identical
    "\u100a": {"\u1009"},  # ည <-> ဉ (both are NYA, commonly confused)
    "\u1009": {"\u100a"},  # ဉ <-> ည (bidirectional)
    # Velar nasals and palatals often confused in colloquial speech
    "\u1004": {"\u100a", "\u1009"},  # င <-> ည/ဉ (NGA vs NYA confusion, includes nga vs nya)
    # Aspirated vs unaspirated consonant pairs (common typing errors)
    "\u1000": {"\u1001"},  # က <-> ခ (Ka vs Kha)
    "\u1001": {"\u1000"},  # ခ <-> က (bidirectional)
    "\u1002": {"\u1003"},  # ဂ <-> ဃ (Ga vs Gha)
    "\u1003": {"\u1002"},  # ဃ <-> ဂ (bidirectional)
    "\u1005": {"\u1006"},  # စ <-> ဆ (Sa vs Ssa)
    "\u1006": {"\u1005"},  # ဆ <-> စ (bidirectional)
    "\u1010": {"\u1011"},  # တ <-> ထ (Ta vs Tha)
    "\u1011": {"\u1010"},  # ထ <-> တ (bidirectional)
    "\u1012": {"\u1013"},  # ဒ <-> ဓ (Da vs Dha)
    "\u1013": {"\u1012"},  # ဓ <-> ဒ (bidirectional)
    "\u1016": {"\u1018"},  # ဖ <-> ဘ (Pha vs Bha)
    "\u1018": {"\u1016"},  # ဘ <-> ဖ (bidirectional)
    # Retroflex consonants (rarely used but sometimes confused)
    "\u100b": {"\u100c"},  # ဋ <-> ဌ (retroflex Ta vs Tha)
    "\u100c": {"\u100b"},  # ဌ <-> ဋ (bidirectional)
    "\u100d": {"\u100e"},  # ဍ <-> ဎ (retroflex Da vs Dha)
    "\u100e": {"\u100d"},  # ဎ <-> ဍ (bidirectional)
    # LA variants (common confusion)
    "\u101c": {"\u1020"},  # လ <-> ဠ (La vs Lla - Pali distinction)
    "\u1020": {"\u101c"},  # ဠ <-> လ (bidirectional)
    # Medial confusions (very common in typing errors)
    "\u103d": {"\u103e"},  # ွ <-> ှ (Wa-hswe vs Ha-htoe)
    "\u103e": {"\u103d"},  # ှ <-> ွ (bidirectional)
    # Wa consonant vs Myanmar zero digit (visually identical)
    "\u101d": {"\u1040"},  # ဝ <-> ၀ (Wa vs Zero)
    "\u1040": {"\u101d"},  # ၀ <-> ဝ (bidirectional)
}


# Myanmar-specific substitution costs for edit distance
# Lower cost means more likely confusion - used for weighted distance
# These costs are applied when neither visual_weight nor keyboard_weight applies
MYANMAR_SUBSTITUTION_COSTS: dict[str, dict[str, float]] = {
    # NYA variants - practically identical, very low cost
    "\u100a": {"\u1009": 0.1},  # ည <-> ဉ
    "\u1009": {"\u100a": 0.1},  # ဉ <-> ည
    # Ra/Ya confusion - common in rapid typing
    "\u101b": {"\u101a": 0.3},  # ရ <-> ယ
    "\u101a": {"\u101b": 0.3},  # ယ <-> ရ
    # Sa/Great Sa - practically identical sound
    "\u101e": {"\u103f": 0.2},  # သ <-> ဿ
    "\u103f": {"\u101e": 0.2},  # ဿ <-> သ
    # Aspirated consonant pairs - phonetically close (includes voiced cross-pairs)
    "\u1000": {"\u1001": 0.4, "\u1002": 0.3, "\u1003": 0.4},  # က <-> ခ, ဂ, ဃ
    "\u1001": {"\u1000": 0.4, "\u1002": 0.4, "\u1003": 0.4},  # ခ <-> က, ဂ, ဃ
    "\u1005": {"\u1006": 0.4, "\u1007": 0.3, "\u1008": 0.4},  # စ <-> ဆ, ဇ, ဈ
    "\u1006": {"\u1005": 0.4, "\u1007": 0.4, "\u1008": 0.4},  # ဆ <-> စ, ဇ, ဈ
    "\u1010": {"\u1011": 0.4, "\u1012": 0.3, "\u1013": 0.4},  # တ <-> ထ, ဒ, ဓ
    "\u1011": {"\u1010": 0.4, "\u1012": 0.4, "\u1013": 0.4},  # ထ <-> တ, ဒ, ဓ
    "\u1015": {"\u1016": 0.4, "\u1017": 0.3, "\u1018": 0.3},  # ပ <-> ဖ, ဗ, ဘ
    "\u1016": {"\u1015": 0.4, "\u1018": 0.4},  # ဖ <-> ပ, ဘ
    # Voiced/voiceless pairs
    "\u1017": {"\u1015": 0.3, "\u1018": 0.3},  # ဗ <-> ပ, ဘ
    "\u1018": {"\u1015": 0.3, "\u1016": 0.4, "\u1017": 0.3},  # ဘ <-> ပ, ဖ, ဗ
    # Additional voiced/voiceless pairs
    "\u1002": {"\u1000": 0.3, "\u1001": 0.4, "\u1003": 0.4},  # ဂ <-> က, ခ, ဃ
    "\u1003": {"\u1000": 0.4, "\u1001": 0.4, "\u1002": 0.4},  # ဃ <-> က, ခ, ဂ
    "\u1007": {"\u1005": 0.3, "\u1006": 0.4, "\u1008": 0.4},  # ဇ <-> စ, ဆ, ဈ
    "\u1008": {"\u1005": 0.4, "\u1006": 0.4, "\u1007": 0.4},  # ဈ <-> စ, ဆ, ဇ
    "\u1012": {"\u1010": 0.3, "\u1011": 0.4, "\u1013": 0.4},  # ဒ <-> တ, ထ, ဓ
    "\u1013": {"\u1010": 0.4, "\u1011": 0.4, "\u1012": 0.4},  # ဓ <-> တ, ထ, ဒ
    # Retroflex pairs
    "\u100b": {"\u100c": 0.3},  # ဋ <-> ဌ
    "\u100c": {"\u100b": 0.3},  # ဌ <-> ဋ
    "\u100d": {"\u100e": 0.3},  # ဍ <-> ဎ
    "\u100e": {"\u100d": 0.3},  # ဎ <-> ဍ
    # LA variants
    "\u101c": {"\u1020": 0.2},  # လ <-> ဠ
    "\u1020": {"\u101c": 0.2},  # ဠ <-> လ
    # Tone mark confusion (aukmyit vs visarga)
    "\u1037": {"\u1038": 0.2},  # ့ <-> း
    "\u1038": {"\u1037": 0.2},  # း <-> ့
    # Medial confusions - very common (same-class + cross-class)
    # Same-class: ျ↔ြ (both Y-medials), ွ↔ှ (both lower medials)
    # Cross-class: ွ↔ျ, ွ↔ြ (e.g. ကွောင်း→ကျောင်း)
    "\u103b": {"\u103c": 0.2, "\u103d": 0.5},  # ျ <-> ြ, ွ
    "\u103c": {"\u103b": 0.2, "\u103d": 0.5},  # ြ <-> ျ, ွ
    "\u103d": {"\u103e": 0.4, "\u103b": 0.5, "\u103c": 0.5},  # ွ <-> ှ, ျ, ြ
    "\u103e": {"\u103d": 0.4},  # ှ <-> ွ
    # Vowel length confusion
    "\u102d": {"\u102e": 0.2},  # ိ <-> ီ (short vs long i)
    "\u102e": {"\u102d": 0.2},  # ီ <-> ိ
    "\u102f": {"\u1030": 0.2},  # ု <-> ူ (short vs long u)
    "\u1030": {"\u102f": 0.2},  # ူ <-> ု
    # Wa consonant vs Myanmar zero digit (visually identical)
    "\u101d": {"\u1040": 0.1},  # ဝ <-> ၀
    "\u1040": {"\u101d": 0.1},  # ၀ <-> ဝ
    # Virama vs Asat - stacked consonant marker vs final consonant marker
    "\u1039": {"\u103a": 0.1},  # ္ <-> ်
    "\u103a": {"\u1039": 0.1},  # ် <-> ္
}

# Tonal groups for variant generation
# Characters that differ only by tone or length, often confused in typing
# Format: canonical_char -> list of tonal variants
TONAL_GROUPS: dict[str, list[str]] = {
    # Vowel 'a'
    "ာ": [
        "ာ",
        "\u1037",
        "\u1038",
        "ား",
    ],  # aa, dot below, visarga, aa + visarga (removed empty variant)
    "\u102b": ["\u102b", "\u1037", "\u1038", "ါး"],  # tall aa variants
    # Vowel 'i'
    "ိ": ["ိ", "ီ", "ိ့", "ီး"],  # i, ii, i + dot, ii + visarga
    "ီ": ["ိ", "ီ", "ိ့", "ီး"],
    # Vowel 'u'
    "ု": ["ု", "ူ", "ု့", "ူး"],  # u, uu, u + dot, uu + visarga
    "ူ": ["ု", "ူ", "ု့", "ူး"],
    # Vowel 'e'
    "ေ": ["ေ", "ေ့", "ေး"],  # e, e + dot, e + visarga
    "ဲ": ["ဲ", "ဲ့"],  # ai, ai + dot
    # Vowel 'o' (combined)
    "ော": [
        "ော",
        "ော့",
        "ော်",
    ],  # o, o + dot, aw (ိုး removed: distinct vowel phoneme /o:/, not a tonal variant of ော /ɔ/)
    # Tone marks themselves
    "\u1037": ["", "\u1038"],  # Dot Below (Auk-myit) -> empty or Visarga
    "\u1038": ["", "\u1037"],  # Visarga (Wit-sa-pauk) -> empty or Dot Below
}

# Colloquial Substitutions
# Common multi-character or whole-word substitutions found in colloquial/social media text
# that cannot be explained by simple edit distance or single-char phonetic groups.
# Format: colloquial_form -> set of standard forms
COLLOQUIAL_SUBSTITUTIONS: dict[str, set[str]] = {
    # Particle variants
    "အုန်း": {"ဦး"},  # Ohn (Coconut) -> U (Particle)
    "အုံး": {"ဦး"},  # Ohn (Pillow) -> U (Particle)
    # Verb ending colloquialisms
    "ပါဘူး": {"မပါဘူး"},  # Shortened negation
    "တာပဲ": {"တာပါပဲ"},  # Shortened emphasis
    # "တဲ့" removed: it is a VALID quotative/reported-speech particle, not a colloquial variant
    # Pronoun colloquialisms
    "ကျနော်": {"ကျွန်တော်"},  # Male first person (colloquial)
    "ကျွနော်": {"ကျွန်တော်"},  # Male first person (variant)
    "ကျမ": {"ကျွန်မ"},  # Female first person (colloquial)
    "မင်း": {"သင်"},  # Second person (informal -> formal)
    "ငါ": {"ကျွန်တော်", "ကျွန်မ"},  # First person (very informal)
    "သူတို့": {"သူများ"},  # Third person plural variants
    # Common word colloquialisms
    "ဟုတ်": {"ဟုတ်ကဲ့"},  # Yes (shortened)
    "အို": {"အိုး"},  # Pot/exclamation (without visarga)
    # "အဲ", "အဲဒါ", "ဘယ်လို", "ဘာကြောင့်" removed: standard modern Burmese
    # demonstratives/question words, not colloquial variants (their literary
    # counterparts ထို, မည်သို့, etc. are the marked forms)
    # Adverb colloquialisms
    "တော်တော်": {"အလွန်"},  # Very (colloquial -> formal)
    "သိပ်": {"အလွန်"},  # Very (colloquial -> formal)
    "ရမ်းရမ်း": {"အလွန်"},  # Very (very colloquial)
    # Reduplication variants
    "ကောင်းကောင်း": {"ကောင်းမွန်စွာ"},  # Well (reduplication -> adverb)
    "မြန်မြန်": {"မြန်ဆန်စွာ"},  # Quickly (reduplication -> adverb)
    "နှေးနှေး": {"နှေးကွေးစွာ"},  # Slowly (reduplication -> adverb)
    # Colloquial contractions
    "လို့ပဲ": {"ထို့ကြောင့်"},  # Because (contracted)
    "ရင်": {"လျှင်"},  # If (colloquial -> formal)
    # "တော့" removed: different particle from "တွင်" (discourse vs locative)
    # Social media / texting abbreviations
    "555": {"ဟာဟာဟာ"},  # Laughing (Thai style)
    # "ရယ်" removed: distinct particle (listing/exclamation) from "လေ" (emphasis/evidential)
    # --- Additional pronoun shortenings ---
    "ကျနော့်": {"ကျွန်တော့်"},  # Male first person possessive (colloquial)
    "ကျနော်တို့": {"ကျွန်တော်တို့"},  # Male first person plural (colloquial)
    "ကျမတို့": {"ကျွန်မတို့"},  # Female first person plural (colloquial)
    "ငါတို့": {"ကျွန်တော်တို့", "ကျွန်မတို့"},  # First person plural (very informal)
    "နင်": {"သင်", "ခင်ဗျား"},  # Second person (rude informal -> formal)
    "သူ့": {"သူ၏"},  # His/her possessive (colloquial -> formal)
    "ခင်ဗျ": {"ခင်ဗျား"},  # Sir (shortened honorific)
    # "ရှင်" removed: standard polite female-register pronoun, not colloquial
    "ရှင့်": {"ခင်ဗျား၏"},  # Your-female polite possessive (colloquial)
    "ငါ့": {"ကျွန်တော့်", "ကျွန်မ၏"},  # My (very informal possessive)
    "မင်းတို့": {"သင်တို့"},  # Second person plural (informal -> formal)
    # --- Common verb/phrase contractions ---
    "လုပ်တယ်": {"လုပ်ပါတယ်"},  # Do (dropping polite particle)
    "သွားတယ်": {"သွားပါတယ်"},  # Go (dropping polite particle)
    "လာတယ်": {"လာပါတယ်"},  # Come (dropping polite particle)
    "စားတယ်": {"စားပါတယ်"},  # Eat (dropping polite particle)
    "ဖတ်တယ်": {"ဖတ်ပါတယ်"},  # Read (dropping polite particle)
    "ရေးတယ်": {"ရေးပါတယ်"},  # Write (dropping polite particle)
    "ပြောတယ်": {"ပြောပါတယ်"},  # Say (dropping polite particle)
    "သိတယ်": {"သိပါတယ်"},  # Know (dropping polite particle)
    "ဖြစ်တယ်": {"ဖြစ်ပါတယ်"},  # Be/happen (dropping polite particle)
    "ရတယ်": {"ရပါတယ်"},  # Get/can (dropping polite particle)
    # --- Negation contractions ---
    "မလုပ်ဘူး": {"မလုပ်ပါဘူး"},  # Don't do (dropping polite particle)
    "မသွားဘူး": {"မသွားပါဘူး"},  # Don't go (dropping polite particle)
    "မလာဘူး": {"မလာပါဘူး"},  # Don't come (dropping polite particle)
    "မသိဘူး": {"မသိပါဘူး"},  # Don't know (dropping polite particle)
    "မရဘူး": {"မရပါဘူး"},  # Can't/don't get (dropping polite particle)
    # --- Particle and ending variants ---
    "ပေါ့": {"ပါ"},  # Casual affirmative -> polite particle
    "နော်": {"နော"},  # Tag question particle (colloquial with asat -> standard without)
    "ဟင်": {"ဟုတ်လား"},  # Huh? (colloquial question -> formal)
    "ဟာ": {"ဟယ်"},  # Exclamation (colloquial)
    "လေ": {"ပါ"},  # Emphasis particle (casual -> polite register)
    "ကွ": {"ကွာ"},  # Sentence-final particle (shortened)
    # --- Demonstrative and question word variants ---
    # Removed 16 entries: ဒီ, ဒီမှာ, အဲဒီ, အဲဒီမှာ, ဘယ်, ဘယ်မှာ, ဘာ,
    # ဘယ်တော့, ဘယ်သူ, ဘယ်နှစ်, ဒီလောက်, အဲလောက်.
    # These are standard modern Burmese (default, unmarked forms), not
    # colloquial variants. Their literary counterparts (ဤ, ထို, မည်သည့်,
    # etc.) are the stylistically marked forms. The POS disambiguator
    # already recognizes ဒီ/အဲဒီ as valid determiners.
    # --- Adverb and intensifier colloquialisms ---
    "အရမ်း": {"အလွန်"},  # Very (colloquial -> formal)
    "အပြင်": {"အပြင်ဘက်"},  # Outside (shortened)
    "အထဲ": {"အတွင်း"},  # Inside (colloquial -> formal)
    # --- Greeting and polite phrase variants ---
    "ဗျ": {"ခင်ဗျား"},  # Casual address (very shortened honorific)
    "ဗျာ": {"ခင်ဗျား"},  # Casual address (shortened honorific variant)
    # "ကြာ" removed: not a colloquial form of ခဏ (different meaning:
    # ကြာ = long duration, ခဏ = short moment)
    "ကိုယ်": {"မိမိ"},  # Self (colloquial -> formal reflexive)
    "ကိုယ့်": {"မိမိ၏"},  # Self's (colloquial possessive -> formal)
    # --- Common noun colloquialisms ---
    "ကိစ္စ": {"ကိစ္စရပ်"},  # Matter (shortened -> formal)
    "မနက်ဖြန်": {"နက်ဖြန်"},  # Tomorrow (colloquial with prefix -> standard)
    "တုန်းက": {"အချိန်က"},  # Back when (colloquial -> formal)
    "ဟိုတုန်းက": {"ထိုအခါက"},  # Back then (colloquial -> formal)
    "ဟိုတစ်ခေါက်": {"ထိုအကြိမ်"},  # That other time (colloquial -> formal)
    "ဟိုနေ့က": {"ထိုနေ့က"},  # That day (colloquial -> formal)
    # --- Verb aspect contractions (past/progressive dropping ပါ) ---
    "သွားခဲ့တယ်": {"သွားခဲ့ပါတယ်"},  # Past: went (dropping ပါ)
    "လုပ်ခဲ့တယ်": {"လုပ်ခဲ့ပါတယ်"},  # Past: did (dropping ပါ)
    "စားခဲ့တယ်": {"စားခဲ့ပါတယ်"},  # Past: ate (dropping ပါ)
    "ပြောနေတယ်": {"ပြောနေပါတယ်"},  # Progressive: is saying (dropping ပါ)
    "လုပ်နေတယ်": {"လုပ်နေပါတယ်"},  # Progressive: is doing (dropping ပါ)
    "သွားနေတယ်": {"သွားနေပါတယ်"},  # Progressive: is going (dropping ပါ)
    # --- Modal/auxiliary colloquial forms (dropping ပါ from modals) ---
    "လုပ်နိုင်တယ်": {"လုပ်နိုင်ပါတယ်"},  # Modal: can do (ability)
    "သွားရမယ်": {"သွားရပါမယ်"},  # Modal: must go (obligation)
    "စားချင်တယ်": {"စားချင်ပါတယ်"},  # Modal: want to eat (desire)
    "ပြောလို့ရတယ်": {"ပြောလို့ရပါတယ်"},  # Modal: can say (permission)
    "လာခဲ့ရမယ်": {"လာခဲ့ရပါမယ်"},  # Modal: had to come (past obligation)
    # --- Reduplication extensions (adjective reduplication -> formal adverb) ---
    "နည်းနည်း": {"အနည်းငယ်"},  # Reduplication: a little
    "ရှည်ရှည်": {"ရှည်လျားစွာ"},  # Reduplication: at length
    "ဖြေးဖြေး": {"ဖြေးညှင်းစွာ"},  # Reduplication: gently/slowly
    "လေးလေး": {"လေးနက်စွာ"},  # Reduplication: heavily/seriously
    # --- Register-marked casual particles ---
    # "ဟုတ်ဘူးလား" removed: meaning-reversing (adds negation prefix မ)
    "တယ်နော်": {"ပါတယ်"},  # Casual tag: sentence-final confirmation
    "ပဲလေ": {"ပါ"},  # Casual: emphatic assertion (double casual marker)
    # --- Colloquial negation extensions (dropping ပါ) ---
    "မဖြစ်ဘူး": {"မဖြစ်ပါဘူး"},  # Negation: won't happen
    "မကြိုက်ဘူး": {"မကြိုက်ပါဘူး"},  # Negation: don't like
    "မရှိဘူး": {"မရှိပါဘူး"},  # Negation: doesn't exist
    # --- Colloquial existence/copula (dropping ပါ) ---
    "ရှိတယ်": {"ရှိပါတယ်"},  # Existence: there is
    "ဖြစ်မယ်": {"ဖြစ်ပါမယ်"},  # Copula: will be (future)
    "ဟုတ်တယ်": {"ဟုတ်ပါတယ်"},  # Copula: that's right
    # --- Discourse connectors ---
    # "ဒါပေမယ့်", "ပြီးတော့", "ဒါကြောင့်" removed: standard modern Burmese
    # connectors, not colloquial (their literary counterparts သို့သော်,
    # ထို့နောက်, ထို့ကြောင့် are the marked forms)
    # --- Additional verb contractions (dropping ပါ) ---
    "ကြည့်တယ်": {"ကြည့်ပါတယ်"},  # Verb: look/watch
    "နေတယ်": {"နေပါတယ်"},  # Verb: stay/live
    "ပေးတယ်": {"ပေးပါတယ်"},  # Verb: give
}

# Reverse mapping: standard form -> set of colloquial variants
# Used to recognize when a colloquial form is intentionally used
STANDARD_TO_COLLOQUIAL: dict[str, set[str]] = {}
for colloquial, standards in COLLOQUIAL_SUBSTITUTIONS.items():
    for standard in standards:
        if standard not in STANDARD_TO_COLLOQUIAL:
            STANDARD_TO_COLLOQUIAL[standard] = set()
        STANDARD_TO_COLLOQUIAL[standard].add(colloquial)


def is_colloquial_variant(word: str) -> bool:
    """
    Check if a word is a known colloquial variant.

    Args:
        word: The word to check.

    Returns:
        True if word is a colloquial variant of a standard form.
    """
    return word in COLLOQUIAL_SUBSTITUTIONS


def get_standard_forms(colloquial: str) -> set[str]:
    """
    Get standard forms for a colloquial variant.

    Args:
        colloquial: The colloquial word to look up.

    Returns:
        Set of standard forms, empty set if not a known colloquial variant.
    """
    return COLLOQUIAL_SUBSTITUTIONS.get(colloquial, set())


# ============================================================
# G2P (Grapheme-to-Phoneme) Integration
# Loads homophone groups from rules/g2p_mappings.yaml to build
# phonetic equivalence classes for improved homophone detection.
# ============================================================


def _load_g2p_homophone_groups() -> list[dict[str, object]]:
    """Load homophone groups from g2p_mappings.yaml.

    Returns a list of group dicts, each with keys: phoneme, graphemes,
    confusion_type, frequency, and optional notes.  Returns an empty
    list if the YAML file cannot be loaded (missing file, missing
    PyYAML, etc.) so the library degrades gracefully.
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    yaml_path = Path(__file__).resolve().parent.parent / "rules" / "g2p_mappings.yaml"
    if not yaml_path.exists():
        logger.debug("G2P mappings file not found: %s", yaml_path)
        return []

    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed; G2P mappings unavailable")
        return []

    try:
        with open(yaml_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception:
        logger.warning("Failed to load G2P mappings from %s", yaml_path, exc_info=True)
        return []

    if not isinstance(data, dict):
        return []

    groups = data.get("homophone_groups")
    if not isinstance(groups, list):
        return []

    return groups


# Loaded once at import time; empty list on failure (graceful degradation).
G2P_HOMOPHONE_GROUPS: list[dict[str, object]] = _load_g2p_homophone_groups()

# Reverse index: character -> set of phonetically equivalent characters.
# Built from G2P homophone_groups so that lookups are O(1).
_G2P_EQUIVALENCE_MAP: dict[str, set[str]] = {}
for _group in G2P_HOMOPHONE_GROUPS:
    _graphemes = _group.get("graphemes")
    if not isinstance(_graphemes, list) or len(_graphemes) < 2:
        continue
    for _g in _graphemes:
        if _g not in _G2P_EQUIVALENCE_MAP:
            _G2P_EQUIVALENCE_MAP[_g] = set()
        for _other in _graphemes:
            if _other != _g:
                _G2P_EQUIVALENCE_MAP[_g].add(_other)


def get_phonetic_equivalents(char: str) -> set[str]:
    """Get characters that are phonetically equivalent to *char*.

    Uses the homophone_groups from ``g2p_mappings.yaml`` to return
    all characters that share the same phoneme and are therefore
    commonly confused.

    This is a superset of the information in :data:`PHONETIC_GROUPS`
    because G2P groups also capture cross-series mergers (e.g.
    retroflex/alveolar, ရ/ယ liquid merger) that phonetic-group
    membership alone does not express.

    Args:
        char: A single Myanmar character (consonant, vowel sign,
              medial, or final marker).

    Returns:
        Set of phonetically equivalent characters.  Empty set if the
        character has no known equivalents in the G2P data.

    Example::

        >>> get_phonetic_equivalents("က")
        {'ခ', 'ဂ', 'ဃ', 'ဋ'}
        >>> get_phonetic_equivalents("ရ")
        {'ယ'}
        >>> get_phonetic_equivalents("ξ")  # non-Myanmar
        set()
    """
    return set(_G2P_EQUIVALENCE_MAP.get(char, set()))
