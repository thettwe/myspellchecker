"""Static data sets for Myanmar text validation.

Contains linguistically-verified word lists, whitelists, and character sets
used by the validation functions.  This module is data-only (no logic) so
that the sets can be shared across sub-modules without circular imports.
"""

from __future__ import annotations

from myspellchecker.core.constants import MEDIALS

__all__ = [
    "KNOWN_INVALID_WORDS",
    "MEDIALS_SET",
    "PALI_WHITELIST",
    "VALID_PALI_BARE_ENDINGS",
]

# ============================================================================
# KNOWN INVALID WORDS
# ============================================================================

# Known invalid words that require lexical verification
# These are specific incorrect variants verified through linguistic analysis
# that cannot be caught by regex patterns alone
KNOWN_INVALID_WORDS: set[str] = {
    "ပျှော်",  # Should be ပျော် (happy) - extra ှ medial
    "န္တရား",  # Fragment - incomplete stacking at start
    "မစ္ဆားတရား",  # Invalid stacking combination
    "သြင်္ကန်",  # Wrong medial-ra position - should be သင်္ကြန်
    "စြင်္ကံ",  # Wrong medial-ra position
    "နိဂြောဓ",  # Incorrect Pali transliteration
    "ခုင်း",  # Should be ခွင်း - u-vowel confused with medial-wa
    "မျှေ",  # Incomplete word fragment
    "င်း",  # Segmentation artifact - common suffix, not standalone word
    "်",  # Floating asat - invalid standalone
    "့",  # Floating dot below - invalid standalone
    # Truncated words verified by frequency comparison
    "ညှိနှိ",  # Should be ညှိနှိုင်း (negotiate) - 169x less frequent
    "တင",  # Should be တင် (present/submit) - 871x less frequent
    "အဓ",  # Should be အဓိက (main) - 392x less frequent
    "က္ကဌ",  # Should be ဥက္ကဌ (chairman) - 104x less frequent
    "ယပ",  # Should be ယပ် (fan) - 2.4x less frequent
    # Missing ေ in ောင pattern (common typos) - verified by frequency
    "ကြာင့်",  # Should be ကြောင့် (because) - 1564x less frequent
    "အာင်",  # Should be အောင် (victory) - 2146x less frequent
    "အရာင်",  # Should be အရောင် (color) - 878x less frequent
    "ရှာင်",  # Should be ရှောင် (avoid) - 645x less frequent
    "လုပ်ဆာင်",  # Should be လုပ်ဆောင် (perform) - 2006x less frequent
    "တာင်",  # Should be တောင် (south/mountain) - 1692x less frequent
    "ဆာင်",  # Should be ဆောင် (carry) - 933x less frequent
    # Truncated Pali words
    "အိန္ဒိ",  # Should be အိန္ဒိယ (India) - 222x less frequent
    # Incomplete ော ending (truncated words)
    "ခြော",  # Should be ခြောက် (dry/six) - 309x less frequent
    # Misplaced vowels / typos
    "ယေန့",  # Should be ယနေ့ (today) - 3273x less frequent
    # Particle + extra syllable (segmentation error)
    "မှာက်",  # Particle မှာ + extra က် - 28899x less frequent than မှာ
    # Garbage consonant sequences
    "အခတ",  # Invalid - three bare consonants, no valid form
    # နူန်း → နှုန်း typos (common spelling error)
    "စျေးနူန်း",  # Should be စျေးနှုန်း (price rate)
    "နူန်း",  # Should be နှုန်း (rate)
    "ရာခိုင်နူန်း",  # Should be ရာခိုင်နှုန်း (percentage)
    # ခြဲ့ → ခဲ့ typos (wrong medial)
    "ခြဲ့ပီး",  # Should be ခဲ့ပြီး (has already)
    "ခြဲ့က",  # Should be ခဲ့က (past tense marker)
}

# ============================================================================
# PALI / SANSKRIT WHITELIST
# ============================================================================

# Pali loanwords whitelist - these end with bare consonants but are valid
# Common Pali/Sanskrit loanwords that end without asat
PALI_WHITELIST: set[str] = {
    # Geographic/administrative Pali terms
    "ဒေသ",  # region/area (deśa)
    "ကာလ",  # time/period (kāla)
    "ဌာန",  # department/place (ṭhāna)
    "ဇာတ",  # birth/tale (jāta)
    "ရာသ",  # zodiac sign (rāśi)
    "သဘာဝ",  # nature (svabhāva)
    "စကား",  # speech/language (sakā) — has visarga
    "ဝေဒ",  # Veda (veda)
    "မာန",  # pride (māna)
    "ဒါန",  # donation (dāna)
    "သီလ",  # morality (sīla)
    "ပညာ",  # wisdom (paññā)
    "သမာဓိ",  # concentration (samādhi)
    "ကမ္မ",  # karma/action (kamma)
    "နိဗ္ဗာန",  # nirvana (nibbāna)
    "ဓမ္မ",  # dhamma/teaching (dhamma)
    "သံဃ",  # sangha/community (saṅgha)
    "ဗုဒ္ဓ",  # Buddha (buddha)
    "အာရုံ",  # object/sense-base (ārammaṇa) - common in everyday Myanmar
    "သုတ",  # sutta/discourse (sutta)
    # Common Pali loanwords from religious corpora (Tipitaka, dictionaries)
    "ကုသလ",  # kusala - wholesome/meritorious
    "အကုသလ",  # akusala - unwholesome
    "သာသန",  # sasana - religion/dispensation
    "အာရာမ",  # arama - monastery/garden
    "ကထိက",  # kathika - preacher
    "ဝိနယ",  # vinaya - discipline
    "အဘိဓမ္မ",  # abhidhamma - higher teaching
    "ပိဋက",  # pitaka - basket (of scripture)
    "သုတ္တန်",  # suttanta - discourse collection
    "ဝိပဿနာ",  # vipassana - insight meditation
    "သမထ",  # samatha - calm meditation
    "စေတန",  # cetana - volition/intention
    "ဝိညာဏ",  # viññāṇa - consciousness
    "ဝေဒနာ",  # vedanā - feeling
    "သညာ",  # saññā - perception
    "သင်္ခါရ",  # saṅkhāra - formations
    "နာမ",  # nāma - name/mentality
    "ရူပ",  # rūpa - form/materiality
    "ခန္ဓ",  # khandha - aggregate
    "ဓာတ",  # dhātu - element
    "အာယတန",  # āyatana - sense base
    "ပစ္စယ",  # paccaya - condition
    "ဗောဓိ",  # bodhi - enlightenment
    # nibbāna - nirvana
    "ပရမတ္ထ",  # paramattha - ultimate reality
    "ပညတ္တိ",  # paññatti - concept/designation
    "ဥပါဒါန",  # upādāna - clinging
    "တဏှာ",  # taṇhā - craving
    "အဝိဇ္ဇာ",  # avijjā - ignorance
    "မဂ္ဂ",  # magga - path
    "ဖလ",  # phala - fruit/result
    "ဝိမုတ္တိ",  # vimutti - liberation
    "ပါရမီ",  # pāramī - perfection
    "မေတ္တာ",  # mettā - loving-kindness
    "ကရုဏာ",  # karuṇā - compassion
    "ဥပေက္ခာ",  # upekkhā - equanimity
    "ပဋိစ္စသမုပ္ပါဒ",  # paṭiccasamuppāda - dependent origination
}

# ============================================================================
# PALI/SANSKRIT WHITELIST (bare-ending words)
# ============================================================================
# Valid words ending with bare consonants (no asat/tone) - primarily Pali/Sanskrit
# loanwords and common particles. These should NOT be filtered as truncated words.
# Compiled from high-frequency analysis (freq > 10,000) and linguistic knowledge.

VALID_PALI_BARE_ENDINGS: set[str] = {
    # Single consonant particles (valid standalone)
    "က",
    "ခ",
    "ဂ",
    "င",
    "စ",
    "ဆ",
    "ည",
    "တ",
    "ထ",
    "န",
    "ပ",
    "ဖ",
    "ဗ",
    "ဘ",
    "မ",
    "ယ",
    "ရ",
    "လ",
    "ဝ",
    "သ",
    "ဟ",
    "အ",
    "ဒ",
    "ဇ",
    # Pali/Sanskrit loanwords - geographic/administrative
    "ဒေသ",  # region (436,426)
    "ဌာန",  # department (221,256)
    "ဝန်ကြီးဌာန",  # ministry (67,720)
    "ဦးစီးဌာန",  # department (51,842)
    # Pali/Sanskrit loanwords - ordinals/numbers
    "ဒုတိယ",  # second (151,791)
    "ပထမ",  # first (96,325)
    "တတိယ",  # third (39,638)
    "ဒသမ",  # tenth (39,871)
    # Pali/Sanskrit loanwords - abstract concepts
    "ကာလ",  # time/period (166,167)
    "အဓိက",  # main (122,999)
    "ဘဝ",  # life (115,276)
    "သမ္မတ",  # president (114,004)
    "သဘာဝ",  # nature (102,730)
    "ပမာဏ",  # quantity (56,170)
    "သုတေသန",  # research (46,956)
    "လောက",  # world (40,122)
    "မူဝါဒ",  # policy (36,322)
    "ဝါဒ",  # doctrine (23,205)
    "ဂီတ",  # music (22,060)
    "ပုဂ္ဂလိက",  # private (17,120)
    "ဒေါသ",  # anger (15,894)
    "ဗေဒ",  # science (13,522)
    "အာကာသ",  # space (12,362)
    "ဇီဝ",  # biology (12,349)
    "သယံဇာတ",  # resources (13,477)
    "ဗဟုသုတ",  # knowledge (10,361)
    "အာဟာရ",  # nutrition (19,659)
    "အနှစ်သာရ",  # essence (10,367)
    "သင်္ကေတ",  # symbol (12,440)
    "သံသယ",  # doubt (22,653)
    # Pali/Sanskrit loanwords - country/region names
    "အိန္ဒိယ",  # India (63,986)
    "ဥရောပ",  # Europe (38,798)
    "အာဖရိက",  # Africa (13,388)
    "အမေရိက",  # America (12,627)
    "ဣသရေလ",  # Israel (10,755)
    # Government/political terms
    "အစိုးရ",  # government (352,237)
    "ဥက္ကဌ",  # chairman (28,539)
    # Common compounds ending with valid bare consonants
    "အရ",  # matter/thing (275,547)
    "ကျင်းပ",  # hold/organize (199,500)
    "ကုသ",  # treatment (94,150)
    "ယူဆ",  # consider (45,480)
    "မူလ",  # original (33,945)
    "ထာဝရ",  # eternal (27,470)
    "ခဏ",  # moment (20,577)
    "အယူအဆ",  # opinion (20,517)
    # Words ending with medial + consonant (valid patterns)
    "ကြွယ်ဝ",  # rich (17,919)
    "ပြည့်ဝ",  # full (15,307)
    "လုံးဝ",  # completely (53,187)
    "တောက်ပ",  # bright (16,225)
    # Person-related
    "သူမ",  # she (103,394)
    "ကျွန်မ",  # I (female) (60,885)
    "မိဘ",  # parents (42,239)
    "ဆရာမ",  # female teacher (33,360)
    "မိန်းမ",  # woman (25,263)
    # Administrative/location
    "ပြည်ပ",  # abroad (47,344)
    "ပြင်ပ",  # outside (29,496)
    "ပုဒ်မ",  # clause/section (55,778)
    "ဧက",  # acre (31,508)
    # Temporal
    "ကတည်းက",  # since (26,259)
    "မကြာခဏ",  # frequently (34,697)
    # Other common valid patterns
    "သာမက",  # not only (23,796)
    "အပြည့်အဝ",  # fully (27,925)
    "မတော်တဆ",  # accident (12,016)
    "သတိရ",  # remember (12,148)
    # Room/place suffixes
    "ခန်းမ",  # hall (23,708)
    "လမ်းမ",  # main road (21,314)
    "မြို့မ",  # main city (10,154)
    "လက်မ",  # inch (17,020)
}

# ============================================================================
# SEGMENTATION FRAGMENT CONSTANTS
# ============================================================================

# Single-character interjections that are valid standalone words
_ALLOWED_SINGLE_CONSONANTS: set[str] = {
    "\u1021",  # အ - common interjection (ah!)
    "\u101f",  # ဟ - interjection (ha!)
}

# Stacking marker (virama)
_STACKING_MARKER = "\u1039"  # ္

# Great Sa - a conjunct character that never appears at word start
# It appears in the middle of words like ပြဿနာ (problem), မနုဿ (human)
_GREAT_SA = "\u103f"  # ဿ

# Dependent vowel signs that cannot start a word
_DEPENDENT_VOWELS: set[str] = {
    "\u102b",  # ါ (tall aa)
    "\u102c",  # ာ (aa)
    "\u102d",  # ိ (i)
    "\u102e",  # ီ (ii)
    "\u102f",  # ု (u)
    "\u1030",  # ူ (uu)
    "\u1031",  # ေ (e) - visually appears before consonant but follows in Unicode
    "\u1032",  # ဲ (ai)
    "\u1033",  # ဳ (mon ii)
    "\u1034",  # ဴ (mon o)
    "\u1035",  # ဵ (e above)
    "\u1036",  # ံ (anusvara)
    "\u1037",  # ့ (dot below - aukmyit)
    "\u1038",  # း (visarga)
    "\u103a",  # ် (asat)
}

# Asat + Anusvara sequence (phonetically impossible)
# Asat (်) closes a syllable, anusvara (ံ) requires a vowel to nasalize
# The sequence ်ံ cannot occur in valid Myanmar text
_ASAT_ANUSVARA = "\u103a\u1036"  # ်ံ

# Independent vowels (standalone vowel characters)
# These represent vowel sounds without a consonant carrier
# Doubled independent vowels (e.g., ဤဤ, ဥဥ) are OCR errors, not valid words
_INDEPENDENT_VOWELS_SET: set[str] = {
    "\u1023",  # ဣ (i)
    "\u1024",  # ဤ (ii)
    "\u1025",  # ဥ (u)
    "\u1026",  # ဦ (uu)
    "\u1027",  # ဧ (e)
    "\u1029",  # ဩ (o)
    "\u102a",  # ဪ (au)
}

# Characters allowed in Myanmar words beyond the core Myanmar block (U+1000-U+109F):
# - U+200B (ZWSP) — may appear in segmented text, stripped elsewhere
# - U+200C/U+200D (ZWNJ/ZWJ) — used in some Myanmar rendering
_WORD_ALLOWED_NON_MYANMAR = frozenset("\u200b\u200c\u200d")

# Re-export MEDIALS so that validator_checks can use it without a separate import
MEDIALS_SET: set[str] = MEDIALS
