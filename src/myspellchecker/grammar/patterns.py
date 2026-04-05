"""
Centralized linguistic rule definitions for Myanmar (Burmese) language.

This module provides a single source of truth for all linguistic patterns,
particles, and rules used across the spellchecker. It eliminates duplication
and ensures consistency across engine.py, morphology.py, and other modules.

Architecture:
    - All particle/suffix lists are defined once here
    - Other modules import from this centralized source
    - Machine-readable format for easy updates and extensions
    - Organized by linguistic category for maintainability

Categories:
    1. Sentence Particles - Sentence-ending markers
    2. Verb Particles - Tense/aspect markers after verbs
    3. Noun Particles - Case markers and postpositions
    4. Suffixes - Derivational suffixes for POS guessing
    5. Grammar Patterns - Common error patterns

Usage:
    >>> from myspellchecker.grammar.patterns import SENTENCE_PARTICLES
    >>> if word in SENTENCE_PARTICLES:
    ...     # Handle sentence-final particle
"""

from __future__ import annotations

from myspellchecker.core.constants.core_constants import (
    ADVERB_SUFFIXES,  # noqa: F401 — re-exported for consumers
    NOUN_SUFFIXES,  # noqa: F401 — re-exported for consumers
    VERB_SUFFIXES,  # noqa: F401 — re-exported for consumers
)
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "ALL_PARTICLES",
    "ASPIRATION_MEDIAL_CONFUSIONS",
    "AUXILIARY_VERBS",
    "COMPLETIVE_ENDINGS",
    "DANGLING_COMPLETION_TEMPLATES",
    "INVALID_POS_SEQUENCES",
    "MALFORMED_QUESTION_ENDINGS",
    "MEDIAL_CONFUSION_PATTERNS",
    "MEDIAL_ORDER_CORRECTIONS",
    "NEGATIVE_ENDINGS",
    "NEGATIVE_INDEFINITE_WORDS",
    "NOUN_PARTICLES",
    "PARTICLE_TYPO_PATTERNS",
    "QUESTION_ENDING_REWRITE_MAP",
    "QUESTION_MERGED_PREFIXES",
    "QUESTION_PARTICLE_SUGGESTIONS",
    "QUESTION_PARTICLES",
    "QUESTION_WORDS",
    "RHYME_REDUPLICATION_PATTERNS",
    "SECOND_PERSON_FUTURE_TIME_WORDS",
    "SECOND_PERSON_MODAL_FUTURE_MARKERS",
    "SECOND_PERSON_PRONOUNS",
    "SENTENCE_FINAL_PARTICLES",
    "SENTENCE_PARTICLES",
    "STATEMENT_ENDINGS",
    "VALID_VERB_AUXILIARY_SEQUENCES",
    "VERB_PARTICLES",
    "detect_malformed_question_ending",
    "detect_reduplication_pattern",
    "detect_sentence_type",
    "ends_with_sfp",
    "find_statement_ending_for_question_rewrite",
    "get_aspiration_confusion",
    "get_dangling_completion_suggestions",
    "get_medial_confusion_correction",
    "get_medial_order_correction",
    "get_particle_typo_correction",
    "get_question_completion_suggestions",
    "get_question_rewrite_suggestions",
    "has_dangling_completion_template",
    "has_enclitic_question_particle",
    "has_question_particle_context",
    "has_question_word_context",
    "is_noun_particle",
    "is_question_particle",
    "is_question_word",
    "is_reduplication",
    "is_second_person_modal_future_question",
    "is_sentence_particle",
    "is_valid_verb_sequence",
    "is_verb_particle",
]

# =============================================================================
# SECTION 1: SENTENCE PARTICLES
# Particles that typically end sentences or clauses
# =============================================================================

SENTENCE_PARTICLES: frozenset[str] = frozenset(
    {
        # Statement endings
        "တယ်",  # Statement (colloquial)
        "သည်",  # Statement (formal)
        "ပါတယ်",  # Polite statement (colloquial)
        "ပါသည်",  # Polite statement (formal)
        # Future endings
        "မယ်",  # Future (colloquial)
        "မည်",  # Future (formal)
        "ပါမယ်",  # Polite future (colloquial)
        "ပါမည်",  # Polite future (formal)
        # Completion endings
        "ပြီ",  # Completion (informal)
        "ပါပြီ",  # Polite completion
        # Question endings
        "လား",  # Yes/no question
        "သလား",  # Yes/no question (alternative)
        "လဲ",  # Wh-question
        "မလဲ",  # Future wh-question
        # Emphasis/softener endings
        # Removed: "ပေ့" (with dot below) — not a recognized Myanmar particle.
        # The standard literary emphatic particle is "ပေ" (without dot below).
        "လေ",  # Emphasis
        "နော်",  # Tag question
    }
)

# =============================================================================
# SECTION 2: VERB PARTICLES
# Particles that follow verbs (tense, aspect, mood markers)
# =============================================================================

VERB_PARTICLES: frozenset[str] = frozenset(
    {
        # Tense markers
        "ခဲ့",  # Past tense
        "မယ်",  # Future tense (colloquial)
        "မည်",  # Future tense (formal)
        "လိမ့်မည်",  # Probability (will likely)
        # Aspect markers
        "နေ",  # Progressive
        "ပြီ",  # Perfective
        "ပြီး",  # After completion
        # Modal markers
        "နိုင်",  # Ability
        "ရ",  # Permission/possibility
        "တတ်",  # Habituality
        "တတ်တယ်",  # Habitually does
        # Directional markers
        "သွား",  # Away/go
        "လာ",  # Toward/come
        # Continuation/manner
        "လျက်",  # While doing
        "လေ",  # Emphasis
        "ပေ",  # Giving action
        # Statement endings (also verb particles)
        "တယ်",  # Statement
        "ပါတယ်",  # Polite statement
        "သည်",  # Formal statement
        "ပါသည်",  # Formal polite
        "ကြ",  # Plural marker
        # Conditional/purpose
        "ရင်",  # Conditional (if/when)
        "လျှင်",  # Formal conditional
        "သော်လည်း",  # Although
        "သော်",  # Relative (which/that)
        "သော",  # Relative (which/that)
        # Nominalization
        "ခြင်း",  # Nominalization
        "မှု",  # Nominalization (formal)
        "ရန်",  # Purpose/infinitive
        "ဖို့",  # Purpose (colloquial)
        "အောင်",  # Until/so that
        "လို့",  # Because/quotative
        "ကြောင့်",  # Because of
        "ကြောင်း",  # That (complement)
    }
)

# =============================================================================
# SECTION 3: NOUN PARTICLES
# Particles that follow nouns (case markers, postpositions)
# =============================================================================

NOUN_PARTICLES: frozenset[str] = frozenset(
    {
        # Core case markers
        "က",  # Subject marker
        "ကို",  # Object marker
        "သည်",  # Topic marker
        # Location/direction
        "မှာ",  # Location (at/in)
        "မှ",  # From/only if
        "တွင်",  # At (formal)
        "၌",  # At (formal written)
        "ထဲ",  # Inside
        "အထဲ",  # Inside (formal)
        "ပေါ်",  # On/upon
        "အပေါ်",  # On/upon (formal)
        "အောက်",  # Under
        "ရှေ့",  # In front
        "အရှေ့",  # In front (formal)
        "နောက်",  # Behind
        "အနောက်",  # Behind (formal)
        "ဘေး",  # Beside
        "အနား",  # Near
        # Accompaniment
        "နဲ့",  # With (colloquial)
        "နှင့်",  # With (formal)
        # Possession
        "ရဲ့",  # Possessive (colloquial)
        "၏",  # Possessive (formal)
        # Comparison/relation
        "အတွက်",  # For/because of
        "လို",  # Like
        "လိုလို",  # Like (emphasis)
        "လိုပဲ",  # Just like
        "ထက်",  # More than
        # Plural markers
        "များ",  # Plural marker
        "တို့",  # Plural (colloquial)
        # Topic/emphasis
        "ဟာ",  # Topic (colloquial)
        "တော့",  # Emphasis/then
        "ပဲ",  # Only/emphasis
        "ပေါ့",  # Softener
        "လေ",  # Emphasis
    }
)

# =============================================================================
# SECTION 4: DERIVATIONAL SUFFIXES (for OOV POS guessing)
# Imported from core.constants.core_constants (single source of truth).
# =============================================================================

# Re-exported: VERB_SUFFIXES, NOUN_SUFFIXES, ADVERB_SUFFIXES
# (imported at top of module from core.constants.core_constants)

# =============================================================================
# SECTION 5: PARTICLE TYPO PATTERNS
# Common typing errors and their corrections
# Format: {typo: (correction, description, confidence)}
# =============================================================================

PARTICLE_TYPO_PATTERNS: dict[str, tuple[str, str, float]] = {
    # Missing asat patterns
    "တယ": ("တယ်", "Statement ending - missing asat", 0.95),
    "သည": ("သည်", "Formal statement - missing asat", 0.95),
    "မယ": ("မယ်", "Future tense - missing asat", 0.95),
    # Note: "ပြီ" was intentionally removed. "ပြီ" (perfective completion, "already done")
    # and "ပြီး" (sequential conjunction, "after doing") are DISTINCT particles.
    # "ပြီ" is NOT a typo of "ပြီး". See SENTENCE_PARTICLES, VERB_PARTICLES above.
    "ခဲ": ("ခဲ့", "Past tense - missing tone", 0.90),
    "နဲ": ("နဲ့", "With - missing tone", 0.90),
    "ပါတယ": ("ပါတယ်", "Polite statement - missing asat", 0.95),
    "ပါသည": ("ပါသည်", "Formal polite - missing asat", 0.95),
    "ရဲ": ("ရဲ့", "Possessive - missing tone", 0.90),
    "ဘူ": ("ဘူး", "Negative ending - missing tone", 0.90),
    # Ha-htoe missing
    # NOTE: "မာ" (hard/arrogant) removed — it's a valid word, not a typo of "မှာ".
    # NOTE: "နင်" (you, pronoun) removed — it's a valid pronoun, not a typo of "နှင်".
    # Subject/object marker confusions
    "ဂ": ("က", "Subject marker - wrong consonant", 0.95),
    "ဂို": ("ကို", "Object marker - wrong consonant", 0.95),
    # Question particles
    "လာ့": ("လား", "Question - wrong form", 0.85),
    "သလာ": ("သလား", "Question - missing tone", 0.90),
    # Emphasis particles
    "တော": ("တော့", "Emphasis - missing asat", 0.90),
    "ပဲ့": ("ပဲ", "Emphasis - extra tone", 0.85),
}

# =============================================================================
# SECTION 6: MEDIAL CONFUSION PATTERNS
# Common ျ (Ya-pin) vs ြ (Ya-yit) confusions
# Format: {typo: (correction, description, context)}
#
# These are common typing errors due to:
# - Visual similarity in some fonts
# - Adjacent keyboard positions
# - Historical dialect mergers
# =============================================================================

# Medial order corrections per UTN #11 canonical ordering
# Correct order: Ya (ျ) < Ra (ြ) < Wa (ွ) < Ha (ှ)
MEDIAL_ORDER_CORRECTIONS: dict[str, str] = {
    "ြျ": "ျြ",  # Ra+Ya → Ya+Ra
    "ွျ": "ျွ",  # Wa+Ya → Ya+Wa
    "ှျ": "ျှ",  # Ha+Ya → Ya+Ha
    "ွြ": "ြွ",  # Wa+Ra → Ra+Wa
    "ှြ": "ြှ",  # Ha+Ra → Ra+Ha
    "ှွ": "ွှ",  # Ha+Wa → Wa+Ha
}

# Aspiration confusions (aspirated vs unaspirated with medials)
ASPIRATION_MEDIAL_CONFUSIONS: dict[str, tuple[str, str]] = {
    # Ka-series
    "ချ": ("ကျ", "aspiration confusion"),
    "ဂျ": ("ကျ", "voicing confusion"),
    # Ca-series
    "ဆျ": ("စျ", "aspiration confusion"),
    "ဇျ": ("စျ", "voicing confusion"),
    # Ta-series
    "ထျ": ("တျ", "aspiration confusion"),
    "ဒျ": ("တျ", "voicing confusion"),
    # Pa-series
    "ဖျ": ("ပျ", "aspiration confusion"),
    # NOTE: "ဗျ" removed — it's valid in polite forms like ဗျာ, ခင်ဗျား.
}

# Word-level medial confusions (context-dependent)
# Format: {typo: (correction, description, context)}
MEDIAL_CONFUSION_PATTERNS: dict[str, tuple[str, str, str]] = {
    # High-frequency word corrections
    # These are common real-world typos
    "ကြေးဇူး": ("ကျေးဇူး", "thanks - Ra→Ya", "any"),
    "ကျည်": ("ကြည့်", "bullet→look - Ya→Ra", "after_verb"),
    "ကျွေး": ("ကြွေး", "feed→owe - context needed", "context_dependent"),
    "ကြွေး": ("ကျွေး", "owe→feed - context needed", "context_dependent"),
    # Note: "ပျော်" (happy) was intentionally removed. "ပျော်" is correctly spelled.
    # "ပြော်" (with ya-yit + asat) is NOT the verb "to say" (which is "ပြော" without asat).
    # These are different words with different vowels/tone marks, not a medial confusion.
    "ကြောက်": ("ကျောက်", "fear→stone - context needed", "context_dependent"),
    "ကျောက်": ("ကြောက်", "stone→fear - context needed", "context_dependent"),
    "ကြောင်": ("ကျောင်း", "because→school - common typo", "any"),
    "ငျက်": ("ငှက်", "bird - Ya→Ha on Nga", "any"),
    "ကွောင်း": ("ကျောင်း", "non-word→school - Wa→Ya", "any"),
}

# =============================================================================
# SECTION 7: INVALID POS SEQUENCES
# Grammatically incorrect POS tag sequences
# =============================================================================

INVALID_POS_SEQUENCES: dict[tuple[str, str], tuple[str, str]] = {
    ("V", "V"): ("info", "Consecutive verbs (Serial Verb Construction?)"),
    ("P", "P"): ("error", "Consecutive particles"),
    ("N", "N"): ("warning", "May need particle between nouns"),
}

# =============================================================================
# SECTION 8: QUESTION WORDS AND PARTICLES
# Words and particles that indicate interrogative sentences
# =============================================================================

QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "ဘာ",  # What
        "ဘယ်",  # Which/where
        "ဘယ်လို",  # How
        "ဘယ်သူ",  # Who
        "ဘယ်အချိန်",  # When
        "အဘယ်ကြောင့်",  # Why (formal)
        "ဘာကြောင့်",  # Why (colloquial)
        "ဘယ်နှစ်",  # How many
        "ဘယ်လောက်",  # How much
        "ဘယ်မှာ",  # Where
        "ဘယ်ကို",  # Where to
        "ဘယ်က",  # Where from
        "ဘယ်တုန်းက",  # Since when
        "ဘယ်တော့",  # When (future)
        "ဘာလို့",  # Why (reason)
    }
)

# Question particles (sentence-ending particles for questions)
QUESTION_PARTICLES: frozenset[str] = frozenset(
    {
        "လား",  # Yes/no question (ပြောသလား - did you say?)
        "သလား",  # Yes/no question (alternative form)
        "လဲ",  # Wh-question (ဘာလဲ - what is it?)
        "မလဲ",  # Future wh-question (ဘာလုပ်မလဲ - what will you do?)
        "လိုက်လား",  # Resultative question
        "ရဲ့လား",  # Confirmation question
        "နော်",  # Tag question (soft)
        "ပဲလား",  # Exclusive question
        # Compound question particles
        "တုန်းလား",  # Since when? / Still? (ရှိတုန်းလား - is it still there?)
        "သေးလား",  # Yet question (စားပြီးပြီလား၊ မစားသေးလား - have you eaten yet?)
        "ပါလား",  # Polite question form (သွားပါလား - shall we go?)
        "မှာလား",  # Will/shall question (သွားမှာလား - will you go?)
    }
)

# Statement endings (subset of SENTENCE_PARTICLES used for question structure validation)
# When these appear at the end of a sentence with question words, suggest question particles
STATEMENT_ENDINGS: frozenset[str] = frozenset(
    {
        "တယ်",  # Statement (colloquial)
        "သည်",  # Statement (formal)
        "ပါတယ်",  # Polite statement (colloquial)
        "ပါသည်",  # Polite statement (formal)
        # Future markers
        "မယ်",  # Future (colloquial)
        "မည်",  # Future (formal)
        "ပါမယ်",  # Polite future (colloquial)
        "ပါမည်",  # Polite future (formal)
        # Completion markers
        "ပြီ",  # Completion (informal)
        "ပါပြီ",  # Polite completion
        "ပြီး",  # After completion
        # Negative endings
        "ဘူး",  # Negative ending
        # Note: "ဘဲ" was intentionally removed. It is a connective meaning "without"
        # (e.g., "မစားဘဲ" = "without eating"), not a sentence-final statement ending.
    }
)

# Negative indefinite words - question words + "မှ" suffix indicating negative indefinite
# These are NOT interrogative when followed by negative verb patterns (မ...ဘူး/ဘဲ)
# Example: "ဘယ်သူမှ မလာဘူး" = "Nobody came" (statement, not question)
NEGATIVE_INDEFINITE_WORDS: frozenset[str] = frozenset(
    {
        "ဘယ်သူမှ",  # anybody/nobody (negative context)
        "ဘာမှ",  # anything/nothing (negative context)
        "ဘယ်မှာမှ",  # anywhere/nowhere (negative context)
        "ဘယ်တော့မှ",  # anytime/never (negative context)
        "ဘယ်လိုမှ",  # anyway/no way (negative context)
        "ဘယ်နေရာမှ",  # any place/no place (negative context)
        "ဘယ်အရာမှ",  # any thing/no thing (negative context)
        "ဘယ်သူ့ကိုမှ",  # to anybody/nobody (negative context)
        "ဘာကိုမှ",  # to anything/nothing (negative context)
    }
)

# Negative endings that follow the negative prefix
NEGATIVE_ENDINGS: frozenset[str] = frozenset(
    {
        "ဘူး",  # Negative ending (colloquial)
        "ဘဲ",  # Negative alternative/without
        "ပါဘူး",  # Polite negative ending
    }
)

# Common question particle suggestions (ordered by frequency)
QUESTION_PARTICLE_SUGGESTIONS: tuple[str, ...] = ("လား", "သလား", "လဲ")

# Merged-token question cue prefixes (e.g., "ဘယ်သွား", "ဘာလုပ်").
# Used to detect question contexts when segmentation does not split question words.
QUESTION_MERGED_PREFIXES: tuple[str, ...] = (
    "ဘယ်အချိန်",
    "ဘယ်တော့",
    "ဘယ်သူ",
    "ဘယ်လို",
    "ဘယ်မှာ",
    "ဘယ်ကို",
    "ဘယ်က",
    "ဘာကြောင့်",
    "ဘာလို့",
    "ဘာလုပ်",
    "ဘယ်",
)

# Statement/question ending rewrite templates (suffix-only forms).
# Full-form candidates are composed by attaching these suffixes to the word stem.
QUESTION_ENDING_REWRITE_MAP: dict[str, tuple[str, ...]] = {
    "မယ်": ("မလဲ", "မလား"),
    "မည်": ("မလဲ", "မလား"),
    "ပါမယ်": ("ပါမလဲ", "ပါမလား"),
    "ပါမည်": ("ပါမလဲ", "ပါမလား"),
    "သည်": ("သလဲ", "သလား"),
    "ပါသည်": ("ပါသလဲ", "ပါသလား"),
    "တယ်": ("လဲ", "မလား", "လား"),
    "ပါတယ်": ("ပါလဲ", "ပါမလား"),
}

# Known malformed enclitic question endings.
MALFORMED_QUESTION_ENDINGS: tuple[str, ...] = (
    "ရဲ့လဲ",
    "ရဲလဲ",
)

# Modal/future cues for second-person implicit question handling.
# Note: "မယ်" and "မည်" were removed — they are neutral future declaratives,
# not modal markers. "မင်း ... မယ်" is a promise/statement, not a question.
# Only keep actual modal markers that strongly imply interrogative intent.
SECOND_PERSON_MODAL_FUTURE_MARKERS: frozenset[str] = frozenset(
    {
        "နိုင်",
        "ရမယ်",
        "ရမည်",
    }
)

SECOND_PERSON_FUTURE_TIME_WORDS: frozenset[str] = frozenset(
    {
        "မနက်ဖြန်",
        "မနက်ဖန်",
        "နောက်နေ့",
        "နောက်လ",
        "နောက်နှစ်",
        "အနာဂတ်",
    }
)

# Targeted concrete completions for known dangling endings.
DANGLING_COMPLETION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "အရမ်းကို": ("အရေးကြီးတယ်", "အရမ်းကို ကောင်းတယ်"),
    "တကယ်ကို": ("တကယ်ကို မှန်တယ်", "တကယ်ကို ကောင်းတယ်"),
    "က": ("ပါ", "လား"),
}

# =============================================================================
# SECTION 9: IMPLICIT QUESTION DETECTION
# Second-person pronouns + completive endings that imply a question
# =============================================================================

SECOND_PERSON_PRONOUNS: frozenset[str] = frozenset(
    {
        "မင်း",  # You (informal, equal/lower status)
        "နင်",  # You (informal, intimate/rude)
        "ခင်ဗျား",  # You (polite, male speaker)
        "ရှင်",  # You (polite, female speaker)
    }
)

# Completive endings that become questions with လား when addressed to 2nd person
COMPLETIVE_ENDINGS: frozenset[str] = frozenset(
    {
        "ပြီ",  # Completion (ပြီ → ပြီလား)
        "ပြီးပြီ",  # Already completed (ပြီးပြီ → ပြီးပြီလား)
        "ပါပြီ",  # Polite completion (ပါပြီ → ပါပြီလား)
    }
)


# Unified SFP set for punctuation detection
SENTENCE_FINAL_PARTICLES: frozenset[str] = (
    STATEMENT_ENDINGS
    | QUESTION_PARTICLES
    | frozenset(
        {
            "ပါ",  # polite imperative (သွားပါ။)
            "ကြ",  # plural imperative (သွားကြ။)
            "စေ",  # causative/optative (သွားစေ။)
            "ပါစေ",  # polite optative (ကျန်းမာပါစေ။)
        }
    )
)


def ends_with_sfp(text: str) -> tuple[str, int] | None:
    """Check if text ends with a sentence-final particle.

    Returns (matched_sfp, start_position) or None.
    Matches longest SFP first to avoid partial matches.
    """
    stripped = text.rstrip()
    if not stripped:
        return None
    for sfp in sorted(SENTENCE_FINAL_PARTICLES, key=len, reverse=True):
        if stripped.endswith(sfp):
            pos = len(stripped) - len(sfp)
            return (sfp, pos)
    return None


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Return unique items while preserving input order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def has_question_word_context(words: list[str]) -> bool:
    """
    Check whether a sentence has question-word cues, including merged-token forms.
    """
    for word in words:
        if word in QUESTION_WORDS:
            return True
        if word in NEGATIVE_INDEFINITE_WORDS:
            continue
        for prefix in QUESTION_MERGED_PREFIXES:
            if word == prefix or (len(word) > len(prefix) and word.startswith(prefix)):
                return True
    return False


def detect_malformed_question_ending(word: str) -> str | None:
    """Return malformed question ending suffix if present."""
    for ending in MALFORMED_QUESTION_ENDINGS:
        if word == ending or (len(word) > len(ending) and word.endswith(ending)):
            return ending
    return None


def find_statement_ending_for_question_rewrite(word: str) -> str | None:
    """Find the statement ending suffix that should be rewritten for question forms."""
    for ending in sorted(QUESTION_ENDING_REWRITE_MAP, key=len, reverse=True):
        if word == ending or (len(word) > len(ending) and word.endswith(ending)):
            return ending
    return None


def get_question_rewrite_suggestions(
    word: str,
    words: list[str] | None = None,
    *,
    prefer_yes_no: bool = False,
    phrase_first: bool = True,
) -> list[str]:
    """
    Build ordered rewrite suggestions for malformed/question statement endings.

    Returns suffix-level suggestions and, for enclitic forms, full-token rewrites.
    """
    if not word:
        return []

    context_words = words or []
    has_question_word = has_question_word_context(context_words)
    malformed = detect_malformed_question_ending(word)
    matched = malformed or find_statement_ending_for_question_rewrite(word)
    if not matched:
        return []

    if malformed:
        if has_question_word and not prefer_yes_no:
            suffix_candidates = ["လဲ", "ရဲ့လား", "လား"]
        else:
            suffix_candidates = ["ရဲ့လား", "လား", "လဲ"]
    else:
        suffix_candidates = list(
            QUESTION_ENDING_REWRITE_MAP.get(matched, QUESTION_PARTICLE_SUGGESTIONS)
        )
        if matched in {"တယ်", "ပါတယ်"}:
            if prefer_yes_no or not has_question_word:
                suffix_candidates = ["မလား", "လား", "လဲ"]
            else:
                suffix_candidates = ["လဲ", "မလား", "လား"]
        elif matched in {"မယ်", "မည်", "ပါမယ်", "ပါမည်"}:
            if prefer_yes_no or not has_question_word:
                suffix_candidates = ["မလား", "မလဲ"]
        elif matched in {"သည်", "ပါသည်"}:
            if prefer_yes_no or not has_question_word:
                suffix_candidates = ["သလား", "သလဲ"]

    # Standalone endings: suffix suggestions are the concrete replacements.
    if word == matched:
        return _dedupe_preserve_order(suffix_candidates)

    # Enclitic endings: provide full-token rewrites plus suffix-only fallbacks.
    stem = word[: -len(matched)]
    full_candidates = [stem + suffix for suffix in suffix_candidates]
    ordered = (
        full_candidates + suffix_candidates if phrase_first else suffix_candidates + full_candidates
    )
    return _dedupe_preserve_order(ordered)


def get_question_completion_suggestions(
    word: str,
    words: list[str] | None = None,
    *,
    prefer_yes_no: bool = False,
    phrase_first: bool = True,
) -> list[str]:
    """
    Get question completion suggestions with rewrite-first, fallback-second behavior.
    """
    context_words = words or []
    rewrites = get_question_rewrite_suggestions(
        word,
        context_words,
        prefer_yes_no=prefer_yes_no,
        phrase_first=phrase_first,
    )
    if rewrites:
        return rewrites

    if not word:
        return []

    has_question_word = has_question_word_context(context_words)
    if has_question_word and not prefer_yes_no:
        fallback = [f"{word}လဲ", f"{word}လား"]
    elif prefer_yes_no:
        fallback = [f"{word}မလား", f"{word}လား"]
    else:
        fallback = [f"{word}လား", f"{word}လဲ"]

    return _dedupe_preserve_order(fallback)


def is_second_person_modal_future_question(words: list[str]) -> bool:
    """
    Detect implicit second-person modal/future questions.
    """
    if len(words) < 2:
        return False

    has_second_person = any(word in SECOND_PERSON_PRONOUNS for word in words[:2])
    if not has_second_person:
        return False

    # Already explicit question form.
    for i in range(min(3, len(words))):
        idx = len(words) - 1 - i
        if is_question_particle(words[idx]) or has_enclitic_question_particle(words[idx]):
            return False

    # Must end in a rewriteable statement ending.
    if find_statement_ending_for_question_rewrite(words[-1]) is None:
        return False

    has_modal_future = any(
        marker in word for word in words for marker in SECOND_PERSON_MODAL_FUTURE_MARKERS
    ) or any(word in SECOND_PERSON_FUTURE_TIME_WORDS for word in words)

    return has_modal_future


def get_dangling_completion_suggestions(word: str, *, enabled: bool | None = None) -> list[str]:
    """Get concrete completion suggestions for dangling sentence-final endings."""
    flag = enabled if enabled is not None else True
    if flag and word in DANGLING_COMPLETION_TEMPLATES:
        return list(DANGLING_COMPLETION_TEMPLATES[word])
    return ["ပါ", "ပြည့်စုံအောင် ဆက်ရေးပါ"]


def has_dangling_completion_template(word: str, *, enabled: bool | None = None) -> bool:
    """Check whether a token has a targeted dangling-completion template."""
    flag = enabled if enabled is not None else True
    return flag and word in DANGLING_COMPLETION_TEMPLATES


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def is_sentence_particle(word: str) -> bool:
    """Check if word is a sentence-ending particle."""
    return word in SENTENCE_PARTICLES


def is_verb_particle(word: str) -> bool:
    """Check if word is a verb-following particle."""
    return word in VERB_PARTICLES


def is_noun_particle(word: str) -> bool:
    """Check if word is a noun-following particle."""
    return word in NOUN_PARTICLES


def is_question_word(word: str) -> bool:
    """Check if word is a question/interrogative word."""
    return word in QUESTION_WORDS


def is_question_particle(word: str) -> bool:
    """Check if word is a question-ending particle."""
    return word in QUESTION_PARTICLES


def has_enclitic_question_particle(word: str) -> bool:
    """
    Check if word ends with a question particle (enclitic form).

    In Myanmar, question particles often attach directly to verbs without spaces.
    For example:
    - "သွားလား" (go+question = did [you] go?)
    - "လာမလား" (come+question = will [you] come?)
    - "ဘာလုပ်မလဲ" (what+do+future+question = what will [you] do?)

    This function detects these enclitic question particles that are not
    separated from the verb by spaces.

    Args:
        word: The word to check for enclitic question particles.

    Returns:
        True if word ends with a question particle, False otherwise.
    """
    if not word:
        return False

    # Malformed endings (e.g., ရဲ့လဲ) should be corrected, not treated as valid.
    if detect_malformed_question_ending(word):
        return False

    # Check standalone particles first (optimization)
    if word in QUESTION_PARTICLES:
        return True

    # Check for enclitic question particles (attached to verb)
    for particle in QUESTION_PARTICLES:
        if word.endswith(particle) and len(word) > len(particle):
            return True

    return False


def has_question_particle_context(
    words: list[str],
    *,
    has_question_word: bool | None = None,
    tail_window: int = 3,
) -> bool:
    """
    Check whether sentence context already contains a valid question particle.

    Question-word sentences may place the particle before final polite/request
    markers (e.g., ``ဘာ လဲ ... ပါ ဦး``), so scan the full sentence in that case.
    Yes/no patterns without question words remain constrained to tail tokens.
    """
    if not words:
        return False

    if has_question_word is None:
        has_question_word = has_question_word_context(words)

    search_words = words if has_question_word else words[-max(1, tail_window) :]
    return any(
        is_question_particle(word) or has_enclitic_question_particle(word) for word in search_words
    )


def detect_sentence_type(words: list[str]) -> str:
    """
    Detect the type of sentence based on question words and particles.

    Handles negative indefinite constructions correctly:
    - "ဘယ်သူမှ မလာဘူး" = "Nobody came" (STATEMENT, not question)
    - "ဘာမှ မသိဘူး" = "I don't know anything" (STATEMENT, not question)

    Args:
        words: List of words in the sentence.

    Returns:
        Sentence type: "question", "statement", or "unknown".
    """
    if not words:
        return "unknown"

    # Check for question particles at the end (including enclitic forms).
    for i in range(min(3, len(words))):
        idx = len(words) - 1 - i
        word = words[idx]
        if is_question_particle(word) or has_enclitic_question_particle(word):
            return "question"

    # Check for negative indefinite pattern BEFORE checking question words
    # Negative indefinite = question word + "မှ" suffix + negative verb pattern
    has_negative_indefinite = any(word in NEGATIVE_INDEFINITE_WORDS for word in words)
    has_negative_ending = any(word in NEGATIVE_ENDINGS for word in words)

    if has_negative_indefinite and has_negative_ending:
        # This is a negative indefinite statement like "Nobody came"
        return "statement"

    # Check for explicit and merged-token question-word contexts.
    if has_question_word_context(words):
        return "question"

    return "statement"


def get_particle_typo_correction(word: str) -> tuple[str, str, float] | None:
    """
    Get correction for a particle typo.

    Args:
        word: The word to check.

    Returns:
        Tuple of (correction, description, confidence) or None.
    """
    return PARTICLE_TYPO_PATTERNS.get(word)


def get_medial_confusion_correction(word: str) -> tuple[str, str, str] | None:
    """
    Get correction for medial confusion.

    Args:
        word: The word to check.

    Returns:
        Tuple of (correction, description, context) or None.
    """
    return MEDIAL_CONFUSION_PATTERNS.get(word)


def get_medial_order_correction(text: str) -> str | None:
    """
    Get correction for wrong medial order per UTN #11.

    Args:
        text: Text to check for medial order errors.

    Returns:
        Corrected text or None if no correction needed.
    """
    corrected = text
    found_error = False
    for wrong, correct in MEDIAL_ORDER_CORRECTIONS.items():
        if wrong in corrected:
            corrected = corrected.replace(wrong, correct)
            found_error = True
    return corrected if found_error else None


def get_aspiration_confusion(syllable: str) -> tuple[str, str] | None:
    """
    Get correction for aspiration confusion in medials.

    Args:
        syllable: Syllable to check.

    Returns:
        Tuple of (correction, description) or None.
    """
    return ASPIRATION_MEDIAL_CONFUSIONS.get(syllable)


# =============================================================================
# SECTION 9: RHYME REDUPLICATION PATTERNS
# Common Myanmar reduplication patterns (အသံထပ်ကွဲ) for grammar validation
# These are valid compound constructions that should not be flagged as errors
# =============================================================================

# Rhyme reduplication patterns (AB-AC or AB-CB forms where B rhymes with C)
# Format: frozenset of common reduplication patterns
RHYME_REDUPLICATION_PATTERNS: frozenset[str] = frozenset(
    {
        # AB-AC type (same first syllable, rhyming final syllables)
        "ရှုပ်ရှက်",  # disorderly (rhyme: -up/-ek)
        "ချိုချဉ်",  # sour-sweet (rhyme: -o/-in)
        "ပူပင်",  # worried (rhyme: -u/-in)
        "ချောချော",  # smoothly (reduplication)
        "လှလှပပ",  # beautifully (compound reduplication)
        "သာသာယာယာ",  # comfortably
        "မြင့်မြင့်မားမား",  # prominently
        "ရိုင်းရိုင်းစိုင်းစိုင်း",  # rudely
        "တောက်တောက်ပပ",  # sparkling
        "ချိုချိုသာသာ",  # sweetly and softly
        "ပျော်ပျော်ရွှင်ရွှင်",  # happily
        # AB-CB type (same final syllable, different first syllables)
        "ရိုးရိုးရှင်းရှင်း",  # simple
        "ရှင်းရှင်းလင်းလင်း",  # clearly
        "တည်တည်ငြိမ်ငြိမ်",  # calmly
        "ညောင့်ညောင့်နောက်နောက်",  # wobbly
        "သန့်သန့်ရှင်းရှင်း",  # clean
        # ABAB type (exact reduplication)
        "ဖြေးဖြေး",  # slowly
        "မြန်မြန်",  # quickly
        "ကောင်းကောင်း",  # well
        "ကြီးကြီး",  # big (adj intensifier)
        "သေးသေး",  # small (adj intensifier)
        "နှေးနှေး",  # slowly
        "သာသာ",  # softly
        "လှလှ",  # beautifully
        "ရှည်ရှည်",  # long (adj intensifier)
        "တိုတို",  # short (adj intensifier)
    }
)


# =============================================================================
# SECTION 10: AUXILIARY VERB PATTERNS
# Common Myanmar auxiliary verbs that modify main verbs
# These form valid V+AUX sequences and should not be flagged as V+V errors
# =============================================================================

# Primary auxiliary verbs (frequently used to modify main verbs)
AUXILIARY_VERBS: frozenset[str] = frozenset(
    {
        # Aspect markers (indicate state/completion)
        "နေ",  # Progressive (ပြောနေတယ် - is speaking)
        "ထား",  # Resultative/have done (လုပ်ထားတယ် - has done)
        "ပြီး",  # After completion (စားပြီး - after eating)
        "ပြီ",  # Completion (perfective)
        "ခဲ့",  # Past tense (လုပ်ခဲ့တယ် - did)
        "ဖူး",  # Experiential (စားဖူးတယ် - have eaten before)
        "သေး",  # Not-yet/still (မစားသေးဘူး - haven't eaten yet)
        "တော့",  # Imminent/about to (သွားတော့မယ် - about to go)
        "ဆဲ",  # Continuative/still (လုပ်နေဆဲ - still doing)
        # Direction/manner markers
        "လိုက်",  # Along with/quickly (သွားလိုက်တယ် - went suddenly)
        "သွား",  # Away (ပြောင်းသွားတယ် - changed away)
        "လာ",  # Coming/toward (ပြောင်းလာတယ် - coming to change)
        "ချ",  # Down (ဆင်းချတယ် - go down)
        "တက်",  # Up (တက်သွားတယ် - went up)
        "ဝင်",  # Enter (ဝင်လာတယ် - come in)
        "ထွက်",  # Exit (ထွက်သွားတယ် - go out)
        "ရောက်",  # Arrive (ရောက်ရှိတယ် - arrive at)
        "ကျ",  # Fall/descend (ကျလာတယ် - fell down)
        # Manner/result markers
        "မိ",  # Accidentally (ကျမိတယ် - fell accidentally)
        "ပစ်",  # Discard/away (ပစ်ချတယ် - throw down)
        "ခံ",  # Passive/endure (ဒဏ်ခံတယ် - suffer)
        "ယူ",  # Take/acquire (သယ်ယူတယ် - carry)
        "ထည့်",  # Put in (ထည့်သွင်းတယ် - insert)
        "ထုတ်",  # Out (ထုတ်ပြတယ် - bring out)
        "လွန်",  # Exceed (များလွန်းတယ် - too many)
        "ပြတ်",  # Sudden/complete (ပြတ်သွားတယ် - snapped)
        "ကုန်",  # Completive/all used up (စားကုန်ပြီ - ate it all)
        "အောင်",  # Until success (ကြိုးစားအောင်မြင် - strive to succeed)
        "ဖြစ်",  # Become/copula (ပြောင်းလဲဖြစ်လာ - became changed)
        # Benefactive/causative
        "ပေး",  # For someone (ကူညီပေးတယ် - helped for someone)
        "စေ",  # Causative (လုပ်စေတယ် - made do)
        "ခိုင်း",  # Order/delegate (ခိုင်းစေတယ် - order to do)
        "ပြ",  # Show by doing (လုပ်ပြတယ် - demonstrated)
        # Modal auxiliaries
        "ရ",  # Permission/can (စားရတယ် - can eat)
        "နိုင်",  # Ability (လုပ်နိုင်တယ် - able to do)
        "ချင်",  # Want (စားချင်တယ် - want to eat)
        "တတ်",  # Know how (ရေးတတ်တယ် - know how to write)
        "ရဲ",  # Dare (ပြောရဲတယ် - dare to say)
        "အပ်",  # Obligation (လုပ်အပ်တယ် - ought to do)
        "ဖို့",  # Purpose (လုပ်ဖို့ - in order to do)
        "သင့်",  # Should/advisory (လုပ်သင့်တယ် - should do)
        "လို",  # Need/want (လုပ်လိုတယ် - need to do)
        "ထိုက်",  # Deserve (ခံထိုက်တယ် - deserves)
        "ခင်",  # Before (temporal) (မသွားခင် - before going)
        # Attempt/effort/sequence
        "ကြည့်",  # Try (လုပ်ကြည့်တယ် - try doing)
        "ကြ",  # Plural action (သွားကြတယ် - they go)
        "စမ်း",  # Try-imperative (လုပ်စမ်း - try doing!)
        "ဦး",  # First (စားဦး - eat first)
        # Repetition/habitual
        "ပြန်",  # Again (ပြောပြန်တယ် - say again)
        "လေ့",  # Habitually (သွားလေ့ရှိတယ် - usually go)
    }
)

# V+AUX sequences that are valid (main_verb + auxiliary)
# These should NOT be flagged as invalid V+V sequences
VALID_VERB_AUXILIARY_SEQUENCES: frozenset[tuple[str, str]] = frozenset(
    {
        # Common verb + auxiliary patterns
        ("ပြော", "နေ"),  # is speaking
        ("လုပ်", "ထား"),  # has done
        ("သွား", "လိုက်"),  # went quickly
        ("ပြောင်း", "သွား"),  # changed away
        ("ပြောင်း", "လာ"),  # coming to change
        ("ကူညီ", "ပေး"),  # helped for someone
        ("စား", "ရ"),  # can eat
        ("လုပ်", "နိုင်"),  # able to do
        ("စား", "ချင်"),  # want to eat
        ("ရေး", "တတ်"),  # know how to write
        ("လုပ်", "ကြည့်"),  # try doing
        ("ပြော", "ပြန်"),  # say again
    }
)


def is_reduplication(word: str) -> bool:
    """
    Check if word is a valid rhyme reduplication pattern.

    Args:
        word: The word to check.

    Returns:
        True if word is a known reduplication pattern.
    """
    return word in RHYME_REDUPLICATION_PATTERNS


def detect_reduplication_pattern(syllables: list) -> str:
    """
    Detect reduplication pattern type from a list of syllables.

    Myanmar reduplication (ထပ်ဆင့်) types:
    - AABB: Syllable-level reduplication, each syllable doubles (သေသေချာချာ)
    - ABAB: Word-level reduplication, the whole unit repeats (ခဏခဏ)
    - AB: Simple reduplication of single syllable (လှလှ, ကြီးကြီး)
    - NONE: No reduplication detected

    Args:
        syllables: List of syllables to analyze.

    Returns:
        Reduplication pattern type: 'AABB', 'ABAB', 'AB', or 'NONE'.

    Example:
        >>> detect_reduplication_pattern(['သေ', 'သေ', 'ချာ', 'ချာ'])
        'AABB'
        >>> detect_reduplication_pattern(['ခ', 'ဏ', 'ခ', 'ဏ'])
        'ABAB'
    """
    n = len(syllables)

    if n < 2:
        return "NONE"

    # AB pattern: simple reduplication (AA)
    if n == 2 and syllables[0] == syllables[1]:
        return "AB"

    # AABB pattern: each syllable doubles individually (A-A-B-B)
    # Example: သေသေချာချာ = [သေ, သေ, ချာ, ချာ]
    if n == 4:
        if syllables[0] == syllables[1] and syllables[2] == syllables[3]:
            if syllables[0] != syllables[2]:
                return "AABB"
        # ABAB pattern: whole disyllabic unit repeats (AB-AB)
        # Example: ခဏခဏ = [ခ, ဏ, ခ, ဏ]
        if syllables[0] == syllables[2] and syllables[1] == syllables[3]:
            if syllables[0] != syllables[1]:
                return "ABAB"

    # 6-syllable patterns: AAABBB (extended AABB)
    if n == 6:
        if (
            syllables[0] == syllables[1] == syllables[2]
            and syllables[3] == syllables[4] == syllables[5]
            and syllables[0] != syllables[3]
        ):
            return "AABB"  # AAA-BBB treated as extended AABB

    return "NONE"


def is_valid_verb_sequence(verb1: str, verb2: str) -> bool:
    """
    Check if a verb-verb sequence is valid (V+AUX pattern).

    This prevents false positives when flagging V+V as an error.

    Args:
        verb1: First verb (main verb).
        verb2: Second verb (potential auxiliary).

    Returns:
        True if verb2 is an auxiliary verb that can follow verb1.
    """
    # If verb2 is an auxiliary verb, the sequence is valid
    if verb2 in AUXILIARY_VERBS:
        return True
    # Check for specific known valid patterns
    return (verb1, verb2) in VALID_VERB_AUXILIARY_SEQUENCES


# =============================================================================
# ALL PARTICLES (combined for quick membership testing)
# =============================================================================

ALL_PARTICLES: frozenset[str] = SENTENCE_PARTICLES | VERB_PARTICLES | NOUN_PARTICLES
