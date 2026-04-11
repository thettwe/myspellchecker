"""
Core Application Constants.

This module contains core application constants including:
- Enums (ValidationLevel, ErrorType)
- Database and file handling constants
- Data schema and statistics keys
- CSV headers and separators
- POS tag constants
- Morphology suffixes
"""

from enum import Enum

# =============================================================================
# Enums
# =============================================================================


class ActionType(str, Enum):
    """
    Recommended action for a detected error.

    Guides consumers (webapp, API, IDE) on how to present each correction.

    - AUTO_FIX: High-confidence structural error — safe to apply silently.
      Examples: encoding normalization, illegal syllable repair, obvious typos
      where only one correction is possible.

    - SUGGEST: Genuine error but correction needs user confirmation.
      Examples: confusable words, homophone choices, context-dependent fixes.

    - INFORM: Advisory note, not a hard error. Do not auto-apply.
      Examples: colloquial variants in formal text, register mixing suggestions.
    """

    AUTO_FIX = "auto_fix"
    SUGGEST = "suggest"
    INFORM = "inform"


class ValidationLevel(str, Enum):
    """
    Defines the depth of spell checking validation.
    """

    SYLLABLE = "syllable"  # Fast, only checks valid syllables
    WORD = "word"  # Comprehensive, checks words and context


class ErrorType(str, Enum):
    """
    Defines specific types of detected errors.

    All error types should be defined here to ensure consistency across modules.
    String literals should not be used directly in error_type parameters.
    """

    # Note: Syllable and Word errors are often just typed by their class,
    # but these constants can be used for serialization or explicit tagging.
    SYLLABLE = "invalid_syllable"
    WORD = "invalid_word"

    # Context errors have specific subtypes
    CONTEXT_PROBABILITY = "context_probability"

    # Grammar errors from grammar checkers
    GRAMMAR = "grammar_error"

    # Specific error types for validators
    PARTICLE_TYPO = "particle_typo"
    MEDIAL_CONFUSION = "medial_confusion"

    # Colloquial variant error types
    COLLOQUIAL_VARIANT = "colloquial_variant"  # Strict mode: flag as error
    COLLOQUIAL_INFO = "colloquial_info"  # Lenient mode: info note

    # Validation strategy error types
    QUESTION_STRUCTURE = "question_structure"
    SYNTAX_ERROR = "syntax_error"
    HOMOPHONE_ERROR = "homophone_error"
    TONE_AMBIGUITY = "tone_ambiguity"
    POS_SEQUENCE_ERROR = "pos_sequence_error"
    SEMANTIC_ERROR = "semantic_error"

    # Encoding error types
    ZAWGYI_ENCODING = "zawgyi_encoding"

    # Grammar checker specific error types
    MIXED_REGISTER = "mixed_register"
    ASPECT_TYPO = "aspect_typo"
    INVALID_SEQUENCE = "invalid_sequence"
    INCOMPLETE_ASPECT = "incomplete_aspect"
    TYPO = "typo"
    AGREEMENT = "agreement"
    COMPOUND_TYPO = "compound_typo"
    INCOMPLETE_REDUPLICATION = "incomplete_reduplication"
    CLASSIFIER_TYPO = "classifier_typo"

    # Text-level detector error types (spellchecker.py)
    COLLOQUIAL_CONTRACTION = "colloquial_contraction"
    PARTICLE_CONFUSION = "particle_confusion"
    HA_HTOE_CONFUSION = "ha_htoe_confusion"
    DANGLING_PARTICLE = "dangling_particle"
    DANGLING_WORD = "dangling_word"
    MISSING_CONJUNCTION = "missing_conjunction"
    TENSE_MISMATCH = "tense_mismatch"
    REGISTER_MIXING = "register_mixing"

    # Grammar checker class-level error types
    NEGATION_ERROR = "negation_error"
    REGISTER_ERROR = "register_error"
    MERGED_WORD = "merged_word"
    ASPECT_ERROR = "aspect_error"
    CLASSIFIER_ERROR = "classifier_error"
    COMPOUND_ERROR = "compound_error"

    # Orthography error types
    MEDIAL_ORDER_ERROR = "medial_order_error"
    MEDIAL_COMPATIBILITY_ERROR = "medial_compatibility_error"
    VOWEL_AFTER_ASAT = "vowel_after_asat"
    BROKEN_VIRAMA = "broken_virama"

    # Confusable variant error (valid word, wrong in context)
    CONFUSABLE_ERROR = "confusable_error"

    # Broken stacking (asat instead of virama before stacked consonant)
    BROKEN_STACKING = "broken_stacking"

    # Broken compound (wrongly split compound word)
    BROKEN_COMPOUND = "broken_compound"

    # Hidden compound typo (compound where one morpheme is a confusable typo
    # but the segmenter already split it into individually-valid tokens)
    HIDDEN_COMPOUND_TYPO = "hidden_compound_typo"

    # Syllable-window OOV (multi-syllable compound typo recovered by joining
    # adjacent syllable windows, checking OOV status, and consulting SymSpell
    # for a high-frequency near-match). Sprint I-1 — detection for FNs in
    # zero-error ("clean") sentences where the segmenter over-splits a typo
    # into individually-valid syllables.
    SYLLABLE_WINDOW_OOV = "syllable_window_oov"

    # Leading vowel-e (Zawgyi-style ေ before consonant)
    LEADING_VOWEL_E = "leading_vowel_e"

    # Incomplete stacking (virama present but stacked consonant missing)
    INCOMPLETE_STACKING = "incomplete_stacking"

    # Negation with affirmative SFP (မ-V-တယ် instead of မ-V-ဘူး)
    NEGATION_SFP_MISMATCH = "negation_sfp_mismatch"

    # Merged SFP + conjunction (တယ်ပြီး instead of ပြီး)
    MERGED_SFP_CONJUNCTION = "merged_sfp_conjunction"

    # Aspect-adverb conflict (habitual adverb + specific-past marker)
    ASPECT_ADVERB_CONFLICT = "aspect_adverb_conflict"

    # Missing asat (detected by text-level detector, not syllable merge)
    MISSING_ASAT = "missing_asat"

    # Particle misuse (semantically wrong particle for verb frame)
    PARTICLE_MISUSE = "particle_misuse"

    # Collocation error (wrong word partner in fixed phrase)
    COLLOCATION_ERROR = "collocation_error"

    # Punctuation error types
    DUPLICATE_PUNCTUATION = "duplicate_punctuation"
    WRONG_PUNCTUATION = "wrong_punctuation"
    MISSING_PUNCTUATION = "missing_punctuation"


# =============================================================================
# Pre-cached ErrorType values (plain strings)
#
# Accessing ErrorType.X.value on every call triggers the descriptor protocol
# (__get__) each time.  On hot paths (per-word validation loops) this adds up
# to millions of redundant enum lookups.  The constants below are resolved once
# at import time so call sites pay zero overhead.
# =============================================================================

ET_SYLLABLE: str = ErrorType.SYLLABLE.value
ET_WORD: str = ErrorType.WORD.value
ET_CONTEXT_PROBABILITY: str = ErrorType.CONTEXT_PROBABILITY.value
ET_GRAMMAR: str = ErrorType.GRAMMAR.value
ET_PARTICLE_TYPO: str = ErrorType.PARTICLE_TYPO.value
ET_MEDIAL_CONFUSION: str = ErrorType.MEDIAL_CONFUSION.value
ET_COLLOQUIAL_VARIANT: str = ErrorType.COLLOQUIAL_VARIANT.value
ET_COLLOQUIAL_INFO: str = ErrorType.COLLOQUIAL_INFO.value
ET_QUESTION_STRUCTURE: str = ErrorType.QUESTION_STRUCTURE.value
ET_SYNTAX_ERROR: str = ErrorType.SYNTAX_ERROR.value
ET_HOMOPHONE_ERROR: str = ErrorType.HOMOPHONE_ERROR.value
ET_TONE_AMBIGUITY: str = ErrorType.TONE_AMBIGUITY.value
ET_POS_SEQUENCE_ERROR: str = ErrorType.POS_SEQUENCE_ERROR.value
ET_SEMANTIC_ERROR: str = ErrorType.SEMANTIC_ERROR.value
ET_ZAWGYI_ENCODING: str = ErrorType.ZAWGYI_ENCODING.value
ET_MIXED_REGISTER: str = ErrorType.MIXED_REGISTER.value
ET_ASPECT_TYPO: str = ErrorType.ASPECT_TYPO.value
ET_INVALID_SEQUENCE: str = ErrorType.INVALID_SEQUENCE.value
ET_INCOMPLETE_ASPECT: str = ErrorType.INCOMPLETE_ASPECT.value
ET_TYPO: str = ErrorType.TYPO.value
ET_AGREEMENT: str = ErrorType.AGREEMENT.value
ET_COMPOUND_TYPO: str = ErrorType.COMPOUND_TYPO.value
ET_INCOMPLETE_REDUPLICATION: str = ErrorType.INCOMPLETE_REDUPLICATION.value
ET_CLASSIFIER_TYPO: str = ErrorType.CLASSIFIER_TYPO.value
ET_COLLOQUIAL_CONTRACTION: str = ErrorType.COLLOQUIAL_CONTRACTION.value
ET_PARTICLE_CONFUSION: str = ErrorType.PARTICLE_CONFUSION.value
ET_HA_HTOE_CONFUSION: str = ErrorType.HA_HTOE_CONFUSION.value
ET_DANGLING_PARTICLE: str = ErrorType.DANGLING_PARTICLE.value
ET_DANGLING_WORD: str = ErrorType.DANGLING_WORD.value
ET_MISSING_CONJUNCTION: str = ErrorType.MISSING_CONJUNCTION.value
ET_TENSE_MISMATCH: str = ErrorType.TENSE_MISMATCH.value
ET_REGISTER_MIXING: str = ErrorType.REGISTER_MIXING.value
ET_NEGATION_ERROR: str = ErrorType.NEGATION_ERROR.value
ET_REGISTER_ERROR: str = ErrorType.REGISTER_ERROR.value
ET_MERGED_WORD: str = ErrorType.MERGED_WORD.value
ET_ASPECT_ERROR: str = ErrorType.ASPECT_ERROR.value
ET_CLASSIFIER_ERROR: str = ErrorType.CLASSIFIER_ERROR.value
ET_COMPOUND_ERROR: str = ErrorType.COMPOUND_ERROR.value
ET_MEDIAL_ORDER_ERROR: str = ErrorType.MEDIAL_ORDER_ERROR.value
ET_MEDIAL_COMPATIBILITY_ERROR: str = ErrorType.MEDIAL_COMPATIBILITY_ERROR.value
ET_VOWEL_AFTER_ASAT: str = ErrorType.VOWEL_AFTER_ASAT.value
ET_BROKEN_VIRAMA: str = ErrorType.BROKEN_VIRAMA.value
ET_CONFUSABLE_ERROR: str = ErrorType.CONFUSABLE_ERROR.value
ET_BROKEN_STACKING: str = ErrorType.BROKEN_STACKING.value
ET_BROKEN_COMPOUND: str = ErrorType.BROKEN_COMPOUND.value
ET_HIDDEN_COMPOUND_TYPO: str = ErrorType.HIDDEN_COMPOUND_TYPO.value
ET_SYLLABLE_WINDOW_OOV: str = ErrorType.SYLLABLE_WINDOW_OOV.value
ET_LEADING_VOWEL_E: str = ErrorType.LEADING_VOWEL_E.value
ET_INCOMPLETE_STACKING: str = ErrorType.INCOMPLETE_STACKING.value
ET_NEGATION_SFP_MISMATCH: str = ErrorType.NEGATION_SFP_MISMATCH.value
ET_MERGED_SFP_CONJUNCTION: str = ErrorType.MERGED_SFP_CONJUNCTION.value
ET_ASPECT_ADVERB_CONFLICT: str = ErrorType.ASPECT_ADVERB_CONFLICT.value
ET_MISSING_ASAT: str = ErrorType.MISSING_ASAT.value
ET_PARTICLE_MISUSE: str = ErrorType.PARTICLE_MISUSE.value
ET_COLLOCATION_ERROR: str = ErrorType.COLLOCATION_ERROR.value
ET_DUPLICATE_PUNCTUATION: str = ErrorType.DUPLICATE_PUNCTUATION.value
ET_WRONG_PUNCTUATION: str = ErrorType.WRONG_PUNCTUATION.value
ET_MISSING_PUNCTUATION: str = ErrorType.MISSING_PUNCTUATION.value

# =============================================================================
# Database Constants
# =============================================================================

DEFAULT_DB_NAME = "mySpellChecker-default.db"

# =============================================================================
# Segmenter Constants
# =============================================================================

SEGMENTER_ENGINE_MYWORD = "myword"
SEGMENTER_ENGINE_CRF = "crf"
SEGMENTER_ENGINE_TRANSFORMER = "transformer"

# =============================================================================
# Data Schema Constants
# =============================================================================

DATA_KEY_SYLLABLE = "syllable"
DATA_KEY_WORD = "word"
DATA_KEY_FREQUENCY = "frequency"
DATA_KEY_PROBABILITY = "probability"
DATA_KEY_SYLLABLE_COUNT = "syllable_count"
DATA_KEY_POS = "pos"
DATA_KEY_WORDS = "words"
DATA_KEY_SYLLABLES = "syllables"
DATA_KEY_BIGRAMS = "bigrams"

# =============================================================================
# Statistics Keys
# =============================================================================

STATS_KEY_SYLLABLE_COUNT = "syllable_count"
STATS_KEY_WORD_COUNT = "word_count"
STATS_KEY_BIGRAM_COUNT = "bigram_count"
STATS_KEY_TRIGRAM_COUNT = "trigram_count"
STATS_KEY_FOURGRAM_COUNT = "fourgram_count"
STATS_KEY_FIVEGRAM_COUNT = "fivegram_count"

# =============================================================================
# CSV Headers
# =============================================================================

CSV_HEADER_WORD1 = "word1"
CSV_HEADER_WORD2 = "word2"
CSV_HEADER_POS = "pos"

# =============================================================================
# File Handling Constants
# =============================================================================

DEFAULT_FILE_ENCODING = "utf-8"
BIGRAM_SEPARATOR = "|"

# =============================================================================
# Algorithm Constants
# =============================================================================

# Damerau-Levenshtein cache size (raised from 4096 — SymSpell generates
# 2M+ unique pairs per run; larger cache reduces recomputation)
DAMERAU_CACHE_SIZE = 16384

# =============================================================================
# Runtime SQLite PRAGMA Constants
# =============================================================================

# 512 MB page cache for runtime (negative = KiB)
RUNTIME_PRAGMA_CACHE_SIZE = -524288

# 2 GB memory-mapped I/O for runtime
RUNTIME_PRAGMA_MMAP_SIZE = 2147483648

# =============================================================================
# Frequency Guard Constants
# =============================================================================

# Minimum word frequency to consider a compound as lexicalized (not a
# segmentation artifact).  Used by register mixing, context probability
# suppression, and missing-diacritic detection.
LEXICALIZED_COMPOUND_MIN_FREQ = 1000

# =============================================================================
# Confidence Constants
# =============================================================================

# Floor for confidence values (prevents zero-confidence results)
CONFIDENCE_FLOOR = 0.1

# =============================================================================
# Schema Check Constants
# =============================================================================

# Timeout for one-off schema version check connection (seconds)
SCHEMA_CHECK_TIMEOUT = 5.0


# =============================================================================
# POS Tag Constants
# =============================================================================

# Granular Particle Tags (for detailed POS tagging)
P_SUBJ = "P_SUBJ"  # Subject/topic marker (က, ကား, ဟာ)
P_OBJ = "P_OBJ"  # Object marker (ကို, အား)
P_SENT = "P_SENT"  # Sentence ending particle (သည်, တယ်, မယ်, ပြီ, လား)
P_MOD = "P_MOD"  # Modifier particle (သော, တဲ့, နဲ့, လို)
P_LOC = "P_LOC"  # Location/direction marker (မှ, သို့, ဆီ, တွင်, ထဲ)

# =============================================================================
# Morphology Constants
# =============================================================================

# Suffixes for Morphology Analyzer (for OOV POS guessing)
# Canonical source — grammar/patterns.py imports from here.

VERB_SUFFIXES: frozenset[str] = frozenset(
    {
        # Core tense/aspect
        "ပြီ",
        "ပြီး",
        "ခဲ့",
        "နေ",
        "သွား",
        "နေသည်",
        # Extended tense/aspect
        "သည့်",
        "မည်",
        "နိုင်",
        "တတ်",
        "ရ",
        "အောင်",
        # Continuation/manner
        "လျက်",
        "ရင်",
        "လျှင်",
        # Plural/direction
        "ကြ",
        "လာ",
        "ထား",
        # Causative/benefactive
        "စေ",
        "ပေး",
    }
)

NOUN_SUFFIXES: frozenset[str] = frozenset(
    {
        # Plural markers
        "များ",  # Formal plural marker
        "တွေ",  # Colloquial plural marker
        # Nominalization
        "ခြင်း",
        "မှု",
        "ရေး",
        "ရာ",
        # Person/agent
        "သူ",
        "သား",
        "သမား",
        # Abstract/manner
        "ချက်",
        "ပုံ",
        "အချက်",
        "လုံး",
        "မျိုး",
        # Classifiers (often form compound nouns)
        "ယောက်",
        "ခု",
        "ဦး",
    }
)

ADVERB_SUFFIXES: frozenset[str] = frozenset(
    {
        "စွာ",
        "တိုင်း",
        "ပြန်",
        "လိုလို",
        "အလိုက်",
        "ချင်း",
        "လုံး",
        "တကူ",
    }
)
