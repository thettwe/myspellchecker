"""
Internationalization (i18n) support for error messages.

This module provides localized error messages for the Myanmar spell checker,
supporting both English and Myanmar language output.

Thread Safety:
    Language settings are stored in thread-local storage, so different threads
    can use different languages concurrently without interference.

Example:
    >>> from myspellchecker.core.i18n import get_message, set_language
    >>> set_language("my")  # Set to Myanmar
    >>> get_message("invalid_syllable")
    'စာလုံးပေါင်း မမှန်ကန်ပါ'
"""

from __future__ import annotations

import threading

from myspellchecker.core.constants.core_constants import ErrorType

__all__ = [
    "format_suggestion_message",
    "get_error_message",
    "get_language",
    "get_message",
    "get_supported_languages",
    "set_language",
]

# Thread-local storage for language setting
# This allows different threads to use different languages concurrently
_thread_local = threading.local()

# Default language (used when thread hasn't set a language)
_DEFAULT_LANGUAGE: str = "en"

# Message translations
# Keys follow the ErrorType values from constants.py
MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        # Error types (keyed by ErrorType values)
        ErrorType.SYLLABLE.value: "Invalid syllable",
        ErrorType.WORD.value: "Invalid word",
        ErrorType.CONTEXT_PROBABILITY.value: "Unlikely word sequence",
        ErrorType.GRAMMAR.value: "Grammar error",
        ErrorType.PARTICLE_TYPO.value: "Particle typo",
        ErrorType.MEDIAL_CONFUSION.value: "Medial character confusion",
        ErrorType.COLLOQUIAL_VARIANT.value: "Colloquial variant",
        ErrorType.COLLOQUIAL_INFO.value: "Colloquial usage note",
        ErrorType.QUESTION_STRUCTURE.value: "Question structure error",
        ErrorType.SYNTAX_ERROR.value: "Syntax error",
        ErrorType.HOMOPHONE_ERROR.value: "Homophone error",
        ErrorType.TONE_AMBIGUITY.value: "Tone mark ambiguity",
        ErrorType.POS_SEQUENCE_ERROR.value: "POS sequence error",
        ErrorType.SEMANTIC_ERROR.value: "Semantic error",
        ErrorType.ZAWGYI_ENCODING.value: "Zawgyi encoding detected",
        ErrorType.MIXED_REGISTER.value: "Mixed register",
        ErrorType.CONFUSABLE_ERROR.value: "Confusable word error",
        ErrorType.PARTICLE_CONFUSION.value: "Particle confusion",
        ErrorType.HA_HTOE_CONFUSION.value: "Ha-htoe confusion",
        ErrorType.DANGLING_PARTICLE.value: "Dangling particle",
        ErrorType.DANGLING_WORD.value: "Dangling word",
        ErrorType.REGISTER_MIXING.value: "Register mixing",
        ErrorType.BROKEN_VIRAMA.value: "Broken virama stacking",
        ErrorType.BROKEN_STACKING.value: "Broken stacking",
        ErrorType.BROKEN_COMPOUND.value: "Broken compound",
        # Grammar checker specific error types
        ErrorType.ASPECT_TYPO.value: "Aspect marker typo",
        ErrorType.INVALID_SEQUENCE.value: "Invalid POS sequence",
        ErrorType.INCOMPLETE_ASPECT.value: "Incomplete aspect marker",
        ErrorType.TYPO.value: "Typo",
        ErrorType.AGREEMENT.value: "Agreement error",
        ErrorType.COMPOUND_TYPO.value: "Compound word typo",
        ErrorType.INCOMPLETE_REDUPLICATION.value: "Incomplete reduplication",
        ErrorType.CLASSIFIER_TYPO.value: "Classifier typo",
        ErrorType.NEGATION_ERROR.value: "Negation error",
        ErrorType.REGISTER_ERROR.value: "Register error",
        ErrorType.MERGED_WORD.value: "Merged word error",
        ErrorType.ASPECT_ERROR.value: "Aspect marker error",
        ErrorType.CLASSIFIER_ERROR.value: "Classifier error",
        ErrorType.COMPOUND_ERROR.value: "Compound word error",
        # Text-level detector error types
        ErrorType.COLLOQUIAL_CONTRACTION.value: "Colloquial contraction",
        ErrorType.MISSING_CONJUNCTION.value: "Missing conjunction",
        ErrorType.TENSE_MISMATCH.value: "Tense mismatch",
        ErrorType.NEGATION_SFP_MISMATCH.value: "Negation with wrong sentence-final particle",
        ErrorType.MERGED_SFP_CONJUNCTION.value: "Merged sentence-final particle and conjunction",
        ErrorType.ASPECT_ADVERB_CONFLICT.value: "Aspect-adverb conflict",
        ErrorType.MISSING_ASAT.value: "Missing asat",
        ErrorType.PARTICLE_MISUSE.value: "Particle misuse",
        ErrorType.COLLOCATION_ERROR.value: "Collocation error",
        # Orthography error types
        ErrorType.MEDIAL_ORDER_ERROR.value: "Medial order error",
        ErrorType.MEDIAL_COMPATIBILITY_ERROR.value: "Medial compatibility error",
        ErrorType.VOWEL_AFTER_ASAT.value: "Vowel after asat",
        ErrorType.LEADING_VOWEL_E.value: "Leading vowel-e error",
        ErrorType.INCOMPLETE_STACKING.value: "Incomplete stacking",
        # Punctuation error types
        ErrorType.DUPLICATE_PUNCTUATION.value: "Duplicate punctuation",
        ErrorType.WRONG_PUNCTUATION.value: "Wrong punctuation",
        ErrorType.MISSING_PUNCTUATION.value: "Missing punctuation",
        # Legacy keys (kept for backward compatibility)
        "context_error": "Unlikely word sequence",
        "tone_error": "Tone mark error",
        "stacking_error": "Stacking consonant error",
        # Error descriptions
        "syllable_not_found": "Syllable not found in dictionary",
        "word_not_found": "Word not found in dictionary",
        "low_probability": "Low probability sequence",
        "missing_particle": "Missing required particle",
        "wrong_particle": "Incorrect particle usage",
        "ya_ra_confusion": "Ya-pin and Ya-yit confusion",
        "tone_mark_missing": "Missing tone mark",
        "tone_mark_wrong": "Incorrect tone mark",
        # Suggestions
        "did_you_mean": "Did you mean",
        "suggestions": "Suggestions",
        "no_suggestions": "No suggestions available",
        # Validation messages
        "text_valid": "Text is valid",
        "errors_found": "Errors found",
    },
    "my": {
        # Error types (keyed by ErrorType values)
        ErrorType.SYLLABLE.value: "စာလုံးပေါင်း မမှန်ကန်ပါ",
        ErrorType.WORD.value: "စကားလုံး မမှန်ကန်ပါ",
        ErrorType.CONTEXT_PROBABILITY.value: "စကားစပ် မသင့်လျော်ပါ",
        ErrorType.GRAMMAR.value: "သဒ္ဒါ အမှား",
        ErrorType.PARTICLE_TYPO.value: "ပစ္စည်း စာလုံးပေါင်းအမှား",
        ErrorType.MEDIAL_CONFUSION.value: "ယပင်/ရရစ် ရောထွေးမှု",
        ErrorType.COLLOQUIAL_VARIANT.value: "ပြောစကား ရေးထုံး",
        ErrorType.COLLOQUIAL_INFO.value: "ပြောစကား ရေးထုံး မှတ်ချက်",
        ErrorType.QUESTION_STRUCTURE.value: "မေးခွန်း ဖွဲ့စည်းပုံ အမှား",
        ErrorType.SYNTAX_ERROR.value: "ဝါကျဖွဲ့စည်းပုံ အမှား",
        ErrorType.HOMOPHONE_ERROR.value: "အသံတူ စာလုံးရောထွေးမှု",
        ErrorType.TONE_AMBIGUITY.value: "အသံပြသင်္ကေတ မရှင်းလင်းမှု",
        ErrorType.POS_SEQUENCE_ERROR.value: "ဝေါဟာရ အစီအစဉ် အမှား",
        ErrorType.SEMANTIC_ERROR.value: "အဓိပ္ပာယ် အမှား",
        ErrorType.ZAWGYI_ENCODING.value: "ဇော်ဂျီ ကုဒ်နံပါတ် တွေ့ရှိပါသည်",
        ErrorType.MIXED_REGISTER.value: "စကားလုံး အဆင့် ရောထွေးမှု",
        ErrorType.CONFUSABLE_ERROR.value: "ရောထွေးလွယ်သော စကားလုံး အမှား",
        ErrorType.PARTICLE_CONFUSION.value: "ပစ္စည်း ရောထွေးမှု",
        ErrorType.HA_HTOE_CONFUSION.value: "ဟထိုး ရောထွေးမှု",
        ErrorType.DANGLING_PARTICLE.value: "ပစ္စည်း ချွတ်ယွင်းနေပါသည်",
        ErrorType.DANGLING_WORD.value: "စကားလုံး ချွတ်ယွင်းနေပါသည်",
        ErrorType.REGISTER_MIXING.value: "စကားလုံး အဆင့် ရောထွေးမှု",
        ErrorType.BROKEN_VIRAMA.value: "ပါဋ်ဆင့် ပျက်စီးနေပါသည်",
        ErrorType.BROKEN_STACKING.value: "ပါဋ်ဆင့် ပျက်စီးနေပါသည်",
        ErrorType.BROKEN_COMPOUND.value: "ပေါင်းစပ်စကားလုံး ပျက်စီးနေပါသည်",
        # Grammar checker specific error types
        ErrorType.ASPECT_TYPO.value: "ကြိယာသရုပ်ပြ စာလုံးပေါင်းအမှား",
        ErrorType.INVALID_SEQUENCE.value: "ဝေါဟာရ အစီအစဉ် မမှန်ကန်ပါ",
        ErrorType.INCOMPLETE_ASPECT.value: "ကြိယာသရုပ်ပြ မပြည့်စုံပါ",
        ErrorType.TYPO.value: "စာလုံးပေါင်းအမှား",
        ErrorType.AGREEMENT.value: "သဒ္ဒါ ညီညွတ်မှု အမှား",
        ErrorType.COMPOUND_TYPO.value: "ပေါင်းစပ်စကားလုံး စာလုံးပေါင်းအမှား",
        ErrorType.INCOMPLETE_REDUPLICATION.value: "ထပ်ကာထပ်ကာ မပြည့်စုံပါ",
        ErrorType.CLASSIFIER_TYPO.value: "ရေတွက်ပုံ စာလုံးပေါင်းအမှား",
        ErrorType.NEGATION_ERROR.value: "ငြင်းဆိုပုံ အမှား",
        ErrorType.REGISTER_ERROR.value: "စကားလုံး အဆင့် အမှား",
        ErrorType.MERGED_WORD.value: "စကားလုံးများ ပေါင်းကပ်နေပါသည်",
        ErrorType.ASPECT_ERROR.value: "ကြိယာသရုပ်ပြ အမှား",
        ErrorType.CLASSIFIER_ERROR.value: "ရေတွက်ပုံ အမှား",
        ErrorType.COMPOUND_ERROR.value: "ပေါင်းစပ်စကားလုံး အမှား",
        # Text-level detector error types
        ErrorType.COLLOQUIAL_CONTRACTION.value: "ပြောစကား အတိုကောက်",
        ErrorType.MISSING_CONJUNCTION.value: "ဆက်စပ်မှု ပျောက်နေပါသည်",
        ErrorType.TENSE_MISMATCH.value: "ကာလ မကိုက်ညီပါ",
        ErrorType.NEGATION_SFP_MISMATCH.value: "ငြင်းဆို ဝါကျအဆုံးသတ် ပစ္စည်း မမှန်ကန်ပါ",
        ErrorType.MERGED_SFP_CONJUNCTION.value: "ဝါကျအဆုံးသတ်နှင့် ဆက်စပ်မှု ပေါင်းကပ်နေပါသည်",
        ErrorType.ASPECT_ADVERB_CONFLICT.value: "ကြိယာသရုပ်ပြနှင့် ကြိယာဝိသေသန မကိုက်ညီပါ",
        ErrorType.MISSING_ASAT.value: "အသတ် ပျောက်နေပါသည်",
        ErrorType.PARTICLE_MISUSE.value: "ပစ္စည်း အသုံးအမှား",
        ErrorType.COLLOCATION_ERROR.value: "စကားတွဲ အမှား",
        # Orthography error types
        ErrorType.MEDIAL_ORDER_ERROR.value: "မီဒီယယ် အစီအစဉ် အမှား",
        ErrorType.MEDIAL_COMPATIBILITY_ERROR.value: "မီဒီယယ် လိုက်ဖက်မှု အမှား",
        ErrorType.VOWEL_AFTER_ASAT.value: "အသတ်ပြီး သရ",
        ErrorType.LEADING_VOWEL_E.value: "ရှေ့ထား ေ အမှား",
        ErrorType.INCOMPLETE_STACKING.value: "ပါဋ်ဆင့် မပြည့်စုံပါ",
        # Punctuation error types
        ErrorType.DUPLICATE_PUNCTUATION.value: "ပုဒ်ဖြတ် ထပ်နေပါသည်",
        ErrorType.WRONG_PUNCTUATION.value: "ပုဒ်ဖြတ် မမှန်ကန်ပါ",
        ErrorType.MISSING_PUNCTUATION.value: "ပုဒ်ဖြတ် ပျောက်နေပါသည်",
        # Legacy keys (kept for backward compatibility)
        "context_error": "စကားစပ် မသင့်လျော်ပါ",
        "tone_error": "အသံပြသင်္ကေတ အမှား",
        "stacking_error": "ပါဋ်ဆင့် အမှား",
        # Error descriptions
        "syllable_not_found": "စာလုံးပေါင်း အဘိဓာန်တွင် မတွေ့ရှိပါ",
        "word_not_found": "စကားလုံး အဘိဓာန်တွင် မတွေ့ရှိပါ",
        "low_probability": "ဖြစ်နိုင်ခြေ နည်းသော စကားစပ်",
        "missing_particle": "ပစ္စည်း ပျောက်နေပါသည်",
        "wrong_particle": "ပစ္စည်း မမှန်ကန်ပါ",
        "ya_ra_confusion": "ယပင်နှင့် ရရစ် ရောထွေးနေပါသည်",
        "tone_mark_missing": "အသံပြသင်္ကေတ ပျောက်နေပါသည်",
        "tone_mark_wrong": "အသံပြသင်္ကေတ မမှန်ကန်ပါ",
        # Suggestions
        "did_you_mean": "ဆိုလိုသည်မှာ",
        "suggestions": "အကြံပြုချက်များ",
        "no_suggestions": "အကြံပြုချက် မရှိပါ",
        # Validation messages
        "text_valid": "စာသား မှန်ကန်ပါသည်",
        "errors_found": "အမှားများ တွေ့ရှိပါသည်",
    },
}


def get_language() -> str:
    """
    Get the current language setting for this thread.

    Returns:
        Current language code ('en' or 'my').

    Note:
        Returns the thread-local language setting, or default if not set.
    """
    return getattr(_thread_local, "language", _DEFAULT_LANGUAGE)


def set_language(language: str) -> None:
    """
    Set the current language for error messages in this thread.

    Args:
        language: Language code ('en' for English, 'my' for Myanmar).

    Raises:
        ValueError: If language code is not supported.

    Note:
        This sets the language for the current thread only.
        Other threads are not affected.

    Example:
        >>> set_language("my")
        >>> get_message("invalid_syllable")
        'စာလုံးပေါင်း မမှန်ကန်ပါ'
    """
    if language not in MESSAGES:
        supported = ", ".join(MESSAGES.keys())
        raise ValueError(f"Unsupported language '{language}'. Supported: {supported}")
    _thread_local.language = language


def get_message(key: str, language: str | None = None) -> str:
    """
    Get a localized message by key.

    Args:
        key: Message key (e.g., 'invalid_syllable').
        language: Optional language override. Uses current thread's language if None.

    Returns:
        Localized message string. Returns key if not found.

    Example:
        >>> get_message("invalid_syllable", "my")
        'စာလုံးပေါင်း မမှန်ကန်ပါ'
        >>> get_message("invalid_syllable", "en")
        'Invalid syllable'
    """
    lang = language or get_language()
    messages = MESSAGES.get(lang, MESSAGES["en"])
    return messages.get(key, key)


def get_error_message(error_type: str, language: str | None = None) -> str:
    """
    Get a localized error type message.

    This is a convenience function that maps error_type values
    to their localized descriptions.

    Args:
        error_type: Error type value (e.g., 'invalid_syllable').
        language: Optional language override.

    Returns:
        Localized error description.

    Example:
        >>> get_error_message("invalid_syllable", "my")
        'စာလုံးပေါင်း မမှန်ကန်ပါ'
    """
    return get_message(error_type, language)


def format_suggestion_message(suggestions: list[str], language: str | None = None) -> str:
    """
    Format a suggestion message with localized text.

    Args:
        suggestions: List of suggested corrections.
        language: Optional language override. Uses current thread's language if None.

    Returns:
        Formatted suggestion string.

    Example:
        >>> format_suggestion_message(["မြန်မာ", "မြန်"], "my")
        'အကြံပြုချက်များ: မြန်မာ, မြန်'
    """
    lang = language or get_language()
    if not suggestions:
        return get_message("no_suggestions", lang)

    label = get_message("suggestions", lang)
    return f"{label}: {', '.join(suggestions)}"


def get_supported_languages() -> list[str]:
    """
    Get list of supported language codes.

    Returns:
        List of supported language codes.

    Example:
        >>> get_supported_languages()
        ['en', 'my']
    """
    return list(MESSAGES.keys())
