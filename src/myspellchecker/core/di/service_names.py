"""
Service name constants for the DI container.

Using constants instead of string literals provides:
- IDE autocomplete support
- Compile-time typo detection
- Easy refactoring of service names
- Single source of truth for service identifiers

Example:
    >>> from myspellchecker.core.di.service_names import SERVICE_PROVIDER
    >>> provider = container.get(SERVICE_PROVIDER)
"""

# Core services
SERVICE_PROVIDER = "provider"
SERVICE_SEGMENTER = "segmenter"
SERVICE_PHONETIC_HASHER = "phonetic_hasher"
SERVICE_SYMSPELL = "symspell"
SERVICE_CONTEXT_CHECKER = "context_checker"
SERVICE_SUGGESTION_STRATEGY = "suggestion_strategy"

# Validators
SERVICE_SYLLABLE_VALIDATOR = "syllable_validator"
SERVICE_WORD_VALIDATOR = "word_validator"
SERVICE_CONTEXT_VALIDATOR = "context_validator"
SERVICE_CONTEXT_VALIDATOR_MINIMAL = "context_validator_minimal"

# Optional services (may not be registered in all configurations)
SERVICE_TONE_DISAMBIGUATOR = "tone_disambiguator"
SERVICE_SYNTACTIC_RULE_CHECKER = "syntactic_rule_checker"
SERVICE_VITERBI_TAGGER = "viterbi_tagger"
SERVICE_HOMOPHONE_CHECKER = "homophone_checker"
SERVICE_SEMANTIC_CHECKER = "semantic_checker"
SERVICE_NAME_HEURISTIC = "name_heuristic"
