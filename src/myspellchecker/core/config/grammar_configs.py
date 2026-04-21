"""
Grammar Configuration Classes.

This module contains configuration classes for grammar checking:
- GrammarEngineConfig: Main grammar checking engine settings
- AspectCheckerConfig: Aspect marker checker settings
- ClassifierCheckerConfig: Classifier checker settings
- CompoundCheckerConfig: Compound word checker settings
- RegisterCheckerConfig: Register consistency checker settings
"""

from pydantic import BaseModel, ConfigDict, Field

# Default rule priorities for grammar conflict resolution.
# Higher value = higher priority (wins when rules overlap).
DEFAULT_RULE_PRIORITIES: dict[str, int] = {
    "sentence_boundary": 100,
    "medial_confusion": 90,
    "particle_typo": 85,
    "verb_particle_agreement": 70,
    "pos_sequence": 65,
    "loan_word": 55,
    "config_pattern": 50,
    "classifier": 45,
    "merged_word": 42,
    "compound": 40,
    "register": 20,
    "aspect": 15,
    "negation": 10,
}


class GrammarEngineConfig(BaseModel):
    """
    Configuration for the grammar checking engine.

    Controls confidence thresholds for different types of grammar checks.

    Attributes:
        default_confidence_threshold: Minimum confidence for grammar errors (default: 0.80).
            Errors below this threshold are not reported.
        exact_match_confidence: Confidence for exact/specific rule matches (default: 0.95).
            Used for very specific medial confusion and context rules.
        high_confidence: Confidence for word corrections with clear patterns (default: 0.90).
            Used for after-noun corrections and config-based typos.
        medium_confidence: Confidence for common error patterns (default: 0.85).
            Used for verb particles, missing asat, and general word corrections.
        pos_sequence_confidence: Confidence for POS sequence errors (default: 0.80).
            Used when invalid POS tag sequences are detected.
        verb_particle_confidence: Confidence for verb-particle agreement errors (default: 0.75).
            Used when particle follows wrong POS.
        sentence_final_confidence: Confidence for sentence-final particle errors (default: 0.70).
            Used when sentence-final particle appears mid-sentence.
        context_confidence_threshold: Confidence for context-dependent corrections (default: 0.65).
            Used for corrections that depend on surrounding words.
        question_confidence: Confidence for question particle errors (default: 0.60).
            Used for missing or incorrect question particles.
        tense_marker_confidence: Confidence for tense marker position errors (default: 0.60).
            Used when tense markers appear after non-verbs.
        low_confidence_threshold: Threshold for speculative suggestions (default: 0.55).
            Used for suggestions that might be intentional (e.g., missing subject marker).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    default_confidence_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for grammar errors to be reported",
    )
    exact_match_confidence: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence for exact/specific rule matches",
    )
    high_confidence: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Confidence for word corrections with clear patterns",
    )
    medium_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence for common error patterns",
    )
    pos_sequence_confidence: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Confidence for POS sequence errors",
    )
    verb_particle_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence for verb-particle agreement errors",
    )
    sentence_final_confidence: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Confidence for sentence-final particle errors",
    )
    context_confidence_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Confidence for context-dependent corrections",
    )
    question_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Confidence for question particle errors",
    )
    tense_marker_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Confidence for tense marker position errors",
    )
    low_confidence_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Threshold for speculative suggestions",
    )
    enable_targeted_grammar_completion_templates: bool = Field(
        default=True,
        description="Enable targeted grammar completion template suggestions",
    )
    rule_priorities: dict[str, int] = Field(
        default_factory=lambda: dict(DEFAULT_RULE_PRIORITIES),
        description=(
            "Priority values for grammar rule conflict resolution. "
            "Higher value = higher priority (wins on overlap). "
            "Keys: sentence_boundary, medial_confusion, particle_typo, "
            "verb_particle_agreement, pos_sequence, loan_word, config_pattern, "
            "classifier, merged_word, compound, register, aspect, negation."
        ),
    )
    rare_word_frequency_threshold: int = Field(
        default=50,
        ge=0,
        description=(
            "Words with corpus frequency below this are considered 'rare'. "
            "Rare words require higher confidence to be flagged by grammar "
            "checkers, reducing FPs on uncommon but valid words."
        ),
    )
    rare_word_min_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum confidence required to flag rare words (freq below "
            "rare_word_frequency_threshold). Set to 0.0 to disable the guard."
        ),
    )


class AspectCheckerConfig(BaseModel):
    """
    Configuration for aspect marker checker.

    Controls confidence levels for aspect marker validation.

    Attributes:
        high_confidence: Confidence for clear, single-marker matches (default: 0.95).
        medium_confidence: Confidence for valid multi-marker patterns (default: 0.85).
        low_confidence: Confidence for ambiguous or context-dependent cases (default: 0.70).
        error_penalty: Score reduction per error in validation (default: 0.20).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    high_confidence: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence for clear, single-marker matches",
    )
    medium_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence for valid multi-marker patterns",
    )
    low_confidence: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Confidence for ambiguous or context-dependent cases",
    )
    error_penalty: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Score reduction per error in validation",
    )


class ClassifierCheckerConfig(BaseModel):
    """
    Configuration for classifier checker.

    Controls confidence levels for classifier validation.

    Attributes:
        default_confidence: Default confidence for classifier corrections (0.90).
        animate_classifier_confidence: Confidence for animate classifiers (0.85).
        inanimate_classifier_confidence: Confidence for inanimate classifiers (0.80).
        context_boost: Confidence boost when context supports correction (0.05).
        max_confidence: Maximum confidence after boost (0.95).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    default_confidence: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Default confidence for classifier corrections",
    )
    animate_classifier_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence for animate classifier corrections",
    )
    inanimate_classifier_confidence: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Confidence for inanimate classifier corrections",
    )
    context_boost: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Confidence boost when context supports correction",
    )
    max_confidence: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after boost",
    )


class CompoundCheckerConfig(BaseModel):
    """
    Configuration for compound word checker.

    Controls confidence levels for compound word validation.

    Attributes:
        exact_match_confidence: Confidence for exact compound matches (default: 0.95).
        pattern_match_confidence: Confidence for pattern-based matches (default: 0.90).
        component_match_confidence: Confidence for component-based matches (default: 0.80).
        partial_match_confidence: Confidence for partial matches (default: 0.75).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    exact_match_confidence: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence for exact compound matches",
    )
    pattern_match_confidence: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Confidence for pattern-based matches",
    )
    component_match_confidence: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Confidence for component-based matches",
    )
    partial_match_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence for partial matches",
    )


class RegisterCheckerConfig(BaseModel):
    """
    Configuration for register consistency checker.

    Controls confidence levels for register (formal/colloquial/polite) validation.

    Attributes:
        register_mismatch_confidence: Confidence for formal+casual mixing errors (default: 0.85).
        register_formality_gap_confidence: Confidence for formal+polite mixing (default: 0.65).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    register_mismatch_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence for formal+casual register mismatch errors",
    )
    register_formality_gap_confidence: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Confidence for formal+polite register gap warnings",
    )


class NegationCheckerConfig(BaseModel):
    """
    Configuration for negation pattern checker.

    Controls confidence levels for negation pattern validation.

    Attributes:
        typo_confidence: Confidence for negation ending typos (default: 0.90).
        missing_visarga_confidence: Confidence for missing visarga corrections (default: 0.85).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    typo_confidence: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Confidence for negation ending typos",
    )
    missing_visarga_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence for missing visarga corrections",
    )


class TenseAgreementCheckerConfig(BaseModel):
    """
    Configuration for tense agreement checker.

    Controls confidence levels for tense-time agreement validation
    between temporal adverbials and aspectual particles.

    Attributes:
        default_confidence: Default confidence for tense mismatch errors (default: 0.75).
        high_confidence: Confidence when both adverb and marker are unambiguous (default: 0.85).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    default_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Default confidence for tense mismatch errors",
    )
    high_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence when both adverb and marker are unambiguous",
    )


class ParticleCheckerConfig(BaseModel):
    """
    Configuration for particle context checker.

    Controls confidence levels for particle context validation.

    Attributes:
        confusion_confidence: Confidence for particle confusion pair detections (default: 0.70).
        frame_violation_confidence: Confidence for verb-particle frame violations (default: 0.70).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    confusion_confidence: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Confidence for particle confusion pair detections",
    )
    frame_violation_confidence: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Confidence for verb-particle frame violations",
    )


class MergedWordCheckerConfig(BaseModel):
    """
    Configuration for merged word checker.

    Controls confidence levels for merged word detection.

    Attributes:
        default_confidence: Default confidence for merged word detections (default: 0.80).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    default_confidence: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Default confidence for merged word detections",
    )
