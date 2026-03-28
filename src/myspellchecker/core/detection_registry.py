"""Detection registry — ordered sequence of post-normalization detectors.

Defines the execution order for all ``_detect_*`` methods that run on
normalized text.  Each entry maps to a method on SpellChecker (inherited
from detector mixins) with the signature ``(self, text: str, errors: list[Error]) -> None``.

Ordering constraints:
    - ``broken_stacking`` must precede ``colloquial_contractions`` to prevent
      stacking errors from being claimed as colloquial.
    - ``particle_confusion`` must follow ``medial_confusion`` (medial errors
      can produce particles as byproducts).
    - ``register_mixing`` runs after individual particle/sentence detectors
      so it can see the full picture.
    - ``punctuation_errors`` runs last among detectors (lowest priority).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetectorEntry:
    """A single detector in the execution sequence."""

    method_name: str
    description: str = ""


# Ordered sequence of post-normalization detectors.
# All methods accept (self, text: str, errors: list[Error]) -> None.
POST_NORM_DETECTOR_SEQUENCE: tuple[DetectorEntry, ...] = (
    DetectorEntry("_detect_broken_stacking", "Asat→virama in Pali words"),
    DetectorEntry("_detect_missing_stacking", "Missing Pali/Sanskrit virama stacking"),
    DetectorEntry("_detect_missing_asat", "Missing asat on normalized text"),
    DetectorEntry("_detect_missing_visarga_suffix", "Missing visarga in clause-linker suffixes"),
    DetectorEntry("_detect_missing_visarga_in_compound", "Missing visarga inside compound words"),
    DetectorEntry("_detect_medial_confusion", "Medial ya-pin/ya-yit confusion"),
    DetectorEntry("_detect_colloquial_contractions", "Colloquial contraction detection"),
    DetectorEntry("_detect_particle_confusion", "Particle confusion (ကိ/ကု → ကို)"),
    DetectorEntry("_detect_compound_confusion_typos", "Compound confusion (ha-htoe + aspirated)"),
    DetectorEntry("_detect_suffix_confusion_typos", "Suffix confusion on invalid compounds"),
    DetectorEntry(
        "_detect_invalid_token_with_strong_candidates",
        "Invalid token repair via strong DB candidates",
    ),
    DetectorEntry(
        "_detect_frequency_dominant_valid_variants",
        "Valid-token variant correction via frequency + semantic",
    ),
    DetectorEntry("_detect_broken_compound_morpheme", "Broken compound morpheme (ed-1 variant)"),
    DetectorEntry("_detect_missegmented_confusable", "Confusable errors hidden by segmentation"),
    DetectorEntry("_detect_ha_htoe_particle_typos", "Ha-htoe particle confusion (မာ → မှာ)"),
    DetectorEntry("_detect_aukmyit_confusion", "Aukmyit confusion (ထည် → ထည့်)"),
    DetectorEntry("_detect_extra_aukmyit_confusion", "Extra aukmyit (ပြော့ → ပြော)"),
    DetectorEntry("_detect_sequential_particle_confusion", "Sequential particle (တော် → တော့)"),
    DetectorEntry("_detect_particle_misuse", "Particle misuse via verb-frame (ကို → မှ/မှာ/တွင်)"),
    DetectorEntry("_detect_homophone_left_context", "Homophone left-context (ဖက် → ဖတ်)"),
    DetectorEntry("_detect_collocation_errors", "Collocation error (wrong word partner)"),
    DetectorEntry("_detect_semantic_agent_implausibility", "Non-human subject implausibility"),
    DetectorEntry("_detect_merged_classifier_mismatch", "Merged NUM+classifier mismatch"),
    DetectorEntry("_detect_dangling_particles", "Dangling sentence-end particles"),
    DetectorEntry("_detect_sentence_structure_issues", "Dangling word, missing conjunction"),
    DetectorEntry("_detect_tense_mismatch", "Temporal adverb vs particle mismatch"),
    DetectorEntry("_detect_formal_yi_in_colloquial_context", "Verb+၏ in colloquial context"),
    DetectorEntry("_detect_negation_sfp_mismatch", "Negation pattern mismatch"),
    DetectorEntry("_detect_merged_sfp_conjunction", "Merged SFP + conjunction"),
    DetectorEntry("_detect_missing_visarga", "Missing visarga (း) via frequency ratio"),
    DetectorEntry("_detect_register_mixing", "Formal/colloquial register mixing"),
    DetectorEntry("_detect_informal_with_honorific", "Informal particle + honorific"),
    DetectorEntry("_detect_informal_h_after_completive", "Terse ဟ after completive"),
    DetectorEntry("_detect_vowel_after_asat", "Vowel after asat (ကျွန်ုတော် → ကျွန်တော်)"),
    DetectorEntry("_detect_missing_diacritic_in_compound", "Missing anusvara/dot-below"),
    DetectorEntry("_detect_unknown_compound_segments", "Unknown freq=0 compound segments"),
    DetectorEntry("_detect_broken_compound_space", "Space inside compound word"),
    DetectorEntry("_detect_punctuation_errors", "Punctuation errors"),
)
