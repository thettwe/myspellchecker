"""Bucket-specific Gemini Flash prompts for Burmese spelling-error generation.

Buckets follow the path-f spelling-only taxonomy ordered by FN count from
`benchmarks/results/path-f-spelling-only/spelling_only_metrics.json`. The
corruption-types under each bucket are the same Path F subtypes that the
spell-checker is currently failing to detect.

Each prompt asks Gemini to:
  1. introduce ONE error matching the bucket
  2. return the corrupted sentence + the exact span that changed
  3. avoid changing meaning beyond the localized error
"""

from __future__ import annotations

# FN-count weights drive sampling proportions when corrupting.
#
# v1.9 architecture decision (2026-05-05): ``compound_spacing`` removed.
# Soft-Masked BERT is length-preserving; spacing edits change length and
# break the soft-masking alignment. Spacing is handled separately by the
# existing Phase 2A probe segmenter (+0.0111 in production). The v1
# weight (148) for compound_spacing is dropped; the sampler renormalizes
# over the remaining 9 lexical buckets automatically.
# See: [[60_Decisions/Soft-Masked v1.9 Whitespace-Stripped Architecture 2026-05-05]]
BUCKET_FN_WEIGHT: dict[str, int] = {
    "real_word_homophone": 84,
    "aukmyit_visarga_asat": 77,
    "consonant_vowel_sub": 58,
    "generic_legacy": 53,
    "tone_marks": 31,
    "synonym_sub": 30,
    "non_word_typo": 29,
    "loanword": 18,
    "zawgyi": 18,
}


SYSTEM_PROMPT = """\
You are an expert in Myanmar (Burmese) orthography helping build training data
for a spell-checker neural network. The user will give you a clean Burmese
sentence and an error category. Your job is to introduce ONE realistic error
matching that category, while keeping every other word unchanged.

CRITICAL RULES:
1. Change ONLY one word/phrase — the rest of the sentence must be byte-identical.
2. The error must be REALISTIC — the kind a Burmese typist or learner would actually make.
3. The error must match the requested category EXACTLY.
4. Identify the exact character span (start, end half-open) of the change.
5. Output strict JSON only — no markdown fences, no commentary.
6. The gold and erroneous forms MUST differ visually after Unicode normalization.
   Codepoint-reorder swaps that render identically (e.g. swapping virama and
   dot-below order, or U+1026 vs U+1025+U+102E for "ဦ") do NOT count as a
   real corruption — the model can't learn anything from them. If you can
   only produce such a reorder, return {"skip": true, "reason": "..."}
   instead. Encoding-level confusions belong in the ``zawgyi`` bucket.

Output schema (single JSON object, no preamble):
{
  "corrupted": "<full sentence with error introduced>",
  "gold":      "<the original word/phrase>",
  "erroneous": "<what the corrupted word/phrase looks like>",
  "span_start": <int — character offset in corrupted sentence>,
  "span_end":   <int — half-open end offset>,
  "subtype":    "<one of the subtypes listed in the user prompt>",
  "rationale":  "<one short sentence explaining the error>"
}

If you CANNOT introduce a realistic error of the requested category in this
sentence, output: {"skip": true, "reason": "<short why>"}
"""


# Per-bucket user prompt templates. The orchestrator fills {clean} with the
# clean sentence and may pick a specific subtype if it wants control.

BUCKET_USER_TEMPLATES: dict[str, str] = {
    "real_word_homophone": """\
Bucket: real_word_homophone
Allowed subtypes:
  - real_word_confusion: swap a word for a real Burmese word that is wrong in
    THIS context but valid in others (e.g., နေ့ → နေ where နေ is real but wrong)
  - homophone_confusion: swap to a homophone that sounds the same but is
    spelled differently and means something different here

The replacement MUST be a valid Burmese word — the goal is for the model to
catch CONTEXTUAL errors, not non-words.

Clean sentence:
{clean}

Introduce exactly ONE real-word/homophone error and return JSON per the schema.""",
    "aukmyit_visarga_asat": """\
Bucket: aukmyit_visarga_asat
Allowed subtypes:
  - aukmyit_confusion: drop or misplace ၏ (the formal possessive) where it's expected
  - missing_visarga: drop the visarga ◌ ့ (e.g., ပေး့ → ပေး — when ့ was needed)
  - missing_asat: drop the asat ် (e.g., ကမ်း → ကမး)
  - missing_anusvara: drop ◌ ံ
  - missing_dot_below: drop ◌ ့

These are "diacritic-omission" errors common in casual typing.

Clean sentence:
{clean}

Introduce exactly ONE aukmyit/visarga/asat-class error and return JSON per the schema.""",
    "consonant_vowel_sub": """\
Bucket: consonant_vowel_sub
Allowed subtypes:
  - consonant_substitution: swap one consonant for a similar-sounding one
    (e.g., က → ခ, ပ → ဖ, တ → ထ)
  - vowel_substitution: swap a vowel sign (e.g., ◌ ု → ◌ ူ, ◌ ိ → ◌ ီ)
  - vowel_medial_substitution: swap a medial (e.g., ြ → ျ)
  - vowel_length_substitution: swap short/long vowel (e.g., ◌ ု ↔ ◌ ူ)
  - missing_medial: drop a medial sign
  - vowel_sign_substitution: swap for a different vowel sign

These are typo-class errors at the syllable level.

Clean sentence:
{clean}

Introduce exactly ONE consonant/vowel substitution error and return JSON per the schema.""",
    "tone_marks": """\
Bucket: tone_marks
Allowed subtypes:
  - tone_mark_error: swap or misplace a tone mark (◌ ် ◌ ့ ◌း)
  - tone_mark_placement: place a tone mark on the wrong syllable
  - tone_error: drop a needed tone mark or add an extraneous one

Clean sentence:
{clean}

Introduce exactly ONE tone-mark error and return JSON per the schema.""",
    "non_word_typo": """\
Bucket: non_word_typo
Allowed subtypes:
  - non_word_typo: swap one character to produce a non-word (a string that's
    not a real Burmese word in any dictionary)
  - complex_typo: 1-2 char insertion/deletion/swap that breaks the word
  - typo: random one-char typo

The result must NOT be a real Burmese word.

Clean sentence:
{clean}

Introduce exactly ONE non-word typo and return JSON per the schema.""",
    "synonym_sub": """\
Bucket: synonym_sub
Allowed subtypes:
  - confusable_semantic: swap to a synonym or near-synonym that's contextually
    INAPPROPRIATE (e.g., formal alternative used in colloquial register; archaic
    word in modern context; technical term in casual writing)

The replacement MUST be a valid Burmese word and a true synonym/near-synonym.
The error is REGISTER/CONTEXT mismatch, not spelling.

Clean sentence:
{clean}

Introduce exactly ONE synonym-substitution error and return JSON per the schema.""",
    "loanword": """\
Bucket: loanword
Allowed subtypes:
  - loan_word_misspelling: misspell a transliterated loan word (English/Pali
    borrowing rendered in Burmese script — e.g., ကွန်ပျူတာ "computer",
    ပရောဂျက် "project", တယ်လီဖုန်း "telephone", အင်တာနက် "internet",
    ရေဒီယို "radio", ဒေါက်တာ "doctor", ဘတ်စ်ကား "bus", ကမ္မဋ္ဌာန်း Pali)
  - loanword_spelling: typo on a loan word

STRICT RULE: the gold word MUST be a transliterated borrowing from English,
Pali, or another foreign language — NOT a native Burmese word. If the
sentence has no actual loan words, return
{"skip": true, "reason": "no loan words"}.

DO NOT corrupt native Burmese words and label them as loanwords. If you're
unsure whether a word is a loan, skip rather than guess.

Clean sentence:
{clean}

Introduce exactly ONE loan-word spelling error and return JSON per the schema.""",
    "zawgyi": """\
Bucket: zawgyi
Allowed subtypes:
  - zawgyi: introduce a Zawgyi-style codepoint substitution (e.g., U+102B ါ
    → U+102C ာ, or vowel-sign placement in pre-Unicode order)
  - zawgyi_artifact: legacy encoding artifact in an otherwise-Unicode word
  - zawgyi_conversion: an error that arises from buggy Zawgyi-to-Unicode
    conversion (e.g., ပေါ် → ပော် when U+102B was mis-converted)
  - zawgyi_vowel_confusion: vowel-sign codepoint confusion from Zawgyi era

These look like normal Burmese to a casual reader but are codepoint-wrong.

Clean sentence:
{clean}

Introduce exactly ONE Zawgyi-class error and return JSON per the schema.""",
    "generic_legacy": """\
Bucket: generic_legacy
Allowed subtypes:
  - spelling: generic spelling error not fitting other buckets
  - orthography: orthographic-rule violation
  - missing_syllable: drop a whole syllable from a word
  - syllable_error: malformed syllable (invalid C(M)V(F)(T) structure)
  - missing_character: drop a single character
  - colloquial_spelling: nonstandard spelling that's accepted in colloquial use

STRICT RULE: the corrupted form MUST be visually distinct from the gold
form. Rendering a different codepoint ORDER for the same visual glyph
(e.g. swapping the order of virama ◌် and dot-below ◌့, or replacing
U+1026 "ဦ" with U+1025 + U+102E which renders identically) does NOT
count — the corruption must produce text that LOOKS different to a
reader. If you can only produce a codepoint-reorder corruption, use the
``zawgyi`` bucket instead — that's where encoding-level confusions belong.

Clean sentence:
{clean}

Introduce exactly ONE generic spelling/orthography error and return JSON per the schema.""",
}


def render_user_prompt(bucket: str, clean: str) -> str:
    """Fill the bucket template with a clean sentence.

    Uses str.replace rather than str.format because templates contain literal
    JSON braces (e.g. ``{"skip": true}``) that .format would treat as fields.
    """
    template = BUCKET_USER_TEMPLATES.get(bucket)
    if template is None:
        raise ValueError(f"unknown bucket: {bucket!r}")
    return template.replace("{clean}", clean)


def all_buckets() -> list[str]:
    return list(BUCKET_FN_WEIGHT.keys())
