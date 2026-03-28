# Rules Directory

YAML rule files for Myanmar linguistic validation, grammar checking, and suggestion reranking. These files are loaded at runtime by `grammar/config.py` and various strategy/checker modules.

## File Inventory

| File | Purpose |
|------|---------|
| `grammar_rules.yaml` | Syntactic grammar rules: invalid POS sequences, particle chains, sentence-final requirements |
| `particles.yaml` | Particle usage rules and valid particle combinations |
| `tone_rules.yaml` | Tone mark disambiguation rules |
| `negation.yaml` | Negation placement and double-negation patterns |
| `classifiers.yaml` | Classifier-noun agreement rules |
| `aspects.yaml` | Aspect marker validation rules |
| `register.yaml` | Formal/informal register mixing detection rules |
| `pronouns.yaml` | Pronoun usage rules |
| `morphology.yaml` | Morphological analysis rules (suffix stripping, POS guessing) |
| `morphotactics.yaml` | Compound word POS pattern rules |
| `pos_inference.yaml` | POS inference rules for ambiguous words |
| `homophones.yaml` | Bidirectional homophone lookup (word to its homophones list) |
| `homophone_confusion.yaml` | Context-dependent homophone disambiguation (left/right triggers) |
| `medial_confusion.yaml` | Medial ya-pin/ya-yit confusion detection (unconditional + contextual) |
| `medial_swap_pairs.yaml` | Medial consonant swap pairs for candidate generation (with weights) |
| `compounds.yaml` | Compound word formation patterns and definitions |
| `compound_confusion.yaml` | Compound-level confusion corrections (ha-htoe, aspiration, consonant, suffix) |
| `confusable_pairs.yaml` | Curated confusable word pairs for MLM-based detection (threshold tiers) |
| `confusion_matrix.yaml` | Character-level confusion probabilities for edit distance |
| `typo_corrections.yaml` | Common typo patterns and their corrections |
| `orthographic_corrections.yaml` | Orthographic correction mappings |
| `ambiguous_words.yaml` | Ambiguous word handling rules |
| `stacking_pairs.yaml` | Consonant stacking (patsin) pair rules |
| `tense_markers.yaml` | Tense marker validation rules |
| `collocations.yaml` | Word collocation data for context validation |
| `corruption_weights.yaml` | Weights for synthetic error generation (training) |
| `detector_confidences.yaml` | Per-detector confidence thresholds |
| `rerank_rules.yaml` | Data-driven suggestion reranking and injection tables |
| `semantic_rules.yaml` | Semantic validation rules (agent implausibility, animacy) |

## Paired Files

Several files work together. Understanding the pairs avoids duplicate edits:

### Homophones
- **`homophones.yaml`** -- Bidirectional lookup: maps each word to its homophone list. Used by `HomophoneStrategy` for candidate generation.
- **`homophone_confusion.yaml`** -- Context-dependent detection: triggers only when specific left/right context tokens appear. Used by `PostNormalizationDetectorsMixin`.

### Medials
- **`medial_confusion.yaml`** -- Detection rules: unconditional corrections (wrong form is never valid) and contextual corrections (both forms valid, context disambiguates).
- **`medial_swap_pairs.yaml`** -- Candidate generation: defines which medial consonants can be swapped and with what weight. Used by SymSpell variant generation.

### Compounds
- **`compounds.yaml`** -- Definitions: compound word formation patterns (noun-noun, verb-verb, reduplication, affixes).
- **`compound_confusion.yaml`** -- Corrections: maps wrong compound forms to their correct components across four categories (ha-htoe, aspiration, consonant, suffix).

## Schema Validation

Every YAML file has a corresponding JSON schema in `../schemas/`:

```
rules/grammar_rules.yaml    <-->  schemas/grammar_rules.schema.json
rules/homophones.yaml        <-->  schemas/homophones.schema.json
...
```

Schemas are validated at load time by `grammar/config.py` (requires `jsonschema` package). The shared base schema is `schemas/_common.schema.json`.

## File Format

All rule files follow a standard header:

```yaml
# Description comment block
# Explains the file's purpose and format

version: "1.0.0"          # Semantic version of this rule file
category: "grammar_rules"  # Machine-readable category identifier
description: "..."         # One-line description (optional but recommended)

metadata:                   # Optional
  created_date: "2025-12-30"
  last_updated: "2026-03-21"
```

The `version` and `category` fields are required. Category must match the schema filename (e.g., `category: "homophones"` validates against `homophones.schema.json`).

## Adding New Rules

1. **Add entries to an existing file** when possible. Check the file's comment header for format details.

2. **Creating a new file**:
   - Add the standard header (`version`, `category`, `description`).
   - Create a matching schema in `../schemas/<category>.schema.json`.
   - Add a loader in the appropriate config/strategy module.
   - All Myanmar text must use post-normalization Unicode forms (tall AA U+102B is normalized to U+102C).

3. **Version bumps**: Increment the patch version for new entries, minor version for structural changes.

4. **Testing**: Run `pytest tests/test_yaml_schema_validation.py` to verify schema compliance after changes.
