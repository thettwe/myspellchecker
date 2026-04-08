"""
Detection Rules Loader.

Loads text-level detection dicts from YAML rule files and returns
typed data structures matching what SpellChecker._detect_* methods expect.

All Myanmar text is normalized at load time via the norm_* helpers
from detector_data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from myspellchecker.core.detector_data import (
    norm_dict,
    norm_dict_context,
    norm_dict_tuple,
    norm_set,
)
from myspellchecker.text.normalize import normalize
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

_RULES_DIR = Path(__file__).parent.parent / "rules"


# ── YAML loading ──


def _load_yaml(path: Path, name: str) -> dict[str, Any] | None:
    """Load a YAML file with error handling. Returns None on failure."""
    if not path.exists():
        logger.debug("Detection rules file not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                logger.debug("Loaded detection rules from %s", path)
                return data
    except (yaml.YAMLError, OSError) as e:
        logger.warning("Failed to load detection rules from %s: %s", path, e)
    return None


# ── Orthographic corrections (orthographic_corrections.yaml) ──


def load_orthographic_corrections(
    path: Path | None = None,
) -> dict[str, Any]:
    """Load orthographic correction rules.

    Returns a dict with keys:
      - medial_confusion_unconditional: dict[str, str | list[str]]
      - medial_confusion_contextual: dict[str, tuple[str, tuple[str, ...]]]
      - aukmyit_context: dict[str, tuple[str, tuple[str, ...]]]
      - extra_aukmyit_context: dict[str, tuple[str, tuple[str, ...]]]
      - vowel_reorder_errors: dict[str, list[str]]
      - colloquial_contractions: dict[str, str | list[str]]
      - stacking_completions: dict[str, str]
    """
    result: dict[str, Any] = {
        "medial_confusion_unconditional": {},
        "medial_confusion_contextual": {},
        "aukmyit_context": {},
        "extra_aukmyit_context": {},
        "vowel_reorder_errors": {},
        "colloquial_contractions": {},
        "stacking_completions": {},
    }

    yaml_path = path or (_RULES_DIR / "orthographic_corrections.yaml")
    data = _load_yaml(yaml_path, "orthographic_corrections")
    if not data:
        return result

    # -- Medial confusion unconditional --
    raw_unconditional = data.get("medial_confusion_unconditional", [])
    unconditional: dict[str, str | list[str]] = {}
    for entry in raw_unconditional:
        incorrect = entry.get("incorrect", "")
        correct = entry.get("correct")
        if incorrect and correct:
            unconditional[incorrect] = correct
    result["medial_confusion_unconditional"] = norm_dict(unconditional)

    # -- Medial confusion contextual --
    raw_contextual = data.get("medial_confusion_contextual", [])
    contextual_raw: dict[str, tuple[str, tuple[str, ...]]] = {}
    for entry in raw_contextual:
        incorrect = entry.get("incorrect", "")
        correct = entry.get("correct", "")
        followed_by = entry.get("followed_by", [])
        if incorrect and correct and followed_by:
            contextual_raw[incorrect] = (correct, tuple(followed_by))
    result["medial_confusion_contextual"] = norm_dict_context(contextual_raw)

    # -- Aukmyit (dot-below) confusion --
    raw_aukmyit = data.get("aukmyit_confusion", [])
    missing_raw: dict[str, tuple[str, tuple[str, ...]]] = {}
    extra_raw: dict[str, tuple[str, tuple[str, ...]]] = {}
    for entry in raw_aukmyit:
        incorrect = entry.get("incorrect", "")
        correct = entry.get("correct", "")
        followed_by = entry.get("followed_by", [])
        if not (incorrect and correct and followed_by):
            continue
        if "\u1037" in incorrect and "\u1037" not in correct:
            extra_raw[incorrect] = (correct, tuple(followed_by))
        else:
            missing_raw[incorrect] = (correct, tuple(followed_by))
    result["aukmyit_context"] = norm_dict_context(missing_raw)
    result["extra_aukmyit_context"] = norm_dict_context(extra_raw)

    # -- Vowel reorder errors --
    raw_vowel = data.get("vowel_reorder", [])
    vowel_raw: dict[str, list[str]] = {}
    for entry in raw_vowel:
        incorrect = entry.get("incorrect", "")
        correct = entry.get("correct")
        if not (incorrect and correct):
            continue
        if isinstance(correct, list):
            vowel_raw[incorrect] = correct
        else:
            vowel_raw[incorrect] = [correct]
    # Vowel reorder patterns must NOT be normalized (they are pre-normalization)
    result["vowel_reorder_errors"] = vowel_raw

    # -- Colloquial contractions --
    raw_colloquial = data.get("colloquial_contractions", [])
    colloquial: dict[str, str | list[str]] = {}
    for entry in raw_colloquial:
        incorrect = entry.get("incorrect", "")
        correct = entry.get("correct")
        if incorrect and correct:
            colloquial[incorrect] = correct
    result["colloquial_contractions"] = norm_dict(colloquial)

    # -- Stacking completions --
    raw_stacking = data.get("stacking_completions", [])
    stacking: dict[str, str] = {}
    for entry in raw_stacking:
        incorrect = entry.get("incorrect", "")
        correct = entry.get("correct", "")
        if incorrect and correct:
            stacking[incorrect] = correct
    # Stacking patterns are pre-normalization (virama sequences)
    result["stacking_completions"] = stacking

    return result


# ── Compound confusion dicts (compounds.yaml) ──


def load_compound_confusion(
    path: Path | None = None,
) -> dict[str, Any]:
    """Load compound confusion dicts from compounds.yaml.

    Returns a dict with keys:
      - ha_htoe_compounds: dict[str, tuple[str, str]]
      - aspirated_compounds: dict[str, tuple[str, str]]
      - consonant_confusion_compounds: dict[str, tuple[str, str]]
    """
    result: dict[str, Any] = {
        "ha_htoe_compounds": {},
        "aspirated_compounds": {},
        "consonant_confusion_compounds": {},
    }

    yaml_path = path or (_RULES_DIR / "compounds.yaml")
    data = _load_yaml(yaml_path, "compounds")
    if not data:
        return result

    for section_key in (
        "ha_htoe_compounds",
        "aspirated_compounds",
        "consonant_confusion_compounds",
    ):
        entries = data.get(section_key, [])
        raw: dict[str, tuple[str, str]] = {}
        for entry in entries:
            pattern = entry.get("pattern", "")
            wrong = entry.get("wrong", "")
            correct = entry.get("correct", "")
            if pattern and wrong and correct:
                raw[pattern] = (wrong, correct)
        result[section_key] = norm_dict_tuple(raw)

    return result


# ── Particle confusion dicts (particles.yaml) ──


def load_particle_confusion(
    path: Path | None = None,
) -> dict[str, Any]:
    """Load particle confusion dicts from particles.yaml.

    Returns a dict with keys:
      - particle_confusion: dict[str, str | list[str]]
      - sequential_particle_left_context: tuple[str, ...]
      - ha_htoe_particles: dict[str, str | list[str]]
      - ha_htoe_exclusions: frozenset[str]
      - honorific_prefixes: frozenset[str]
      - dangling_particles: frozenset[str]
      - suppression_suffixes: tuple[str, ...]
      - syllable_finals: frozenset[str]
      - visarga_exclude_pronouns: frozenset[str]
      - subject_pronouns: frozenset[str]
      - missing_asat_particles: dict[str, str | list[str]]
      - missing_visarga_suffixes: dict[str, str | list[str]]
      - particle_misuse_rules: list[dict] — verb-frame particle correction rules
    """
    result: dict[str, Any] = {
        "particle_confusion": {},
        "sequential_particle_left_context": (),
        "ha_htoe_particles": {},
        "ha_htoe_exclusions": frozenset(),
        "honorific_prefixes": frozenset(),
        "dangling_particles": frozenset(),
        "suppression_suffixes": (),
        "syllable_finals": frozenset(),
        "visarga_exclude_pronouns": frozenset(),
        "subject_pronouns": frozenset(),
        "missing_asat_particles": {},
        "missing_visarga_suffixes": {},
        "particle_misuse_rules": [],
        "locative_exempt_prefixes": (),
    }

    yaml_path = path or (_RULES_DIR / "particles.yaml")
    data = _load_yaml(yaml_path, "particles")
    if not data:
        return result

    # -- Confusion pairs (ကိ→ကို, ကု→ကို, နဲ→နဲ့) --
    raw_confusion = data.get("confusion_pairs", [])
    confusion: dict[str, str | list[str]] = {}
    for entry in raw_confusion:
        incorrect = entry.get("incorrect", "")
        correct = entry.get("correct")
        if incorrect and correct:
            confusion[incorrect] = correct
    result["particle_confusion"] = norm_dict(confusion)

    # -- Sequential particle confusion --
    seq_data = data.get("sequential_particle_confusion")
    if seq_data and isinstance(seq_data, dict):
        triggers = seq_data.get("left_context_triggers", [])
        result["sequential_particle_left_context"] = tuple(normalize(p) for p in triggers)

    # -- Ha-htoe particle confusion --
    htoe_data = data.get("ha_htoe_particle_confusion")
    if htoe_data and isinstance(htoe_data, dict):
        # Corrections
        corrections = htoe_data.get("corrections", [])
        htoe_dict: dict[str, str | list[str]] = {}
        for entry in corrections:
            incorrect = entry.get("incorrect", "")
            correct = entry.get("correct")
            if incorrect and correct:
                htoe_dict[incorrect] = correct
        result["ha_htoe_particles"] = norm_dict(htoe_dict)

        # Exclusions
        exclusions = htoe_data.get("exclusions", [])
        result["ha_htoe_exclusions"] = norm_set(exclusions)

        # Honorific prefixes
        prefixes = htoe_data.get("honorific_prefixes", [])
        result["honorific_prefixes"] = norm_set(prefixes)

    # -- Dangling particles --
    raw_dangling = data.get("dangling_particles", [])
    dangling = [entry.get("particle", "") for entry in raw_dangling]
    result["dangling_particles"] = norm_set([p for p in dangling if p])

    # -- Suppression suffixes --
    raw_suppression = data.get("suppression_suffixes", [])
    suffixes = [entry.get("suffix", "") for entry in raw_suppression]
    result["suppression_suffixes"] = tuple(normalize(s) for s in suffixes if s)

    # -- Syllable finals --
    raw_finals = data.get("syllable_finals", [])
    finals = [entry.get("char", "") for entry in raw_finals]
    result["syllable_finals"] = frozenset(c for c in finals if c)

    # -- Visarga exclude pronouns --
    raw_ve = data.get("visarga_exclude_pronouns", [])
    result["visarga_exclude_pronouns"] = norm_set(raw_ve)

    # -- Subject pronouns --
    raw_sp = data.get("subject_pronouns", [])
    result["subject_pronouns"] = norm_set(raw_sp)

    # -- Missing asat particles --
    raw_asat = data.get("missing_asat_particles", [])
    asat_dict: dict[str, str | list[str]] = {}
    for entry in raw_asat:
        incorrect = entry.get("incorrect", "")
        correct = entry.get("correct")
        if incorrect and correct:
            asat_dict[incorrect] = correct
    result["missing_asat_particles"] = norm_dict(asat_dict)

    # -- Missing visarga suffixes --
    raw_vis = data.get("missing_visarga_suffixes", [])
    vis_dict: dict[str, str | list[str]] = {}
    for entry in raw_vis:
        incorrect = entry.get("incorrect", "")
        correct = entry.get("correct")
        if incorrect and correct:
            vis_dict[incorrect] = correct
    result["missing_visarga_suffixes"] = norm_dict(vis_dict)

    # -- Locative exempt prefixes --
    raw_locative = data.get("locative_exempt_prefixes", [])
    if isinstance(raw_locative, list) and raw_locative:
        result["locative_exempt_prefixes"] = tuple(normalize(p) for p in raw_locative if p)

    # -- Particle misuse rules (verb-frame-based) --
    raw_misuse = data.get("particle_misuse_rules", [])
    misuse_rules: list[dict[str, Any]] = []
    for entry in raw_misuse:
        wrong = entry.get("wrong_particle", "")
        correct = entry.get("correct_particle", "")
        triggers = entry.get("verb_triggers", [])
        max_window = entry.get("max_window", 5)
        require_prior = entry.get("require_prior_marker", "")
        if wrong and correct and triggers:
            rule_dict: dict[str, Any] = {
                "wrong_particle": normalize(wrong),
                "correct_particle": normalize(correct),
                "verb_triggers": tuple(normalize(t) for t in triggers),
                "max_window": max_window,
            }
            if require_prior:
                rule_dict["require_prior_marker"] = normalize(require_prior)
            misuse_rules.append(rule_dict)
    result["particle_misuse_rules"] = misuse_rules

    return result


# ── Homophone context rules (homophones.yaml) ──


def load_homophone_context(
    path: Path | None = None,
) -> dict[str, Any]:
    """Load context-dependent homophone rules from homophones.yaml.

    Returns a dict with keys:
      - homophone_left_context: dict[str, tuple[str, tuple[str, ...]]]
      - homophone_left_suffixes: dict[str, tuple[str, tuple[str, ...]]]
    """
    result: dict[str, Any] = {
        "homophone_left_context": {},
        "homophone_left_suffixes": {},
    }

    yaml_path = path or (_RULES_DIR / "homophones.yaml")
    data = _load_yaml(yaml_path, "homophones")
    if not data:
        return result

    ctx_section = data.get("context_dependent")
    if not ctx_section or not isinstance(ctx_section, dict):
        return result

    # -- Left context rules --
    left_rules = ctx_section.get("left_context_rules", [])
    context_raw: dict[str, tuple[str, tuple[str, ...]]] = {}
    suffix_raw: dict[str, tuple[str, tuple[str, ...]]] = {}

    for rule in left_rules:
        wrong = rule.get("wrong", "")
        correct = rule.get("correct", "")
        if not (wrong and correct):
            continue

        words = rule.get("left_context_words", [])
        if words:
            context_raw[wrong] = (correct, tuple(words))

        suffixes = rule.get("left_context_suffixes", [])
        if suffixes:
            suffix_raw[wrong] = (correct, tuple(suffixes))

    result["homophone_left_context"] = norm_dict_context(context_raw)
    result["homophone_left_suffixes"] = norm_dict_context(suffix_raw)

    return result


# ── Collocation rules (collocations.yaml) ──


def load_collocation_rules(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load collocation error detection rules from collocations.yaml.

    Returns a list of collocation rule dicts, each with keys:
      - wrong_word: str  (the wrong word to look for)
      - correct_word: str  (the suggested correction)
      - context_words: tuple[str, ...]  (context words to match)
      - direction: str  ("left", "right", or "both")
      - window: int  (how many tokens to scan)
    """
    yaml_path = path or (_RULES_DIR / "collocations.yaml")
    data = _load_yaml(yaml_path, "collocations")
    if not data:
        return []

    rules: list[dict[str, Any]] = []
    for entry in data.get("collocations", []):
        wrong = entry.get("wrong_word", "")
        correct = entry.get("correct_word", "")
        context = entry.get("context_words", [])
        direction = entry.get("direction", "both")
        window = entry.get("window", 3)
        if not (wrong and correct and context):
            continue
        rules.append(
            {
                "wrong_word": normalize(wrong),
                "correct_word": normalize(correct),
                "context_words": tuple(normalize(w) for w in context),
                "direction": direction,
                "window": window,
            }
        )

    logger.debug("Loaded %d collocation rules", len(rules))
    return rules


# ── Convenience: load all detection rules at once ──


class DetectionRules:
    """Container for all text-level detection rule dicts.

    Loads from YAML at construction time. Falls back to empty dicts
    if any file is missing or invalid.

    Attributes mirror the class-level dicts in SpellChecker:
      - medial_confusion_unconditional
      - medial_confusion_contextual
      - aukmyit_context
      - extra_aukmyit_context
      - vowel_reorder_errors
      - colloquial_contractions
      - stacking_completions
      - ha_htoe_compounds
      - aspirated_compounds
      - consonant_confusion_compounds
      - particle_confusion
      - sequential_particle_left_context
      - ha_htoe_particles
      - ha_htoe_exclusions
      - honorific_prefixes
      - dangling_particles
      - suppression_suffixes
      - syllable_finals
      - visarga_exclude_pronouns
      - subject_pronouns
      - missing_asat_particles
      - missing_visarga_suffixes
      - homophone_left_context
      - homophone_left_suffixes
      - collocation_rules
    """

    def __init__(
        self,
        orthographic_path: Path | None = None,
        compounds_path: Path | None = None,
        particles_path: Path | None = None,
        homophones_path: Path | None = None,
        collocations_path: Path | None = None,
    ) -> None:
        ortho = load_orthographic_corrections(orthographic_path)
        self.medial_confusion_unconditional: dict[str, str | list[str]] = ortho[
            "medial_confusion_unconditional"
        ]
        self.medial_confusion_contextual: dict[str, tuple[str, tuple[str, ...]]] = ortho[
            "medial_confusion_contextual"
        ]
        self.aukmyit_context: dict[str, tuple[str, tuple[str, ...]]] = ortho["aukmyit_context"]
        self.extra_aukmyit_context: dict[str, tuple[str, tuple[str, ...]]] = ortho[
            "extra_aukmyit_context"
        ]
        self.vowel_reorder_errors: dict[str, list[str]] = ortho["vowel_reorder_errors"]
        self.colloquial_contractions: dict[str, str | list[str]] = ortho["colloquial_contractions"]
        self.stacking_completions: dict[str, str] = ortho["stacking_completions"]

        compounds = load_compound_confusion(compounds_path)
        self.ha_htoe_compounds: dict[str, tuple[str, str]] = compounds["ha_htoe_compounds"]
        self.aspirated_compounds: dict[str, tuple[str, str]] = compounds["aspirated_compounds"]
        self.consonant_confusion_compounds: dict[str, tuple[str, str]] = compounds[
            "consonant_confusion_compounds"
        ]

        particles = load_particle_confusion(particles_path)
        self.particle_confusion: dict[str, str | list[str]] = particles["particle_confusion"]
        self.sequential_particle_left_context: tuple[str, ...] = particles[
            "sequential_particle_left_context"
        ]
        self.ha_htoe_particles: dict[str, str | list[str]] = particles["ha_htoe_particles"]
        self.ha_htoe_exclusions: frozenset[str] = particles["ha_htoe_exclusions"]
        self.honorific_prefixes: frozenset[str] = particles["honorific_prefixes"]
        self.dangling_particles: frozenset[str] = particles["dangling_particles"]
        self.suppression_suffixes: tuple[str, ...] = particles["suppression_suffixes"]
        self.syllable_finals: frozenset[str] = particles["syllable_finals"]
        self.visarga_exclude_pronouns: frozenset[str] = particles["visarga_exclude_pronouns"]
        self.subject_pronouns: frozenset[str] = particles["subject_pronouns"]
        self.missing_asat_particles: dict[str, str | list[str]] = particles[
            "missing_asat_particles"
        ]
        self.missing_visarga_suffixes: dict[str, str | list[str]] = particles[
            "missing_visarga_suffixes"
        ]
        self.particle_misuse_rules: list[dict[str, Any]] = particles["particle_misuse_rules"]
        self.locative_exempt_prefixes: tuple[str, ...] = particles["locative_exempt_prefixes"]

        homophones = load_homophone_context(homophones_path)
        self.homophone_left_context: dict[str, tuple[str, tuple[str, ...]]] = homophones[
            "homophone_left_context"
        ]
        self.homophone_left_suffixes: dict[str, tuple[str, tuple[str, ...]]] = homophones[
            "homophone_left_suffixes"
        ]

        self.collocation_rules: list[dict[str, Any]] = load_collocation_rules(collocations_path)


# ── Rerank rules (rerank_rules.yaml) ──


def _norm_cue(cue: dict[str, Any]) -> dict[str, Any]:
    """Normalize text fields in a contextual cue entry."""
    out: dict[str, Any] = {}
    for k, v in cue.items():
        if k == "target" and isinstance(v, str):
            out[k] = normalize(v)
        elif k in ("sentence_contains", "sentence_contains_all") and isinstance(v, list):
            out[k] = tuple(normalize(s) for s in v)
        else:
            out[k] = v
    return out


def load_rerank_rules(
    path: Path | None = None,
) -> dict[str, Any]:
    """Load rerank rule data from rerank_rules.yaml.

    Returns a dict with keys:
      - simple_injections: dict[str, str]  (norm_error -> target_surface)
      - simple_injections_no_suggestions: dict[str, str]
      - simple_injections_conditional: list[dict]  (match_mode/error_type variants)
      - contextual_promotions: list[dict]
      - semantic_context_injections: list[dict]
      - disambiguation_promotions: list[dict]
      - error_type_injections: list[dict]
      - delete_on_error_types: frozenset[str]
      - surface_rewrites: list[dict]
      - post_normalizers: list[dict]
      - sequential_particle_noun_exclusions: tuple[str, ...]
      - school_context_tail_cues: tuple[str, ...]
      - second_person_pronouns: tuple[str, ...]
      - targeted_top1_hints: dict[str, tuple[str, ...]]
      - parametric_rules: list[dict]
    """
    result: dict[str, Any] = {
        "simple_injections": {},
        "simple_injections_no_suggestions": {},
        "simple_injections_conditional": [],
        "contextual_promotions": [],
        "semantic_context_injections": [],
        "disambiguation_promotions": [],
        "error_type_injections": [],
        "delete_on_error_types": frozenset(),
        "delete_on_error_type_rule_ids": {},
        "surface_rewrites": [],
        "post_normalizers": [],
        "sequential_particle_noun_exclusions": (),
        "school_context_tail_cues": (),
        "second_person_pronouns": (),
        "targeted_top1_hints": {},
        "parametric_rules": [],
    }

    yaml_path = path or (_RULES_DIR / "rerank_rules.yaml")
    data = _load_yaml(yaml_path, "rerank_rules")
    if not data:
        return result

    # -- Simple injections --
    simple: dict[str, str] = {}
    simple_no_sugg: dict[str, str] = {}
    conditional: list[dict[str, Any]] = []
    for entry in data.get("simple_injections", []):
        error = entry.get("error", "")
        target = entry.get("target", "")
        if not (error and target is not None):
            continue
        match_mode = entry.get("match_mode", "exact")
        error_type = entry.get("error_type")
        no_suggestions = entry.get("no_suggestions", False)

        if match_mode != "exact" or error_type:
            conditional.append(
                {
                    "error": normalize(error),
                    "target": target,
                    "match_mode": match_mode,
                    "error_type": error_type,
                }
            )
        elif no_suggestions:
            simple_no_sugg[normalize(error)] = target
        else:
            simple[normalize(error)] = target
    result["simple_injections"] = simple
    result["simple_injections_no_suggestions"] = simple_no_sugg
    result["simple_injections_conditional"] = conditional

    # -- Contextual promotions --
    ctx_promos: list[dict[str, Any]] = []
    for entry in data.get("contextual_promotions", []):
        error = entry.get("error", "")
        default_target = entry.get("default_target", "")
        if not error:
            continue
        raw_cues = entry.get("cues", [])
        cues = [_norm_cue(c) for c in raw_cues]
        ctx_promos.append(
            {
                "error": normalize(error),
                "cues": cues,
                "default_target": normalize(default_target) if default_target else "",
                "promote_if_needed": entry.get("promote_if_needed", False),
                "default_rule_id": entry.get("default_rule_id"),
            }
        )
    result["contextual_promotions"] = ctx_promos

    # -- Semantic context injections --
    semantic: list[dict[str, Any]] = []
    for entry in data.get("semantic_context_injections", []):
        requires = entry.get("sentence_requires_all", [])
        target = entry.get("target", "")
        if not (requires and target):
            continue
        semantic.append(
            {
                "sentence_requires_all": tuple(normalize(s) for s in requires),
                "target_surface": target,
                "target_normalized": normalize(target),
            }
        )
    result["semantic_context_injections"] = semantic

    # -- Disambiguation promotions --
    _DISAMBIG_TEXT_KEYS = (
        "error",
        "suggestion_contains",
        "suggestion_prefix",
        "target",
        "sentence_contains",
        "top1_was",
    )
    _DISAMBIG_LIST_KEYS = (
        "suggestion_contains_any",
        "suggestion_contains_all",
        "suggestion_exact",
    )
    disambig: list[dict[str, Any]] = []
    for entry in data.get("disambiguation_promotions", []):
        error = entry.get("error", "")
        target = entry.get("target", "")
        if not (error and target):
            continue
        out: dict[str, Any] = {"action": entry.get("action", "promote")}
        for k in _DISAMBIG_TEXT_KEYS:
            v = entry.get(k)
            if v is not None and isinstance(v, str):
                out[k] = normalize(v)
        for k in _DISAMBIG_LIST_KEYS:
            v = entry.get(k)
            if v is not None and isinstance(v, list):
                out[k] = tuple(normalize(s) for s in v)
        disambig.append(out)
    result["disambiguation_promotions"] = disambig

    # -- Error-type conditioned injections --
    _ERRTYPE_TEXT_KEYS = ("error", "token_contains")
    _ERRTYPE_LIST_KEYS = ("sentence_contains", "sentence_contains_all", "suggestion_exact")
    errtype: list[dict[str, Any]] = []
    for entry in data.get("error_type_injections", []):
        error = entry.get("error", "")
        target = entry.get("target", "")
        if not (error and target is not None):
            continue
        out_e: dict[str, Any] = {
            "target": target,
            "match_mode": entry.get("match_mode", "exact"),
            "error_type": entry.get("error_type"),
            "no_suggestions": entry.get("no_suggestions", False),
            "second_person_context": entry.get("second_person_context", False),
            "rule_id": entry.get("rule_id"),
        }
        for k in _ERRTYPE_TEXT_KEYS:
            v = entry.get(k)
            if v is not None and isinstance(v, str):
                out_e[k] = normalize(v)
        for k in _ERRTYPE_LIST_KEYS:
            v = entry.get(k)
            if v is not None and isinstance(v, list):
                out_e[k] = tuple(normalize(s) for s in v)
        errtype.append(out_e)
    result["error_type_injections"] = errtype

    # -- Delete-on-error-type --
    raw_delete = data.get("delete_on_error_type", [])
    delete_types: list[str] = []
    delete_rule_ids: dict[str, str] = {}
    for entry in raw_delete:
        etype = entry.get("error_type", "") if isinstance(entry, dict) else str(entry)
        if etype:
            delete_types.append(etype)
            rule_id = entry.get("rule_id", "") if isinstance(entry, dict) else ""
            if rule_id:
                delete_rule_ids[etype] = rule_id
    result["delete_on_error_types"] = frozenset(delete_types)
    result["delete_on_error_type_rule_ids"] = delete_rule_ids

    # -- Surface rewrites --
    rewrites: list[dict[str, Any]] = []
    for entry in data.get("surface_rewrites", []):
        trigger = entry.get("trigger_error", "")
        malformed = entry.get("malformed", "")
        corrected = entry.get("corrected", "")
        if not (trigger and malformed and corrected):
            continue
        rewrites.append(
            {
                "trigger_error": normalize(trigger),
                "malformed": normalize(malformed),
                "corrected_surface": corrected,
                "corrected_normalized": normalize(corrected),
            }
        )
    result["surface_rewrites"] = rewrites

    # -- Post-injection normalizers --
    post: list[dict[str, Any]] = []
    for entry in data.get("post_normalizers", []):
        error = entry.get("error", "")
        target = entry.get("target", "")
        if not error:
            continue
        post.append(
            {
                "error": normalize(error),
                "target_surface": target,
                "target_normalized": normalize(target) if target else "",
                "top1_was": normalize(entry["top1_was"]) if entry.get("top1_was") else None,
                "action": entry.get("action", "inject"),
            }
        )
    result["post_normalizers"] = post

    # -- Auxiliary data --
    result["sequential_particle_noun_exclusions"] = tuple(
        normalize(s) for s in data.get("sequential_particle_noun_exclusions", [])
    )
    result["school_context_tail_cues"] = tuple(
        normalize(s) for s in data.get("school_context_tail_cues", [])
    )
    result["second_person_pronouns"] = tuple(
        normalize(s) for s in data.get("second_person_pronouns", [])
    )

    # -- Targeted top-1 hints --
    hints: dict[str, tuple[str, ...]] = {}
    for entry in data.get("targeted_top1_hints", []):
        error = entry.get("error", "")
        targets = entry.get("targets", [])
        if error and targets:
            hints[normalize(error)] = tuple(normalize(t) for t in targets)
    result["targeted_top1_hints"] = hints

    # -- Parametric rules --
    parametric: list[dict[str, Any]] = []
    for entry in data.get("parametric_rules", []):
        rule_id = entry.get("id", "")
        template = entry.get("template", "")
        params = entry.get("params", {})
        if not (rule_id and template):
            continue
        parametric.append(
            {
                "id": rule_id,
                "template": template,
                "enabled": entry.get("enabled", True),
                "params": params,
            }
        )
    result["parametric_rules"] = parametric

    return result


class RerankRulesData:
    """Container for rerank rule data loaded from YAML.

    Attributes match the sections in rerank_rules.yaml.
    All Myanmar text is normalized at load time.
    """

    def __init__(self, path: Path | None = None) -> None:
        data = load_rerank_rules(path)

        self.simple_injections: dict[str, str] = data["simple_injections"]
        self.simple_injections_no_suggestions: dict[str, str] = data[
            "simple_injections_no_suggestions"
        ]
        self.simple_injections_conditional: list[dict[str, Any]] = data[
            "simple_injections_conditional"
        ]
        self.contextual_promotions: list[dict[str, Any]] = data["contextual_promotions"]
        self.semantic_context_injections: list[dict[str, Any]] = data["semantic_context_injections"]
        self.disambiguation_promotions: list[dict[str, Any]] = data["disambiguation_promotions"]
        self.error_type_injections: list[dict[str, Any]] = data["error_type_injections"]
        self.delete_on_error_types: frozenset[str] = data["delete_on_error_types"]
        self.delete_on_error_type_rule_ids: dict[str, str] = data["delete_on_error_type_rule_ids"]
        self.surface_rewrites: list[dict[str, Any]] = data["surface_rewrites"]
        self.post_normalizers: list[dict[str, Any]] = data["post_normalizers"]
        self.sequential_particle_noun_exclusions: tuple[str, ...] = data[
            "sequential_particle_noun_exclusions"
        ]
        self.school_context_tail_cues: tuple[str, ...] = data["school_context_tail_cues"]
        self.second_person_pronouns: tuple[str, ...] = data["second_person_pronouns"]
        self.targeted_top1_hints: dict[str, tuple[str, ...]] = data["targeted_top1_hints"]
        self.parametric_rules: list[dict[str, Any]] = data["parametric_rules"]


# ── Stacking pairs (stacking_pairs.yaml) ──


def load_stacking_pairs(path: Path | None = None) -> set[tuple[str, str]]:
    """Load valid virama stacking pairs from YAML config.

    Returns a set of (upper, lower) consonant tuples that can legally stack
    via virama (U+1039). If the YAML file is missing or fails to load,
    falls back to the hardcoded ``STACKING_EXCEPTIONS`` from myanmar_constants.

    The YAML file organizes pairs into categories (gemination, cross_aspirated,
    cross_row, etc.), each with an ``enabled`` flag allowing categories to be
    toggled without removing entries.

    Args:
        path: Optional path to stacking_pairs.yaml. Defaults to
            ``rules/stacking_pairs.yaml``.

    Returns:
        Frozenset of (upper_consonant, lower_consonant) string tuples.
    """
    from myspellchecker.core.constants.myanmar_constants import STACKING_EXCEPTIONS

    yaml_path = path or (_RULES_DIR / "stacking_pairs.yaml")
    data = _load_yaml(yaml_path, "stacking_pairs")
    if not data:
        return set(STACKING_EXCEPTIONS)

    stacking_data = data.get("stacking_pairs", {})
    if not stacking_data:
        return set(STACKING_EXCEPTIONS)

    pairs: set[tuple[str, str]] = set()
    for category_name, category in stacking_data.items():
        if not isinstance(category, dict):
            continue
        if not category.get("enabled", True):
            logger.debug("Stacking category '%s' disabled, skipping", category_name)
            continue
        for pair in category.get("pairs", []):
            if isinstance(pair, list) and len(pair) == 2:
                pairs.add((pair[0], pair[1]))

    if not pairs:
        logger.warning("No stacking pairs loaded from YAML, using hardcoded defaults")
        return set(STACKING_EXCEPTIONS)

    return pairs


# ── Curated confusable pairs (confusable_pairs.yaml) ──


def load_confusable_pairs(
    path: Path | None = None,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Load curated confusable word pairs from YAML config.

    Returns two bidirectional mappings:
    1. General curated pairs (non-near-synonym)
    2. Near-synonym pairs (confusion_type == "near_synonym")

    Both have the shape ``{word: {variant1, variant2, ...}, ...}``.
    These pairs represent known real-word confusions that cannot be generated
    by the standard phonetic/visual variant rules. They are used by
    ConfusableSemanticStrategy with different MLM thresholds:
    - General curated: ``curated_logit_diff_threshold`` (default 2.0)
    - Near-synonym: ``near_synonym_logit_diff_threshold`` (default 3.0)

    Args:
        path: Optional path to confusable_pairs.yaml. Defaults to
            ``rules/confusable_pairs.yaml``.

    Returns:
        Tuple of (curated_pairs, near_synonym_pairs) dicts, each mapping
        words to their sets of curated confusable variants.
    """
    yaml_path = path or (_RULES_DIR / "confusable_pairs.yaml")
    data = _load_yaml(yaml_path, "confusable_pairs")
    if not data:
        return {}, {}

    curated: dict[str, set[str]] = {}
    near_synonym: dict[str, set[str]] = {}
    for entry in data.get("pairs", []):
        word = entry.get("word", "")
        correction = entry.get("correction", "")
        if not (word and correction):
            continue

        word_norm = normalize(word)
        correction_norm = normalize(correction)

        # Route to near_synonym or general curated dict
        confusion_type = entry.get("confusion_type", "")
        target = near_synonym if confusion_type == "near_synonym" else curated

        # Add forward direction
        target.setdefault(word_norm, set()).add(correction_norm)

        # Add reverse direction if bidirectional
        if entry.get("bidirectional", False):
            target.setdefault(correction_norm, set()).add(word_norm)

    if curated:
        total = sum(len(v) for v in curated.values())
        logger.debug(
            "Loaded %d curated confusable pair entries (%d words)",
            total,
            len(curated),
        )
    if near_synonym:
        total = sum(len(v) for v in near_synonym.values())
        logger.debug(
            "Loaded %d near-synonym confusable pair entries (%d words)",
            total,
            len(near_synonym),
        )

    return curated, near_synonym


# ── Confusion matrix (confusion_matrix.yaml) ──


def load_confusion_matrix(
    path: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Load data-driven substitution costs from YAML and merge with defaults.

    Returns a dict in the same format as ``MYANMAR_SUBSTITUTION_COSTS``:
    ``{from_char: {to_char: cost, ...}, ...}``.

    The YAML defines additional pairs not in the hardcoded matrix plus
    cost refinements for existing pairs. The returned dict contains
    the merged result (YAML additions override hardcoded values when
    both define the same pair).

    Args:
        path: Optional path to confusion_matrix.yaml. Defaults to
            ``rules/confusion_matrix.yaml``.

    Returns:
        Merged substitution cost dict.
    """
    from myspellchecker.text.phonetic_data import MYANMAR_SUBSTITUTION_COSTS

    # Start with a deep copy of the hardcoded costs
    merged: dict[str, dict[str, float]] = {}
    for k, v in MYANMAR_SUBSTITUTION_COSTS.items():
        merged[k] = dict(v)

    yaml_path = path or (_RULES_DIR / "confusion_matrix.yaml")
    data = _load_yaml(yaml_path, "confusion_matrix")
    if not data:
        return merged

    # Merge additional substitution costs
    for entry in data.get("additional_substitution_costs", []):
        from_char = entry.get("from", "")
        to_char = entry.get("to", "")
        cost = float(entry.get("cost", 0.5))
        bidirectional = entry.get("bidirectional", False)
        if from_char and to_char:
            merged.setdefault(from_char, {})[to_char] = cost
            if bidirectional:
                merged.setdefault(to_char, {})[from_char] = cost

    # Apply cost refinements
    for entry in data.get("cost_refinements", []):
        from_char = entry.get("from", "")
        to_char = entry.get("to", "")
        suggested = float(entry.get("suggested_cost", 0.5))
        if from_char and to_char:
            merged.setdefault(from_char, {})[to_char] = suggested
            # Refinements are always bidirectional
            merged.setdefault(to_char, {})[from_char] = suggested

    additions = len(data.get("additional_substitution_costs", []))
    refinements = len(data.get("cost_refinements", []))
    logger.debug(
        "Confusion matrix: %d additions, %d refinements merged",
        additions,
        refinements,
    )

    return merged
