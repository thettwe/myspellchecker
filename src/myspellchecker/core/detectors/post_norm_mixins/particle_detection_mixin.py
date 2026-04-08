"""Particle detection mixin for PostNormalizationDetectorsMixin.

Provides particle-related detection methods and their data constants:
``_detect_missing_asat``, ``_detect_missing_visarga_suffix``,
``_detect_particle_confusion``, ``_detect_sequential_particle_confusion``,
``_detect_particle_misuse``,
``_detect_ha_htoe_particle_typos``, ``_detect_dangling_particles``.

Data is loaded from ``rules/particles.yaml`` at module level, with
fallback to hardcoded defaults if the YAML file is missing or invalid.
At runtime, ``SpellChecker.__init__`` may further override these with
values from ``detection_rules.load_particle_confusion()``.

Extracted from ``post_normalization.py`` to reduce file size while
preserving the exact same method signatures and behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from myspellchecker.core.constants import (
    CORE_RESPECTFUL_TITLES,
    ET_DANGLING_PARTICLE,
    ET_HA_HTOE_CONFUSION,
    ET_MISSING_ASAT,
    ET_PARTICLE_CONFUSION,
    ET_PARTICLE_MISUSE,
    ET_SYLLABLE,
    ET_WORD,
)
from myspellchecker.core.constants.detector_thresholds import (
    DEFAULT_PARTICLE_THRESHOLDS,
    ParticleDetectionThresholds,
)

if TYPE_CHECKING:
    from myspellchecker.providers.base import DictionaryProvider
from myspellchecker.core.detector_data import (
    STRUCTURAL_ERROR_TYPES as _STRUCTURAL_ERROR_TYPES,
)
from myspellchecker.core.detector_data import (
    TEXT_DETECTOR_CONFIDENCES,
)
from myspellchecker.core.detector_data import (
    norm_dict as _norm_dict,
)
from myspellchecker.core.detector_data import (
    norm_set as _norm_set,
)
from myspellchecker.core.detectors.utils import (
    get_existing_positions,
    iter_occurrences,
    try_replace_syllable_error,
)
from myspellchecker.core.response import Error, SyllableError
from myspellchecker.text.normalize import normalize
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Hardcoded defaults (fallback when YAML is unavailable) ──

_DEFAULT_DANGLING_PARTICLES: frozenset[str] = _norm_set(
    {
        "ကြောင့်",  # because (causal)
        "ကြောင်း",  # that / because (complement/causal)
        "ပေမယ့်",  # although
        "ဒါပေမယ့်",  # but/however
        "အတွက်",  # for / in order to
    }
)

_DEFAULT_MISSING_ASAT_PARTICLES: dict[str, str | list[str]] = _norm_dict(
    {
        "တယ": "တယ်",
        "မယ": "မယ်",
        "သည": "သည်",
    }
)

_DEFAULT_MISSING_VISARGA_SUFFIXES: dict[str, str | list[str]] = _norm_dict({"လို": "လို့"})

_DEFAULT_HA_HTOE_EXCLUSIONS: frozenset[str] = _norm_set(
    {
        "ဥပမာ",  # discourse marker "for example"
        "မြန်မာ",  # Myanmar (Unicode)
        "ျမန္မာ",  # Myanmar (Zawgyi)
        "ျမန်မာ",  # Myanmar (mixed encoding)
        "အမာ",  # tumor, hard thing
        "စိတ်မာ",  # stubborn/hard-hearted
        "ခေါင်းမာ",  # stubborn (lit. hard-headed)
        "အသားမာ",  # tough (meat)
        "ကျောက်မာ",  # gem, hard stone
        "လက်မာ",  # skilled/heavy-handed
        "သံမာ",  # hard metal/iron
        "နှလုံးမာ",  # hard-hearted
        "ကြေးမာ",  # hardened copper
        "ခြေမာ",  # strong-footed
        "သတ္တိမာ",  # brave/courageous
        "အရိုးမာ",  # hard-boned
        "ဉာဏ်မာ",  # sharp-minded
        "ခင်မာ",  # common female name (Khin Ma)
    }
)

_DEFAULT_LOCATIVE_EXEMPT_PREFIXES: tuple[str, ...] = (
    "ထဲ",  # inside
    "ပေါ်",  # on top
    "အောက်",  # under
    "ရှေ့",  # in front
    "နောက်",  # behind
    "ဘေး",  # beside
    "အနား",  # near
    "အထဲ",  # inside (formal)
    "အပေါ်",  # on top (formal)
)

_DEFAULT_SEQUENTIAL_PARTICLE_LEFT_CONTEXT: tuple[str, ...] = tuple(
    normalize(p)
    for p in (
        "ပြီးတော်",  # completed + then
        "ရောက်တော်",  # arrive + then
        "တော်ပြီ",  # (wrong) should be: တော့ + completed
        "ရတော်",  # get/obtain + then
        "ပြောတော်",  # say + then
        "လာတော်",  # come + then
        "သွားတော်",  # go + then
        "ဖြစ်တော်",  # happen/be + then
        "ပေးတော်",  # give + then
        "လုပ်တော်",  # do + then
        "ထားတော်",  # put/keep + then
    )
)

_DEFAULT_VISARGA_EXCLUDE_PRONOUNS: frozenset[str] = _norm_set(
    {"ငါ", "နင်", "မင်း", "သူ", "သူမ", "ကျွန်တော်", "ကျွန်မ"}
)

# ── YAML loading ──

_YAML_PATH = Path(__file__).resolve().parent.parent.parent.parent / "rules" / "particles.yaml"


def _load_particle_data() -> tuple[
    frozenset[str],  # dangling_particles
    dict[str, str | list[str]],  # missing_asat_particles
    dict[str, str | list[str]],  # missing_visarga_suffixes
    frozenset[str],  # ha_htoe_exclusions
    tuple[str, ...],  # locative_exempt_prefixes
    tuple[str, ...],  # sequential_particle_left_context
    frozenset[str],  # visarga_exclude_pronouns
]:
    """Load particle detection data from YAML with fallback to defaults.

    Returns:
        Tuple of data structures for each particle data category.
    """
    defaults = (
        _DEFAULT_DANGLING_PARTICLES,
        _DEFAULT_MISSING_ASAT_PARTICLES,
        _DEFAULT_MISSING_VISARGA_SUFFIXES,
        _DEFAULT_HA_HTOE_EXCLUSIONS,
        _DEFAULT_LOCATIVE_EXEMPT_PREFIXES,
        _DEFAULT_SEQUENTIAL_PARTICLE_LEFT_CONTEXT,
        _DEFAULT_VISARGA_EXCLUDE_PRONOUNS,
    )

    if not _YAML_PATH.exists():
        logger.debug(
            "Particles YAML not found at %s, using defaults",
            _YAML_PATH,
        )
        return defaults

    try:
        import yaml

        with open(_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            logger.warning("Particles YAML empty or invalid, using defaults")
            return defaults

        # -- Dangling particles --
        raw_dangling = data.get("dangling_particles", [])
        if isinstance(raw_dangling, list) and raw_dangling:
            dangling = [
                entry.get("particle", "") if isinstance(entry, dict) else ""
                for entry in raw_dangling
            ]
            dangling_particles = _norm_set([p for p in dangling if p])
        else:
            dangling_particles = _DEFAULT_DANGLING_PARTICLES

        # -- Missing asat particles --
        raw_asat = data.get("missing_asat_particles", [])
        if isinstance(raw_asat, list) and raw_asat:
            asat_dict: dict[str, str | list[str]] = {}
            for entry in raw_asat:
                if isinstance(entry, dict):
                    incorrect = entry.get("incorrect", "")
                    correct = entry.get("correct")
                    if incorrect and correct:
                        asat_dict[incorrect] = correct
            missing_asat = _norm_dict(asat_dict) if asat_dict else _DEFAULT_MISSING_ASAT_PARTICLES
        else:
            missing_asat = _DEFAULT_MISSING_ASAT_PARTICLES

        # -- Missing visarga suffixes --
        raw_vis = data.get("missing_visarga_suffixes", [])
        if isinstance(raw_vis, list) and raw_vis:
            vis_dict: dict[str, str | list[str]] = {}
            for entry in raw_vis:
                if isinstance(entry, dict):
                    incorrect = entry.get("incorrect", "")
                    correct = entry.get("correct")
                    if incorrect and correct:
                        vis_dict[incorrect] = correct
            missing_visarga = (
                _norm_dict(vis_dict) if vis_dict else _DEFAULT_MISSING_VISARGA_SUFFIXES
            )
        else:
            missing_visarga = _DEFAULT_MISSING_VISARGA_SUFFIXES

        # -- Ha-htoe exclusions --
        htoe_data = data.get("ha_htoe_particle_confusion", {})
        if isinstance(htoe_data, dict):
            raw_excl = htoe_data.get("exclusions", [])
            ha_htoe_exclusions = (
                _norm_set(raw_excl)
                if isinstance(raw_excl, list) and raw_excl
                else _DEFAULT_HA_HTOE_EXCLUSIONS
            )
        else:
            ha_htoe_exclusions = _DEFAULT_HA_HTOE_EXCLUSIONS

        # -- Locative exempt prefixes --
        raw_locative = data.get("locative_exempt_prefixes", [])
        if isinstance(raw_locative, list) and raw_locative:
            locative_exempt = tuple(raw_locative)
        else:
            locative_exempt = _DEFAULT_LOCATIVE_EXEMPT_PREFIXES

        # -- Sequential particle left context --
        seq_data = data.get("sequential_particle_confusion", {})
        if isinstance(seq_data, dict):
            triggers = seq_data.get("left_context_triggers", [])
            seq_context = (
                tuple(normalize(p) for p in triggers)
                if isinstance(triggers, list) and triggers
                else _DEFAULT_SEQUENTIAL_PARTICLE_LEFT_CONTEXT
            )
        else:
            seq_context = _DEFAULT_SEQUENTIAL_PARTICLE_LEFT_CONTEXT

        # -- Visarga exclude pronouns --
        raw_ve = data.get("visarga_exclude_pronouns", [])
        if isinstance(raw_ve, list) and raw_ve:
            visarga_exclude = _norm_set(raw_ve)
        else:
            visarga_exclude = _DEFAULT_VISARGA_EXCLUDE_PRONOUNS

        logger.debug(
            "Loaded particle data from YAML: "
            "%d dangling, %d missing_asat, %d missing_visarga, "
            "%d ha_htoe_exclusions, %d locative_exempt, "
            "%d seq_context, %d visarga_exclude",
            len(dangling_particles),
            len(missing_asat),
            len(missing_visarga),
            len(ha_htoe_exclusions),
            len(locative_exempt),
            len(seq_context),
            len(visarga_exclude),
        )
        return (
            dangling_particles,
            missing_asat,
            missing_visarga,
            ha_htoe_exclusions,
            locative_exempt,
            seq_context,
            visarga_exclude,
        )

    except Exception:
        logger.warning(
            "Failed to load particles YAML, using defaults",
            exc_info=True,
        )
        return defaults


# Load at module level (once, at import time)
(
    _DANGLING_PARTICLES_LOADED,
    _MISSING_ASAT_PARTICLES_LOADED,
    _MISSING_VISARGA_SUFFIXES_LOADED,
    _HA_HTOE_EXCLUSIONS_LOADED,
    _LOCATIVE_EXEMPT_PREFIXES_LOADED,
    _SEQUENTIAL_PARTICLE_LEFT_CONTEXT_LOADED,
    _VISARGA_EXCLUDE_PRONOUNS_LOADED,
) = _load_particle_data()


class ParticleDetectionMixin:
    """Mixin providing particle-related detection methods.

    Detects missing asat, missing visarga, particle confusion,
    sequential particle confusion, ha-htoe particles, and
    dangling particles.

    Data is loaded from ``rules/particles.yaml`` at module level
    (with hardcoded fallbacks). At runtime, ``SpellChecker.__init__``
    may further override via ``detection_rules.load_particle_confusion()``.
    """

    # --- Type stubs for attributes provided by SpellChecker ---
    provider: "DictionaryProvider"
    _COLLOQUIAL_ENDINGS_WITH_STRIPPED: frozenset[str]
    _FORMAL_ENDINGS_WITH_STRIPPED: frozenset[str]

    # ----- Syllable finals -----
    # Myanmar syllable-final markers that indicate a syllable boundary.
    # Used by particle confusion detection to identify left boundaries.
    # Includes both consonant finals (asat, visarga, dot-below) and
    # vowel endings (AA, tall-AA, UU, II, AI).
    _SYLLABLE_FINALS: frozenset[str] = frozenset(
        "\u103a\u1038\u1037"  # asat, visarga, dot below
        "\u102c\u102b\u1030\u102e\u1032"  # AA, tall-AA, UU, II, AI
    )

    # ----- Locative particles -----
    # Spatial locative particles exempt from particle-misuse when preceded
    # by a spatial postposition (e.g. ထဲမှာ "inside-at").
    _LOCATIVE_PARTICLES: frozenset[str] = frozenset({"မှာ", "မှ", "တွင်", "၌"})

    _particle_thresholds: ParticleDetectionThresholds = DEFAULT_PARTICLE_THRESHOLDS

    # ----- Loaded from rules/particles.yaml at module level (with hardcoded fallbacks) -----

    # Visarga exclude pronouns (they validly end with "လို").
    _VISARGA_EXCLUDE_PRONOUNS: frozenset[str] = _VISARGA_EXCLUDE_PRONOUNS_LOADED

    # ----- Subject pronouns -----
    # Subject pronouns used in contextual object-marker (က→ကို)
    # confusion detection.
    _SUBJECT_PRONOUNS: frozenset[str] = _norm_set(
        {
            "ငါ",
            "ငါတို့",
            "နင်",
            "နင်တို့",
            "မင်း",
            "ကျွန်တော်",
            "ကျွန်မ",
            "သူ",
            "သူမ",
            "သူတို့",
        }
    )

    # ----- Particle confusion detection -----
    # Extremely common particle ကို (object marker, freq=217K) is
    # often mis-typed as the much rarer ကိ (freq=12) or ကု (freq=124).
    # Only flag when these appear as standalone particles (at word
    # boundaries).
    _PARTICLE_CONFUSION: dict[str, str | list[str]] = _norm_dict(
        {
            "ကိ": "ကို",  # vowel length confusion
            "ကု": "ကို",  # vowel u/uu confusion
            "နဲ": "နဲ့",  # missing dot-below in conjunction
        }
    )

    # ----- Sequential particle confusion -----
    # Loaded from rules/particles.yaml at module level (with hardcoded fallbacks).
    # တော် (honorific suffix) vs တော့ (sequential/emphatic particle).
    _SEQUENTIAL_PARTICLE_LEFT_CONTEXT: tuple[str, ...] = _SEQUENTIAL_PARTICLE_LEFT_CONTEXT_LOADED

    # ----- Ha-htoe particle confusion -----
    # Suffix particles that are missing ha-htoe (U+103E).
    # Only flag when attached to a preceding word (not standalone).
    _HA_HTOE_PARTICLES: dict[str, str | list[str]] = _norm_dict(
        {
            "မာ": "မှာ",  # "hard" (adj) vs "at/in" (particle)
        }
    )
    # Loaded from rules/particles.yaml at module level (with hardcoded fallbacks).
    # Words ending with the same suffix that should NOT trigger ha-htoe correction.
    _HA_HTOE_EXCLUSIONS: frozenset[str] = _HA_HTOE_EXCLUSIONS_LOADED
    # Myanmar honorific prefixes that mark the following text as a
    # personal name. When မာ appears in a token starting with one of
    # these honorifics, it is part of a name (e.g., ဒေါ်ခင်မာ),
    # not the locative particle မှာ.
    # Extends CORE_RESPECTFUL_TITLES (ဒေါ်, ဦး, ဆရာ, ဆရာမ) with
    # additional name prefixes used in particle ha-htoe detection.
    _HONORIFIC_PREFIXES: frozenset[str] = _norm_set(
        CORE_RESPECTFUL_TITLES | {"ကို", "ဒေါက်တာ", "ပရော်ဖက်ဆာ"}
    )

    # ----- Particle misuse rules (verb-frame-based) -----
    # Rules loaded from particles.yaml at runtime.  Each entry is a dict:
    #   wrong_particle, correct_particle, verb_triggers, max_window.
    # Default is empty; SpellChecker.__init__ overrides from YAML.
    _PARTICLE_MISUSE_RULES: list[dict] = []

    # Loaded from rules/particles.yaml at module level (with hardcoded fallbacks).
    # Spatial postpositions that exempt locative particles from misuse detection.
    _LOCATIVE_EXEMPT_PREFIXES: tuple[str, ...] = _LOCATIVE_EXEMPT_PREFIXES_LOADED

    # ----- Dangling particle detection -----
    # Loaded from rules/particles.yaml at module level (with hardcoded fallbacks).
    # Clause-linking particles that should NOT appear at sentence end.
    _DANGLING_PARTICLES: frozenset[str] = _DANGLING_PARTICLES_LOADED

    # ----- Context probability suppression suffixes -----
    # Particle/connector suffixes where context_probability inside a
    # token can still be meaningful (e.g., ကွောင်း + ကို).
    _KEEP_ATTACHED_SUFFIXES: tuple[str, ...] = tuple(
        normalize(s)
        for s in (
            "ကို",
            "က",
            "မှာ",
            "မှ",
            "နှင့်",
            "နဲ့",
            "တွေ",
            "ပါ",
            "မယ်",
            "တယ်",
            "သည်",
        )
    )

    # ----- Missing asat particles -----
    # Loaded from rules/particles.yaml at module level (with hardcoded fallbacks).
    _MISSING_ASAT_PARTICLES: dict[str, str | list[str]] = _MISSING_ASAT_PARTICLES_LOADED

    # ----- Missing visarga suffixes -----
    # Loaded from rules/particles.yaml at module level (with hardcoded fallbacks).
    _MISSING_VISARGA_SUFFIXES: dict[str, str | list[str]] = _MISSING_VISARGA_SUFFIXES_LOADED

    def _detect_missing_asat(self, text: str, errors: list[Error]) -> None:
        """Detect missing asat (virama) on sentence-final particles.

        Scans the text for known particles (e.g. တယ) that appear at
        word boundaries (end of text or before space) but are missing
        the asat mark (်). This catches errors that the segmenter may
        merge into compound tokens, making them invisible to the word
        validator.
        """
        existing_positions = get_existing_positions(errors)
        for bare, corrected in self._MISSING_ASAT_PARTICLES.items():
            if bare == corrected:
                continue
            for idx, end in iter_occurrences(text, bare):
                if idx > 0 and text[idx - 1] != " ":
                    continue
                if end < len(text) and text[end] != " ":
                    continue
                sugg = [corrected] if isinstance(corrected, str) else corrected
                new_err = SyllableError(
                    text=bare,
                    position=idx,
                    suggestions=sugg,
                    confidence=TEXT_DETECTOR_CONFIDENCES["missing_asat"],
                    error_type=ET_MISSING_ASAT,
                )
                if idx in existing_positions:
                    try_replace_syllable_error(errors, idx, new_err)
                    continue
                errors.append(new_err)

        # Suffix form in merged tokens, e.g. ဖက်တယ / ဖေဆိုတယ.
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        for span in tokenized:
            token = span.text
            token_pos = span.position
            for bare, corrected in self._MISSING_ASAT_PARTICLES.items():
                if len(token) <= len(bare) or not token.endswith(bare):
                    continue
                idx = token_pos + len(token) - len(bare)
                sugg = [corrected] if isinstance(corrected, str) else corrected
                new_err = SyllableError(
                    text=bare,
                    position=idx,
                    suggestions=sugg,
                    confidence=TEXT_DETECTOR_CONFIDENCES["missing_asat"],
                    error_type=ET_MISSING_ASAT,
                )
                if idx in existing_positions:
                    try_replace_syllable_error(errors, idx, new_err)
                    continue
                errors.append(new_err)

        # Phase 3: General standalone missing-asat detection.
        # For standalone space-delimited tokens that are NOT valid words,
        # check if token+asat produces a valid word (catches ညက→ညက်).
        if self.provider and hasattr(self.provider, "is_valid_word"):
            for span in tokenized:
                token = span.text
                token_pos = span.position
                # Only check tokens with Myanmar chars that aren't valid words
                if not any("\u1000" <= ch <= "\u109f" for ch in token):
                    continue
                if self.provider.is_valid_word(token):
                    continue
                # Skip if already ends with asat
                if token.endswith("\u103a"):
                    continue
                # Skip if token ends with a known bare particle
                # (Phase 2 suffix match already handles those)
                if any(
                    token.endswith(bare) and len(token) > len(bare)
                    for bare in self._MISSING_ASAT_PARTICLES
                ):
                    continue
                asat_form = token + "\u103a"
                if not self.provider.is_valid_word(asat_form):
                    continue
                asat_freq = self.provider.get_word_frequency(asat_form)
                if (
                    isinstance(asat_freq, (int, float))
                    and asat_freq >= self._particle_thresholds.missing_asat_standalone_min_freq
                ):
                    # Try to narrow to suffix level for compound tokens.
                    # E.g., ကျောင်းတက → narrow to တက (prefix ကျောင်း is valid,
                    # suffix+asat တက် is valid).
                    err_text = token
                    err_pos = token_pos
                    err_sugg = asat_form
                    from myspellchecker.tokenizers.syllable import SyllableTokenizer

                    _syls = SyllableTokenizer().tokenize(token)
                    if len(_syls) >= 2:
                        for k in range(len(_syls) - 1, 0, -1):
                            _suffix = "".join(_syls[k:])
                            _prefix = "".join(_syls[:k])
                            _suffix_asat = _suffix + "\u103a"
                            if self.provider.is_valid_word(_suffix_asat):
                                _sf = self.provider.get_word_frequency(_suffix_asat)
                                if (
                                    isinstance(_sf, (int, float))
                                    and _sf
                                    >= self._particle_thresholds.missing_asat_suffix_min_freq
                                ):
                                    # Only narrow if the suffix form is at least
                                    # as common as the whole-word form.  When
                                    # token+asat is a single dictionary entry
                                    # (e.g., လိုင်စင်) with higher frequency,
                                    # report the full token as the error span.
                                    if isinstance(asat_freq, (int, float)) and asat_freq > _sf:
                                        break
                                    err_text = _suffix
                                    err_pos = token_pos + len(_prefix)
                                    err_sugg = _suffix_asat
                                    break
                    new_err = SyllableError(
                        text=err_text,
                        position=err_pos,
                        suggestions=[err_sugg],
                        confidence=TEXT_DETECTOR_CONFIDENCES["missing_asat"],
                        error_type=ET_MISSING_ASAT,
                    )
                    if err_pos in existing_positions:
                        try_replace_syllable_error(errors, err_pos, new_err)
                        continue
                    errors.append(new_err)

            # Phase 4: Missing asat in stems with attached postpositions.
            # Detect forms like "စနစကို" where the stem is missing asat
            # but carries a grammatical suffix/particle.
            attached_suffixes = tuple(
                s
                for s in self._KEEP_ATTACHED_SUFFIXES
                if s and len(s) >= 1 and not s.endswith("\u103a")
            )
            for span in tokenized:
                token = span.text
                token_pos = span.position

                if not any("\u1000" <= ch <= "\u109f" for ch in token):
                    continue

                for suffix in sorted(attached_suffixes, key=len, reverse=True):
                    if len(token) <= len(suffix) + 1 or not token.endswith(suffix):
                        continue
                    stem = token[: -len(suffix)]
                    if not stem or stem.endswith("\u103a"):
                        continue
                    if self.provider.is_valid_word(stem):
                        continue
                    stem_with_asat = stem + "\u103a"
                    if not self.provider.is_valid_word(stem_with_asat):
                        continue
                    with_asat_freq = self.provider.get_word_frequency(stem_with_asat)
                    if (
                        not isinstance(with_asat_freq, (int, float))
                        or with_asat_freq < self._particle_thresholds.missing_asat_stem_min_freq
                    ):
                        continue
                    stem_freq = self.provider.get_word_frequency(stem)
                    stem_base = (
                        float(stem_freq)
                        if isinstance(stem_freq, (int, float)) and stem_freq > 0
                        else 1.0
                    )
                    if (
                        float(with_asat_freq) / stem_base
                        < self._particle_thresholds.missing_asat_stem_min_ratio
                    ):
                        continue

                    stem_pos = token_pos
                    new_err = SyllableError(
                        text=stem,
                        position=stem_pos,
                        suggestions=[stem_with_asat],
                        confidence=TEXT_DETECTOR_CONFIDENCES["missing_asat"],
                        error_type=ET_MISSING_ASAT,
                    )
                    if stem_pos in existing_positions:
                        try_replace_syllable_error(errors, stem_pos, new_err)
                    else:
                        errors.append(new_err)
                    existing_positions.add(stem_pos)
                    break

    def _detect_missing_visarga_suffix(self, text: str, errors: list[Error]) -> None:
        """Detect missing visarga in clause-linker suffixes.

        Example: မိုးရွာလို -> လို့
        """
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 2:
            return

        tokens = tokenized.tokens
        token_positions = tokenized.positions
        existing_positions = get_existing_positions(errors)

        def _is_verbish(stem: str) -> bool:
            if not stem:
                return False
            if self.provider and hasattr(self.provider, "get_word_pos"):
                pos = self.provider.get_word_pos(stem)
                if isinstance(pos, str) and "V" in pos:
                    return True
            return stem.endswith(normalize("ရွာ")) or stem.endswith(normalize("သွား"))

        for i, token in enumerate(tokens[:-1]):
            for bare, corrected in self._MISSING_VISARGA_SUFFIXES.items():
                if len(token) <= len(bare) or not token.endswith(bare):
                    continue
                stem = token[: -len(bare)]
                if stem in self._VISARGA_EXCLUDE_PRONOUNS:
                    continue
                if not _is_verbish(stem):
                    continue
                err_pos = token_positions[i] + len(stem)
                sugg = [corrected] if isinstance(corrected, str) else corrected
                new_err = SyllableError(
                    text=bare,
                    position=err_pos,
                    suggestions=sugg,
                    confidence=TEXT_DETECTOR_CONFIDENCES["missing_visarga"],
                    error_type=ET_MISSING_ASAT,
                )
                if err_pos in existing_positions:
                    try_replace_syllable_error(errors, err_pos, new_err)
                    continue
                errors.append(new_err)

    def _detect_missing_visarga_in_compound(self, text: str, errors: list[Error]) -> None:
        """Detect missing visarga (း) inside compound words.

        Scans each space-delimited token that is NOT a valid word.
        For each syllable that ends in asat (်), checks whether the
        syllable+visarga form is a valid word with significantly higher
        frequency.  If so, flags the missing visarga.

        Example: ဒီလမ်ဘေးမှာ — လမ် (freq=2167) should be လမ်း (freq=135951)
        """
        if not self.provider or not hasattr(self.provider, "is_valid_word"):
            return

        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        existing_positions = get_existing_positions(errors)

        for span in tokenized:
            token = span.text
            token_pos = span.position

            if not any("\u1000" <= ch <= "\u109f" for ch in token):
                continue
            # Only check tokens that are NOT valid words (compound with error)
            if self.provider.is_valid_word(token):
                continue

            from myspellchecker.tokenizers.syllable import SyllableTokenizer

            syls = SyllableTokenizer().tokenize(token)
            if len(syls) < 2:
                continue

            # Check each non-final syllable that ends with asat.
            # Only check non-final syllables: final syllable missing
            # visarga is handled by _detect_missing_asat.
            syl_offset = 0
            for _si, syl in enumerate(syls[:-1]):
                syl_pos = token_pos + syl_offset
                syl_offset += len(syl)

                if not syl.endswith("\u103a"):
                    continue

                # Build the visarga form: syl already ends with asat, add visarga
                vis_form = syl + "\u1038"

                # Visarga form must be a valid word
                if not self.provider.is_valid_word(vis_form):
                    continue

                # Check frequency ratio: visarga form must be much more common
                syl_freq = self.provider.get_word_frequency(syl)
                vis_freq = self.provider.get_word_frequency(vis_form)

                if (
                    not isinstance(vis_freq, (int, float))
                    or vis_freq < self._particle_thresholds.visarga_compound_min_freq
                ):
                    continue

                # Skip syllables that are already valid words with high frequency.
                # These are legitimate word forms (e.g., လည် "to rotate" freq=33K)
                # not missing visarga errors, even though the visarga form is more common.
                syl_f = (
                    float(syl_freq) if isinstance(syl_freq, (int, float)) and syl_freq > 0 else 1.0
                )
                if syl_f >= self._particle_thresholds.visarga_compound_skip_freq:
                    continue

                if vis_freq / syl_f < self._particle_thresholds.visarga_compound_min_ratio:
                    continue

                if syl_pos in existing_positions:
                    continue

                new_err = SyllableError(
                    text=syl,
                    position=syl_pos,
                    suggestions=[vis_form],
                    confidence=TEXT_DETECTOR_CONFIDENCES["missing_visarga"],
                    error_type=ET_MISSING_ASAT,
                )
                errors.append(new_err)
                existing_positions.add(syl_pos)

    def _detect_particle_confusion(self, text: str, errors: list[Error]) -> None:
        """Detect commonly confused particles (ကိ/ကု -> ကို).

        These rare particles (freq < 200) are almost always typos for
        the extremely common object marker ကို (freq > 200K).  Flagged
        when they appear in particle position: after an asat/visarga
        (syllable-final) and before a space or end of text.
        """
        existing_positions = get_existing_positions(errors)
        for wrong, correct in self._PARTICLE_CONFUSION.items():
            for idx, end in iter_occurrences(text, wrong):
                # Left boundary: start, space, or syllable-final
                if idx > 0 and text[idx - 1] != " " and text[idx - 1] not in self._SYLLABLE_FINALS:
                    continue
                # Right boundary: end of text or before space
                if end < len(text) and text[end] != " ":
                    continue
                sugg_pc = [correct] if isinstance(correct, str) else correct
                new_err = SyllableError(
                    text=wrong,
                    position=idx,
                    suggestions=sugg_pc,
                    confidence=TEXT_DETECTOR_CONFIDENCES["particle_confusion"],
                    error_type=ET_PARTICLE_CONFUSION,
                )
                if idx in existing_positions:
                    try_replace_syllable_error(errors, idx, new_err)
                    continue
                errors.append(new_err)

        # Contextual object-marker confusion: repeated/ambiguous က
        # before a verb.
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 3:
            return
        tokens = tokenized.tokens
        token_positions = tokenized.positions

        def _looks_verbal(token: str) -> bool:
            if self.provider and hasattr(self.provider, "get_word_pos"):
                pos = self.provider.get_word_pos(token)
                if isinstance(pos, str) and "V" in pos:
                    return True
            return any(
                token.endswith(s)
                for s in (
                    self._COLLOQUIAL_ENDINGS_WITH_STRIPPED | self._FORMAL_ENDINGS_WITH_STRIPPED
                )
            )

        marker_k = normalize("က")
        existing_positions = get_existing_positions(errors)

        for i, token in enumerate(tokens):
            marker_pos: int | None = None
            if token == marker_k:
                marker_pos = token_positions[i]
            elif len(token) > len(marker_k) and token.endswith(marker_k):
                marker_pos = token_positions[i] + len(token) - len(marker_k)
            if marker_pos is None:
                continue
            if i + 1 >= len(tokens) or not _looks_verbal(tokens[i + 1]):
                continue
            pronoun_object_pattern = (
                token == marker_k and i >= 2 and tokens[i - 2] in self._SUBJECT_PRONOUNS
            ) or (token != marker_k and i >= 1 and tokens[i - 1] in self._SUBJECT_PRONOUNS)
            if not pronoun_object_pattern:
                continue
            new_err = SyllableError(
                text=marker_k,
                position=marker_pos,
                suggestions=[normalize("ကို")],
                confidence=TEXT_DETECTOR_CONFIDENCES["particle_confusion"],
                error_type=ET_PARTICLE_CONFUSION,
            )
            if marker_pos in existing_positions:
                try_replace_syllable_error(errors, marker_pos, new_err)
                continue
            errors.append(new_err)

        # Contextual subject-marker confusion: pronoun+ကို should often be pronoun+က
        # when followed by another object marker and a verb.
        marker_ko = normalize("ကို")

        def _has_ko_marker(tok: str) -> bool:
            return tok == marker_ko or (len(tok) > len(marker_ko) and tok.endswith(marker_ko))

        def _looks_verbal_with_punct(token: str) -> bool:
            clean = token.rstrip("၊။,.!?;:")
            if self.provider and hasattr(self.provider, "get_word_pos"):
                pos = self.provider.get_word_pos(clean)
                if isinstance(pos, str) and "V" in pos:
                    return True
            # Fallback for merged verb phrases that may be OOV in DB (e.g., စာအုပ်ပေး၏).
            if clean.endswith(
                (
                    normalize("၏"),
                    normalize("သည်"),
                    normalize("မည်"),
                    normalize("တယ်"),
                    normalize("မယ်"),
                    normalize("ခဲ့"),
                    normalize("ခဲ့သည်"),
                    normalize("လိုက်တယ်"),
                    normalize("နေတယ်"),
                    normalize("ပါသည်"),
                    normalize("ပါမည်"),
                )
            ):
                return True
            return any(
                clean.endswith(s)
                for s in (
                    self._COLLOQUIAL_ENDINGS_WITH_STRIPPED | self._FORMAL_ENDINGS_WITH_STRIPPED
                )
            )

        for i, token in enumerate(tokens):
            if len(token) <= len(marker_ko) or not token.endswith(marker_ko):
                continue

            base = token[: -len(marker_ko)]
            if base not in self._SUBJECT_PRONOUNS:
                continue
            if i > self._particle_thresholds.subject_pronoun_max_index:
                continue

            # Require a second object marker nearby to avoid flagging valid object usage:
            # e.g., "သူက ကျွန်တော်ကို စာအုပ် ပေးတယ်" should stay valid.
            lookahead_end = min(
                len(tokens), i + self._particle_thresholds.particle_confusion_lookahead_window
            )
            has_second_object_marker = any(
                _has_ko_marker(tokens[j]) for j in range(i + 1, lookahead_end)
            )
            if not has_second_object_marker:
                continue

            # Require a verb-like token in a short window to keep precision.
            verb_window_end = min(len(tokens), i + 7)
            if not any(_looks_verbal_with_punct(tokens[j]) for j in range(i + 1, verb_window_end)):
                continue

            token_pos = token_positions[i]
            new_err = SyllableError(
                text=token,
                position=token_pos,
                suggestions=[base + marker_k],
                confidence=TEXT_DETECTOR_CONFIDENCES["particle_confusion"],
                error_type=ET_PARTICLE_CONFUSION,
            )
            if token_pos in existing_positions:
                try_replace_syllable_error(errors, token_pos, new_err)
                continue
            errors.append(new_err)

    def _detect_sequential_particle_confusion(self, text: str, errors: list[Error]) -> None:
        """Detect တော်->တော့ sequential particle confusion.

        Inclusion-based: only flags တော် when it appears inside a
        known left-context pattern (e.g. ပြီးတော်, ရောက်တော်) where
        the sequential particle တော့ is required.  This avoids FPs on
        compound words like ကျွန်တော်, မြို့တော်, ပွဲတော် etc.
        """
        existing_positions = get_existing_positions(errors)
        wrong = normalize("တော်")
        correct = normalize("တော့")
        for pattern in self._SEQUENTIAL_PARTICLE_LEFT_CONTEXT:
            start = 0
            while True:
                idx = text.find(pattern, start)
                if idx == -1:
                    break
                start = idx + 1
                # Find position of တော် within the pattern
                wrong_offset = pattern.find(wrong)
                if wrong_offset == -1:
                    break
                wrong_pos = idx + wrong_offset
                new_err = SyllableError(
                    text=wrong,
                    position=wrong_pos,
                    suggestions=[correct],
                    confidence=TEXT_DETECTOR_CONFIDENCES.get("sequential_particle", 0.85),
                    error_type=ET_PARTICLE_CONFUSION,
                )
                if wrong_pos in existing_positions:
                    try_replace_syllable_error(errors, wrong_pos, new_err)
                    continue
                errors.append(new_err)

    def _detect_ha_htoe_particle_typos(self, text: str, errors: list[Error]) -> None:
        """Detect missing ha-htoe on suffix particles.

        Example: မာ -> မှာ

        Scans for particles like မာ that appear as suffixes on a
        preceding word (not standalone). When attached to a noun, မာ
        is almost always the location particle မှာ ("at/in") rather
        than the adjective "hard".
        """
        existing_positions = get_existing_positions(errors)
        for wrong, correct in self._HA_HTOE_PARTICLES.items():
            for idx, end in iter_occurrences(text, wrong):
                if idx == 0 or text[idx - 1] == " ":
                    continue
                if end < len(text) and text[end] != " ":
                    continue
                # Skip if part of a known word ending with this
                # particle
                excluded = False
                for exc in self._HA_HTOE_EXCLUSIONS:
                    if end >= len(exc) and text[end - len(exc) : end] == exc:
                        excluded = True
                        break
                if excluded:
                    continue
                # Skip if the containing token starts with a Myanmar
                # honorific prefix -- the suffix is part of a personal
                # name (e.g., ဒေါ်ခင်မာ = "Daw Khin Ma"), not the
                # locative particle မှာ.
                token_start_pos = text.rfind(" ", 0, idx)
                token_start_pos = token_start_pos + 1 if token_start_pos >= 0 else 0
                containing_token = text[token_start_pos:end]
                if any(containing_token.startswith(h) for h in self._HONORIFIC_PREFIXES):
                    continue
                sugg_hh = [correct] if isinstance(correct, str) else correct
                new_err = SyllableError(
                    text=wrong,
                    position=idx,
                    suggestions=sugg_hh,
                    confidence=TEXT_DETECTOR_CONFIDENCES["ha_htoe_confusion"],
                    error_type=ET_HA_HTOE_CONFUSION,
                )
                if idx in existing_positions:
                    try_replace_syllable_error(errors, idx, new_err)
                    continue
                errors.append(new_err)

    def _detect_dangling_particles(self, text: str, errors: list[Error]) -> None:
        """Detect clause-linking particles at sentence end.

        Particles like ကြောင့် (because) and ကြောင်း (that/because)
        are clause linkers that require a following clause.  When they
        appear at the very end of the text they signal an incomplete
        sentence.

        Note: particle strings are normalized at lookup time because
        Myanmar text normalization may reorder combining characters.
        """
        from myspellchecker.text.normalize import normalize_myanmar_text

        stripped = text.rstrip()
        if not stripped:
            return

        # In structurally garbled sentences, dangling particle
        # detection is unreliable. But only *structural* errors
        # indicate garbled structure -- typo-level errors (missing
        # asat, wrong medial, zero-wa) don't affect clause boundaries
        # and shouldn't suppress detection.
        structural_count = sum(
            1 for e in errors if getattr(e, "error_type", "") in _STRUCTURAL_ERROR_TYPES
        )
        if structural_count >= self._particle_thresholds.structural_error_max_count:
            return

        existing_positions = get_existing_positions(errors)
        for particle in self._DANGLING_PARTICLES:
            norm_particle = normalize_myanmar_text(particle)
            if stripped.endswith(norm_particle):
                idx = len(stripped) - len(norm_particle)
                if idx > 0 and stripped[idx - 1] != " ":
                    continue
                new_err = SyllableError(
                    text=norm_particle,
                    position=idx,
                    suggestions=[""],
                    confidence=TEXT_DETECTOR_CONFIDENCES["dangling_particle"],
                    error_type=ET_DANGLING_PARTICLE,
                )
                if idx in existing_positions:
                    try_replace_syllable_error(errors, idx, new_err)
                    continue
                errors.append(new_err)

    def _detect_particle_misuse(self, text: str, errors: list[Error]) -> None:
        """Detect semantically wrong particles using verb-frame heuristics.

        Certain verbs require specific case markers on their arguments.
        For example, ပြန်လည်ရယူ (retrieve) takes an ablative source
        (မှ = "from"), not an object marker (ကို = "to/the").

        This detector scans for particles attached to or following nouns,
        then checks whether a disambiguating verb appears within a
        short right-context window.  Only fires when a verb trigger
        is found, keeping precision high.

        Rules are loaded from ``particles.yaml`` via
        ``_PARTICLE_MISUSE_RULES``.
        """
        if not self._PARTICLE_MISUSE_RULES:
            return

        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 2:
            return

        tokens = tokenized.tokens
        token_positions = tokenized.positions
        existing_positions = get_existing_positions(errors)

        for rule in self._PARTICLE_MISUSE_RULES:
            wrong_p: str = rule["wrong_particle"]
            correct_p: str = rule["correct_particle"]
            verb_triggers: tuple[str, ...] = rule["verb_triggers"]
            max_window: int = rule.get("max_window", 5)
            require_prior: str = rule.get("require_prior_marker", "")

            for i, token in enumerate(tokens):
                # Check if this token ends with the wrong particle
                # (particle attached to noun) or IS the standalone particle.
                particle_pos: int | None = None
                particle_text: str | None = None

                if token == wrong_p:
                    # Standalone particle token
                    particle_pos = token_positions[i]
                    particle_text = wrong_p
                elif len(token) > len(wrong_p) and token.endswith(wrong_p):
                    # Particle attached to preceding noun (e.g. အိမ်ထဲကို)
                    particle_pos = token_positions[i] + len(token) - len(wrong_p)
                    particle_text = wrong_p
                else:
                    continue

                if particle_pos is None:
                    continue

                # Locative exemption: spatial postposition + locative
                # particle is always valid (e.g. ထဲမှာ "inside-at").
                # Only applies when wrong_particle IS the locative particle
                # (not object marker ကို — ထဲကို IS wrong and should be flagged).
                if token != wrong_p and wrong_p in self._LOCATIVE_PARTICLES:
                    prefix = token[: -len(wrong_p)]
                    if any(prefix.endswith(sp) for sp in self._LOCATIVE_EXEMPT_PREFIXES):
                        continue

                # Check if position is already claimed by a non-upgradeable
                # error.  We allow upgrading particle_confusion and
                # replaceable (syllable/word) errors to particle_misuse.
                pos_occupied = particle_pos in existing_positions
                if pos_occupied:
                    # Check if the existing error can be upgraded
                    upgradeable = False
                    for existing in errors:
                        if existing.position != particle_pos:
                            continue
                        if existing.error_type in (
                            ET_PARTICLE_CONFUSION,
                            ET_SYLLABLE,
                            ET_WORD,
                        ):
                            upgradeable = True
                        break
                    if not upgradeable:
                        continue

                # If require_prior_marker is set, check that the same
                # marker already appeared on an earlier token (e.g., two
                # nouns marked with က means the second is likely the
                # recipient, not another subject).
                if require_prior:
                    has_prior = False
                    for j in range(i):
                        t = tokens[j]
                        if t == require_prior or (
                            len(t) > len(require_prior) and t.endswith(require_prior)
                        ):
                            has_prior = True
                            break
                    if not has_prior:
                        continue

                # Scan right context for verb triggers.
                # Look ahead up to max_window tokens.
                # For multi-word triggers (e.g. "စစ်ဆေးမှု ပြုလုပ်"),
                # join consecutive tokens and check substring match.
                found_trigger = False
                lookahead_end = min(len(tokens), i + 1 + max_window)
                right_text = " ".join(tokens[i + 1 : lookahead_end])

                for trigger in verb_triggers:
                    if trigger in right_text:
                        found_trigger = True
                        break

                if not found_trigger:
                    continue

                # Build error: the span covers just the wrong particle.
                new_err = SyllableError(
                    text=particle_text,
                    position=particle_pos,
                    suggestions=[correct_p],
                    confidence=TEXT_DETECTOR_CONFIDENCES.get("particle_misuse", 0.80),
                    error_type=ET_PARTICLE_MISUSE,
                )
                if pos_occupied:
                    # Replace the less-specific error in place
                    for ei, existing in enumerate(errors):
                        if existing.position == particle_pos and existing.error_type in (
                            ET_PARTICLE_CONFUSION,
                            ET_SYLLABLE,
                            ET_WORD,
                        ):
                            errors[ei] = new_err
                            break
                    continue
                errors.append(new_err)
