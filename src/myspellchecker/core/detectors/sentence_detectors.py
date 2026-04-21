"""Sentence-level text detectors.

Detectors for sentence structure, tense mismatch, register mixing,
negation patterns, and related issues.  These run on normalized text
and mutate the errors list in place.

Extracted from ``spellchecker.py`` to reduce file size while preserving
the exact same method signatures and behaviour.

Method groups are factored into sub-mixins under ``sentence_mixins``:
- RegisterMixingMixin: register mixing detection
- StructureDetectionMixin: sentence structure (G02/G03) detection
- TenseDetectionMixin: tense/aspect mismatch detection
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from myspellchecker.core.constants import (
    CORE_RESPECTFUL_TITLES,
    ET_CLASSIFIER_ERROR,
    ET_MERGED_SFP_CONJUNCTION,
    ET_NEGATION_SFP_MISMATCH,
    ET_REGISTER_MIXING,
    ET_SEMANTIC_ERROR,
)

if TYPE_CHECKING:
    from myspellchecker.providers.base import DictionaryProvider
from myspellchecker.core.detector_data import (
    TEXT_DETECTOR_CONFIDENCES,
)
from myspellchecker.core.detector_data import (
    norm_dict as _norm_dict,
)
from myspellchecker.core.detector_data import (
    norm_dict_context as _norm_dict_context,
)
from myspellchecker.core.detector_data import (
    norm_set as _norm_set,
)
from myspellchecker.core.detectors.sentence_mixins import (
    RegisterMixingMixin,
    StructureDetectionMixin,
    TenseDetectionMixin,
)
from myspellchecker.core.detectors.utils import get_existing_positions
from myspellchecker.core.response import Error, SyllableError
from myspellchecker.text.normalize import normalize
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Hardcoded defaults: register endings (fallback when YAML is unavailable) ──

_DEFAULT_FORMAL_ENDINGS: frozenset[str] = _norm_set(
    {
        "သည်",
        "ပါသည်",
        "မည်",
        "ပါမည်",
    }
)

_DEFAULT_POLITE_ENDINGS: frozenset[str] = _norm_set(
    {
        "ပါတယ်",
        "ပါမယ်",
    }
)

_DEFAULT_COLLOQUIAL_ENDINGS: frozenset[str] = _norm_set(
    {
        "တယ်",
        "မယ်",
    }
)

_DEFAULT_DISCOURSE_ENDINGS: frozenset[str] = _norm_set(
    {
        "နော်",  # tag question ("right?", "okay?")
        "လား",  # yes/no question
        "လဲ",  # wh-question
        "ပေါ့",  # softener
        "ပဲ",  # emphatic/restrictive
        "လေ",  # emphasis
        "ဟ",  # informal affirmation
        "ပြီ",  # completive
        "ပါပြီ",  # polite completive
        "ဘူး",  # negative
        "ပါဘူး",  # polite negative
    }
)

_DEFAULT_INFORMAL_PARTICLES: frozenset[str] = _norm_set({"ဟ", "ကွာ", "ကွ", "ဟေ့"})


# ── YAML loading: register endings ──

_REGISTER_YAML_PATH = Path(__file__).resolve().parent.parent.parent / "rules" / "register.yaml"


def _load_register_endings() -> tuple[
    frozenset[str],  # formal_endings
    frozenset[str],  # polite_endings
    frozenset[str],  # colloquial_endings
    frozenset[str],  # discourse_endings
    frozenset[str],  # informal_particles
]:
    """Load sentence-final particle sets from register.yaml with fallback.

    Returns:
        Tuple of (formal_endings, polite_endings, colloquial_endings,
        discourse_endings, informal_particles), all normalized via _norm_set.
    """
    defaults = (
        _DEFAULT_FORMAL_ENDINGS,
        _DEFAULT_POLITE_ENDINGS,
        _DEFAULT_COLLOQUIAL_ENDINGS,
        _DEFAULT_DISCOURSE_ENDINGS,
        _DEFAULT_INFORMAL_PARTICLES,
    )

    if not _REGISTER_YAML_PATH.exists():
        logger.debug(
            "Register YAML not found at %s, using defaults for endings",
            _REGISTER_YAML_PATH,
        )
        return defaults

    try:
        import yaml

        with open(_REGISTER_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            logger.warning("Register YAML empty or invalid, using defaults for endings")
            return defaults

        # -- Sentence final particles --
        sfp = data.get("sentence_final_particles", {})
        if isinstance(sfp, dict):
            formal_endings = (
                _norm_set(sfp["formal"])
                if isinstance(sfp.get("formal"), list) and sfp["formal"]
                else _DEFAULT_FORMAL_ENDINGS
            )
            polite_endings = (
                _norm_set(sfp["polite"])
                if isinstance(sfp.get("polite"), list) and sfp["polite"]
                else _DEFAULT_POLITE_ENDINGS
            )
            colloquial_endings = (
                _norm_set(sfp["colloquial"])
                if isinstance(sfp.get("colloquial"), list) and sfp["colloquial"]
                else _DEFAULT_COLLOQUIAL_ENDINGS
            )
            discourse_endings = (
                _norm_set(sfp["discourse"])
                if isinstance(sfp.get("discourse"), list) and sfp["discourse"]
                else _DEFAULT_DISCOURSE_ENDINGS
            )
        else:
            formal_endings = _DEFAULT_FORMAL_ENDINGS
            polite_endings = _DEFAULT_POLITE_ENDINGS
            colloquial_endings = _DEFAULT_COLLOQUIAL_ENDINGS
            discourse_endings = _DEFAULT_DISCOURSE_ENDINGS

        # -- Informal particles --
        raw_informal = data.get("informal_particles", [])
        informal_particles = (
            _norm_set(raw_informal)
            if isinstance(raw_informal, list) and raw_informal
            else _DEFAULT_INFORMAL_PARTICLES
        )

        logger.debug(
            "Loaded register endings from YAML: "
            "%d formal, %d polite, %d colloquial, %d discourse, %d informal",
            len(formal_endings),
            len(polite_endings),
            len(colloquial_endings),
            len(discourse_endings),
            len(informal_particles),
        )
        return (
            formal_endings,
            polite_endings,
            colloquial_endings,
            discourse_endings,
            informal_particles,
        )

    except Exception:
        logger.warning(
            "Failed to load register YAML for endings, using defaults",
            exc_info=True,
        )
        return defaults


# Load at module level (once, at import time)
(
    _FORMAL_ENDINGS_LOADED,
    _POLITE_ENDINGS_LOADED,
    _COLLOQUIAL_ENDINGS_LOADED,
    _DISCOURSE_ENDINGS_LOADED,
    _INFORMAL_PARTICLES_LOADED,
) = _load_register_endings()


# ── Hardcoded defaults: semantic rules (fallback when YAML is unavailable) ──

_DEFAULT_IMPLAUSIBLE_SUBJECTS: dict[str, tuple[str, tuple[str, ...]]] = _norm_dict_context(
    {
        "ကုလားထိုင်": ("သူ", ("စာဖတ်", "ချက်ပေး", "စား")),
        "စားပွဲ": ("သူ", ("စာဖတ်", "ချက်ပေး", "စား")),
        "ငါး": ("လူ", ("ကော်ဖီ", "ကိတ်မုန့်", "စာမေးပွဲ", "ချက်ပေး", "စာဖတ်", "စား")),
        "ကြောင်": ("ကျောင်းသား", ("စာမေးပွဲ", "ဖြေ")),
        "ကျောက်တုံး": ("တောင်သူ", ("ရိတ်", "စိုက်", "ရေလောင်း", "ထွန်")),
    }
)

_DEFAULT_ANIMATE_NOUN_HINTS: frozenset[str] = _norm_set(
    {
        "လူ",
        "သူ",
        "သား",
        "သမီး",
        "ဝင်",
        "ဆရာ",
        "ဆရာမ",
        "ဝန်ထမ်း",
        "အရာရှိ",
        "ကျောင်းသား",
        "ကျောင်းသူ",
    }
)

_DEFAULT_INANIMATE_NOUN_HINTS: frozenset[str] = _norm_set(
    {
        "စက်",
        "ယာဉ်",
        "ဆာဗာ",
        "ရူတာ",
        "ဖုန်း",
        "ကွန်ပျူတာ",
        "ဒေတာ",
        "စနစ်",
        "ဖိုင်",
        "ကိရိယာ",
    }
)

# ── YAML loading ──

_YAML_PATH = Path(__file__).resolve().parent.parent.parent / "rules" / "semantic_rules.yaml"


def _load_semantic_rules() -> tuple[
    dict[str, tuple[str, tuple[str, ...]]],
    frozenset[str],
    frozenset[str],
]:
    """Load semantic rules from YAML with fallback to hardcoded defaults.

    Returns:
        Tuple of (implausible_subjects, animate_noun_hints, inanimate_noun_hints),
        all with keys/values normalized.
    """
    if not _YAML_PATH.exists():
        logger.debug(
            "Semantic rules YAML not found at %s, using defaults",
            _YAML_PATH,
        )
        return (
            _DEFAULT_IMPLAUSIBLE_SUBJECTS,
            _DEFAULT_ANIMATE_NOUN_HINTS,
            _DEFAULT_INANIMATE_NOUN_HINTS,
        )

    try:
        import yaml

        with open(_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            logger.warning("Semantic rules YAML empty or invalid, using defaults")
            return (
                _DEFAULT_IMPLAUSIBLE_SUBJECTS,
                _DEFAULT_ANIMATE_NOUN_HINTS,
                _DEFAULT_INANIMATE_NOUN_HINTS,
            )

        # -- Implausible subjects --
        raw_subjects = data.get("implausible_subjects", {})
        if isinstance(raw_subjects, dict) and raw_subjects:
            parsed: dict[str, tuple[str, tuple[str, ...]]] = {}
            for k, v in raw_subjects.items():
                if isinstance(v, dict):
                    suggestion = v.get("suggestion", "")
                    action_cues = v.get("action_cues", [])
                    if suggestion and action_cues:
                        parsed[k] = (suggestion, tuple(action_cues))
            implausible_subjects = _norm_dict_context(parsed)
        else:
            implausible_subjects = _DEFAULT_IMPLAUSIBLE_SUBJECTS

        # -- Animacy hints --
        raw_animacy = data.get("animacy_hints", {})
        if isinstance(raw_animacy, dict):
            raw_animate = raw_animacy.get("animate", [])
            raw_inanimate = raw_animacy.get("inanimate", [])
            animate = (
                _norm_set(raw_animate)
                if isinstance(raw_animate, list) and raw_animate
                else _DEFAULT_ANIMATE_NOUN_HINTS
            )
            inanimate = (
                _norm_set(raw_inanimate)
                if isinstance(raw_inanimate, list) and raw_inanimate
                else _DEFAULT_INANIMATE_NOUN_HINTS
            )
        else:
            animate = _DEFAULT_ANIMATE_NOUN_HINTS
            inanimate = _DEFAULT_INANIMATE_NOUN_HINTS

        logger.debug(
            "Loaded semantic rules from YAML: "
            "%d implausible subjects, %d animate hints, %d inanimate hints",
            len(implausible_subjects),
            len(animate),
            len(inanimate),
        )
        return implausible_subjects, animate, inanimate

    except Exception:
        logger.warning(
            "Failed to load semantic rules YAML, using defaults",
            exc_info=True,
        )
        return (
            _DEFAULT_IMPLAUSIBLE_SUBJECTS,
            _DEFAULT_ANIMATE_NOUN_HINTS,
            _DEFAULT_INANIMATE_NOUN_HINTS,
        )


# Load at module level (once, at import time)
(
    _IMPLAUSIBLE_SUBJECTS_LOADED,
    _ANIMATE_NOUN_HINTS_LOADED,
    _INANIMATE_NOUN_HINTS_LOADED,
) = _load_semantic_rules()


class SentenceDetectorsMixin(
    RegisterMixingMixin,
    StructureDetectionMixin,
    TenseDetectionMixin,
):
    """Mixin providing sentence-level detector methods and their data constants.

    These detectors run on normalized text and mutate the errors list in place.
    They check for structural issues like register mixing, tense mismatch,
    negation patterns, and dangling particles.

    Method groups are factored into sub-mixins:
    - RegisterMixingMixin: ``_detect_register_mixing``
    - StructureDetectionMixin: ``_detect_sentence_structure_issues``
    - TenseDetectionMixin: ``_detect_tense_mismatch``
    """

    # --- Type stubs for attributes provided by SpellChecker ---
    provider: "DictionaryProvider"

    # ----- Register ending data (shared across sub-mixins) -----
    # Loaded from rules/register.yaml at module level (with hardcoded fallbacks).
    # Sentence-final particles with register marking.
    # Checked across the entire text (not per-sentence) because the sentence
    # segmenter splits at sentence-final particles.
    _FORMAL_ENDINGS: frozenset[str] = _FORMAL_ENDINGS_LOADED
    _POLITE_ENDINGS: frozenset[str] = _POLITE_ENDINGS_LOADED
    _COLLOQUIAL_ENDINGS: frozenset[str] = _COLLOQUIAL_ENDINGS_LOADED

    # Discourse/question particles that validly end sentences
    # These follow the main sentence-final particle (e.g., "သွားမယ် နော်")
    _DISCOURSE_ENDINGS: frozenset[str] = _DISCOURSE_ENDINGS_LOADED

    # Asat-stripped variants for matching tokens with missing asat (common typo)
    # Note: comprehensions can't access class vars in Python 3, so use literals
    _ALL_ENDINGS_WITH_STRIPPED: frozenset[str] = (
        _FORMAL_ENDINGS_LOADED
        | _POLITE_ENDINGS_LOADED
        | _COLLOQUIAL_ENDINGS_LOADED
        | _DISCOURSE_ENDINGS_LOADED
        | _norm_set(
            {
                "သည",
                "ပါသည",
                "မည",
                "ပါမည",  # formal without asat
                "ပါတယ",
                "ပါမယ",  # polite without asat
                "တယ",
                "မယ",  # colloquial without asat
            }
        )
    )
    _FORMAL_ENDINGS_WITH_STRIPPED: frozenset[str] = _FORMAL_ENDINGS_LOADED | _norm_set(
        {
            "သည",
            "ပါသည",
            "မည",
            "ပါမည",
        }
    )
    _POLITE_ENDINGS_WITH_STRIPPED: frozenset[str] = _POLITE_ENDINGS_LOADED | _norm_set(
        {
            "ပါတယ",
            "ပါမယ",
        }
    )
    _COLLOQUIAL_ENDINGS_WITH_STRIPPED: frozenset[str] = _COLLOQUIAL_ENDINGS_LOADED | _norm_set(
        {
            "တယ",
            "မယ",
        }
    )

    _IMPLAUSIBLE_SUBJECTS: dict[str, tuple[str, tuple[str, ...]]] = _IMPLAUSIBLE_SUBJECTS_LOADED

    # Common suffixes to strip when matching implausible subjects
    _SUBJECT_SUFFIXES: tuple[str, ...] = tuple(normalize(s) for s in ("များ", "ကြီး", "ငယ်"))

    # Merged numeral+classifier agreement fallback.
    # Used when the segmenter emits single tokens like "ငါးယောက်" and
    # grammar classifier validation cannot see separate NUM/CLF tokens.
    _ANIMATE_NOUN_HINTS: frozenset[str] = _ANIMATE_NOUN_HINTS_LOADED
    _INANIMATE_NOUN_HINTS: frozenset[str] = _INANIMATE_NOUN_HINTS_LOADED
    _FORMAL_YI = normalize("၏")
    _VERB_OBJECT_MARKERS: frozenset[str] = _norm_set({"ကို", "အား"})
    _COLLOQUIAL_CLAUSE_MARKERS: frozenset[str] = _norm_set({"ဆိုပြီး", "လို့", "ဆို"})
    _COLLOQUIAL_CASE_MARKERS: frozenset[str] = _norm_set({"မှာ", "နဲ့"})
    _NON_VERB_YI_STEMS: frozenset[str] = _norm_set(
        {
            "သူ",
            "သူမ",
            "သူတို့",
            "ကျွန်တော်",
            "ကျွန်မ",
            "ကျွန်ုပ်",
            "ငါ",
            "နင်",
            "မင်း",
            "ဒေသ",
            "နိုင်ငံ",
            "အဖွဲ့",
            "ကော်မတီ",
            "ဌာန",
        }
    )

    def _detect_semantic_agent_implausibility(self, text: str, errors: list[Error]) -> None:
        """Rule-based fallback for obvious non-human subject + human-action patterns."""
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 2:
            return

        existing_positions = get_existing_positions(errors)

        for i, span in enumerate(tokenized):
            if span.position in existing_positions:
                continue
            if not span.text.endswith(normalize("က")) or len(span.text) <= 1:
                continue
            subject = span.text[:-1]
            # Strip plural/size suffixes for compound subjects (e.g., ကျောက်တုံးများက)
            matched_subject = subject
            if subject not in self._IMPLAUSIBLE_SUBJECTS:
                for suffix in self._SUBJECT_SUFFIXES:
                    if subject.endswith(suffix):
                        matched_subject = subject[: -len(suffix)]
                        break
            if matched_subject not in self._IMPLAUSIBLE_SUBJECTS:
                continue
            suggestion, cues = self._IMPLAUSIBLE_SUBJECTS[matched_subject]
            tail = " ".join(s.text for s in tokenized.spans[i + 1 :])
            if not any(cue in tail for cue in cues):
                continue
            errors.append(
                SyllableError(
                    text=subject,
                    position=span.position,
                    suggestions=[suggestion],
                    confidence=TEXT_DETECTOR_CONFIDENCES["semantic_implausibility"],
                    error_type=ET_SEMANTIC_ERROR,
                )
            )

    def _detect_merged_classifier_mismatch(self, text: str, errors: list[Error]) -> None:
        """Detect NUM+CLF tokens with likely classifier mismatch by context.

        Examples:
        - ရူတာ ငါးယောက် -> ငါးလုံး
        - ကော်မတီဝင် သုံးကောင် -> သုံးဦး
        """
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 2:
            return

        tokens = tokenized.tokens
        token_positions = tokenized.positions
        existing_positions = get_existing_positions(errors)

        from myspellchecker.grammar.checkers.classifier import get_classifier_checker

        checker = get_classifier_checker()
        classifiers = sorted(checker.classifiers, key=len, reverse=True)

        def _split_num_classifier(token: str) -> tuple[str, str] | None:
            for clf in classifiers:
                if len(token) <= len(clf) or not token.endswith(clf):
                    continue
                num = token[: -len(clf)]
                if checker.is_numeral(num):
                    return num, clf
            return None

        def _choose_context_noun(index: int) -> str | None:
            prev_token = tokens[index - 1] if index > 0 else None
            next_token = tokens[index + 1] if index + 1 < len(tokens) else None
            for candidate in (prev_token, next_token):
                if not candidate:
                    continue
                if any("\u1000" <= ch <= "\u109f" for ch in candidate):
                    return candidate
            return None

        def _default_classifier_for(noun: str) -> str | None:
            if any(h in noun for h in self._ANIMATE_NOUN_HINTS):
                return normalize("ဦး")
            if any(h in noun for h in self._INANIMATE_NOUN_HINTS):
                return normalize("လုံး")
            return None

        for i, token in enumerate(tokens):
            split = _split_num_classifier(token)
            if split is None:
                continue
            num, clf = split
            token_pos = token_positions[i]
            if token_pos in existing_positions:
                continue

            noun = _choose_context_noun(i)
            if noun is None:
                continue

            suggested_classifier: str | None = None

            compatible = checker.get_compatible_classifiers(noun)
            if compatible:
                if clf in compatible:
                    continue
                suggested_classifier = compatible[0]
            else:
                suggested_classifier = _default_classifier_for(noun)
                if suggested_classifier is None or suggested_classifier == clf:
                    continue

            errors.append(
                SyllableError(
                    text=token,
                    position=token_pos,
                    suggestions=[num + suggested_classifier],
                    confidence=TEXT_DETECTOR_CONFIDENCES.get("semantic_implausibility", 0.85),
                    error_type=ET_CLASSIFIER_ERROR,
                )
            )

    def _detect_formal_yi_in_colloquial_context(self, text: str, errors: list[Error]) -> None:
        """Detect verb+၏ inside colloquial context and suggest colloquial ending.

        Examples:
        - ... စာအုပ်ပေး၏ ဆိုပြီး ... -> စာအုပ်ပေးတယ်
        - ... အသိပေး၏ လို့ ... -> အသိပေးတယ်
        """
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 2:
            return

        tokens = tokenized.tokens
        token_positions = tokenized.positions

        boundary_punct = frozenset("၊။,.!?;:\"'()[]{}")

        def _strip_punct(token: str) -> str:
            start = 0
            end = len(token)
            while start < end and token[start] in boundary_punct:
                start += 1
            while end > start and token[end - 1] in boundary_punct:
                end -= 1
            return token[start:end]

        has_colloquial_signal = any(
            (clean := _strip_punct(token)) in self._COLLOQUIAL_CLAUSE_MARKERS
            or any(clean.endswith(s) for s in self._COLLOQUIAL_ENDINGS_WITH_STRIPPED)
            or any(clean.endswith(marker) for marker in self._COLLOQUIAL_CASE_MARKERS)
            for token in tokens
        )
        if not has_colloquial_signal:
            return

        existing_positions = get_existing_positions(errors)

        for i, token in enumerate(tokens):
            clean_token = _strip_punct(token)
            if len(clean_token) <= 1 or not clean_token.endswith(self._FORMAL_YI):
                continue

            lead_trim = 0
            while lead_trim < len(token) and token[lead_trim] in boundary_punct:
                lead_trim += 1
            pos = token_positions[i]
            clean_pos = pos + lead_trim
            if clean_pos in existing_positions:
                continue

            stem = clean_token[: -len(self._FORMAL_YI)]
            if not stem or stem in self._NON_VERB_YI_STEMS:
                continue

            prev = _strip_punct(tokens[i - 1]) if i > 0 else ""
            next_token = _strip_punct(tokens[i + 1]) if i + 1 < len(tokens) else ""

            has_object_context = prev in self._VERB_OBJECT_MARKERS or any(
                prev.endswith(marker) for marker in self._VERB_OBJECT_MARKERS
            )
            has_clause_tail = next_token in self._COLLOQUIAL_CLAUSE_MARKERS or any(
                next_token.startswith(marker) for marker in self._COLLOQUIAL_CLAUSE_MARKERS
            )
            if not (has_object_context or has_clause_tail):
                continue

            if self.provider and hasattr(self.provider, "get_word_pos"):
                try:
                    stem_pos = self.provider.get_word_pos(stem)
                except (RuntimeError, ValueError, TypeError):
                    stem_pos = None
                if isinstance(stem_pos, str) and stem_pos and not stem_pos.startswith("V"):
                    if not has_object_context:
                        continue

            errors.append(
                SyllableError(
                    text=clean_token,
                    position=clean_pos,
                    suggestions=[stem + normalize("တယ်")],
                    confidence=TEXT_DETECTOR_CONFIDENCES["register_mixing"],
                    error_type=ET_REGISTER_MIXING,
                )
            )

    # ----- Negation pattern detection -----
    # Myanmar negation is a circumfix: မ + V + ဘူး (colloquial) / ပါ (formal).
    # Using an affirmative SFP (တယ်, သည်) after negation prefix မ is always wrong.
    _AFFIRMATIVE_SFPS: frozenset[str] = _norm_set({"တယ်", "ပါတယ်", "သည်", "ပါသည်"})
    _NEGATION_SFP_CORRECTIONS: dict[str, str | list[str]] = _norm_dict(
        {
            "တယ်": "ဘူး",
            "ပါတယ်": "ပါဘူး",
            "သည်": "ပါ",
            "ပါသည်": "ပါ",
        }
    )

    def _detect_negation_sfp_mismatch(self, text: str, errors: list[Error]) -> None:
        """Detect negation prefix မ with affirmative SFP.

        Myanmar negation requires circumfix pattern: မ-V-ဘူး (colloquial),
        မ-V-ပါ (formal). Using affirmative SFPs like တယ် or သည် after
        negation prefix is always ungrammatical.
        """
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 1:
            return

        neg_prefix = normalize("မ")

        # Allow negation detection to see through L1 syllable errors
        # at function morpheme positions (e.g., bare မ flagged as invalid_syllable)
        existing_positions = get_existing_positions(
            errors, ignore_types=frozenset({"invalid_syllable"})
        )

        for span in tokenized:
            # Token must start with မ (negation prefix) and end with
            # an affirmative SFP — e.g., မသွားတယ်
            if not span.text.startswith(neg_prefix):
                continue
            # Guard: the remainder after မ should be >= 2 syllables
            # (verb + SFP), otherwise it's a word starting with မ.
            remainder = span.text[len(neg_prefix) :]
            if len(remainder) < 3:  # min: 1-syl verb + 1-syl SFP
                continue

            for sfp in sorted(self._AFFIRMATIVE_SFPS, key=len, reverse=True):
                if not remainder.endswith(sfp):
                    continue
                # Verify that the part between neg prefix and SFP is a verb-like segment
                verb_part = remainder[: -len(sfp)]
                if not verb_part:
                    continue
                # Check that the verb part or full neg+verb is not a valid word on its own
                # (e.g., မြင် starts with မ but is not negation)
                if self.provider and self.provider.is_valid_word(span.text):
                    break
                # Check verb_part looks like a valid verb (in dictionary)
                if self.provider and not self.provider.is_valid_word(verb_part):
                    continue
                # Found: negation + verb + affirmative SFP
                sfp_pos = span.position + len(span.text) - len(sfp)
                if sfp_pos not in existing_positions:
                    neg_corr = self._NEGATION_SFP_CORRECTIONS.get(sfp, normalize("ဘူး"))
                    sugg_neg = [neg_corr] if isinstance(neg_corr, str) else neg_corr
                    errors.append(
                        SyllableError(
                            text=sfp,
                            position=sfp_pos,
                            suggestions=sugg_neg,
                            confidence=TEXT_DETECTOR_CONFIDENCES["negation_sfp_mismatch"],
                            error_type=ET_NEGATION_SFP_MISMATCH,
                        )
                    )
                return  # one error per text

    # ----- Informal particle + honorific mismatch -----
    # Loaded from rules/register.yaml at module level (with hardcoded fallbacks).
    # Casual sentence-ending particles that clash with respectful address terms.
    _INFORMAL_PARTICLES: frozenset[str] = _INFORMAL_PARTICLES_LOADED
    # Respectful address terms that require polite particles.
    # Extends CORE_RESPECTFUL_TITLES (ဒေါ်, ဦး, ဆရာ, ဆရာမ) with compound
    # honorific titles specific to politeness-context detection.
    _HONORIFIC_TERMS: frozenset[str] = _norm_set(
        CORE_RESPECTFUL_TITLES | {"ဆရာကြီး", "ဆရာမကြီး", "ဆရာဝန်ကြီး"}
    )

    def _detect_informal_with_honorific(self, text: str, errors: list[Error]) -> None:
        """Detect informal sentence-ending particles used with respectful address.

        Flags casual particles (ဟ, ကွာ) at sentence end when the text contains
        an honorific term (ဆရာကြီး, ဒေါ်, etc.), suggesting a respectful alternative.
        """
        # Normalize input so honorific membership works against the
        # post-normalize ``_HONORIFIC_TERMS`` set regardless of caller
        # (production pipeline already normalizes; direct unit-test calls do not).
        text = normalize(text)
        has_honorific = any(h in text for h in self._HONORIFIC_TERMS)
        if not has_honorific:
            return

        # Check last token for informal particle (exact or suffix match)
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if not tokenized:
            return
        last_span = tokenized[-1]

        matched_particle: str | None = None
        if last_span.text in self._INFORMAL_PARTICLES:
            matched_particle = last_span.text
        else:
            for particle in sorted(self._INFORMAL_PARTICLES, key=len, reverse=True):
                if len(particle) < len(last_span.text) and last_span.text.endswith(particle):
                    matched_particle = particle
                    break

        if matched_particle is None:
            return

        # Position of the last token is pre-computed
        last_token_pos = last_span.position
        last_token = last_span.text
        if last_token_pos < 0:
            return
        particle_pos = last_token_pos + len(last_token) - len(matched_particle)

        existing_positions = get_existing_positions(errors)
        if particle_pos in existing_positions:
            return

        if matched_particle in {normalize("ကွာ"), normalize("ကွ")}:
            prefix = text[:particle_pos].rstrip()
            if prefix.endswith(normalize("ပြီးပြီ")):
                polite_suggestions = ["ရှင့်", "ရှင်", "ခင်ဗျား"]
            else:
                polite_suggestions = ["ရှင်", "ရှင့်", "ခင်ဗျား"]
        elif matched_particle == normalize("ဟ"):
            question_endings = (
                normalize("လား"),
                normalize("လဲ"),
                normalize("မလား"),
                normalize("သလား"),
                normalize("မလဲ"),
                normalize("သလဲ"),
            )
            question_prefix = text[:particle_pos].rstrip()
            if any(question_prefix.endswith(ending) for ending in question_endings):
                polite_suggestions = ["ခင်ဗျား", "ရှင်", "ပါ"]
            else:
                polite_suggestions = ["ပါ", "ရှင်", "ခင်ဗျား"]
        else:
            polite_suggestions = ["ခင်ဗျား", "ရှင့်"]

        errors.append(
            SyllableError(
                text=matched_particle,
                position=particle_pos,
                suggestions=polite_suggestions,
                confidence=TEXT_DETECTOR_CONFIDENCES["informal_with_honorific"],
                error_type=ET_REGISTER_MIXING,
            )
        )

    def _detect_informal_h_after_completive(self, text: str, errors: list[Error]) -> None:
        """Detect terse trailing `ဟ` after completive clauses.

        Example: `ညစာစားပြီးပြီ ဟ` -> suggest softer final particle `နော်`.
        """
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 2:
            return

        last_span = tokenized[-1]
        if last_span.text != normalize("ဟ"):
            return

        prev_span = tokenized[-2]
        completive_endings = (normalize("ပြီးပြီ"), normalize("ပါပြီ"), normalize("ပြီ"))
        if not any(
            prev_span.text == ending
            or (len(prev_span.text) > len(ending) and prev_span.text.endswith(ending))
            for ending in completive_endings
        ):
            return

        particle_pos = last_span.position

        existing_positions = get_existing_positions(errors)
        if particle_pos in existing_positions:
            return

        errors.append(
            SyllableError(
                text=last_span.text,
                position=particle_pos,
                suggestions=["နော်", "ပါ"],
                confidence=TEXT_DETECTOR_CONFIDENCES["informal_h_after_completive"],
                error_type=ET_REGISTER_MIXING,
            )
        )

    # ----- Merged SFP + conjunction detection -----
    # SFPs that should not be glued onto a conjunction (e.g. တယ်ပြီး → ပြီး)
    _SFP_SET: frozenset[str] = _norm_set({"တယ်", "ပါတယ်", "သည်", "ပါသည်", "ဘူး", "ပါဘူး"})
    _CONJUNCTION_SET: frozenset[str] = _norm_set(
        {"ပြီး", "ပြီးတော့", "ရင်", "လျှင်", "လို့", "ကြောင့်", "သော်လည်း", "ပေမယ့်", "၍"}
    )

    def _detect_merged_sfp_conjunction(self, text: str, errors: list[Error]) -> None:
        """Detect SFP merged with conjunction.

        Catches patterns like `စားတယ်ပြီး` where the SFP `တယ်` is incorrectly
        glued onto the conjunction `ပြီး`. The first verb in a sequential clause
        should take the conjunctive form directly, not SFP+conjunction.
        """
        from myspellchecker.core.detectors.utils import get_tokenized

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 1:
            return

        existing_positions = get_existing_positions(errors)

        for span in tokenized:
            # Check each SFP+conjunction combination
            for sfp in sorted(self._SFP_SET, key=len, reverse=True):
                for conj in sorted(self._CONJUNCTION_SET, key=len, reverse=True):
                    merged = sfp + conj
                    if not span.text.endswith(merged):
                        continue
                    # Found merged pattern — flag the SFP portion
                    sfp_start = span.position + len(span.text) - len(merged)
                    if sfp_start not in existing_positions:
                        errors.append(
                            SyllableError(
                                text=merged,
                                position=sfp_start,
                                suggestions=[conj],
                                confidence=TEXT_DETECTOR_CONFIDENCES["merged_sfp_conjunction"],
                                error_type=ET_MERGED_SFP_CONJUNCTION,
                            )
                        )
                    return  # one error per text
