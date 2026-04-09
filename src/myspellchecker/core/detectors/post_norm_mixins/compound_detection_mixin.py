"""Compound detection mixin for PostNormalizationDetectorsMixin.

Provides compound-related detection methods and their data dicts:
``_detect_compound_confusion_typos``, ``_detect_broken_compound_space``.

Extracted from ``post_normalization.py`` to reduce file size while
preserving the exact same method signatures and behaviour.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from myspellchecker.core.constants import (
    CONFUSABLE_EXEMPT_PAIRS,
    CONFUSABLE_EXEMPT_SUFFIX_PAIRS,
    ET_BROKEN_COMPOUND,
    ET_CONFUSABLE_ERROR,
    ET_HA_HTOE_CONFUSION,
    ET_WORD,
    MYANMAR_NUMERAL_WORDS,
    ValidationLevel,
)
from myspellchecker.core.constants.detector_thresholds import (
    DEFAULT_COMPOUND_THRESHOLDS,
    CompoundDetectionThresholds,
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
    norm_dict_tuple as _norm_dict_tuple,
)
from myspellchecker.core.detectors.utils import (
    get_existing_positions,
    get_tokenized,
    iter_occurrences,
)
from myspellchecker.core.response import Error, SyllableError
from myspellchecker.core.validation_strategies.confusable_strategy import (
    generate_confusable_variants,
)
from myspellchecker.text.phonetic import PhoneticHasher
from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

_TOKEN_BOUNDARY_PUNCT = frozenset("၊။,.!?;:\"'()[]{}")

# ── Hardcoded defaults (fallback when YAML is unavailable) ──

_DEFAULT_HA_HTOE_COMPOUNDS: dict[str, tuple[str, str]] = _norm_dict_tuple(
    {
        "နာခောင်း": ("နာ", "နှာ"),
        # Removed: "နူနာ" — MLC dictionary lists နူနာ as standard form
        "ငက်ကို": ("ငက်", "ငှက်"),
        "လျာက်": ("လျာက်", "လျှောက်"),
        "ပိုးမွား": ("မွား", "မွှား"),
    }
)

_DEFAULT_ASPIRATED_COMPOUNDS: dict[str, tuple[str, str]] = _norm_dict_tuple(
    {
        "ကောင်းဆောင်": ("ကောင်း", "ခောင်း"),
        "ကောင်းကိုက်": ("ကောင်း", "ခောင်း"),
        "ကောက်ဆွဲ": ("ကောက်", "ခောက်"),
        "\u1018\u103d\u1004\u103a\u1037\u1015\u102b": (
            "\u1018\u103d\u1004\u103a\u1037",
            "\u1016\u103d\u1004\u103a\u1037",
        ),
    }
)

_DEFAULT_CONSONANT_CONFUSION_COMPOUNDS: dict[str, tuple[str, str]] = _norm_dict_tuple(
    {
        # --- Unique entries (not covered by other algorithmic detectors) ---
        "ဂစားကွင်း": ("ဂစား", "ကစား"),  # visual similarity ga→ka
        "ကျမ်းမာ": ("ကျမ်းမာ", "ကျန်းမာ"),  # scripture→health
        "သဘာ\u1040": ("သဘာ\u1040", "သဘာ\u101d"),  # digit zero→wa
        "ဒီမိုကရေဆီ": ("ဒီမိုကရေဆီ", "ဒီမိုကရေစီ"),  # loanword sa→ca
        "ပြောင်းလည်း": ("ပြောင်းလည်း", "ပြောင်းလဲ"),  # wrong component
        "ကာ ရေး": ("ကာ", "ကား"),  # context: screen→picture
        "ကာ ဆွဲ": ("ကာ", "ကား"),
        "ကာ ရိုက်": ("ကာ", "ကား"),
        "ကြန်း": ("ကြန်း", "ကြမ်း"),  # nasal swap
        "ဂုန်သိက္ခာ": ("ဂုန်", "ဂုဏ်"),  # Pali consonant na→retroflex
        "သတ်မတ်": ("မတ်", "မှတ်"),  # missing ha-htoe
        "လာကို ကိုက်": ("လာ", "လှာ"),  # ha-htoe on la
        "ဖမ်တီး": ("ဖမ်", "ဖန်"),  # ma→na
        # Removed: "ကုမ်ပဏီ" — correct form is ကုမ္ပဏီ (virama stacking)
        "ခွမ်း": ("ခွမ်း", "ခွန်း"),  # ma→na
        "ဆမ်းစစ်": ("ဆမ်း", "ဆန်း"),  # ma→na
        "ဒါတ်": ("ဒါ", "ဓာ"),  # da→dha aspiration
        "ဖွံဖြိုး": ("ဖွံ", "ဖွံ့"),  # missing dot-below
        "ယဉ်ကျေးမူ": ("မူ", "မှု"),  # suffix confusion
        "ရောဂကို": ("ရောဂ", "ရောဂါ"),  # Pali missing vowel
        "ရောဂကြောင့်": ("ရောဂ", "ရောဂါ"),
        "ပျိုးကျင်း": ("ကျင်း", "ခင်း"),  # two-edit compound
        "ဆာဘာ": ("ဆာဘာ", "ဆာဗာ"),  # loanword ba/ba
        "ဖွဖြိုး": ("ဖွ", "ဖွံ့"),  # missing diacritics
        "ချွဲခြမ်း": ("ချွဲ", "ခွဲ"),
        "ထင်သွင်း": ("ထင်", "တင်"),
        # Removed: "ထင်ရှား" — valid standard compound (prominent/outstanding)
        "သပ်မှတ်": ("သပ်", "သတ်"),
        "ဂိုးသင်း": ("သင်း", "သွင်း"),
        "ခေါင်ယူ": ("ခေါင်", "ခေါ်"),
        "တွမ်းထုပ်": ("ထုပ်", "ထုတ်"),
        "ဆရာ သိ သွား": ("သိ", "ဆီ"),
        "ဆရာ သိ လာ": ("သိ", "ဆီ"),
        # Additional patterns handled by other detectors:
        # _detect_missing_visarga, _detect_medial_confusion,
        # confusable variant generation, _detect_missing_diacritic_in_compound
    }
)

_DEFAULT_SUFFIX_CONFUSION_REPLACEMENTS: dict[str, str | list[str]] = _norm_dict(
    {
        "ရည": "ရည်",
        "မူ": "မှု",
        "ချတ်": "ချက်",
    }
)


# ── YAML loading ──

_YAML_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "rules" / "compound_confusion.yaml"
)


def _load_compound_confusion() -> tuple[
    dict[str, tuple[str, str]],
    dict[str, tuple[str, str]],
    dict[str, tuple[str, str]],
    dict[str, str | list[str]],
]:
    """Load compound confusion dicts from YAML with fallback.

    Returns:
        Tuple of (ha_htoe, aspirated, consonant_confusion, suffix_replacements),
        all with keys/values normalized via _norm_dict_tuple/_norm_dict.
    """
    if not _YAML_PATH.exists():
        logger.debug(
            "Compound confusion YAML not found at %s, using defaults",
            _YAML_PATH,
        )
        return (
            _DEFAULT_HA_HTOE_COMPOUNDS,
            _DEFAULT_ASPIRATED_COMPOUNDS,
            _DEFAULT_CONSONANT_CONFUSION_COMPOUNDS,
            _DEFAULT_SUFFIX_CONFUSION_REPLACEMENTS,
        )

    try:
        import yaml

        with open(_YAML_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            logger.warning("Compound confusion YAML empty or invalid, using defaults")
            return (
                _DEFAULT_HA_HTOE_COMPOUNDS,
                _DEFAULT_ASPIRATED_COMPOUNDS,
                _DEFAULT_CONSONANT_CONFUSION_COMPOUNDS,
                _DEFAULT_SUFFIX_CONFUSION_REPLACEMENTS,
            )

        def _parse_compound_section(
            raw: dict | None,
            default: dict[str, tuple[str, str]],
        ) -> dict[str, tuple[str, str]]:
            if not isinstance(raw, dict) or not raw:
                return default
            parsed: dict[str, tuple[str, str]] = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    wrong = v.get("wrong", "")
                    correct = v.get("correct", "")
                    if wrong and correct:
                        parsed[k] = (wrong, correct)
            return _norm_dict_tuple(parsed) if parsed else default

        ha_htoe = _parse_compound_section(
            data.get("ha_htoe_compounds"),
            _DEFAULT_HA_HTOE_COMPOUNDS,
        )
        aspirated = _parse_compound_section(
            data.get("aspirated_compounds"),
            _DEFAULT_ASPIRATED_COMPOUNDS,
        )
        consonant = _parse_compound_section(
            data.get("consonant_confusion_compounds"),
            _DEFAULT_CONSONANT_CONFUSION_COMPOUNDS,
        )

        raw_suffix = data.get("suffix_confusion_replacements", {})
        if isinstance(raw_suffix, dict) and raw_suffix:
            suffix = _norm_dict(raw_suffix)
        else:
            suffix = _DEFAULT_SUFFIX_CONFUSION_REPLACEMENTS

        logger.debug(
            "Loaded compound confusion rules from YAML: "
            "%d ha_htoe, %d aspirated, %d consonant, %d suffix",
            len(ha_htoe),
            len(aspirated),
            len(consonant),
            len(suffix),
        )
        return ha_htoe, aspirated, consonant, suffix

    except Exception:
        logger.warning(
            "Failed to load compound confusion YAML, using defaults",
            exc_info=True,
        )
        return (
            _DEFAULT_HA_HTOE_COMPOUNDS,
            _DEFAULT_ASPIRATED_COMPOUNDS,
            _DEFAULT_CONSONANT_CONFUSION_COMPOUNDS,
            _DEFAULT_SUFFIX_CONFUSION_REPLACEMENTS,
        )


# Load at module level (once, at import time)
(
    _LOADED_HA_HTOE_COMPOUNDS,
    _LOADED_ASPIRATED_COMPOUNDS,
    _LOADED_CONSONANT_CONFUSION_COMPOUNDS,
    _LOADED_SUFFIX_CONFUSION_REPLACEMENTS,
) = _load_compound_confusion()


def _is_confusable_exempt(word: str, alt: str, provider=None) -> bool:
    """Check if word↔alt pair is exempt from confusable detection.

    First checks the DB confusable_pairs table (if provider supports it),
    then falls back to hardcoded CONFUSABLE_EXEMPT_PAIRS.

    Also checks if any exempt pair ``(a, b)`` is a prefix of
    ``(word, alt)`` with equal remainders for compound handling
    (e.g. ဖေးမှာ↔ဘေးမှာ → stems ဖေး↔ဘေး are exempt).
    """
    # DB-driven suppression (preferred path)
    if provider is not None and hasattr(provider, "is_confusable_suppressed"):
        if provider.is_confusable_suppressed(word, alt) is True:
            return True
        if provider.is_confusable_suppressed(alt, word) is True:
            return True

    # Fallback to hardcoded constants
    if (word, alt) in CONFUSABLE_EXEMPT_PAIRS:
        return True
    for exempt_w, exempt_a in CONFUSABLE_EXEMPT_PAIRS:
        if (
            word.startswith(exempt_w)
            and alt.startswith(exempt_a)
            and word[len(exempt_w) :] == alt[len(exempt_a) :]
        ):
            return True
    return False


# Common particles/function words that are valid as space-separated
# sequences.  When BOTH tokens in a potential broken-compound are in
# this set, the space is intentional (e.g. "တွေ ကို", "လို့ ပြော").
_PARTICLE_TOKENS: frozenset[str] = frozenset(
    {
        # Plural / collective
        "တွေ",
        "များ",
        "တို့",
        # Case markers
        "ကို",
        "က",
        "မှာ",
        "မှ",
        "နဲ့",
        "နှင့်",
        "တွင်",
        "သို့",
        "ဆီ",
        "ထဲ",
        "ပေါ်",
        "အား",
        # Sentence-final / aspect
        "တယ်",
        "မယ်",
        "သည်",
        "ပြီ",
        "ပြီး",
        "ပါ",
        "ဘူး",
        # Nominalization
        "ခြင်း",
        "မှု",
        # Quotative / causal / conditional
        "လို့",
        "ကြောင့်",
        "ကြောင်း",
        "ရင်",
        "လျှင်",
        # Emphasis / topic
        "ပဲ",
        "တော့",
        "ပေါ့",
        # Relational
        "အတွက်",
        "လို",
        "ထက်",
        # Honorific / plural verb
        "ကြ",
        # Common verbs that follow particles (quotative + verb patterns)
        "ပြော",
        "ဆို",
        "ထင်",
        "ယူဆ",
        "သိ",
        "ကြား",
        "မြင်",
    }
)
# Quotative verbs included in _PARTICLE_TOKENS for quotative-pattern
# handling but excluded from standalone-particle checks.
_QUOTATIVE_VERBS: frozenset[str] = frozenset(
    {
        "ပြော",
        "ဆို",
        "ထင်",
        "ယူဆ",
        "သိ",
        "ကြား",
        "မြင်",
    }
)

# Pure grammatical particles that should never be joined to the
# following word in broken-compound detection.  Computed as
# _PARTICLE_TOKENS minus quotative verbs.
_STANDALONE_PARTICLE_TOKENS: frozenset[str] = _PARTICLE_TOKENS - _QUOTATIVE_VERBS

_SUFFIX_CONFUSION_ATTACHED_SUFFIXES: tuple[str, ...] = (
    "အတွက်",
    "အကြောင်း",
    "တွင်",
    "မှာ",
    "မှ",
    "ကို",
    "က",
    "၏",
)


class CompoundDetectionMixin:
    """Mixin providing compound confusion and broken compound detection.

    Detects ha-htoe, aspirated, and consonant confusion in compounds,
    and space-separated broken compound words.
    """

    # --- Type stubs for attributes provided by SpellChecker or sibling mixins ---
    provider: "DictionaryProvider"
    semantic_checker: Any
    symspell: Any

    # ----- False compound suppression keys (loaded from compound_morphology.yaml) -----
    _FALSE_COMPOUND_UNLOADED: object = object()  # sentinel: distinct from None and empty set
    _false_compound_keys_cache: set[tuple[str, str]] | object = _FALSE_COMPOUND_UNLOADED

    @classmethod
    def _get_false_compound_keys(cls) -> set[tuple[str, str]] | None:
        """Load curated false_compound pairs from compound_morphology.yaml.

        Caches at class level (loaded once). Returns None if YAML unavailable.
        """
        if cls._false_compound_keys_cache is not cls._FALSE_COMPOUND_UNLOADED:
            cache = cls._false_compound_keys_cache
            return cache if isinstance(cache, set) else None  # type: ignore[return-value]

        try:
            from myspellchecker.core.validation_strategies.broken_compound_strategy import (
                _load_morphology_data,
            )

            data = _load_morphology_data()
            if data is not None:
                cls._false_compound_keys_cache = data.false_compound_keys
                return cls._false_compound_keys_cache  # type: ignore[return-value]
        except Exception:
            logger.debug("Failed to load false compound keys from morphology data", exc_info=True)

        cls._false_compound_keys_cache = set()
        return cls._false_compound_keys_cache  # type: ignore[return-value]

    # ----- Compound confusion data (loaded from YAML with hardcoded fallback) -----
    _HA_HTOE_COMPOUNDS: dict[str, tuple[str, str]] = _LOADED_HA_HTOE_COMPOUNDS
    _ASPIRATED_COMPOUNDS: dict[str, tuple[str, str]] = _LOADED_ASPIRATED_COMPOUNDS
    _CONSONANT_CONFUSION_COMPOUNDS: dict[str, tuple[str, str]] = (
        _LOADED_CONSONANT_CONFUSION_COMPOUNDS
    )
    _SUFFIX_CONFUSION_REPLACEMENTS: dict[str, str | list[str]] = (
        _LOADED_SUFFIX_CONFUSION_REPLACEMENTS
    )
    _STRIP_SUFFIXES: tuple[str, ...] = (
        "အတွက်",
        "အကြောင်း",
        "ဆိုင်ရာ",
        "တွင်",
        "မှာ",
        "မှ",
        "ကို",
        "က",
        "၏",
    )
    _FINITE_PREDICATE_ENDINGS: tuple[str, ...] = (
        "ခဲ့ကြသည်",
        "ခဲ့ပါသည်",
        "ခဲ့သည်",
        "ပါသည်",
        "ပါတယ်",
        "ရသည်",
        "နေသည်",
        "ကြသည်",
        "ခဲ့တယ်",
        "ရတယ်",
        "နေတယ်",
        "တယ်",
        "သည်",
    )
    _NONFINITE_PREDICATE_ENDINGS: tuple[str, ...] = (
        "ခဲ့သည့်",
        "ပါသည့်",
        "သည့်",
        "ရန်",
    )
    _GRAMMAR_ENDINGS: tuple[str, ...] = tuple(
        sorted(
            {*_FINITE_PREDICATE_ENDINGS, *_NONFINITE_PREDICATE_ENDINGS},
            key=len,
            reverse=True,
        )
    )
    _VARIANT_HASHER = PhoneticHasher(ignore_tones=False)
    _compound_thresholds: CompoundDetectionThresholds = DEFAULT_COMPOUND_THRESHOLDS

    def _detect_compound_confusion_typos(self, text: str, errors: list[Error]) -> None:
        """Detect confusion in compound words (ha-htoe and aspirated consonants).

        Scans raw text for known compound patterns where the first component has
        a common confusion (missing ha-htoe or wrong aspiration).  Flags only
        the wrong component, not the entire compound.

        Note: No existing_positions guard here — compound detections should
        override syllable-level errors at the same position.  The position
        dedup stage handles conflicts by preferring longer text spans.
        """
        compound_maps: list[tuple[dict[str, tuple[str, str]], str, float]] = [
            (self._HA_HTOE_COMPOUNDS, ET_HA_HTOE_CONFUSION, 0.90),
            (self._ASPIRATED_COMPOUNDS, ET_HA_HTOE_CONFUSION, 0.90),
            (self._CONSONANT_CONFUSION_COMPOUNDS, ET_CONFUSABLE_ERROR, 0.90),
        ]
        for mapping, error_type, default_conf in compound_maps:
            for pattern, (wrong_part, correct_part) in mapping.items():
                # Compute offset of wrong_part within pattern for accurate position
                part_offset = pattern.find(wrong_part) if wrong_part != pattern else 0
                for idx, end in iter_occurrences(text, pattern):
                    # Skip if the text continues with the correction suffix
                    # e.g. pattern "တင်သွင်" should not match "တင်သွင်း"
                    if end < len(text) and wrong_part != pattern:
                        suffix = correct_part[len(wrong_part) :]
                        if suffix and text[end:].startswith(suffix):
                            continue
                    # Build suggestion list: full compound first, then morpheme
                    # The full compound correction matches the gold annotation span
                    suggestions = []
                    if wrong_part != pattern:
                        full_correction = pattern.replace(wrong_part, correct_part, 1)
                        if full_correction != correct_part:
                            suggestions.append(full_correction)
                    suggestions.append(correct_part)
                    # Use full pattern as error text when wrong_part is a
                    # substring. This ensures the error span covers the full
                    # compound, matching gold annotation spans.
                    # E.g., pattern "စိတ်ဒါတ်" with wrong="ဒါ" → error text is
                    # "စိတ်ဒါတ်" at idx, not "ဒါ" at idx+4.
                    error_text = pattern if wrong_part != pattern else wrong_part
                    error_pos = idx if wrong_part != pattern else idx + part_offset
                    errors.append(
                        SyllableError(
                            text=error_text,
                            position=error_pos,
                            suggestions=suggestions,
                            confidence=default_conf,
                            error_type=error_type,
                        )
                    )

    def _detect_suffix_confusion_typos(self, text: str, errors: list[Error]) -> None:
        """Detect invalid tokens with common abstract-suffix confusions.

        Examples:
        - စွမ်းဆောင်ရည -> စွမ်းဆောင်ရည်
        - ထိရောက်မူ -> ထိရောက်မှု
        - ညွှန်ကြားချတ် -> ညွှန်ကြားချက်
        """
        if not self.provider:
            return

        tokenized = get_tokenized(self, text)
        if not tokenized:
            return

        existing_positions = get_existing_positions(errors)

        for span in tokenized:
            token = span.text
            pos = span.position

            lead = 0
            tail = len(token)
            while lead < tail and token[lead] in _TOKEN_BOUNDARY_PUNCT:
                lead += 1
            while tail > lead and token[tail - 1] in _TOKEN_BOUNDARY_PUNCT:
                tail -= 1
            core_token = token[lead:tail]
            core_pos = pos + lead

            if not core_token or core_pos in existing_positions:
                continue
            if not any("\u1000" <= ch <= "\u109f" for ch in core_token):
                continue
            attached_suffix = ""
            base_token = core_token
            for suffix in sorted(_SUFFIX_CONFUSION_ATTACHED_SUFFIXES, key=len, reverse=True):
                if len(core_token) <= len(suffix) + 1 or not core_token.endswith(suffix):
                    continue
                maybe_base = core_token[: -len(suffix)]
                if not maybe_base:
                    continue
                base_token = maybe_base
                attached_suffix = suffix
                break

            if self.provider.is_valid_word(base_token):
                continue

            token_freq = self.provider.get_word_frequency(base_token)
            token_freq_num = token_freq if isinstance(token_freq, (int, float)) else 0

            for wrong_suffix, corrected_suffix in sorted(
                self._SUFFIX_CONFUSION_REPLACEMENTS.items(), key=lambda x: len(x[0]), reverse=True
            ):
                if len(base_token) <= len(wrong_suffix) + 1 or not base_token.endswith(
                    wrong_suffix
                ):
                    continue
                candidate = base_token[: -len(wrong_suffix)] + corrected_suffix
                if not self.provider.is_valid_word(candidate):
                    continue

                candidate_freq = self.provider.get_word_frequency(candidate)
                if not isinstance(candidate_freq, (int, float)):
                    continue
                if candidate_freq < self._compound_thresholds.suffix_confusion_min_freq:
                    continue
                if (
                    token_freq_num > 0
                    and candidate_freq
                    < token_freq_num * self._compound_thresholds.suffix_confusion_min_ratio
                ):
                    continue

                errors.append(
                    SyllableError(
                        text=base_token,
                        position=core_pos,
                        suggestions=[candidate, candidate + attached_suffix]
                        if attached_suffix
                        else [candidate],
                        confidence=TEXT_DETECTOR_CONFIDENCES.get(
                            "missing_diacritic_compound", 0.88
                        ),
                        error_type=ET_WORD,
                    )
                )
                break

    def _strip_boundary_punct(self, token: str) -> tuple[str, int]:
        """Return token without boundary punctuation and the leading trim size."""
        lead = 0
        tail = len(token)
        while lead < tail and token[lead] in _TOKEN_BOUNDARY_PUNCT:
            lead += 1
        while tail > lead and token[tail - 1] in _TOKEN_BOUNDARY_PUNCT:
            tail -= 1
        return token[lead:tail], lead

    def _split_suffix(self, token: str) -> tuple[str, str]:
        """Split attached grammatical/derivational suffixes from a token."""
        for suffix in sorted(self._STRIP_SUFFIXES, key=len, reverse=True):
            if len(token) <= len(suffix) + 1 or not token.endswith(suffix):
                continue
            base = token[: -len(suffix)]
            if base:
                return base, suffix
        return token, ""

    def _split_grammar_ending(self, token: str) -> tuple[str, str]:
        """Split broad grammatical endings used in finite/non-finite alternations."""
        for ending in self._GRAMMAR_ENDINGS:
            if len(token) <= len(ending) + 1 or not token.endswith(ending):
                continue
            stem = token[: -len(ending)]
            if stem:
                return stem, ending
        return token, ""

    def _is_predicate_ending_only_variant(self, source: str, candidate: str) -> bool:
        """Return True when source/candidate differ only by finite↔non-finite ending."""
        source_stem, source_ending = self._split_grammar_ending(source)
        cand_stem, cand_ending = self._split_grammar_ending(candidate)
        if not source_ending or not cand_ending:
            return False
        if source_stem != cand_stem:
            return False
        return (
            source_ending in self._FINITE_PREDICATE_ENDINGS
            and cand_ending in self._NONFINITE_PREDICATE_ENDINGS
        )

    # Sentence-final suffixes that mark verb chains when they appear at the
    # end of a token.  Sorted longest-first to match greedily.
    _VERB_CHAIN_SUFFIXES: tuple[str, ...] = (
        "ခဲ့ပါတယ်",
        "ခဲ့ပါသည်",
        "ခဲ့သည်",
        "ခဲ့တယ်",
        "နေတယ်",
        "နေသည်",
        "ချင်တယ်",
        "ချင်သည်",
        "ပါတယ်",
        "ပါသည်",
        "ပါမယ်",
        "ပါမည်",
        "ရပါတယ်",
        "ရမယ်",
        "ရမည်",
        "သည်",
        "တယ်",
        "မယ်",
        "မည်",
        "ပြီ",
        "ပြီး",
    )

    # Lock for thread-safe lazy init of _fallback_segmenter.
    _segmenter_lock: threading.Lock = threading.Lock()

    def _get_compound_segmenter(self) -> Any | None:
        """Return a segmenter for compound resolution (thread-safe).

        Prefers the injected ``self.segmenter``; falls back to a lazily
        created ``DefaultSegmenter`` (guarded by ``_segmenter_lock``).
        """
        seg = getattr(self, "segmenter", None)
        if seg is not None and hasattr(seg, "segment_words"):
            return seg
        seg = getattr(self, "_fallback_segmenter", None)
        if seg is not None:
            return seg
        with self._segmenter_lock:
            # Double-check after acquiring lock.
            seg = getattr(self, "_fallback_segmenter", None)
            if seg is not None:
                return seg
            from myspellchecker.segmenters.default import DefaultSegmenter

            try:
                seg = DefaultSegmenter()
                self._fallback_segmenter = seg
                return seg
            except (ImportError, OSError, ValueError):
                return None

    def _resolve_compound_split(self, token: str) -> tuple[list[str], list[int]] | None:
        """Split *token* via the segmenter and validate all parts.

        Returns ``(parts, freqs)`` when every part is a valid dictionary
        word and there are at least 2 parts.  Returns ``None`` otherwise.
        This is the single source of truth for "is this a valid compound
        concatenation?" — used by both the pre-check and the Option C guard.
        """
        seg = self._get_compound_segmenter()
        if seg is None:
            return None
        try:
            parts = seg.segment_words(token)
        except Exception:
            return None
        if len(parts) < 2:
            return None
        if not all(self.provider.is_valid_word(p) for p in parts):
            return None
        freqs = [int(self.provider.get_word_frequency(p)) for p in parts]
        return parts, freqs

    def _is_valid_verb_chain_or_compound(self, token: str) -> bool:
        """Check if an OOV token is a valid verb chain or compound.

        Returns True if the token ends with a known sentence-final suffix
        pattern AND the stem before it is a valid word, OR if it is a
        high-frequency 2-part compound.
        """
        if not self.provider:
            return False

        # Check 1: verb chain — token ends with a sentence-final pattern
        # and the stem (everything before) is a valid word.
        for suffix in self._VERB_CHAIN_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix):
                stem = token[: -len(suffix)]
                if self.provider.is_valid_word(stem):
                    return True

        # Check 2: compound via shared resolution.
        result = self._resolve_compound_split(token)
        if result is not None:
            parts, freqs = result
            if len(parts) == 2:
                # Both parts high-frequency → confident compound.
                if min(freqs) >= self._compound_thresholds.compound_both_parts_min_freq:
                    return True
                # Leading part is a known multi-syllable word → accept.
                if (
                    freqs[0] >= self._compound_thresholds.compound_leading_part_min_freq
                    and len(parts[0]) > 3
                ):
                    return True

        return False

    def _detect_invalid_token_with_strong_candidates(self, text: str, errors: list[Error]) -> None:
        """Detect invalid tokens and recover best high-confidence candidate.

        Designed for OOV compounds that consist of valid-looking subparts,
        e.g. စမံကိန်းအတွက် / ခုန်ကျစရိတ်အကြောင်း / ဗီတမင်အတွက်.
        """
        if not self.provider:
            return

        symspell = getattr(getattr(self, "syllable_validator", None), "symspell", None)
        if symspell is None:
            symspell = getattr(getattr(self, "word_validator", None), "symspell", None)
        if symspell is None:
            return

        tokenized = get_tokenized(self, text)
        if not tokenized:
            return

        existing_positions = get_existing_positions(errors)

        for span in tokenized:
            raw_token = span.text
            token_pos = span.position

            core_token, lead_trim = self._strip_boundary_punct(raw_token)
            core_pos = token_pos + lead_trim
            if not core_token or core_pos in existing_positions:
                continue
            # NOTE: span-aware overlap guard was evaluated but removed —
            # it blocks valid SymSpell recoveries for compound
            # tokens where an earlier detector flagged a sub-span.
            if not any("\u1000" <= ch <= "\u109f" for ch in core_token):
                continue
            # Skip virama-stacked fragments (e.g., ဏ္ဍ from ကဏ္ဍ).
            # The segmenter splits stacked consonant words at the virama
            # boundary, creating short fragments that are not real words.
            if "\u1039" in core_token and len(core_token) <= 4:
                continue
            if self.provider.is_valid_word(core_token):
                continue

            base_token, attached_suffix = self._split_suffix(core_token)
            if not base_token or self.provider.is_valid_word(base_token):
                continue

            # Guard: if the token is a verb chain ending with a known
            # sentence-final pattern, or a high-frequency 2-part compound,
            # skip it.  Also check the suffix-stripped base when a short
            # grammatical suffix (case marker etc.) was stripped.
            if self._is_valid_verb_chain_or_compound(core_token):
                continue
            if (
                attached_suffix
                and base_token != core_token
                and self._is_valid_verb_chain_or_compound(base_token)
            ):
                continue

            candidate_meta: dict[str, tuple[int, int]] = {}

            def _register_candidate(
                candidate: str,
                *,
                freq: int,
                edit_distance: int,
                store: dict[str, tuple[int, int]] = candidate_meta,
            ) -> None:
                if not candidate:
                    return
                prev = store.get(candidate)
                score = (freq, -edit_distance)
                if prev is None or score > prev:
                    store[candidate] = score

            # NOTE: B3 (SymSpell length cap) was evaluated but removed —
            # evaluation showed it causes FNs on compound tokens where a
            # short morpheme error is embedded in a long compound (e.g.,
            # ရွှဲထည်ပစ္စည်းတွေက at 18 chars contains 4-char error ရွှဲ).
            # Optimization A (LRU cache + nasal guard) provides sufficient
            # latency improvement without the FN risk.
            try:
                lookup_results = symspell.lookup(
                    base_token,
                    level=ValidationLevel.WORD.value,
                    max_suggestions=self._compound_thresholds.invalid_token_max_suggestions,
                    use_phonetic=False,
                )
            except (RuntimeError, ValueError, TypeError):
                lookup_results = []

            for suggestion in lookup_results:
                term = getattr(suggestion, "term", "")
                if not term or term == base_token or not self.provider.is_valid_word(term):
                    continue
                edit_distance = int(getattr(suggestion, "edit_distance", 2))
                if edit_distance > self._compound_thresholds.invalid_token_max_edit_distance:
                    continue

                freq = int(getattr(suggestion, "frequency", 0))
                if freq <= 0:
                    freq = int(self.provider.get_word_frequency(term))
                if freq < self._compound_thresholds.invalid_repair_min_freq and edit_distance > 1:
                    continue

                final_candidate = term
                if attached_suffix and self.provider.is_valid_word(term + attached_suffix):
                    final_candidate = term + attached_suffix
                    freq = max(freq, int(self.provider.get_word_frequency(final_candidate)))

                _register_candidate(final_candidate, freq=freq, edit_distance=edit_distance)

            if "\u1039" in base_token:
                asat_candidate = base_token.replace("\u1039", "\u103a")
                if self.provider.is_valid_word(asat_candidate):
                    freq = int(self.provider.get_word_frequency(asat_candidate))
                    if freq >= self._compound_thresholds.invalid_repair_virama_min_freq:
                        final_candidate = asat_candidate
                        if attached_suffix and self.provider.is_valid_word(
                            asat_candidate + attached_suffix
                        ):
                            final_candidate = asat_candidate + attached_suffix
                            freq = max(freq, int(self.provider.get_word_frequency(final_candidate)))
                        _register_candidate(final_candidate, freq=freq, edit_distance=1)

            for variant in generate_confusable_variants(base_token, self._VARIANT_HASHER):
                if not self.provider.is_valid_word(variant):
                    continue
                freq = int(self.provider.get_word_frequency(variant))
                if freq < self._compound_thresholds.invalid_repair_min_freq:
                    continue
                final_candidate = variant
                if attached_suffix and self.provider.is_valid_word(variant + attached_suffix):
                    final_candidate = variant + attached_suffix
                    freq = max(freq, int(self.provider.get_word_frequency(final_candidate)))
                _register_candidate(final_candidate, freq=freq, edit_distance=2)

            if not candidate_meta:
                continue

            ranked = sorted(
                candidate_meta.items(),
                key=lambda item: (-item[1][0], item[1][1], len(item[0])),
            )
            ranked = [
                item
                for item in ranked
                if not self._is_predicate_ending_only_variant(base_token, item[0])
            ]
            if not ranked:
                continue
            top_candidate, (top_freq, top_edit_neg) = ranked[0]
            top_edit = -top_edit_neg
            if top_freq < self._compound_thresholds.invalid_repair_min_freq and top_edit > 1:
                continue

            if len(ranked) > 1:
                second_freq, second_edit_neg = ranked[1][1]
                if (
                    second_freq > 0
                    and top_freq
                    < int(second_freq * self._compound_thresholds.invalid_token_min_dominance_ratio)
                    and top_edit >= -second_edit_neg
                ):
                    continue

            # Segmenter-based compound guard (Option C): if the word
            # segmenter splits this OOV token into ALL valid words AND
            # the best correction candidate has edit distance > 1 (weak
            # evidence), suppress the error.  Strong corrections (edit
            # distance <= 1) override the guard — the misspelling is
            # close enough to a known word that it's likely real.
            if top_edit > 1 and self._resolve_compound_split(core_token) is not None:
                continue

            suggestions = [
                candidate
                for candidate, _meta in ranked[: self._compound_thresholds.invalid_token_top_n]
            ]
            errors.append(
                SyllableError(
                    text=base_token,
                    position=core_pos,
                    suggestions=suggestions,
                    confidence=TEXT_DETECTOR_CONFIDENCES.get("unknown_compound_segment", 0.85),
                    error_type=ET_WORD,
                )
            )

    def _detect_frequency_dominant_valid_variants(self, text: str, errors: list[Error]) -> None:
        """Detect valid but low-preference variants using DB frequency and semantic score."""
        if not self.provider:
            return

        tokenized = get_tokenized(self, text)
        if not tokenized:
            return

        semantic_checker = getattr(self, "_semantic_checker", None)
        existing_positions = get_existing_positions(errors)
        token_occurrences: dict[str, int] = {}

        for span in tokenized:
            raw_token = span.text
            token_pos = span.position

            core_token, lead_trim = self._strip_boundary_punct(raw_token)
            core_pos = token_pos + lead_trim
            if not core_token:
                continue
            occurrence = token_occurrences.get(core_token, 0)
            token_occurrences[core_token] = occurrence + 1

            if core_pos in existing_positions:
                continue
            if not any("\u1000" <= ch <= "\u109f" for ch in core_token):
                continue

            base_token, attached_suffix = self._split_suffix(core_token)
            if not self.provider.is_valid_word(base_token):
                continue

            base_freq = int(self.provider.get_word_frequency(base_token))
            if base_freq <= 0 or base_freq > self._compound_thresholds.variant_max_base_freq:
                continue

            variant_candidates: list[tuple[str, int, float]] = []
            for variant in generate_confusable_variants(base_token, self._VARIANT_HASHER):
                if variant == base_token or not self.provider.is_valid_word(variant):
                    continue
                # Skip exempt word-variant pairs (e.g. ဖေး↔ဘေး loanword)
                if _is_confusable_exempt(base_token, variant, self.provider):
                    continue
                variant_freq = int(self.provider.get_word_frequency(variant))
                if variant_freq < self._compound_thresholds.variant_min_freq:
                    continue
                ratio = variant_freq / max(base_freq, 1)
                if ratio < self._compound_thresholds.variant_min_freq_ratio:
                    continue

                suggestion = variant
                if attached_suffix and self.provider.is_valid_word(variant + attached_suffix):
                    suggestion = variant + attached_suffix
                    variant_freq = max(
                        variant_freq, int(self.provider.get_word_frequency(suggestion))
                    )

                variant_candidates.append((suggestion, variant_freq, ratio))

            if not variant_candidates:
                continue

            variant_candidates.sort(key=lambda item: (-item[1], -item[2], len(item[0])))
            suggestion, cand_freq, ratio = variant_candidates[0]

            semantic_confirmed = False
            if semantic_checker is not None:
                try:
                    scores = semantic_checker.score_mask_candidates(
                        text,
                        core_token,
                        [core_token, suggestion],
                        occurrence=occurrence,
                    )
                    current_score = scores.get(core_token)
                    candidate_score = scores.get(suggestion)
                    if (
                        isinstance(current_score, (int, float))
                        and isinstance(candidate_score, (int, float))
                        and candidate_score
                        > current_score + self._compound_thresholds.variant_min_semantic_delta
                    ):
                        semantic_confirmed = True
                except (RuntimeError, ValueError, TypeError):
                    semantic_confirmed = False

            if semantic_confirmed:
                pass
            elif semantic_checker is not None:
                if ratio < self._compound_thresholds.variant_min_freq_ratio_with_semantic:
                    continue
            elif ratio < self._compound_thresholds.variant_min_freq_ratio_without_semantic:
                continue

            if len(variant_candidates) > 1:
                second_freq, second_ratio = variant_candidates[1][1], variant_candidates[1][2]
                if (
                    cand_freq
                    < int(
                        second_freq * self._compound_thresholds.variant_second_candidate_multiplier
                    )
                    and ratio <= second_ratio + 0.1
                ):
                    continue

            # Skip exempt suffix pairs (e.g. သည်↔သည့် syntactic distinction)
            _sfx_exempt = False
            for sfx_a, sfx_b in CONFUSABLE_EXEMPT_SUFFIX_PAIRS:
                if core_token.endswith(sfx_a) and suggestion.endswith(sfx_b):
                    _sfx_exempt = True
                    break
            if _sfx_exempt:
                continue

            errors.append(
                SyllableError(
                    text=core_token,
                    position=core_pos,
                    suggestions=[suggestion],
                    confidence=TEXT_DETECTOR_CONFIDENCES.get("semantic_implausibility", 0.82),
                    error_type=ET_CONFUSABLE_ERROR,
                )
            )

    # Myanmar numeral words — these should never be prefix-joined with the
    # preceding word.  They form part of numeral+classifier units
    # (e.g., "လေးယောက်" = "four persons"), not compound prefixes.
    # Derived from myanmar_constants.MYANMAR_NUMERAL_WORDS keys, plus the
    # alternative spelling ခုနှစ် ("seven") which is not in the canonical dict.
    _MYANMAR_NUMERAL_WORDS: frozenset[str] = frozenset(MYANMAR_NUMERAL_WORDS.keys()) | frozenset(
        {"ခုနှစ်"}
    )

    def _detect_broken_compound_space(self, text: str, errors: list[Error]) -> None:
        """Detect space-separated words that form a known compound.

        Scans adjacent space-separated tokens and checks whether their
        concatenation is a high-frequency dictionary word.  If so, the
        space is likely an error (broken compound).

        Example: ``မနက် ဖြန်`` → ``မနက်ဖြန်`` (tomorrow)
        """
        if not self.provider:
            return

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 2:
            return

        tokens = tokenized.tokens
        token_positions = tokenized.positions

        # No existing_positions guard — compound detections should override
        # syllable-level errors at the same position.  The position dedup
        # stage handles conflicts by preferring longer text spans.

        segmenter = getattr(self, "segmenter", None)

        # Load curated false_compound suppression set from compound_morphology.yaml.
        # These are word pairs (e.g., verb+SFP, negation+verb, numeral+classifier)
        # that should never be flagged as broken compounds.
        false_compound_keys = self._get_false_compound_keys()

        # Walk through adjacent pairs
        for i in range(len(tokens) - 1):
            left = tokens[i]
            right = tokens[i + 1]

            # Both parts must contain Myanmar characters
            if not any("\u1000" <= ch <= "\u109f" for ch in left):
                continue
            if not any("\u1000" <= ch <= "\u109f" for ch in right):
                continue

            # Skip curated false_compound pairs from compound_morphology.yaml
            if false_compound_keys and (left, right) in false_compound_keys:
                continue

            # Skip when both tokens are particles/function words — their
            # space-separated form is valid Myanmar orthography (e.g.
            # "တွေ ကို", "လို့ ပြော").
            if left in _PARTICLE_TOKENS and right in _PARTICLE_TOKENS:
                continue

            # Skip when the left token is a grammatical particle —
            # particles are intentionally space-separated and should not
            # be joined to the following word (e.g. "လို့ ပြောထားတယ်").
            if left in _STANDALONE_PARTICLE_TOKENS:
                continue

            # Skip reduplication patterns — same word repeated with space
            # (e.g. "တိုင် တိုင်", "ဆင် ဆင်") is emphatic/adverbial, not broken.
            if left == right:
                continue

            left_pos = token_positions[i]
            right_pos = token_positions[i + 1]
            left_freq = self.provider.get_word_frequency(left)
            right_freq = self.provider.get_word_frequency(right)
            right_base, right_suffix = self._split_suffix(right)

            # Candidate 1: full token join (legacy behavior).
            # Candidate 2+: prefix join against the first 1..N right syllables.
            # This recovers patterns like "အိမ် စာအကြောင်း" -> "အိမ်စာ အကြောင်း"
            # without requiring hardcoded word lists.
            candidates: list[tuple[str, str, int]] = [(right, "", right_pos + len(right))]
            if right_suffix and right_base and right_base != right:
                candidates.append((right_base, right_suffix, right_pos + len(right_base)))
            if segmenter is not None and hasattr(segmenter, "segment_syllables"):
                right_syllables = segmenter.segment_syllables(right)
                if len(right_syllables) >= 2:
                    for k in range(1, len(right_syllables)):
                        right_prefix = "".join(right_syllables[:k])
                        right_tail = "".join(right_syllables[k:])
                        if not right_prefix or not right_tail:
                            continue
                        prefix_end = right_pos + len(right_prefix)
                        candidates.append((right_prefix, right_tail, prefix_end))

            for right_part, right_tail, span_end in candidates:
                # Fix B: Numeral prefix-join guard — Myanmar numeral words
                # (e.g., လေး "four") form numeral+classifier units, not
                # compound prefixes.  Skip prefix-join when right_part is
                # a numeral.
                if right_tail and right_part in self._MYANMAR_NUMERAL_WORDS:
                    continue

                compound = left + right_part
                if not self.provider.is_valid_word(compound):
                    continue

                # Enrichment guard: when DB compound_confusions shows the
                # split form is clearly dominant over the joined form AND
                # the compound is low-frequency, the space is intentional.
                # E.g., ယဉ်ကျေးမှု အမွေအနှစ် (split=1404 >> compound=819).
                # Guard: compound_freq < 5000 prevents suppressing common
                # compounds like ကျောင်းသား (freq=60K+).
                if hasattr(self.provider, "get_compound_confusion"):
                    cc = self.provider.get_compound_confusion(compound)
                    if (
                        cc
                        and cc[3] > 0  # split_freq > 0
                        and cc[2] < 5000  # compound_freq < 5000
                        and cc[3] >= cc[2] * 1.5  # split clearly dominant
                        and left_freq >= 5000  # left part is established
                    ):
                        continue

                # Require lexically established joined form.
                freq = self.provider.get_word_frequency(compound)
                min_freq = (
                    self._compound_thresholds.broken_compound_prefix_min_freq
                    if right_tail
                    else self._compound_thresholds.broken_compound_full_min_freq
                )
                if freq < min_freq:
                    continue

                right_part_freq = self.provider.get_word_frequency(right_part)

                if not right_tail:
                    # Legacy standalone guard for full-token join.
                    if (
                        left_freq >= self._compound_thresholds.rare_standalone_threshold
                        and right_part_freq >= self._compound_thresholds.rare_standalone_threshold
                        and freq < self._compound_thresholds.dominant_compound_threshold
                    ):
                        continue
                else:
                    # Prefix-join guard:
                    # only when right token is not strongly lexicalized on its own,
                    # and the remaining tail is a plausible standalone word/particle.
                    if right_freq > self._compound_thresholds.broken_compound_right_token_max_freq:
                        continue
                    if hasattr(self.provider, "get_word_pos"):
                        compound_pos = self.provider.get_word_pos(compound)
                        if isinstance(compound_pos, str):
                            pos_tags = {t.strip() for t in compound_pos.split("|") if t.strip()}
                            # Avoid over-joining verbal/adverbial phrases such as
                            # အလိုအလျောက် ပြုလုပ်(သည်).
                            if "V" in pos_tags or "ADV" in pos_tags:
                                continue
                    if right_tail in self._STRIP_SUFFIXES:
                        pass
                    else:
                        if not self.provider.is_valid_word(right_tail):
                            continue
                        tail_freq = self.provider.get_word_frequency(right_tail)
                        if tail_freq < self._compound_thresholds.broken_compound_tail_min_freq:
                            continue

                span_text = text[left_pos:span_end]
                errors.append(
                    SyllableError(
                        text=span_text,
                        position=left_pos,
                        suggestions=[compound],
                        confidence=TEXT_DETECTOR_CONFIDENCES["broken_compound_space"],
                        error_type=ET_BROKEN_COMPOUND,
                    )
                )
                break

    def _detect_broken_compound_morpheme(self, text: str, errors: list[Error]) -> None:
        """Detect compound words where one morpheme is an ed-1 typo.

        For each adjacent word pair (W_i, W_{i+1}), check if an edit-distance-1
        variant of W_i concatenated with W_{i+1} forms a known high-frequency
        compound. If so, W_i is likely a typo for that variant.

        Example: ``ဂိုးသွင်း`` (goal-scoring) is a valid compound, but if
        ``ကိုးသွင်း`` appears, we check ed-1 variants of ``ကိုး`` and find
        ``ဂိုး``, whose compound ``ဂိုးသွင်း`` is a known word.
        """
        if not self.provider:
            return

        # Need SymSpell for ed-1 lookup — access via SpellChecker's property
        symspell = getattr(self, "symspell", None)
        if not symspell:
            return

        tokenized = get_tokenized(self, text)
        if len(tokenized) < 2:
            return

        tokens = tokenized.tokens
        positions = tokenized.positions
        existing = get_existing_positions(errors)

        lookups_done = 0

        for i in range(len(tokens) - 1):
            if lookups_done >= self._compound_thresholds.broken_morpheme_max_lookups:
                break

            left, right = tokens[i], tokens[i + 1]
            if positions[i] in existing:
                continue

            # Skip long tokens — SymSpell is O(L^d) and long tokens are
            # likely compound segments, not single-morpheme typos.
            myanmar_chars = sum(1 for ch in left if "\u1000" <= ch <= "\u109f")
            if (
                myanmar_chars == 0
                or myanmar_chars > self._compound_thresholds.broken_morpheme_max_chars
            ):
                continue
            if not any("\u1000" <= ch <= "\u109f" for ch in right):
                continue

            # Skip valid standalone words (most tokens are valid → fast exit)
            left_freq = self.provider.get_word_frequency(left)
            if isinstance(left_freq, (int, float)) and left_freq > 0:
                continue

            # Check if the current compound is already valid and frequent
            compound = left + right
            compound_freq = self.provider.get_word_frequency(compound)
            if (
                isinstance(compound_freq, (int, float))
                and compound_freq >= self._compound_thresholds.broken_morpheme_compound_freq
            ):
                continue  # Already a valid compound, not broken

            # Get ed-1 variants via SymSpell (only for unknown/zero-freq short tokens)
            lookups_done += 1
            candidates = symspell.lookup(left, level="word", max_suggestions=5)
            for cand in candidates:
                if cand.term == left:
                    continue
                if cand.edit_distance > 1:
                    continue
                test_compound = cand.term + right
                test_freq = self.provider.get_word_frequency(test_compound)
                if not isinstance(test_freq, (int, float)):
                    continue
                if test_freq < self._compound_thresholds.broken_morpheme_compound_freq:
                    continue
                # The wrong compound should be absent or rare
                if (
                    isinstance(compound_freq, (int, float))
                    and compound_freq > self._compound_thresholds.broken_morpheme_wrong_max_freq
                ):
                    continue
                # Flag: left should be cand.term
                errors.append(
                    SyllableError(
                        text=left,
                        position=positions[i],
                        suggestions=[cand.term],
                        confidence=TEXT_DETECTOR_CONFIDENCES.get("broken_compound_morpheme", 0.88),
                        error_type=ET_CONFUSABLE_ERROR,
                    )
                )
                break  # One fix per token

    # Aspirated consonant pairs for re-join scanning.
    _REJOIN_ASPIRATION_PAIRS: list[tuple[str, str]] = [
        ("\u1000", "\u1001"),  # က ↔ ခ
        ("\u1002", "\u1003"),  # ဂ ↔ ဃ
        ("\u1005", "\u1006"),  # စ ↔ ဆ
        ("\u1010", "\u1011"),  # တ ↔ ထ
        ("\u1012", "\u1013"),  # ဒ ↔ ဓ
        ("\u1015", "\u1016"),  # ပ ↔ ဖ
        ("\u1015", "\u1017"),  # ပ ↔ ဗ
        ("\u1016", "\u1018"),  # ဖ ↔ ဘ
    ]

    def _detect_missegmented_confusable(self, text: str, errors: list[Error]) -> None:
        """Detect confusable errors hidden by word segmentation.

        When the word segmenter splits an error word into fragments
        (e.g., တမင်း → ['တ', 'မင်း']), neither fragment is flagged.
        This scanner re-joins adjacent syllables and checks for two
        specific high-precision patterns:

        1. **Aspirated consonant swap**: If swapping the initial consonant
           with its aspirated pair produces a known high-frequency word.
           Example: တမင်း (freq=83) → ထမင်း (rice, freq=24025)

        2. **Missing asat**: If adding asat (်) to the re-joined form
           produces a known high-frequency word. Strict guards prevent
           FPs from two standalone words being concatenated.
           Example: ပိတ (freq=0) → ပိတ် (closed, freq=57907)
        """
        if not self.provider:
            return

        segmenter = getattr(self, "segmenter", None)
        if not segmenter:
            return

        existing = get_existing_positions(errors)

        # Process each space-separated chunk independently
        tokenized = get_tokenized(self, text)

        for span in tokenized:
            chunk = span.text
            chunk_pos = span.position

            # Skip non-Myanmar chunks
            if not any("\u1000" <= ch <= "\u109f" for ch in chunk):
                continue

            # Skip chunks that are already valid words — no mis-segmentation
            if self.provider.is_valid_word(chunk):
                continue

            # Syllable-segment the chunk
            syllables = segmenter.segment_syllables(chunk)
            if len(syllables) < 2:
                continue

            # Build syllable positions within text
            syl_positions: list[int] = []
            syl_cursor = 0
            for syl in syllables:
                pos = chunk.find(syl, syl_cursor)
                syl_positions.append(chunk_pos + pos)
                syl_cursor = pos + len(syl)

            # Try re-joining adjacent syllable pairs
            for i in range(len(syllables) - 1):
                joined = syllables[i] + syllables[i + 1]
                joined_pos = syl_positions[i]

                if joined_pos in existing:
                    continue

                # Skip long re-joins (unlikely to be a single word)
                myanmar_chars = sum(1 for ch in joined if "\u1000" <= ch <= "\u109f")
                if myanmar_chars > self._compound_thresholds.rejoin_max_myanmar_chars:
                    continue

                joined_freq = self.provider.get_word_frequency(joined)
                joined_f = joined_freq if isinstance(joined_freq, (int, float)) else 0

                # Skip if re-joined form is already a common word
                if joined_f > self._compound_thresholds.rejoin_max_joined_freq:
                    continue

                # --- Check 1: aspirated consonant swap at initial position ---
                if joined and "\u1000" <= joined[0] <= "\u1021":
                    for char_a, char_b in self._REJOIN_ASPIRATION_PAIRS:
                        variant = None
                        if joined[0] == char_a:
                            variant = char_b + joined[1:]
                        elif joined[0] == char_b:
                            variant = char_a + joined[1:]
                        if variant is None:
                            continue
                        if not self.provider.is_valid_word(variant):
                            continue
                        v_freq = self.provider.get_word_frequency(variant)
                        if not isinstance(v_freq, (int, float)):
                            continue
                        if v_freq < self._compound_thresholds.rejoin_min_correction_freq:
                            continue
                        if (
                            joined_f > 0
                            and v_freq < joined_f * self._compound_thresholds.rejoin_min_freq_ratio
                        ):
                            continue
                        # Skip exempt pairs (e.g. ဖေးမှာ↔ဘေးမှာ → ဖေး↔ဘေး exempt)
                        if _is_confusable_exempt(joined, variant, self.provider):
                            continue
                        errors.append(
                            SyllableError(
                                text=joined,
                                position=joined_pos,
                                suggestions=[variant],
                                confidence=TEXT_DETECTOR_CONFIDENCES.get(
                                    "missegmented_confusable", 0.88
                                ),
                                error_type=ET_CONFUSABLE_ERROR,
                            )
                        )
                        break
                    else:
                        # No aspirated match — fall through to asat check
                        pass
                    if any(e.position == joined_pos for e in errors):
                        continue  # Already flagged by aspirated check

                # --- Check 2: missing asat at syllable boundary ---
                if joined_f >= self._compound_thresholds.rejoin_asat_max_joined_freq:
                    continue
                # Guard: first syllable must not be a common standalone word
                left_freq = self.provider.get_word_frequency(syllables[i])
                left_f = left_freq if isinstance(left_freq, (int, float)) else 0
                if left_f > self._compound_thresholds.rejoin_max_first_syllable_freq:
                    continue
                asat_form = joined + "\u103a"
                if not self.provider.is_valid_word(asat_form):
                    continue
                asat_freq = self.provider.get_word_frequency(asat_form)
                if (
                    isinstance(asat_freq, (int, float))
                    and asat_freq >= self._compound_thresholds.rejoin_min_asat_freq
                ):
                    errors.append(
                        SyllableError(
                            text=joined,
                            position=joined_pos,
                            suggestions=[asat_form],
                            confidence=TEXT_DETECTOR_CONFIDENCES.get(
                                "missegmented_confusable", 0.88
                            ),
                            error_type=ET_CONFUSABLE_ERROR,
                        )
                    )

    # ------------------------------------------------------------------
    # Missing visarga (း) detector
    # ------------------------------------------------------------------

    # Myanmar visarga character
    _VISARGA: str = "\u1038"  # း

    def _detect_missing_visarga(self, text: str, errors: list[Error]) -> None:
        """Detect words missing a final visarga (း) via frequency comparison.

        For each segmented word, checks if appending visarga produces a
        valid word with significantly higher corpus frequency.  This catches
        common tone mark omissions like သတင်→သတင်း, အရေ→အရေး.

        The frequency ratio threshold is configurable via
        ``_MISSING_VISARGA_FREQ_RATIO`` (default 5.0).
        """
        if not self.provider:
            return

        existing_positions = get_existing_positions(errors)

        # Check both space-delimited tokens (user's original word boundaries)
        # and segmenter output.  Space tokens catch cases like "သတင်" where
        # the segmenter incorrectly splits the rare form.
        tokenized = get_tokenized(self, text)
        _segmenter = getattr(self, "segmenter", None)
        try:
            segmented = (
                _segmenter.segment_words(text)
                if _segmenter is not None and hasattr(_segmenter, "segment_words")
                else []
            )
        except Exception:
            segmented = []

        # Merge space-split tokens (from tokenized) and segmenter output,
        # deduplicate by position
        seen_positions: set[int] = set()
        merged_words: list[tuple[str, int]] = []
        # Space tokens: pre-computed positions from TokenizedText
        for span in tokenized:
            if span.position not in seen_positions:
                seen_positions.add(span.position)
                merged_words.append((span.text, span.position))
        # Segmenter tokens: compute positions via cursor (different tokenization)
        seg_cursor = 0
        for word in segmented:
            word_pos = text.find(word, seg_cursor)
            if word_pos < 0:
                continue
            seg_cursor = word_pos + len(word)
            if word_pos not in seen_positions:
                seen_positions.add(word_pos)
                merged_words.append((word, word_pos))

        for word, word_pos in merged_words:
            # Strip boundary punctuation from space tokens
            core_word, lead_trim = self._strip_boundary_punct(word)
            if not core_word:
                continue
            core_pos = word_pos + lead_trim

            # Skip if already errored with a high-confidence detection.
            # Low-confidence errors (e.g., confusable_error at 0.72) may be
            # filtered by the output confidence gate — the visarga detector
            # should still produce its own higher-confidence error.
            if core_pos in existing_positions:
                existing_conf = max(
                    (getattr(e, "confidence", 1.0) for e in errors if e.position == core_pos),
                    default=1.0,
                )
                if existing_conf >= self._compound_thresholds.visarga_existing_confidence_gate:
                    continue
            if core_word.endswith(self._VISARGA):
                continue
            # Skip very short words (particles, single characters)
            if len(core_word) < 2:
                continue

            # Also try suffix-stripped form (e.g., သတင်ကို → base=သတင်)
            base_word, _ = self._split_suffix(core_word)
            word = base_word if base_word and base_word != core_word else core_word

            # Check if word + visarga is a much higher-frequency word
            with_visarga = word + self._VISARGA
            if not self.provider.is_valid_word(with_visarga):
                continue

            orig_freq = self.provider.get_word_frequency(word)
            if not isinstance(orig_freq, (int, float)):
                orig_freq = 0
            visarga_freq = self.provider.get_word_frequency(with_visarga)
            if not isinstance(visarga_freq, (int, float)):
                continue

            # Require the visarga form to be significantly more frequent
            effective_orig = max(int(orig_freq), 1)
            ratio = int(visarga_freq) / effective_orig

            if ratio >= self._compound_thresholds.missing_visarga_freq_ratio:
                errors.append(
                    SyllableError(
                        text=word,
                        position=core_pos,
                        suggestions=[with_visarga],
                        confidence=min(
                            self._compound_thresholds.visarga_confidence_max,
                            self._compound_thresholds.visarga_confidence_base
                            + 0.05 * min(ratio / 100, 4.0),
                        ),
                        error_type=ET_CONFUSABLE_ERROR,
                    )
                )
