"""
Named Entity Recognition (NER) and heuristic utilities for Myanmar text.

This module provides tools to detect potential named entities, especially names,
which are frequent false positives in spell checking.

Strategies:
1. **Honorific-based Heuristics**: Detects names by looking for preceding honorifics
   (e.g., U, Daw, Ko, Maung, Dr.).
2. **Whitelist**: Checks against a list of common names or whitelisted entities.
3. **Pattern Matching**: Uses Regex to identify numbers, dates, and English text.
4. **Gazetteer Lookup**: Checks against curated named_entities.yaml for known entities.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from myspellchecker.core.constants import HONORIFICS
from myspellchecker.utils.logging_utils import get_logger

__all__ = [
    "GazetteerData",
    "NameHeuristic",
    "get_gazetteer_data",
    "is_known_entity",
    "load_gazetteer",
]

logger = get_logger(__name__)

_RULES_DIR = Path(__file__).resolve().parent.parent / "rules"


def _collect_strings(obj: object) -> set[str]:
    """Recursively collect all string values from a nested dict/list structure."""
    result: set[str] = set()
    if isinstance(obj, str):
        result.add(obj)
    elif isinstance(obj, list):
        for item in obj:
            result.update(_collect_strings(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            result.update(_collect_strings(value))
    return result


@dataclass(frozen=True)
class GazetteerData:
    """Structured NER gazetteer loaded from ``rules/named_entities.yaml``.

    Provides typed access to individual entity categories rather than
    flattening everything into a single set.
    """

    # Person name data
    person_prefixes: frozenset[str] = field(default_factory=frozenset)
    common_name_syllables: frozenset[str] = field(default_factory=frozenset)

    # NER heuristic config
    ambiguous_honorifics: frozenset[str] = field(default_factory=frozenset)
    location_suffixes: frozenset[str] = field(default_factory=frozenset)
    org_patterns: frozenset[str] = field(default_factory=frozenset)

    # Place data (categorical)
    states_regions: frozenset[str] = field(default_factory=frozenset)
    major_cities: frozenset[str] = field(default_factory=frozenset)
    townships: frozenset[str] = field(default_factory=frozenset)
    historical_places: frozenset[str] = field(default_factory=frozenset)
    international_places: frozenset[str] = field(default_factory=frozenset)
    countries: frozenset[str] = field(default_factory=frozenset)
    geographic_features: frozenset[str] = field(default_factory=frozenset)

    # Other categories
    organizations: frozenset[str] = field(default_factory=frozenset)
    religious: frozenset[str] = field(default_factory=frozenset)
    historical_figures: frozenset[str] = field(default_factory=frozenset)
    ethnic_groups: frozenset[str] = field(default_factory=frozenset)
    temporal: frozenset[str] = field(default_factory=frozenset)
    pali_sanskrit: frozenset[str] = field(default_factory=frozenset)

    # Combined sets
    all_places: frozenset[str] = field(default_factory=frozenset)
    all_entities: frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Thread-safe lazy singleton (double-checked locking, matching rerank_rules.py)
# ---------------------------------------------------------------------------

_GAZETTEER_DATA: GazetteerData | None = None
_gazetteer_lock = threading.Lock()


def _parse_gazetteer_yaml(path: Path | None = None) -> GazetteerData:
    """Parse ``named_entities.yaml`` into a structured :class:`GazetteerData`."""
    yaml_path = path or (_RULES_DIR / "named_entities.yaml")
    if not yaml_path.exists():
        logger.warning("Named entity gazetteer not found: %s", yaml_path)
        return _hardcoded_fallback()

    try:
        import yaml

        with open(yaml_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except ImportError:
        logger.debug("PyYAML not installed; using hardcoded NER data")
        return _hardcoded_fallback()
    except Exception:
        logger.warning("Failed to load named entity gazetteer", exc_info=True)
        return _hardcoded_fallback()

    if not isinstance(data, dict):
        return _hardcoded_fallback()

    # --- Extract categories ---
    person = data.get("person_names", {})
    person_prefixes = frozenset(person.get("prefixes", []))

    ner_cfg = data.get("ner_heuristics", {})
    ambiguous = frozenset(ner_cfg.get("ambiguous_honorifics", ["ကို", "မ"]))
    # common_name_syllables comes from ner_heuristics (17 high-frequency
    # components), NOT from person_names.common_names (broader gazetteer list).
    common_names = frozenset(
        ner_cfg.get(
            "common_name_syllables",
            [
                "အောင်",
                "ကျော်",
                "မောင်",
                "ထွန်း",
                "မြင့်",
                "ခင်",
                "သန်း",
                "လှ",
                "အေး",
                "ဝင်း",
                "တင်",
                "ဆွေ",
                "မြ",
                "စိုး",
                "နွယ်",
                "သီ",
                "တာ",
            ],
        )
    )
    loc_suffixes = frozenset(
        ner_cfg.get(
            "location_suffixes",
            ["မြို့", "ရွာ", "ပြည်နယ်", "တိုင်း", "ခရိုင်", "မြို့နယ်"],
        )
    )
    org_pats = frozenset(
        ner_cfg.get(
            "org_patterns",
            ["ကုမ္ပဏီ", "ဘဏ်", "တက္ကသိုလ်", "ကျောင်း", "ဆေးရုံ", "ရုံး"],
        )
    )

    places = data.get("places", {})
    states_regions = frozenset(places.get("states_regions", []))
    major_cities = frozenset(places.get("major_cities", []))
    townships_data = frozenset(places.get("townships", []))
    historical_pl = frozenset(places.get("historical_places", []))
    international = frozenset(places.get("international", []))
    countries_data = frozenset(places.get("countries", []))
    geo = frozenset(places.get("geographic_features", []))

    all_places = (
        states_regions
        | major_cities
        | townships_data
        | historical_pl
        | international
        | countries_data
        | geo
    )

    orgs = frozenset(_collect_strings(data.get("organizations", {})))
    religious = frozenset(_collect_strings(data.get("religious", {})))
    historical_fig = frozenset(_collect_strings(data.get("historical", {})))
    ethnics = frozenset(data.get("ethnic_groups", []))
    temporal = frozenset(_collect_strings(data.get("temporal", {})))
    pali = frozenset(data.get("pali_sanskrit", []))

    # Build flat entity set (all strings from all entity sections)
    skip_keys = {"version", "category", "description", "metadata", "ner_heuristics"}
    all_entities: set[str] = set()
    for key, value in data.items():
        if key in skip_keys:
            continue
        all_entities.update(_collect_strings(value))

    return GazetteerData(
        person_prefixes=person_prefixes,
        common_name_syllables=common_names,
        ambiguous_honorifics=ambiguous,
        location_suffixes=loc_suffixes,
        org_patterns=org_pats,
        states_regions=states_regions,
        major_cities=major_cities,
        townships=townships_data,
        historical_places=historical_pl,
        international_places=international,
        countries=countries_data,
        geographic_features=geo,
        organizations=orgs,
        religious=religious,
        historical_figures=historical_fig,
        ethnic_groups=ethnics,
        temporal=temporal,
        pali_sanskrit=pali,
        all_places=all_places,
        all_entities=frozenset(all_entities),
    )


def _hardcoded_fallback() -> GazetteerData:
    """Return hardcoded defaults when YAML/PyYAML is unavailable."""
    return GazetteerData(
        person_prefixes=frozenset(HONORIFICS),
        common_name_syllables=frozenset(
            {
                "အောင်",
                "ကျော်",
                "မောင်",
                "ထွန်း",
                "မြင့်",
                "ခင်",
                "သန်း",
                "လှ",
                "အေး",
                "ဝင်း",
                "တင်",
                "ဆွေ",
                "မြ",
                "စိုး",
                "နွယ်",
                "သီ",
                "တာ",
            }
        ),
        ambiguous_honorifics=frozenset({"ကို", "မ"}),
        location_suffixes=frozenset({"မြို့", "ရွာ", "ပြည်နယ်", "တိုင်း", "ခရိုင်", "မြို့နယ်"}),
        org_patterns=frozenset({"ကုမ္ပဏီ", "ဘဏ်", "တက္ကသိုလ်", "ကျောင်း", "ဆေးရုံ", "ရုံး"}),
    )


def get_gazetteer_data(path: Path | None = None) -> GazetteerData:
    """Return the structured :class:`GazetteerData` singleton.

    Thread-safe lazy initialization.  Pass *path* only in tests.
    """
    global _GAZETTEER_DATA
    if _GAZETTEER_DATA is None:
        with _gazetteer_lock:
            if _GAZETTEER_DATA is None:
                _GAZETTEER_DATA = _parse_gazetteer_yaml(path)
    return _GAZETTEER_DATA


@lru_cache(maxsize=1)
def load_gazetteer() -> frozenset[str]:
    """Load the named entity gazetteer from ``rules/named_entities.yaml``.

    Returns a frozen set of all entity strings across every category.
    The result is cached after the first call.

    Returns
    -------
    frozenset[str]
        All known named entity strings.
    """
    return get_gazetteer_data().all_entities


def is_known_entity(word: str) -> bool:
    """Check if a word is a known named entity that should not be flagged.

    Looks up the word in the curated ``named_entities.yaml`` gazetteer.

    Parameters
    ----------
    word:
        The word to check.

    Returns
    -------
    bool
        ``True`` if the word is a recognized named entity.
    """
    return word in load_gazetteer()


class NameHeuristic:
    """
    Heuristic-based Named Entity Recognition.

    This class identifies potential personal names in Myanmar text by analyzing
    context, primarily preceding honorifics. It helps reduce false positives
    in spell checking by 'whitelisting' words that appear to be names.
    """

    def __init__(self, whitelist: set[str] | None = None):
        """
        Initialize the NameHeuristic detector.

        Args:
            whitelist: Optional set of known names to always treat as valid.
        """
        self.whitelist = whitelist if whitelist is not None else set()

        gaz = get_gazetteer_data()

        # Honorifics that also function as common particles/prefixes.
        # ကို = "Ko" (honorific) vs object marker particle
        # မ = "Ma" (honorific) vs negation prefix
        # These only act as honorifics in name-like context (sentence-initial
        # or preceded by another honorific), not after content words.
        self._AMBIGUOUS_HONORIFICS: set[str] = set(gaz.ambiguous_honorifics)

        self.patterns = {
            # English words (A-Z, a-z)
            "english": re.compile(r"^[A-Za-z]+$"),
            # Digits (English 0-9 and Myanmar 0-9)
            "number": re.compile(r"^[\d\u1040-\u1049]+$"),
            # Simple date-like patterns (e.g., 12/12/2024 or 12-12-2024)
            "date_symbol": re.compile(
                r"^[\d\u1040-\u1049]+[/-][\d\u1040-\u1049]+[/-][\d\u1040-\u1049]+$"
            ),
        }

        # Common Myanmar Name Syllables (High frequency name components)
        # Used as a soft signal when combined with other indicators
        self._common_name_syllables: set[str] = set(gaz.common_name_syllables)

    def is_potential_name(
        self,
        word: str,
        prev_word: str | None = None,
        prev_prev_word: str | None = None,
    ) -> bool:
        """
        Check if a word is likely part of a name or valid entity.

        Args:
            word: The word to check.
            prev_word: The preceding word in the sentence (for context).
            prev_prev_word: The word two positions back (for disambiguation).

        Returns:
            True if the word is likely a name/entity, False otherwise.
        """
        if not word:
            return False

        # 1. Whitelist check
        if word in self.whitelist:
            return True

        # 2. Regex checks (English, Numbers, Dates)
        # These are not "Names" strictly, but are valid entities to ignore
        if self.patterns["english"].match(word):
            return True
        if self.patterns["number"].match(word):
            return True
        if self.patterns["date_symbol"].match(word):
            return True

        # 3. Honorific check (heuristic)
        # If previous word is a known honorific, current word is likely a name start
        if prev_word and prev_word in HONORIFICS:
            # Ambiguous honorifics (ကို, မ) also serve as particles:
            # - ကို = object marker (after nouns), e.g., သတင်းကို
            # - မ = negation prefix (before verbs), e.g., မသွားဘူး
            # Only treat as honorific when in name-like position:
            # sentence-initial or preceded by another honorific/name syllable.
            if prev_word in self._AMBIGUOUS_HONORIFICS:
                if prev_prev_word is not None and prev_prev_word not in HONORIFICS:
                    return False
            return True

        # 4. Common Name Component Check (Soft heuristic)
        # If the word is a very common name component AND follows an honorific context,
        # we might be lenient. However, "အောင်" (Aung) can also be a verb (succeed).
        # We use this as a secondary signal only when prev_word is also a common name syllable.
        if prev_word and prev_word in self._common_name_syllables:
            if word in self._common_name_syllables:
                return True

        # Note: This is a simplified heuristic. A full NER system would check
        # word sequences, multi-word names, and suffix markers.
        # For V1, checking immediately after an honorific is a high-precision signal.

        return False

    def analyze_sentence(self, words: list[str]) -> list[bool]:
        """
        Analyze a sentence and mark potential names.

        Args:
            words: List of words in the sentence.

        Returns:
            List of booleans, corresponding to input words.
            True indicates the word is likely a name/entity.
        """
        is_name = [False] * len(words)

        for i, word in enumerate(words):
            prev = words[i - 1] if i > 0 else None
            prev_prev = words[i - 2] if i > 1 else None

            # Check for name start
            if self.is_potential_name(word, prev, prev_prev):
                is_name[i] = True

        return is_name
