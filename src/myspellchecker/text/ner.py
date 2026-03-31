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

import logging
import re
from functools import lru_cache
from pathlib import Path

from myspellchecker.core.constants import HONORIFICS

__all__ = [
    "NameHeuristic",
    "is_known_entity",
    "load_gazetteer",
]

logger = logging.getLogger(__name__)

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
    yaml_path = _RULES_DIR / "named_entities.yaml"
    if not yaml_path.exists():
        logger.warning("Named entity gazetteer not found: %s", yaml_path)
        return frozenset()

    try:
        import yaml  # noqa: PLC0415

        with open(yaml_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except ImportError:
        logger.debug("PyYAML not installed; gazetteer lookup unavailable")
        return frozenset()
    except Exception:
        logger.warning("Failed to load named entity gazetteer", exc_info=True)
        return frozenset()

    if not isinstance(data, dict):
        return frozenset()

    # Skip non-entity keys
    skip_keys = {"version", "category", "description", "metadata"}
    entities: set[str] = set()
    for key, value in data.items():
        if key in skip_keys:
            continue
        entities.update(_collect_strings(value))

    return frozenset(entities)


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

    # Honorifics that also function as common particles/prefixes.
    # ကို = "Ko" (honorific) vs object marker particle
    # မ = "Ma" (honorific) vs negation prefix
    # These only act as honorifics in name-like context (sentence-initial
    # or preceded by another honorific), not after content words.
    _AMBIGUOUS_HONORIFICS: set[str] = {"ကို", "မ"}

    def __init__(self, whitelist: set[str] | None = None):
        """
        Initialize the NameHeuristic detector.

        Args:
            whitelist: Optional set of known names to always treat as valid.
        """
        self.whitelist = whitelist if whitelist is not None else set()

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
        self._common_name_syllables: set[str] = {
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
