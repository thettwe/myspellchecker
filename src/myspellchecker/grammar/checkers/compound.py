"""
Myanmar Compound Word Detection and Validation.

This module implements compound word detection for Myanmar text.
Myanmar has several compound word formation patterns:

1. Noun-Noun compounds: ပန်း + ခြံ = ပန်းခြံ (flower garden)
2. Verb-Verb compounds: စား + သောက် = စားသောက် (dine)
3. Reduplication: ဖြေး → ဖြေးဖြေး (slowly)
4. Affixed compounds: အ + လုပ် = အလုပ် (work)

Features:
    - Detect compound word patterns
    - Identify compound components
    - Validate compound formations
    - Detect compound typos
    - Check reduplication patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field

from myspellchecker.core.config import CompoundCheckerConfig
from myspellchecker.core.constants import (
    ET_COMPOUND_ERROR,
    ET_COMPOUND_TYPO,
    ET_INCOMPLETE_REDUPLICATION,
)
from myspellchecker.core.response import GrammarError
from myspellchecker.grammar.config import get_grammar_config
from myspellchecker.utils.singleton import Singleton

# Default Compound Checker configuration (module-level singleton)
_default_compound_config = CompoundCheckerConfig()

__all__ = [
    "CompoundChecker",
    "CompoundError",
    "CompoundInfo",
    "get_compound_checker",
    "is_compound",
    "is_reduplication",
]

# Singleton registry for CompoundChecker
_singleton: Singleton["CompoundChecker"] = Singleton()


@dataclass
class CompoundInfo:
    """
    Information about a compound word.

    Attributes:
        word: The compound word.
        compound_type: Type of compound (noun_noun, verb_verb, etc.).
        components: List of component words.
        pattern: Pattern description.
        is_valid: Whether it's a recognized valid compound.
        confidence: Detection confidence (0.0-1.0).
    """

    word: str
    compound_type: str
    components: list[str] = field(default_factory=list)
    pattern: str = ""
    is_valid: bool = True
    confidence: float = 0.8

    def __str__(self) -> str:
        """Return string representation."""
        comp_str = " + ".join(self.components) if self.components else "unknown"
        return f"CompoundInfo({self.word}: {comp_str})"


@dataclass
class CompoundError(GrammarError):
    """
    Represents a compound word error.

    Extends GrammarError to integrate with the spell checker's error hierarchy.

    Attributes:
        text: The erroneous word (inherited from Error).
        position: Index of the error in the word list (inherited from Error).
        suggestions: List of suggested corrections (inherited from Error).
        error_type: Type of error (compound_typo, invalid_compound, incomplete_reduplication).
        confidence: Confidence score (0.0-1.0) (inherited from Error).
        reason: Human-readable explanation (inherited from GrammarError).
        word: Alias for 'text' (inherited from GrammarError).
        suggestion: First suggestion (inherited from GrammarError).
    """

    # Override default error_type
    error_type: str = field(default=ET_COMPOUND_ERROR)

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"CompoundError(pos={self.position}, {self.error_type}: "
            f"{self.text} → {self.suggestion})"
        )


# Compound word types
COMPOUND_NOUN_NOUN = "noun_noun"  # N + N
COMPOUND_VERB_VERB = "verb_verb"  # V + V
COMPOUND_REDUPLICATION = "reduplication"  # X + X
COMPOUND_AFFIXED = "affixed"  # Prefix/Suffix compounds


class CompoundChecker:
    """
    Validates compound word usage in Myanmar text.

    This checker identifies:
    - Valid compound word patterns
    - Compound typos
    - Incomplete or malformed compounds
    - Reduplication patterns and errors
    """

    def __init__(
        self,
        config_path: str | None = None,
        compound_config: CompoundCheckerConfig | None = None,
    ):
        """
        Initialize the compound checker with pattern data.

        Args:
            config_path: Path to grammar/compounds config.
            compound_config: CompoundCheckerConfig for confidence settings.
        """
        self.config = get_grammar_config(config_path)
        self.compound_config = compound_config or _default_compound_config

        # Initialize internal structures
        self.prefixes: dict[str, tuple[str, str]] = {}
        self.suffixes: dict[str, tuple[str, str]] = {}
        self.noun_compounds: dict[tuple[str, str], str] = {}  # (first, second) -> compound
        self.verb_compounds: dict[tuple[str, str], str] = {}  # (first, second) -> compound
        self.reduplication: dict[str, str] = {}  # base -> reduplicated
        self.typo_map: dict[str, str] = {}
        self.valid_compounds: set[str] = set()
        self.boundary_hints: frozenset[str] = frozenset({"်", "း"})  # Default boundary hints

        self._load_from_config()

        # Build reverse lookups
        self._build_lookups()

    def _load_from_config(self) -> None:
        """Load compound data from config."""
        comp_config = self.config.compounds_config

        # Load prefixes
        if "prefixes" in comp_config:
            for p in comp_config["prefixes"]:
                # map: prefix -> (type, description)
                self.prefixes[p["prefix"]] = (p["type"], p.get("description", ""))

        # Load suffixes
        if "suffixes" in comp_config:
            for s in comp_config["suffixes"]:
                # map: suffix -> (type, description)
                self.suffixes[s["suffix"]] = (s["type"], s.get("description", ""))

        # Load noun compounds
        if "noun_compounds" in comp_config:
            for item in comp_config["noun_compounds"]:
                components = tuple(item["components"])
                compound = item["compound"]
                self.noun_compounds[components] = compound
                self.valid_compounds.add(compound)

        # Load verb compounds
        if "verb_compounds" in comp_config:
            for item in comp_config["verb_compounds"]:
                components = tuple(item["components"])
                compound = item["compound"]
                self.verb_compounds[components] = compound
                self.valid_compounds.add(compound)

        # Load reduplication
        if "reduplication" in comp_config:
            for item in comp_config["reduplication"]:
                base = item["base"]
                redup = item["reduplicated"]
                self.reduplication[base] = redup
                self.valid_compounds.add(redup)

        # Load adjective compounds
        if "adjective_compounds" in comp_config:
            for item in comp_config["adjective_compounds"]:
                components = tuple(item["components"])
                compound = item["compound"]
                self.valid_compounds.add(compound)
                # Store in noun_compounds for component lookup (adjective compounds
                # follow the same structure as noun compounds)
                self.noun_compounds[components] = compound

        # Load typos
        if "typos" in comp_config:
            for item in comp_config["typos"]:
                self.typo_map[item["incorrect"]] = item["correct"]

    def _build_lookups(self) -> None:
        """Build reverse lookup dictionaries."""
        # Component to compound mappings
        self.first_component_map: dict[str, list[tuple[str, str]]] = {}
        self.second_component_map: dict[str, list[tuple[str, str]]] = {}

        # Build from noun-noun compounds
        for (first, second), compound in self.noun_compounds.items():
            if first not in self.first_component_map:
                self.first_component_map[first] = []
            self.first_component_map[first].append((second, compound))

            if second not in self.second_component_map:
                self.second_component_map[second] = []
            self.second_component_map[second].append((first, compound))

        # Build from verb-verb compounds (guard against duplicates if a component
        # pair appears in both noun_compounds and verb_compounds)
        for (first, second), compound in self.verb_compounds.items():
            if first not in self.first_component_map:
                self.first_component_map[first] = []
            entry = (second, compound)
            if entry not in self.first_component_map[first]:
                self.first_component_map[first].append(entry)

            if second not in self.second_component_map:
                self.second_component_map[second] = []
            entry = (first, compound)
            if entry not in self.second_component_map[second]:
                self.second_component_map[second].append(entry)

    def is_valid_compound(self, word: str) -> bool:
        """
        Check if a word is a recognized valid compound.

        Args:
            word: The word to check.

        Returns:
            True if the word is a valid compound.
        """
        return word in self.valid_compounds

    def is_compound_typo(self, word: str) -> bool:
        """
        Check if a word is a common compound typo.

        Args:
            word: The word to check.

        Returns:
            True if the word is a known compound typo.
        """
        return word in self.typo_map

    def get_typo_correction(self, word: str) -> str | None:
        """
        Get the correct form for a compound typo.

        Args:
            word: The misspelled compound.

        Returns:
            The correct spelling, or None if not a known typo.
        """
        return self.typo_map.get(word)

    def has_compound_prefix(self, word: str) -> tuple[str, str, str] | None:
        """
        Check if a word starts with a compound-forming prefix.

        Args:
            word: The word to check.

        Returns:
            Tuple of (prefix, prefix_type, description) or None.
        """
        for prefix, (prefix_type, desc) in self.prefixes.items():
            if word.startswith(prefix) and len(word) > len(prefix):
                return (prefix, prefix_type, desc)
        return None

    def has_compound_suffix(self, word: str) -> tuple[str, str, str] | None:
        """
        Check if a word ends with a compound-forming suffix.

        Args:
            word: The word to check.

        Returns:
            Tuple of (suffix, suffix_type, description) or None.
        """
        for suffix, (suffix_type, desc) in self.suffixes.items():
            if word.endswith(suffix) and len(word) > len(suffix):
                return (suffix, suffix_type, desc)
        return None

    def is_reduplication(self, word: str) -> bool:
        """
        Check if a word is a reduplication pattern.

        Args:
            word: The word to check.

        Returns:
            True if the word is a valid reduplication.
        """
        # Check known reduplication patterns
        if word in self.reduplication.values():
            return True

        # Check for character-level repetition (first half equals second half)
        # Require >= 4 codepoints so each half is a meaningful Myanmar segment
        # (a Myanmar syllable is at least 2 codepoints: consonant + vowel/asat)
        length = len(word)
        if length >= 4 and length % 2 == 0:
            half = length // 2
            if word[:half] == word[half:]:
                return True

        return False

    def get_reduplication_base(self, word: str) -> str | None:
        """
        Get the base word from a reduplication.

        Args:
            word: The reduplicated word.

        Returns:
            The base word, or None if not a reduplication.
        """
        # Check known patterns
        for base, reduplicated in self.reduplication.items():
            if word == reduplicated:
                return base

        # Check ABAB pattern
        length = len(word)
        if length >= 4 and length % 2 == 0:
            half = length // 2
            if word[:half] == word[half:]:
                return word[:half]

        return None

    def detect_compound_pattern(self, word: str) -> CompoundInfo | None:
        """
        Detect the compound pattern of a word.

        Args:
            word: The word to analyze.

        Returns:
            CompoundInfo if a pattern is detected, None otherwise.
        """
        # Check if it's a known valid compound
        if word in self.valid_compounds:
            # Find the components
            for (first, second), compound in self.noun_compounds.items():
                if compound == word:
                    return CompoundInfo(
                        word=word,
                        compound_type=COMPOUND_NOUN_NOUN,
                        components=[first, second],
                        pattern="N + N",
                        is_valid=True,
                        confidence=self.compound_config.exact_match_confidence,
                    )
            for (first, second), compound in self.verb_compounds.items():
                if compound == word:
                    return CompoundInfo(
                        word=word,
                        compound_type=COMPOUND_VERB_VERB,
                        components=[first, second],
                        pattern="V + V",
                        is_valid=True,
                        confidence=self.compound_config.exact_match_confidence,
                    )

        # Check for reduplication
        if self.is_reduplication(word):
            base = self.get_reduplication_base(word)
            return CompoundInfo(
                word=word,
                compound_type=COMPOUND_REDUPLICATION,
                components=[base, base] if base else [],
                pattern="X + X",
                is_valid=True,
                confidence=self.compound_config.pattern_match_confidence,
            )

        # Check for prefix compounds
        prefix_info = self.has_compound_prefix(word)
        if prefix_info:
            prefix, prefix_type, _desc = prefix_info
            stem = word[len(prefix) :]
            return CompoundInfo(
                word=word,
                compound_type=COMPOUND_AFFIXED,
                components=[prefix, stem],
                pattern=f"PREFIX({prefix_type}) + STEM",
                is_valid=True,
                confidence=self.compound_config.component_match_confidence,
            )

        # Check for suffix compounds
        suffix_info = self.has_compound_suffix(word)
        if suffix_info:
            suffix, suffix_type, _desc = suffix_info
            stem = word[: -len(suffix)]
            return CompoundInfo(
                word=word,
                compound_type=COMPOUND_AFFIXED,
                components=[stem, suffix],
                pattern=f"STEM + SUFFIX({suffix_type})",
                is_valid=True,
                confidence=self.compound_config.component_match_confidence,
            )

        return None

    def validate_sequence(self, words: list[str]) -> list[CompoundError]:
        """
        Validate compound word usage in a word sequence.

        This method checks for:
        - Compound typos
        - Incomplete reduplications
        - Potentially joinable compounds

        Args:
            words: List of words to validate.

        Returns:
            List of CompoundError objects for any issues found.
        """
        errors: list[CompoundError] = []

        for i, word in enumerate(words):
            # Check for compound typos
            if self.is_compound_typo(word):
                correction = self.get_typo_correction(word)
                if correction:
                    errors.append(
                        CompoundError(
                            text=word,
                            position=i,
                            suggestions=[correction],
                            error_type=ET_COMPOUND_TYPO,
                            confidence=self.compound_config.pattern_match_confidence,
                            reason=f"'{word}' appears to be a typo for '{correction}'",
                        )
                    )

            # Check for incomplete reduplication
            # If word appears to be a partial reduplication
            base = self.get_reduplication_base(word)
            if base and word != base + base:
                # Might be incomplete
                full_redup = base + base
                if full_redup in self.reduplication.values():
                    errors.append(
                        CompoundError(
                            text=word,
                            position=i,
                            suggestions=[full_redup],
                            error_type=ET_INCOMPLETE_REDUPLICATION,
                            confidence=self.compound_config.partial_match_confidence,
                            reason=(f"'{word}' may be incomplete reduplication of '{base}'"),
                        )
                    )

        return errors

    def analyze_word(self, word: str) -> dict[str, CompoundInfo | list[str] | bool | float | None]:
        """
        Perform comprehensive compound analysis on a word.

        Args:
            word: The word to analyze.

        Returns:
            Dictionary with analysis results.
        """
        result: dict[str, CompoundInfo | list[str] | bool | float | None] = {
            "is_compound": False,
            "compound_info": None,
            "components": [],
            "has_prefix": False,
            "has_suffix": False,
            "is_reduplication": False,
            "confidence": 0.0,
        }

        # Detect compound pattern
        info = self.detect_compound_pattern(word)
        if info:
            result["is_compound"] = True
            result["compound_info"] = info
            result["components"] = info.components
            result["confidence"] = info.confidence

            if info.compound_type == COMPOUND_REDUPLICATION:
                result["is_reduplication"] = True
            elif info.compound_type == COMPOUND_AFFIXED:
                if info.pattern.startswith("PREFIX"):
                    result["has_prefix"] = True
                else:
                    result["has_suffix"] = True

        return result


# Module-level singleton for convenience
def get_compound_checker() -> CompoundChecker:
    """
    Get the module-level CompoundChecker instance.

    Returns:
        The singleton CompoundChecker instance.
    """
    return _singleton.get(CompoundChecker)


# Convenience functions for direct usage
def is_compound(word: str) -> bool:
    """Check if a word is a recognized compound."""
    return get_compound_checker().is_valid_compound(word)


def is_reduplication(word: str) -> bool:
    """Check if a word is a reduplication pattern."""
    return get_compound_checker().is_reduplication(word)
