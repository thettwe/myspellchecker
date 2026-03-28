"""
Homophone checker for Myanmar (Burmese) text.

This module provides a mechanism to detect and correct "Real-Word Errors"
arising from homophone confusion (words that sound the same but have different
meanings and spellings).

Data sources (merged at lookup time):
- **homophones.yaml** (via GrammarRuleConfig): Manually curated pairs.
- **confusable_pairs DB table** (via provider): Corpus-mined pairs from the
  enrichment pipeline — aspiration, medial, nasal, tone swaps.

Common confusions involve:
- Medials: ျ (Ya-pin) vs ြ (Ya-yit)
- Finals: န် (Na-that) vs ံ (Anusvara) vs မ် (Ma-that)
- Vowels: ိ (I) vs ည် (I-like sound in some contexts)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from myspellchecker.grammar.config import get_grammar_config
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.providers.interfaces import NgramRepository

logger = get_logger(__name__)

__all__ = [
    "HomophoneChecker",
]


class HomophoneChecker:
    """
    Detects and suggests homophones for Myanmar words.

    Homophones are words that sound the same but have different spellings and
    meanings. This checker merges two data sources:

    1. **YAML** (``homophones.yaml``): Manually curated pairs loaded at init.
    2. **DB** (``confusable_pairs`` table): Corpus-mined pairs from the
       enrichment pipeline, queried at lookup time via the provider's
       ``get_confusable_pairs()`` method.

    The DB source provides ~21K pairs (aspiration, medial, nasal, tone swaps)
    and is the primary source. The YAML source is a curated fallback for
    pairs that corpus mining cannot discover.

    Algorithm:
        1. Load YAML homophone mappings via GrammarRuleConfig
        2. For a given word, look up YAML map AND query DB confusable_pairs
        3. Merge results, excluding suppressed pairs
        4. Return all known alternate spellings (excluding the input word)

    Example:
        >>> checker = HomophoneChecker(provider=sqlite_provider)
        >>> checker.get_homophones("ဆား")  # Returns {"စား"} from DB pairs

    Attributes:
        homophone_map: Dict mapping words to their known homophones (YAML).
        provider: Optional NgramRepository for DB confusable_pairs lookup.
    """

    def __init__(
        self,
        config_path: str | None = None,
        homophone_map: dict[str, set[str]] | None = None,
        provider: "NgramRepository | None" = None,
    ):
        """
        Initialize HomophoneChecker.

        Args:
            config_path: Path to grammar rules config.
            homophone_map: Optional override map (for backward compatibility/testing).
            provider: Optional dictionary provider with ``get_confusable_pairs()``
                for DB-driven confusable pair lookup.
        """
        if homophone_map:
            self.homophone_map = homophone_map
        else:
            self.config = get_grammar_config(config_path)
            self.homophone_map = self.config.homophones_map

        self._provider = provider
        self._ensure_symmetry()

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        db_label = "+DB" if self._provider is not None else "YAML-only"
        return f"HomophoneChecker(entries={len(self.homophone_map)}, {db_label})"

    def _ensure_symmetry(self) -> None:
        """Ensure all homophone mappings are bidirectional.

        For each word → {h1, h2}, ensures reverse edges h1 → {word}
        and h2 → {word} also exist. Logs a warning for any asymmetric
        entries found so YAML maintainers can fix the source data.
        """
        additions: dict[str, set[str]] = {}
        for word, homophones in self.homophone_map.items():
            for h in homophones:
                if h not in self.homophone_map or word not in self.homophone_map.get(h, set()):
                    additions.setdefault(h, set()).add(word)

        if additions:
            logger.debug(
                "Homophone symmetry: added %d reverse edges for %d words",
                sum(len(v) for v in additions.values()),
                len(additions),
            )
        for word, new_homophones in additions.items():
            if word in self.homophone_map:
                self.homophone_map[word].update(new_homophones)
            else:
                self.homophone_map[word] = new_homophones

    def get_homophones(self, word: str) -> set[str]:
        """
        Get known homophones for a given word.

        Merges results from YAML (curated) and DB confusable_pairs
        (corpus-mined). DB pairs marked as suppressed are excluded.

        Args:
            word: Input word.

        Returns:
            Set of homophone candidates (excluding the input word).
        """
        if not word:
            return set()

        # YAML source
        candidates = set(self.homophone_map.get(word, set()))

        # DB source: confusable_pairs table (corpus-mined)
        if self._provider is not None and hasattr(self._provider, "get_confusable_pairs"):
            for variant, _ctype, _overlap, _fratio, suppress in self._provider.get_confusable_pairs(
                word
            ):
                if not suppress:
                    candidates.add(variant)

        # Exclude the input word from results
        return candidates - {word}
