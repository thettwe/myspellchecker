"""
Stemmer/Lemmatizer for Myanmar (Burmese) language.

This module provides rule-based stemming to strip common suffixes (particles,
markers) from words to identify their root form. This is useful for:
1. Identifying OOV words that are just conjugated forms of known words.
2. Aggregating statistics for root words.
3. Improving POS tagging accuracy by mapping to root POS.
"""

from __future__ import annotations

from functools import lru_cache

from myspellchecker.core.config.text_configs import StemmerConfig
from myspellchecker.grammar.config import get_grammar_config
from myspellchecker.text.normalize import normalize

__all__ = [
    "Stemmer",
]

# Default Stemmer configuration (module-level singleton)
_default_stemmer_config = StemmerConfig()


class Stemmer:
    """
    Rule-based stemmer for Myanmar text.

    Performance optimizations:
    - Pre-computed combined suffix list sorted by length
    - LRU cache for frequently stemmed words
    - O(n) suffix collection using append() + reverse
    """

    def __init__(
        self,
        config: StemmerConfig | None = None,
        grammar_config_path: str | None = None,
    ):
        """
        Initialize the stemmer.

        Args:
            config: StemmerConfig instance for settings.
            grammar_config_path: Path to grammar config.
        """
        self.stemmer_config = config or _default_stemmer_config
        self.config = get_grammar_config(grammar_config_path)
        morph_config = self.config.morphology_config

        # Load and normalize suffixes
        v_suffixes = []
        n_suffixes = []
        adv_suffixes = []

        if "suffixes" in morph_config:
            suffixes_data = morph_config["suffixes"]
            if "verb_suffixes" in suffixes_data:
                v_suffixes = [item["suffix"] for item in suffixes_data["verb_suffixes"]]
            if "noun_suffixes" in suffixes_data:
                n_suffixes = [item["suffix"] for item in suffixes_data["noun_suffixes"]]
            if "adverb_suffixes" in suffixes_data:
                adv_suffixes = [item["suffix"] for item in suffixes_data["adverb_suffixes"]]

        # Sort suffixes by length (longest first) to ensure greedy matching
        self.verb_suffixes = sorted([normalize(s) for s in v_suffixes], key=len, reverse=True)
        self.noun_suffixes = sorted([normalize(s) for s in n_suffixes], key=len, reverse=True)
        self.adverb_suffixes = sorted([normalize(s) for s in adv_suffixes], key=len, reverse=True)

        # Particles
        self.particle_suffixes = sorted(
            [normalize(s) for s in self.config.particle_tags.keys()], key=len, reverse=True
        )

        # Pre-compute sorted unique suffix list (longest first for greedy matching)
        all_suffixes = set()
        all_suffixes.update(self.particle_suffixes)
        all_suffixes.update(self.verb_suffixes)
        all_suffixes.update(self.noun_suffixes)
        all_suffixes.update(self.adverb_suffixes)
        self._all_suffixes_sorted = sorted(all_suffixes, key=len, reverse=True)

        # Configure LRU cache
        self._cache_size = self.stemmer_config.cache_size
        self._stem_cached = lru_cache(maxsize=self._cache_size)(self._stem_impl)

    def stem(self, word: str) -> tuple[str, list[str]]:
        """
        Stem a word by removing known suffixes recursively.

        Uses LRU caching for frequently stemmed words.

        Args:
            word: The word to stem.

        Returns:
            Tuple containing:
            - root (str): The stemmed root word.
            - suffixes (list[str]): List of stripped suffixes (in order from root outward).

        Example:
            >>> stemmer = Stemmer()
            >>> root, suffixes = stemmer.stem("စားနေသည်")
            >>> print(root)  # "စား"
            >>> print(suffixes)  # ["နေ", "သည်"]
        """
        normalized_word = normalize(word)
        root, suffixes_tuple = self._stem_cached(normalized_word)
        return root, list(suffixes_tuple)

    def _stem_impl(self, word: str) -> tuple[str, tuple[str, ...]]:
        """
        Internal stemming implementation (cached).

        Returns tuple of suffixes for hashability in LRU cache.
        """
        root = word
        stripped_suffixes: list[str] = []

        # Iteratively strip suffixes
        # Multiple suffixes can be stacked: "စား" + "နေ" + "သည်" -> "စားနေသည်"
        # We strip from outer to inner: "သည်" -> "နေ" -> done

        max_iterations = 10  # Defense-in-depth guard
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            found_suffix = False

            # Use pre-computed combined suffix list (sorted by length, longest first)
            # This ensures longest match is found first
            for suffix in self._all_suffixes_sorted:
                if len(root) > len(suffix) and root.endswith(suffix):
                    root = root[: -len(suffix)]
                    # Append to list - O(1) amortized
                    # Suffixes are added in outer-to-inner order
                    stripped_suffixes.append(suffix)
                    found_suffix = True
                    break

            if not found_suffix:
                break

        # Reverse to get inner-to-outer order (root -> first suffix -> second suffix -> ...)
        # This is O(n) once, vs O(n²) for insert(0) in loop
        stripped_suffixes.reverse()

        return root, tuple(stripped_suffixes)

    def clear_cache(self) -> None:
        """Clear the stemming cache."""
        self._stem_cached.cache_clear()

    def cache_info(self) -> dict:
        """Get cache statistics."""
        info = self._stem_cached.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "maxsize": info.maxsize,
            "currsize": info.currsize,
        }
