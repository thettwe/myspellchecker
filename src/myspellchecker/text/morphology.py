"""
Morphology analyzer for OOV (Out-Of-Vocabulary) POS tagging and word analysis.

This module provides a rule-based system to guess the Part-of-Speech (POS)
of words not found in the main dictionary, primarily by analyzing their suffixes.

Features:
- Suffix-based POS guessing
- Prefix-based POS inference (e.g., အ prefix → Noun)
- Numeral detection for Myanmar digits and numeral words
- Proper noun suffix detection (country, city, university names)
- Confidence scoring based on suffix/prefix length and pattern reliability
- Multi-POS support for ambiguous words (N|V, ADJ|V, etc.)
- Ranked results for disambiguation
- Word analysis for root extraction and suffix identification
- OOV recovery through morphological decomposition
- Integration with Stemmer for cached suffix stripping
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from myspellchecker.core.config.algorithm_configs import MorphologyConfig
from myspellchecker.core.constants import (
    MYANMAR_NUMERAL_WORDS,
    MYANMAR_NUMERALS,
)
from myspellchecker.grammar.config import get_grammar_config
from myspellchecker.text.normalize import normalize

if TYPE_CHECKING:
    from myspellchecker.text.stemmer import Stemmer

__all__ = [
    "MorphologyAnalyzer",
    "POSGuess",
    "WordAnalysis",
    "analyze_word",
    "get_cached_analyzer",
    "get_numeral_pos_guess",
    "is_numeral_word",
]


# ============================================================
# Numeral Detection Functions
# ============================================================


def is_numeral_word(word: str) -> bool:
    """
    Check if a word is a numeral (Myanmar digits or numeral words).

    Detects both:
    - Myanmar digit characters (၀-၉)
    - Myanmar numeral words (တစ်, နှစ်, သုံး, etc.)

    Args:
        word: The word to check.

    Returns:
        True if the word is a numeral, False otherwise.

    Example:
        >>> is_numeral_word("၁၂၃")
        True
        >>> is_numeral_word("သုံး")
        True
        >>> is_numeral_word("စား")
        False
    """
    if not word:
        return False

    if all(char in MYANMAR_NUMERALS for char in word):
        return True

    if word in MYANMAR_NUMERAL_WORDS:
        return True

    return False


def get_numeral_pos_guess(word: str) -> POSGuess | None:
    """
    Get a POS guess for numeral words.

    Args:
        word: The word to analyze.

    Returns:
        POSGuess with NUM tag if word is a numeral, None otherwise.

    Example:
        >>> guess = get_numeral_pos_guess("၁၂၃")
        >>> guess.tag
        'NUM'
    """
    if is_numeral_word(word):
        # Myanmar digits have very high confidence
        if all(char in MYANMAR_NUMERALS for char in word):
            return POSGuess(
                tag="NUM",
                confidence=0.99,
                reason=f'Myanmar numeral digits "{word}"',
            )
        # Numeral words have slightly lower confidence
        return POSGuess(
            tag="NUM",
            confidence=0.95,
            reason=f'numeral word "{word}"',
        )
    return None


@dataclass
class POSGuess:
    """
    A POS tag guess with confidence score and reasoning.

    Attributes:
        tag: The guessed POS tag (e.g., "V", "N", "P_SENT")
        confidence: Confidence score between 0.0 and 1.0
        reason: Human-readable explanation for the guess
    """

    tag: str
    confidence: float
    reason: str

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"POSGuess(tag='{self.tag}', confidence={self.confidence:.2f}, reason='{self.reason}')"
        )


@dataclass
class WordAnalysis:
    """
    Result of morphological word analysis.

    Contains the decomposed parts of a word: root, identified suffixes,
    and potential POS guesses based on the morphological structure.

    Attributes:
        original: The original input word.
        root: The extracted root/stem after suffix stripping (may be same as original).
        suffixes: List of identified suffixes in order of removal.
        pos_guesses: POS guesses based on morphological analysis.
        confidence: Overall confidence in the analysis (0.0-1.0).
        is_compound: True if the word appears to be a compound.

    Example:
        >>> analysis = analyze_word("စားခဲ့သည်")
        >>> print(analysis.root)
        'စား'
        >>> print(analysis.suffixes)
        ['ခဲ့', 'သည်']
    """

    original: str
    root: str
    suffixes: list[str] = field(default_factory=list)
    pos_guesses: list[POSGuess] = field(default_factory=list)
    confidence: float = 0.0
    is_compound: bool = False

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        suffix_str = "+".join(self.suffixes) if self.suffixes else "none"
        return (
            f"WordAnalysis(root='{self.root}', suffixes=[{suffix_str}], "
            f"confidence={self.confidence:.2f})"
        )


class MorphologyAnalyzer:
    """
    Analyzes word morphology (suffixes/prefixes) to guess its Part-of-Speech.

    Features:
        - Suffix-based POS guessing
        - Prefix-based POS inference (e.g., အ prefix → Noun)
        - Numeral detection for Myanmar digits and numeral words
        - Proper noun suffix detection (country, city names)
        - Multi-POS support for ambiguous words
        - Confidence scoring based on suffix/prefix length ratio
        - Multiple output formats: set, ranked list, or single best guess
        - Optional Stemmer integration for cached suffix stripping

    Example:
        >>> analyzer = MorphologyAnalyzer()
        >>> analyzer.guess_pos("စားပြီ")  # Returns set of all matching tags
        {'V', 'P_SENT'}
        >>> analyzer.guess_pos_ranked("စားပြီ")  # Returns ranked list with confidence
        [POSGuess(tag='P_SENT', confidence=0.67, reason='particle suffix "ပြီ"'), ...]
        >>> analyzer.guess_pos_best("စားပြီ")  # Returns single best tag
        'P_SENT'
        >>> analyzer.guess_pos_multi("ကြီး")  # Returns multi-POS for ambiguous words
        {'N', 'V', 'ADJ'}

        >>> # With Stemmer integration for caching
        >>> from myspellchecker.text.stemmer import Stemmer
        >>> analyzer = MorphologyAnalyzer(stemmer=Stemmer())
    """

    # Priority order for tags when multiple match with similar confidence
    TAG_PRIORITY = {
        "NUM": 0,  # Numerals highest (unambiguous)
        "P_SENT": 1,  # Sentence particles (usually outer layer)
        "P_MOD": 2,
        "P_LOC": 3,
        "P_SUBJ": 4,
        "P_OBJ": 5,
        "V": 6,  # Verbs before nouns
        "N": 7,
        "ADJ": 8,  # Adjectives
        "ADV": 9,
    }

    def __init__(
        self,
        stemmer: Stemmer | None = None,
        config_path: str | None = None,
        morphology_config: MorphologyConfig | None = None,
    ):
        """
        Initialize the MorphologyAnalyzer.

        Args:
            stemmer: Optional Stemmer instance for cached suffix stripping.
                    If provided, uses Stemmer's LRU cache for better performance.
            config_path: Optional path to grammar config.
            morphology_config: Optional MorphologyConfig for confidence tuning.
        """
        self._stemmer = stemmer
        self.morphology_config = morphology_config or MorphologyConfig()
        self.config = get_grammar_config(config_path)

        # Extract suffixes from morphology config
        morph_config = self.config.morphology_config
        verb_suffixes = []
        noun_suffixes = []
        adverb_suffixes = []

        if "suffixes" in morph_config:
            suffixes_data = morph_config["suffixes"]
            if "verb_suffixes" in suffixes_data:
                verb_suffixes = [item["suffix"] for item in suffixes_data["verb_suffixes"]]
            if "noun_suffixes" in suffixes_data:
                noun_suffixes = [item["suffix"] for item in suffixes_data["noun_suffixes"]]
            if "adverb_suffixes" in suffixes_data:
                adverb_suffixes = [item["suffix"] for item in suffixes_data["adverb_suffixes"]]

        self.normalized_verb_suffixes = sorted(
            [normalize(s) for s in verb_suffixes], key=len, reverse=True
        )
        self.normalized_noun_suffixes = sorted(
            [normalize(s) for s in noun_suffixes], key=len, reverse=True
        )
        self.normalized_adverb_suffixes = sorted(
            [normalize(s) for s in adverb_suffixes], key=len, reverse=True
        )

        # Load other inference data
        self.ambiguous_words = self.config.ambiguous_words_map

        # Process POS inference config
        self.prefix_patterns = {}
        self.proper_noun_suffixes = {}

        pos_config = self.config.pos_inference_config
        if "prefixes" in pos_config:
            for item in pos_config["prefixes"]:
                self.prefix_patterns[item["prefix"]] = item

        if "proper_noun_suffixes" in pos_config:
            for item in pos_config["proper_noun_suffixes"]:
                self.proper_noun_suffixes[item["suffix"]] = item

    def guess_pos(self, word: str) -> set[str]:
        """
        Guesses the POS tag(s) of a given word based on its suffixes.

        This is the original method for backward compatibility. Returns all
        matching tags without disambiguation.

        Args:
            word: The word to analyze.

        Returns:
            A set of guessed POS tags (e.g., {"V", "N"}). Returns an empty set
            if no specific guess can be made.
        """
        guessed_pos: set[str] = set()

        # Check Particle Suffixes (longest first)
        for suffix_word, granular_tag in sorted(
            self.config.particle_tags.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if word.endswith(suffix_word):
                guessed_pos.add(granular_tag)

        # Check Verb Suffixes
        for suffix in self.normalized_verb_suffixes:
            if word.endswith(suffix):
                guessed_pos.add("V")

        # Check Noun Suffixes
        for suffix in self.normalized_noun_suffixes:
            if word.endswith(suffix):
                guessed_pos.add("N")

        # Check Adverb Suffixes
        for suffix in self.normalized_adverb_suffixes:
            if word.endswith(suffix):
                guessed_pos.add("ADV")

        return guessed_pos

    def guess_pos_ranked(self, word: str) -> list[POSGuess]:
        """
        Guesses POS tags with confidence scores, sorted by confidence.

        Confidence is calculated based on:
        - Numeral detection (highest priority, 0.95-0.99 confidence)
        - Prefix patterns (e.g., အ prefix → Noun, 0.60-0.75 confidence)
        - Proper noun suffix patterns (0.85-0.95 confidence)
        - Suffix length / word length ratio (longer matches = higher confidence)
        - Tag priority for tie-breaking

        Args:
            word: The word to analyze.

        Returns:
            List of POSGuess objects sorted by confidence (highest first).
            Empty list if no matches found.

        Example:
            >>> analyzer = MorphologyAnalyzer()
            >>> guesses = analyzer.guess_pos_ranked("လာခဲ့သည်")
            >>> for g in guesses:
            ...     print(f"{g.tag}: {g.confidence:.2f} ({g.reason})")
        """
        if not word:
            return []

        guesses: list[POSGuess] = []
        word_len = len(word)

        # 1. Check for numerals first (highest confidence)
        numeral_guess = get_numeral_pos_guess(word)
        if numeral_guess:
            guesses.append(numeral_guess)
            # Return immediately - numerals are unambiguous
            return guesses

        # 2. Check proper noun suffix patterns (high confidence, before prefix)
        # Proper nouns like "မြန်မာနိုင်ငံ" should not be caught by "မ" prefix
        for suffix, pattern_info in self.proper_noun_suffixes.items():
            if word.endswith(suffix) and len(word) > len(suffix):
                guesses.append(
                    POSGuess(
                        tag=pattern_info["pos"],
                        confidence=pattern_info["confidence"],
                        reason=f'{pattern_info.get("description", "")} (suffix "{suffix}")',
                    )
                )
                # Return early for proper nouns to avoid prefix interference
                guesses.sort(key=lambda g: (-g.confidence, self.TAG_PRIORITY.get(g.tag, 99)))
                return guesses

        # 3. Check prefix patterns (e.g., အ prefix → Noun)
        for prefix, pattern_info in self.prefix_patterns.items():
            if word.startswith(prefix) and len(word) > len(prefix):
                # Check if it's in the exceptions list
                if word not in pattern_info.get("exceptions", []):
                    guesses.append(
                        POSGuess(
                            tag=pattern_info["pos"],
                            confidence=pattern_info["confidence"],
                            reason=f'{pattern_info.get("description", "")} (prefix "{prefix}")',
                        )
                    )

        # 4. Check Particle Suffixes (longer suffix = higher confidence)
        mc = self.morphology_config
        for suffix_word, granular_tag in self.config.particle_tags.items():
            if word.endswith(suffix_word):
                confidence = len(suffix_word) / word_len
                # Boost particle confidence slightly (they're usually reliable)
                confidence = min(
                    confidence * mc.particle_confidence_boost,
                    mc.particle_confidence_cap,
                )
                guesses.append(
                    POSGuess(
                        tag=granular_tag,
                        confidence=confidence,
                        reason=f'particle suffix "{suffix_word}"',
                    )
                )

        # 5. Check Verb Suffixes
        for suffix in self.normalized_verb_suffixes:
            if word.endswith(suffix):
                confidence = len(suffix) / word_len * mc.verb_suffix_weight
                guesses.append(
                    POSGuess(
                        tag="V",
                        confidence=confidence,
                        reason=f'verb suffix "{suffix}"',
                    )
                )

        # 6. Check Noun Suffixes
        for suffix in self.normalized_noun_suffixes:
            if word.endswith(suffix):
                confidence = len(suffix) / word_len * mc.noun_suffix_weight
                guesses.append(
                    POSGuess(
                        tag="N",
                        confidence=confidence,
                        reason=f'noun suffix "{suffix}"',
                    )
                )

        # 7. Check Adverb Suffixes
        for suffix in self.normalized_adverb_suffixes:
            if word.endswith(suffix):
                confidence = len(suffix) / word_len * mc.adverb_suffix_weight
                guesses.append(
                    POSGuess(
                        tag="ADV",
                        confidence=confidence,
                        reason=f'adverb suffix "{suffix}"',
                    )
                )

        # Sort by confidence (descending), then by priority for tie-breaking
        guesses.sort(key=lambda g: (-g.confidence, self.TAG_PRIORITY.get(g.tag, 99)))

        return guesses

    def guess_pos_best(self, word: str) -> str | None:
        """
        Returns the single most likely POS tag for a word.

        This is a convenience method that returns the highest-confidence
        guess from guess_pos_ranked().

        Args:
            word: The word to analyze.

        Returns:
            The most likely POS tag string, or None if no matches found.

        Example:
            >>> analyzer = MorphologyAnalyzer()
            >>> analyzer.guess_pos_best("စားပြီ")
            'P_SENT'
        """
        ranked = self.guess_pos_ranked(word)
        return ranked[0].tag if ranked else None

    def guess_pos_multi(self, word: str) -> tuple[frozenset[str], float, str]:
        """
        Returns all possible POS tags for an ambiguous word with confidence.

        For words in the AMBIGUOUS_WORDS registry, returns all possible tags.
        For other words, falls back to guess_pos_ranked() to determine tags.

        This method is essential for multi-POS support where words can
        function as multiple parts of speech depending on context.

        Args:
            word: The word to analyze.

        Returns:
            Tuple of (frozenset of POS tags, confidence, source):
            - tags: Frozenset of possible POS tags (e.g., {'N', 'V'})
            - confidence: Overall confidence score (0.0-1.0)
            - source: How the POS was determined ('ambiguous_registry',
                      'morphological_inference', 'unknown')

        Example:
            >>> analyzer = MorphologyAnalyzer()
            >>> tags, conf, src = analyzer.guess_pos_multi("ကြီး")
            >>> print(tags)
            frozenset({'N', 'V', 'ADJ'})
            >>> print(src)
            'ambiguous_registry'

            >>> tags, conf, src = analyzer.guess_pos_multi("အလုပ်")
            >>> print(tags)
            frozenset({'N'})
            >>> print(src)
            'morphological_inference'
        """
        if not word:
            return frozenset(), 0.0, "unknown"

        # 1. Check if word is in the ambiguous words registry
        # We check the instance variable now, not the global function
        if word in self.ambiguous_words:
            ambig_tags = frozenset(self.ambiguous_words.get(word, []))
            # Ambiguous words have medium confidence (need context for disambiguation)
            return ambig_tags, 0.70, "ambiguous_registry"

        # 2. Check for numerals (unambiguous, high confidence)
        if is_numeral_word(word):
            return frozenset({"NUM"}), 0.99, "numeral_detection"

        # 3. Use morphological inference
        ranked = self.guess_pos_ranked(word)
        if ranked:
            # Collect all unique tags from ranked guesses
            tags = frozenset(g.tag for g in ranked)
            # Use the highest confidence from ranked guesses
            confidence = ranked[0].confidence
            return tags, confidence, "morphological_inference"

        # 4. No POS could be determined
        return frozenset(), 0.0, "unknown"

    def analyze_word(
        self,
        word: str,
        dictionary_check: Callable[[str], bool] | None = None,
    ) -> WordAnalysis:
        """
        Perform morphological analysis on a word to extract root and suffixes.

        This method attempts to decompose a word into its root form and any
        grammatical suffixes. This is useful for OOV (Out-of-Vocabulary)
        recovery, where an unknown word might have a known root.

        Args:
            word: The word to analyze.
            dictionary_check: Optional callable that returns True if a word
                            is in the dictionary. Used to validate extracted roots.

        Returns:
            WordAnalysis object containing root, suffixes, and POS guesses.

        Example:
            >>> analyzer = MorphologyAnalyzer()
            >>> result = analyzer.analyze_word("စားခဲ့သည်")
            >>> print(result.root)
            'စား'
            >>> print(result.suffixes)
            ['ခဲ့', 'သည်']
        """
        if not word:
            return WordAnalysis(original="", root="")

        word = normalize(word)
        original = word
        suffixes: list[str] = []
        confidence = 0.0

        # Phase 1: Greedily strip all known suffixes
        # Use Stemmer if available (has LRU cache for performance)
        if self._stemmer is not None:
            # Use cached Stemmer for fast suffix stripping
            current, temp_suffixes = self._stemmer.stem(word)
        else:
            # Fallback: Manual suffix stripping
            # Collect all suffix patterns sorted by length (longest first)
            all_suffixes = []

            # Particle suffixes (highest priority for sentence endings)
            for suffix_word in sorted(self.config.particle_tags.keys(), key=len, reverse=True):
                all_suffixes.append((normalize(suffix_word), "particle", suffix_word))

            # Verb suffixes
            for suffix in self.normalized_verb_suffixes:
                all_suffixes.append((suffix, "verb", suffix))

            # Noun suffixes
            for suffix in self.normalized_noun_suffixes:
                all_suffixes.append((suffix, "noun", suffix))

            # Adverb suffixes
            for suffix in self.normalized_adverb_suffixes:
                all_suffixes.append((suffix, "adverb", suffix))

            # Sort by length descending to match longest suffixes first
            all_suffixes.sort(key=lambda x: len(x[0]), reverse=True)

            # Iteratively strip suffixes from the end
            # Strategy: Strip suffixes greedily first, then validate final root
            current = word
            max_iterations = self.morphology_config.max_suffix_strip_iterations
            iterations = 0
            temp_suffixes = []

            while current and iterations < max_iterations:
                found_suffix = False
                for suffix_norm, _suffix_type, suffix_orig in all_suffixes:
                    if current.endswith(suffix_norm) and len(current) > len(suffix_norm):
                        potential_root = current[: -len(suffix_norm)]
                        # Use append() O(1) instead of insert(0, ...) O(n)
                        temp_suffixes.append(suffix_orig)
                        current = potential_root
                        found_suffix = True
                        break

                if not found_suffix:
                    break
                iterations += 1

            # Reverse to restore correct order (suffixes were collected end-to-start)
            temp_suffixes.reverse()

        # Phase 2: If dictionary check provided, validate the final root
        # If root is valid, keep the analysis. Otherwise, try progressively
        # adding suffixes back until we find a valid root
        if dictionary_check is not None and temp_suffixes:
            if dictionary_check(current):
                # Root is valid, use the full analysis
                suffixes = temp_suffixes
            else:
                # Root not valid, try adding suffixes back to find valid root
                # This handles cases where intermediate forms are valid
                test_root = current
                valid_suffixes: list[str] = []
                for idx, suffix in enumerate(temp_suffixes):
                    test_root = test_root + suffix
                    if dictionary_check(test_root):
                        # Found a valid intermediate root
                        current = test_root
                        valid_suffixes = temp_suffixes[idx + 1 :]
                        break
                suffixes = valid_suffixes
        else:
            # No dictionary check, accept the greedy analysis
            suffixes = temp_suffixes

        root = current

        # Calculate confidence based on analysis quality
        mc = self.morphology_config
        if suffixes:
            # More suffixes found = higher confidence the root is valid
            suffix_ratio = len("".join(suffixes)) / len(original)
            confidence = min(
                mc.oov_base_confidence + suffix_ratio * mc.oov_scale_factor, mc.oov_cap
            )

            # Boost confidence if dictionary check confirmed the root
            if dictionary_check is not None and dictionary_check(root):
                confidence = min(confidence + mc.dictionary_boost, mc.dictionary_cap)
        else:
            # No suffixes found - word is probably a root or unknown structure
            confidence = (
                mc.fallback_with_dict
                if dictionary_check and dictionary_check(root)
                else mc.fallback_without_dict
            )

        # Get POS guesses for the original word
        pos_guesses = self.guess_pos_ranked(original)

        return WordAnalysis(
            original=original,
            root=root,
            suffixes=suffixes,
            pos_guesses=pos_guesses,
            confidence=confidence,
            is_compound=len(suffixes) > 2,  # Rough heuristic: 3+ suffixes may indicate compound
        )


# Module-level singleton for convenience functions
# Using @lru_cache for thread-safe lazy initialization


@lru_cache(maxsize=1)
def _get_default_analyzer() -> MorphologyAnalyzer:
    """Get or create the default MorphologyAnalyzer singleton (thread-safe)."""
    return MorphologyAnalyzer()


@lru_cache(maxsize=1)
def get_cached_analyzer() -> MorphologyAnalyzer:
    """
    Get a MorphologyAnalyzer with Stemmer integration for better performance.

    The Stemmer provides LRU caching for suffix stripping operations,
    which significantly improves performance for repeated analysis of
    similar words.

    This function is thread-safe using functools.lru_cache.

    Returns:
        MorphologyAnalyzer instance with Stemmer integration.

    Example:
        >>> from myspellchecker.text.morphology import get_cached_analyzer
        >>> analyzer = get_cached_analyzer()
        >>> result = analyzer.analyze_word("စားခဲ့သည်")
    """
    from myspellchecker.text.stemmer import Stemmer

    return MorphologyAnalyzer(stemmer=Stemmer())


def analyze_word(
    word: str,
    dictionary_check: Callable[[str], bool] | None = None,
    use_cache: bool = False,
) -> WordAnalysis:
    """
    Analyze a word's morphological structure to extract root and suffixes.

    This is a convenience function that uses a module-level MorphologyAnalyzer
    singleton. For batch processing, consider creating your own MorphologyAnalyzer
    instance.

    Args:
        word: The word to analyze.
        dictionary_check: Optional callable that returns True if a word
                         is in the dictionary. Used to validate extracted roots.
        use_cache: If True, uses a cached analyzer with Stemmer integration
                  for better performance on repeated calls.

    Returns:
        WordAnalysis object containing root, suffixes, and POS guesses.

    Example:
        >>> from myspellchecker.text.morphology import analyze_word
        >>> result = analyze_word("စားခဲ့သည်")
        >>> print(f"Root: {result.root}, Suffixes: {result.suffixes}")
        Root: စား, Suffixes: ['ခဲ့', 'သည်']

        >>> # With dictionary validation
        >>> def is_valid(w):
        ...     return w in {"စား", "သွား", "လာ"}
        >>> result = analyze_word("စားခဲ့သည်", dictionary_check=is_valid)
        >>> print(result.confidence)  # Higher due to dictionary validation
        0.85

        >>> # With caching for better performance
        >>> result = analyze_word("စားခဲ့သည်", use_cache=True)
    """
    analyzer = get_cached_analyzer() if use_cache else _get_default_analyzer()
    return analyzer.analyze_word(word, dictionary_check)
