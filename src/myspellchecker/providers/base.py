"""
Abstract base class for dictionary data access.

This module defines the DictionaryProvider interface that all dictionary
implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from myspellchecker.core.exceptions import ProviderError

__all__ = [
    "DictionaryProvider",
]


class DictionaryProvider(ABC):
    """
    Abstract interface for dictionary data storage and retrieval.

    DictionaryProvider implementations abstract the storage backend for
    dictionary data, enabling pluggable storage solutions (SQLite, JSON,
    CSV, in-memory, etc.).

    All providers must support:
    1. Syllable validation (Layer 1)
    2. Word validation (Layer 2)
    3. Frequency retrieval for suggestion ranking
    4. Bigram probability retrieval for context checking (Layer 3)

    Thread Safety:
        Implementations should be thread-safe for read operations.
        Write operations (if supported) may require external synchronization.

    API Naming Conventions:
        Method names follow these patterns:
        - Single item: ``is_valid_word(word)``
        - Multiple items: ``is_valid_words_bulk(words)`` - uses "bulk" suffix
        - Batch internal: Methods like ``_batch_validate`` use "batch" prefix

        The distinction:
        - "bulk" = public API methods for validating multiple items
        - "batch" = internal implementation detail (SQL batching, etc.)

        This naming convention is intentional to distinguish user-facing
        batch operations from internal database batching strategies.
    """

    @abstractmethod
    def is_valid_syllable(self, syllable: str) -> bool:
        """
        Check if a syllable exists in the dictionary.

        This is the core validation method for Layer 1 (syllable-level)
        spell checking. Implementation should be highly optimized as this
        is called for every syllable in the input text.

        Args:
            syllable: Myanmar syllable (Unicode string) to validate.
                     Should be pre-normalized using Myanmar3 or equivalent.

        Returns:
            True if syllable exists in dictionary, False otherwise.

        Expected Performance:
            <1ms per query (with proper indexing)

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.is_valid_syllable("မြန်")
            True
            >>> provider.is_valid_syllable("မျန်")  # Typo (wrong diacritic)
            False

        Notes:
            - Input should be normalized before calling
            - Case sensitivity: Myanmar script doesn't have case
            - Empty strings should return False
        """
        raise NotImplementedError

    @abstractmethod
    def is_valid_word(self, word: str) -> bool:
        """
        Check if a multi-syllable word exists in the dictionary.

        This method supports Layer 2 (word-level) spell checking. A word
        may consist of valid syllables but still be invalid as a combination.

        Args:
            word: Myanmar word (Unicode string) to validate.
                 May contain multiple syllables.

        Returns:
            True if word exists in dictionary, False otherwise.

        Expected Performance:
            <1ms per query (with proper indexing)

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.is_valid_word("မြန်မာ")  # "Myanmar" - valid
            True
            >>> provider.is_valid_word("မြန်စာ")  # Invalid combination
            False

        Notes:
            - A valid word must also have all valid syllables
            - Word validation is more strict than syllable validation
            - Words may be single-syllable
        """
        raise NotImplementedError

    @abstractmethod
    def get_syllable_frequency(self, syllable: str) -> int:
        """
        Get corpus frequency count for a syllable.

        Frequency data is used by SymSpell and other algorithms to rank
        correction suggestions by likelihood. Higher frequency syllables
        are more common and should be ranked higher.

        Args:
            syllable: Myanmar syllable (Unicode string).

        Returns:
            Integer frequency count (number of occurrences in corpus).
            Returns 0 if syllable not found in dictionary.

        Expected Performance:
            <1ms per query

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.get_syllable_frequency("မြန်")
            15432  # Common syllable
            >>> provider.get_syllable_frequency("ဩ")
            23  # Rare syllable
            >>> provider.get_syllable_frequency("xyz")
            0  # Not found

        Notes:
            - Used by SymSpell for ranking correction suggestions
            - Frequency should reflect actual corpus statistics
            - Zero frequency means syllable not in dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def get_word_frequency(self, word: str) -> int:
        """
        Get corpus frequency count for a word.

        Similar to get_syllable_frequency but for multi-syllable words.
        Used for word-level suggestion ranking.

        Args:
            word: Myanmar word (Unicode string).

        Returns:
            Integer frequency count (number of occurrences in corpus).
            Returns 0 if word not found in dictionary.

        Expected Performance:
            <1ms per query

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.get_word_frequency("မြန်မာ")  # "Myanmar"
            8752
            >>> provider.get_word_frequency("နိုငံ")  # "country"
            12341
            >>> provider.get_word_frequency("unknown")
            0

        Notes:
            - Word frequency is typically lower than syllable frequency
            - Used for word-level suggestion ranking
            - Zero frequency means word not in dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def get_word_pos(self, word: str) -> str | None:
        """
        Get Part-of-Speech (POS) tag(s) for a word.

        Used by SyntacticRuleChecker (Layer 2.5) to enforce grammatical rules
        (e.g., Verb-Particle agreement).

        Words with multiple POS tags return a pipe-separated string ordered by
        frequency (most common first). Use ``tag in pos.split("|")`` or the
        grammar engine's ``_has_tag()`` helper to check for a specific tag.

        Args:
            word: Myanmar word (Unicode string).

        Returns:
            POS tag string or None if not found/unknown.
            Single-POS: ``'N'``, ``'V'``, ``'ADJ'``
            Multi-POS: ``'N|V'``, ``'N|ADJ'``

        Expected Performance:
            <1ms per query

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.get_word_pos("ကျောင်း")
            'N'
            >>> provider.get_word_pos("ရေ")
            'N|V'
        """
        raise NotImplementedError

    @abstractmethod
    def get_bigram_probability(self, prev_word: str, current_word: str) -> float:
        """
        Get conditional probability P(current_word | prev_word).

        This method supports Layer 3 (context-aware) spell checking by
        providing bigram probabilities for detecting unlikely word sequences.

        The probability represents: "Given that prev_word occurred, what is
        the probability that current_word follows?"

        Args:
            prev_word: Previous word in sequence (Unicode string).
            current_word: Current word in sequence (Unicode string).

        Returns:
            Probability as float in range [0.0, 1.0].
            Returns 0.0 if bigram not found (unseen combination).

        Expected Performance:
            <5ms per query (with JOIN optimization and indexing)

        Example:
            >>> provider = SQLiteProvider()
            >>> provider.get_bigram_probability("သူ", "သွား")
            0.234  # "He goes" - common sequence
            >>> provider.get_bigram_probability("သူ", "ဘယ်")
            0.012  # Less common sequence
            >>> provider.get_bigram_probability("abc", "xyz")
            0.0  # Unknown words

        Calculation:
            P(w2|w1) = count(w1, w2) / count(w1)

        Notes:
            - Used by N-gram Context Checker (Layer 3)
            - Low probability (<threshold) indicates potential context error
            - Zero probability means bigram never seen in corpus
            - May apply smoothing (Laplace, Good-Turing) for unseen bigrams
            - Should handle edge cases (start/end of sentence)
        """
        raise NotImplementedError

    @abstractmethod
    def get_trigram_probability(self, w1: str, w2: str, w3: str) -> float:
        """
        Get conditional probability P(w3 | w1, w2).

        Args:
            w1: First word in sequence.
            w2: Second word in sequence.
            w3: Third word in sequence (target).

        Returns:
            Probability as float in range [0.0, 1.0].
            Returns 0.0 if trigram not found.
        """
        raise NotImplementedError

    @abstractmethod
    def get_fourgram_probability(self, word1: str, word2: str, word3: str, word4: str) -> float:
        """Get P(word4 | word1, word2, word3). Returns 0.0 if not found."""
        raise NotImplementedError

    @abstractmethod
    def get_fivegram_probability(
        self, word1: str, word2: str, word3: str, word4: str, word5: str
    ) -> float:
        """Get P(word5 | word1, word2, word3, word4). Returns 0.0 if not found."""
        raise NotImplementedError

    @abstractmethod
    def get_pos_unigram_probabilities(self) -> dict[str, float]:
        """
        Get all POS unigram probabilities from the database.

        Returns:
            Dictionary mapping pos_tag (str) to probability (float).
        """
        raise NotImplementedError

    @abstractmethod
    def get_pos_bigram_probabilities(self) -> dict[tuple[str, str], float]:
        """
        Get all POS bigram probabilities.

        Returns:
            Dictionary mapping (pos1, pos2) tuple to probability (float).
        """
        raise NotImplementedError

    @abstractmethod
    def get_pos_trigram_probabilities(self) -> dict[tuple[str, str, str], float]:
        """
        Get all POS trigram probabilities.

        Returns:
            Dictionary mapping (pos1, pos2, pos3) tuple to probability (float).
        """
        raise NotImplementedError

    @abstractmethod
    def get_top_continuations(self, prev_word: str, limit: int = 20) -> list[tuple[str, float]]:
        """
        Get the most likely words to follow a given word.

        This method retrieves the top N words that commonly follow prev_word
        based on bigram probabilities. Used by NgramContextChecker to generate
        candidate suggestions for context-aware spell checking.

        Args:
            prev_word: Previous word in sequence (Unicode string).
            limit: Maximum number of continuations to return (default: 20).

        Returns:
            List of (word, probability) tuples, sorted by probability (descending).
            Returns empty list if prev_word not found or has no continuations.

        Expected Performance:
            <10ms per query (with proper indexing)

        Example:
            >>> provider = SQLiteProvider()
            >>> continuations = provider.get_top_continuations("သူ", limit=5)
            >>> for word, prob in continuations:
            ...     print(f"{word}: {prob:.3f}")
            သွား: 0.234  # "goes"
            သည်: 0.189  # "is/am/are"
            ရှိ: 0.156   # "has/have"
            က: 0.089     # [subject marker]
            တို: 0.067   # "and similar"

        Notes:
            - Used by NgramContextChecker for candidate generation
            - Results are sorted by probability (highest first)
            - Returns empty list for unknown words
            - Limit controls query performance and memory usage
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_syllables(self) -> Iterator[tuple[str, int]]:
        """
        Get iterator over all syllables in dictionary.

        Used for building SymSpell index.

        Returns:
            Iterator yielding (syllable, frequency) tuples.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_words(self) -> Iterator[tuple[str, int]]:
        """
        Get iterator over all words in dictionary.

        Used for building SymSpell index.

        Returns:
            Iterator yielding (word, frequency) tuples.
        """
        raise NotImplementedError

    # ==========================================================================
    # Bulk Operations - Default implementations for backward compatibility
    # Providers can override these for optimized batch queries
    # ==========================================================================

    def is_valid_syllables_bulk(self, syllables: list[str]) -> dict[str, bool]:
        """
        Check validity of multiple syllables in a single operation.

        This bulk operation can be significantly faster than individual calls
        when checking many syllables, especially with database providers that
        can optimize multi-value queries.

        Args:
            syllables: List of Myanmar syllables to validate.

        Returns:
            Dictionary mapping syllable to validity (True/False).

        Expected Performance:
            Optimized implementations should achieve <0.1ms per syllable
            for batches of 100+ syllables.

        Example:
            >>> provider = SQLiteProvider()
            >>> syllables = ["မြန်", "မာ", "xyz"]
            >>> results = provider.is_valid_syllables_bulk(syllables)
            >>> results
            {"မြန်": True, "မာ": True, "xyz": False}

        Notes:
            - Default implementation calls is_valid_syllable individually
            - Database providers should override with optimized batch queries
            - Results preserve input order when iterating dict (Python 3.7+)
        """
        return {s: self.is_valid_syllable(s) for s in syllables}

    def is_valid_words_bulk(self, words: list[str]) -> dict[str, bool]:
        """
        Check validity of multiple words in a single operation.

        Args:
            words: List of Myanmar words to validate.

        Returns:
            Dictionary mapping word to validity (True/False).

        Example:
            >>> provider = SQLiteProvider()
            >>> words = ["မြန်မာ", "နိုင်ငံ", "invalid"]
            >>> results = provider.is_valid_words_bulk(words)
            >>> results
            {"မြန်မာ": True, "နိုင်ငံ": True, "invalid": False}

        Notes:
            - Default implementation calls is_valid_word individually
            - Override for optimized batch queries
        """
        return {w: self.is_valid_word(w) for w in words}

    def get_syllable_frequencies_bulk(self, syllables: list[str]) -> dict[str, int]:
        """
        Get corpus frequencies for multiple syllables in a single operation.

        Args:
            syllables: List of Myanmar syllables.

        Returns:
            Dictionary mapping syllable to frequency count (0 if not found).

        Example:
            >>> provider = SQLiteProvider()
            >>> syllables = ["မြန်", "မာ", "xyz"]
            >>> results = provider.get_syllable_frequencies_bulk(syllables)
            >>> results
            {"မြန်": 15432, "မာ": 8234, "xyz": 0}

        Notes:
            - Default implementation calls get_syllable_frequency individually
            - Override for optimized batch queries
        """
        return {s: self.get_syllable_frequency(s) for s in syllables}

    def get_word_frequencies_bulk(self, words: list[str]) -> dict[str, int]:
        """
        Get corpus frequencies for multiple words in a single operation.

        Args:
            words: List of Myanmar words.

        Returns:
            Dictionary mapping word to frequency count (0 if not found).

        Example:
            >>> provider = SQLiteProvider()
            >>> words = ["မြန်မာ", "နိုင်ငံ"]
            >>> results = provider.get_word_frequencies_bulk(words)
            >>> results
            {"မြန်မာ": 8752, "နိုင်ငံ": 12341}

        Notes:
            - Default implementation calls get_word_frequency individually
            - Override for optimized batch queries
        """
        return {w: self.get_word_frequency(w) for w in words}

    def get_word_pos_bulk(self, words: list[str]) -> dict[str, str | None]:
        """
        Get POS tags for multiple words in a single operation.

        Args:
            words: List of Myanmar words.

        Returns:
            Dictionary mapping word to POS tag string (None if not found).
            Multi-POS words return pipe-separated tags, e.g. ``'N|V'``.

        Example:
            >>> provider = SQLiteProvider()
            >>> words = ["ကျောင်း", "ရေ", "unknown"]
            >>> results = provider.get_word_pos_bulk(words)
            >>> results
            {"ကျောင်း": "N", "ရေ": "N|V", "unknown": None}

        Notes:
            - Default implementation calls get_word_pos individually
            - Override for optimized batch queries
        """
        return {w: self.get_word_pos(w) for w in words}

    def get_word_syllable_count(self, word: str) -> int | None:
        """
        Get syllable count for a word.

        This method returns the number of syllables in a word if stored
        in the dictionary. Useful for morphological analysis and validation.

        Args:
            word: Myanmar word (Unicode string).

        Returns:
            Syllable count, or None if not supported by this provider.
            Use supports_syllable_count property to check support before calling.

        Notes:
            - Default implementation returns None (not supported)
            - Override in providers that store syllable count data (e.g., JSON, CSV)
            - SQLiteProvider and MemoryProvider may not have this data

        Example:
            >>> if provider.supports_syllable_count:
            ...     count = provider.get_word_syllable_count(word)
            ...     if count is not None:
            ...         print(f"Word has {count} syllables")
        """
        return None

    @property
    def supports_syllable_count(self) -> bool:
        """
        Whether this provider supports syllable count retrieval.

        Returns:
            True if get_word_syllable_count returns actual counts,
            False if it always returns None.

        Example:
            >>> if provider.supports_syllable_count:
            ...     count = provider.get_word_syllable_count("မြန်မာ")
        """
        return False

    # ==========================================================================
    # Existence Check Methods
    # ==========================================================================

    def has_syllable(self, syllable: str) -> bool:
        """
        Check if syllable exists in dictionary (pure existence check).

        Unlike is_valid_syllable() which may perform additional validation,
        this method only checks dictionary membership.

        Args:
            syllable: Myanmar syllable (Unicode string) to check.

        Returns:
            True if syllable exists in dictionary, False otherwise.

        Example:
            >>> provider.has_syllable("မြန်")
            True
            >>> provider.has_syllable("xyz")
            False

        Note:
            Default implementation delegates to is_valid_syllable().
            Override if validation includes additional rule checks.
        """
        return self.is_valid_syllable(syllable)

    def has_word(self, word: str) -> bool:
        """
        Check if word exists in dictionary (pure existence check).

        Unlike is_valid_word() which may perform additional validation,
        this method only checks dictionary membership.

        Args:
            word: Myanmar word (Unicode string) to check.

        Returns:
            True if word exists in dictionary, False otherwise.

        Example:
            >>> provider.has_word("မြန်မာ")
            True
            >>> provider.has_word("xyz")
            False

        Note:
            Default implementation delegates to is_valid_word().
            Override if validation includes additional rule checks.
        """
        return self.is_valid_word(word)

    def __contains__(self, item: str) -> bool:
        """
        Support Python's `in` operator for membership testing.

        Checks if the item exists as either a syllable or word in the dictionary.

        Args:
            item: Myanmar text (syllable or word) to check.

        Returns:
            True if item exists as syllable or word, False otherwise.

        Example:
            >>> "မြန်" in provider
            True
            >>> "xyz" in provider
            False

        Note:
            Checks syllables first, then words. For explicit checking,
            use has_syllable() or has_word() directly.
        """
        return self.has_syllable(item) or self.has_word(item)

    # ==========================================================================
    # Factory Method
    # ==========================================================================

    @classmethod
    def create(
        cls,
        provider_type: str = "sqlite",
        **kwargs,
    ) -> DictionaryProvider:
        """
        Factory method to create provider instances.

        Provides a unified way to create different provider types without
        importing specific provider classes directly.

        Args:
            provider_type: Type of provider to create. One of:
                - "sqlite": SQLiteProvider (default, production use)
                - "memory": MemoryProvider (testing, in-memory)
                - "json": JSONProvider (simple file-based)
                - "csv": CSVProvider (CSV file-based)
            **kwargs: Provider-specific configuration passed to constructor.

        Returns:
            Configured DictionaryProvider instance.

        Raises:
            ValueError: If provider_type is not recognized.

        Example:
            >>> # Create SQLite provider
            >>> provider = DictionaryProvider.create("sqlite", database_path="myspell.db")

            >>> # Create memory provider for testing
            >>> provider = DictionaryProvider.create(
            ...     "memory",
            ...     syllables={"မြန်": 100, "မာ": 50},
            ...     words={"မြန်မာ": 80},
            ... )

            >>> # Create JSON provider
            >>> provider = DictionaryProvider.create("json", json_path="dict.json")
        """
        provider_type = provider_type.lower()

        if provider_type == "sqlite":
            from myspellchecker.providers.sqlite import SQLiteProvider

            return SQLiteProvider(**kwargs)
        elif provider_type == "memory":
            from myspellchecker.providers.memory import MemoryProvider

            return MemoryProvider(**kwargs)
        elif provider_type == "json":
            from myspellchecker.providers.json_provider import JSONProvider

            return JSONProvider(**kwargs)
        elif provider_type == "csv":
            from myspellchecker.providers.csv_provider import CSVProvider

            return CSVProvider(**kwargs)
        else:
            available = ["sqlite", "memory", "json", "csv"]
            raise ProviderError(
                f"Unknown provider type: {provider_type!r}. Available types: {available}"
            )
