"""
In-memory dictionary provider implementation.

This module provides a simple in-memory implementation of DictionaryProvider,
ideal for testing, prototyping, and small-scale applications.
"""

from __future__ import annotations

from collections.abc import Iterator

from myspellchecker.providers.base import DictionaryProvider

__all__ = [
    "MemoryProvider",
]


class MemoryProvider(DictionaryProvider):
    """
    In-memory dictionary provider using Python dictionaries.

    MemoryProvider stores all dictionary data in RAM using native Python
    dictionaries. It's optimized for fast lookups with O(1) average case
    complexity for all operations.

    This provider is ideal for:
    - Testing and development
    - Small dictionaries (<10,000 entries)
    - Applications where fast startup is critical
    - Scenarios where data persistence is not required

    For production use with large dictionaries, consider SQLiteProvider.

    Thread Safety:
        Read operations are thread-safe (Python dict reads are atomic).
        Write operations require external synchronization.

    Example:
        >>> # Initialize with custom data
        >>> syllables = {
        ...     "မြန်": 1500,
        ...     "မာ": 2300,
        ...     "နိုင်": 1800,
        ...     "ငံ": 900
        ... }
        >>> words = {
        ...     "မြန်မာ": 850,
        ...     "နိုင်ငံ": 720
        ... }
        >>> bigrams = {
        ...     ("မြန်မာ", "နိုင်ငံ"): 0.45
        ... }
        >>> provider = MemoryProvider(
        ...     syllables=syllables,
        ...     words=words,
        ...     bigrams=bigrams
        ... )
        >>> provider.is_valid_syllable("မြန်")
        True
        >>> provider.get_syllable_frequency("မာ")
        2300

        >>> # Initialize empty and populate later
        >>> provider = MemoryProvider()
        >>> provider.add_syllable("သူ", frequency=1000)
        >>> provider.add_word("သူတို့", frequency=450)
    """

    def __init__(
        self,
        syllables: dict[str, int] | None = None,
        words: dict[str, int] | None = None,
        bigrams: dict[tuple[str, str], float] | None = None,
        trigrams: dict[tuple[str, str, str], float] | None = None,
        fourgrams: dict[tuple[str, str, str, str], float] | None = None,
        fivegrams: dict[tuple[str, str, str, str, str], float] | None = None,
        word_pos: dict[str, str] | None = None,
    ):
        """
        Initialize MemoryProvider with optional pre-populated data.

        Args:
            syllables: Dictionary mapping syllable -> frequency count.
                      Defaults to empty dict if not provided.
            words: Dictionary mapping word -> frequency count.
                  Defaults to empty dict if not provided.
            bigrams: Dictionary mapping (prev_word, curr_word) -> probability.
                    Defaults to empty dict if not provided.
            trigrams: Dictionary mapping (word1, word2, word3) -> probability.
                     Defaults to empty dict if not provided.
            fourgrams: Dictionary mapping (w1, w2, w3, w4) -> probability.
                      Defaults to empty dict if not provided.
            fivegrams: Dictionary mapping (w1, w2, w3, w4, w5) -> probability.
                      Defaults to empty dict if not provided.
            word_pos: Dictionary mapping word -> POS tag.
                     Defaults to empty dict if not provided.

        Example:
            >>> # Empty provider
            >>> provider = MemoryProvider()

            >>> # Pre-populated provider
            >>> provider = MemoryProvider(
            ...     syllables={"မြန်": 100, "မာ": 200},
            ...     words={"မြန်မာ": 50}
            ... )
        """
        self._syllables: dict[str, int] = syllables if syllables is not None else {}
        self._words: dict[str, int] = words if words is not None else {}
        self._bigrams: dict[tuple[str, str], float] = bigrams if bigrams is not None else {}
        self._trigrams: dict[tuple[str, str, str], float] = trigrams if trigrams is not None else {}
        self._fourgrams: dict[tuple[str, str, str, str], float] = (
            fourgrams if fourgrams is not None else {}
        )
        self._fivegrams: dict[tuple[str, str, str, str, str], float] = (
            fivegrams if fivegrams is not None else {}
        )
        self._word_pos: dict[str, str] = word_pos if word_pos is not None else {}
        # Index for O(1) bigram prefix lookups
        self._bigram_index: dict[str, list[tuple[str, float]]] = {}
        self._rebuild_bigram_index()

        # Enrichment data (populated via set_* methods or directly)
        self._confusable_pairs: dict[str, list[tuple[str, str, float, int]]] = {}
        self._compound_confusions: dict[str, tuple[str, str, int, int, float]] = {}
        self._collocations: dict[tuple[str, str], float] = {}
        self._register_tags: dict[str, str] = {}

    def _rebuild_bigram_index(self) -> None:
        """
        Rebuild the bigram prefix index for O(1) lookups.

        This creates an index mapping each first word to a sorted list of
        (second_word, probability) tuples, enabling efficient get_top_continuations.

        Thread Safety:
            This method is thread-safe for concurrent reads. The new index is built
            completely before atomically replacing the old one (Python dict assignment
            is atomic). Concurrent readers always see either the old complete index
            or the new complete index, never a partial state.

            Note: This assumes external synchronization for concurrent writes to
            _bigrams, as documented in the class docstring.
        """
        # Build new index without modifying the existing one
        # This ensures concurrent readers always see a complete index
        new_index: dict[str, list[tuple[str, float]]] = {}
        for (w1, w2), prob in self._bigrams.items():
            if w1 not in new_index:
                new_index[w1] = []
            new_index[w1].append((w2, prob))

        # Sort each list by probability (descending) for efficient limit queries
        for w1 in new_index:
            new_index[w1].sort(key=lambda x: x[1], reverse=True)

        # Atomic replacement: Python dict assignment is atomic at the bytecode level
        # Concurrent readers will see either the old index or the new index,
        # never a partially built index
        self._bigram_index = new_index

    def is_valid_syllable(self, syllable: str) -> bool:
        """
        Check if a syllable exists in the dictionary.

        Args:
            syllable: Myanmar syllable (Unicode string) to validate.

        Returns:
            True if syllable exists in dictionary, False otherwise.

        Performance:
            O(1) average case - Python dict lookup

        Example:
            >>> provider = MemoryProvider(syllables={"မြန်": 100})
            >>> provider.is_valid_syllable("မြန်")
            True
            >>> provider.is_valid_syllable("xyz")
            False
            >>> provider.is_valid_syllable("")
            False
        """
        if not syllable:  # Empty string check
            return False
        return syllable in self._syllables

    def is_valid_word(self, word: str) -> bool:
        """
        Check if a multi-syllable word exists in the dictionary.

        Args:
            word: Myanmar word (Unicode string) to validate.

        Returns:
            True if word exists in dictionary, False otherwise.

        Performance:
            O(1) average case - Python dict lookup

        Example:
            >>> provider = MemoryProvider(words={"မြန်မာ": 50})
            >>> provider.is_valid_word("မြန်မာ")
            True
            >>> provider.is_valid_word("invalid")
            False
            >>> provider.is_valid_word("")
            False
        """
        if not word:  # Empty string check
            return False
        return word in self._words

    def get_syllable_frequency(self, syllable: str) -> int:
        """
        Get corpus frequency count for a syllable.

        Args:
            syllable: Myanmar syllable (Unicode string).

        Returns:
            Integer frequency count. Returns 0 if syllable not found.

        Performance:
            O(1) average case - Python dict lookup

        Example:
            >>> provider = MemoryProvider(syllables={"မြန်": 1500, "မာ": 2300})
            >>> provider.get_syllable_frequency("မြန်")
            1500
            >>> provider.get_syllable_frequency("unknown")
            0
        """
        return self._syllables.get(syllable, 0)

    def get_word_frequency(self, word: str) -> int:
        """
        Get corpus frequency count for a word.

        Args:
            word: Myanmar word (Unicode string).

        Returns:
            Integer frequency count. Returns 0 if word not found.

        Performance:
            O(1) average case - Python dict lookup

        Example:
            >>> provider = MemoryProvider(words={"မြန်မာ": 850, "နိုင်ငံ": 720})
            >>> provider.get_word_frequency("မြန်မာ")
            850
            >>> provider.get_word_frequency("unknown")
            0
        """
        return self._words.get(word, 0)

    def get_word_pos(self, word: str) -> str | None:
        """
        Get Part-of-Speech (POS) tag(s) for a word.

        Args:
            word: Myanmar word (Unicode string).

        Returns:
            POS tag string (possibly pipe-separated for multi-POS words,
            e.g. ``'N|V'``) or None if not found.

        Performance:
            O(1) average case - Python dict lookup
        """
        return self._word_pos.get(word)

    def get_bigram_probability(self, prev_word: str, current_word: str) -> float:
        """
        Get conditional probability P(current_word | prev_word).

        Args:
            prev_word: Previous word in sequence (Unicode string).
            current_word: Current word in sequence (Unicode string).

        Returns:
            Probability as float in range [0.0, 1.0].
            Returns 0.0 if bigram not found.

        Performance:
            O(1) average case - Python dict lookup with tuple key

        Example:
            >>> bigrams = {("သူ", "သွား"): 0.234, ("သူ", "ဘယ်"): 0.012}
            >>> provider = MemoryProvider(bigrams=bigrams)
            >>> provider.get_bigram_probability("သူ", "သွား")
            0.234
            >>> provider.get_bigram_probability("abc", "xyz")
            0.0
        """
        return self._bigrams.get((prev_word, current_word), 0.0)

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
        return self._trigrams.get((w1, w2, w3), 0.0)

    def get_fourgram_probability(self, word1: str, word2: str, word3: str, word4: str) -> float:
        """
        Get conditional probability P(word4 | word1, word2, word3).

        Args:
            word1: First word in sequence.
            word2: Second word in sequence.
            word3: Third word in sequence.
            word4: Fourth word in sequence (target).

        Returns:
            Probability as float in range [0.0, 1.0].
            Returns 0.0 if fourgram not found.
        """
        return self._fourgrams.get((word1, word2, word3, word4), 0.0)

    def get_fivegram_probability(
        self, word1: str, word2: str, word3: str, word4: str, word5: str
    ) -> float:
        """
        Get conditional probability P(word5 | word1, word2, word3, word4).

        Args:
            word1: First word in sequence.
            word2: Second word in sequence.
            word3: Third word in sequence.
            word4: Fourth word in sequence.
            word5: Fifth word in sequence (target).

        Returns:
            Probability as float in range [0.0, 1.0].
            Returns 0.0 if fivegram not found.
        """
        return self._fivegrams.get((word1, word2, word3, word4, word5), 0.0)

    def get_top_continuations(self, prev_word: str, limit: int = 20) -> list[tuple[str, float]]:
        """
        Get the most likely words to follow a given word.

        This method retrieves the top N words that commonly follow prev_word
        based on bigram probabilities.

        Args:
            prev_word: Previous word in sequence (Unicode string).
            limit: Maximum number of continuations to return (default: 20).

        Returns:
            List of (word, probability) tuples, sorted by probability (descending).

        Performance:
            O(1) lookup + O(limit) slice using pre-built index

        Example:
            >>> bigrams = {
            ...     ("သူ", "သွား"): 0.234,
            ...     ("သူ", "သည်"): 0.189,
            ...     ("သူ", "ရှိ"): 0.156
            ... }
            >>> provider = MemoryProvider(bigrams=bigrams)
            >>> continuations = provider.get_top_continuations("သူ", limit=2)
            >>> for word, prob in continuations:
            ...     print(f"{word}: {prob:.3f}")
            သွား: 0.234
            သည်: 0.189

        Notes:
            - Returns empty list for unknown words
            - Results are sorted by probability (highest first)
        """
        if not prev_word:
            return []

        # Use pre-built index for O(1) lookup
        continuations = self._bigram_index.get(prev_word, [])
        return continuations[:limit]

    def get_all_syllables(self) -> Iterator[tuple[str, int]]:
        """Get iterator over all syllables."""
        return ((s, f) for s, f in self._syllables.items())

    def get_all_words(self) -> Iterator[tuple[str, int]]:
        """Get iterator over all words."""
        return ((w, f) for w, f in self._words.items())

    # Additional helper methods for dynamic data manipulation

    def add_syllable(self, syllable: str, frequency: int = 1) -> None:
        """
        Add or update a syllable in the dictionary.

        Args:
            syllable: Myanmar syllable to add.
            frequency: Frequency count (default: 1).

        Example:
            >>> provider = MemoryProvider()
            >>> provider.add_syllable("သူ", frequency=1000)
            >>> provider.is_valid_syllable("သူ")
            True
            >>> provider.get_syllable_frequency("သူ")
            1000

        Notes:
            - If syllable already exists, frequency is overwritten
            - Not thread-safe for concurrent writes
        """
        if not syllable:
            raise ValueError("Syllable cannot be empty")
        if frequency < 0:
            raise ValueError("Frequency must be non-negative")
        self._syllables[syllable] = frequency

    def add_word(self, word: str, frequency: int = 1) -> None:
        """
        Add or update a word in the dictionary.

        Args:
            word: Myanmar word to add.
            frequency: Frequency count (default: 1).

        Example:
            >>> provider = MemoryProvider()
            >>> provider.add_word("မြန်မာ", frequency=850)
            >>> provider.is_valid_word("မြန်မာ")
            True
            >>> provider.get_word_frequency("မြန်မာ")
            850

        Notes:
            - If word already exists, frequency is overwritten
            - Not thread-safe for concurrent writes
        """
        if not word:
            raise ValueError("Word cannot be empty")
        if frequency < 0:
            raise ValueError("Frequency must be non-negative")
        self._words[word] = frequency

    def add_word_pos(self, word: str, pos: str) -> None:
        """
        Add or update a POS tag for a word.

        Args:
            word: Myanmar word.
            pos: POS tag (e.g., 'N', 'V').
        """
        if not word:
            raise ValueError("Word cannot be empty")
        self._word_pos[word] = pos

    def add_bigram(self, prev_word: str, current_word: str, probability: float) -> None:
        """
        Add or update a bigram probability in the dictionary.

        Args:
            prev_word: Previous word in sequence.
            current_word: Current word in sequence.
            probability: Conditional probability P(current | prev) in range [0.0, 1.0].

        Example:
            >>> provider = MemoryProvider()
            >>> provider.add_bigram("သူ", "သွား", 0.234)
            >>> provider.get_bigram_probability("သူ", "သွား")
            0.234

        Notes:
            - If bigram already exists, probability is overwritten
            - Not thread-safe for concurrent writes

        Raises:
            ValueError: If probability is not in range [0.0, 1.0]
        """
        if not prev_word or not current_word:
            raise ValueError("Words cannot be empty")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be in range [0.0, 1.0]")

        is_update = (prev_word, current_word) in self._bigrams
        self._bigrams[(prev_word, current_word)] = probability

        # Update the index
        if prev_word not in self._bigram_index:
            self._bigram_index[prev_word] = []

        if is_update:
            # Remove old entry and add new one
            self._bigram_index[prev_word] = [
                (w, p) for w, p in self._bigram_index[prev_word] if w != current_word
            ]
        self._bigram_index[prev_word].append((current_word, probability))
        # Re-sort after adding
        self._bigram_index[prev_word].sort(key=lambda x: x[1], reverse=True)

    def add_trigram(self, word1: str, word2: str, word3: str, probability: float) -> None:
        """
        Add or update a trigram probability in the dictionary.

        Args:
            word1: First word in sequence.
            word2: Second word in sequence.
            word3: Third word in sequence (target).
            probability: Conditional probability P(word3 | word1, word2).

        Raises:
            ValueError: If any word is empty or probability is out of range.
        """
        if not word1 or not word2 or not word3:
            raise ValueError("Words cannot be empty")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be in range [0.0, 1.0]")
        self._trigrams[(word1, word2, word3)] = probability

    def add_fourgram(
        self, word1: str, word2: str, word3: str, word4: str, probability: float
    ) -> None:
        """
        Add or update a fourgram probability in the dictionary.

        Args:
            word1: First word in sequence.
            word2: Second word in sequence.
            word3: Third word in sequence.
            word4: Fourth word in sequence (target).
            probability: Conditional probability P(word4 | word1, word2, word3).

        Raises:
            ValueError: If any word is empty or probability is out of range.
        """
        if not word1 or not word2 or not word3 or not word4:
            raise ValueError("Words cannot be empty")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be in range [0.0, 1.0]")
        self._fourgrams[(word1, word2, word3, word4)] = probability

    def add_fivegram(
        self,
        word1: str,
        word2: str,
        word3: str,
        word4: str,
        word5: str,
        probability: float,
    ) -> None:
        """
        Add or update a fivegram probability in the dictionary.

        Args:
            word1: First word in sequence.
            word2: Second word in sequence.
            word3: Third word in sequence.
            word4: Fourth word in sequence.
            word5: Fifth word in sequence (target).
            probability: Conditional probability P(word5 | word1, word2, word3, word4).

        Raises:
            ValueError: If any word is empty or probability is out of range.
        """
        if not word1 or not word2 or not word3 or not word4 or not word5:
            raise ValueError("Words cannot be empty")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be in range [0.0, 1.0]")
        self._fivegrams[(word1, word2, word3, word4, word5)] = probability

    def remove_syllable(self, syllable: str) -> bool:
        """
        Remove a syllable from the dictionary.

        Args:
            syllable: Myanmar syllable to remove.

        Returns:
            True if syllable was removed, False if it didn't exist.

        Example:
            >>> provider = MemoryProvider(syllables={"သူ": 100})
            >>> provider.remove_syllable("သူ")
            True
            >>> provider.is_valid_syllable("သူ")
            False
            >>> provider.remove_syllable("သူ")  # Already removed
            False
        """
        if syllable in self._syllables:
            del self._syllables[syllable]
            return True
        return False

    def remove_word(self, word: str) -> bool:
        """
        Remove a word from the dictionary.

        Args:
            word: Myanmar word to remove.

        Returns:
            True if word was removed, False if it didn't exist.

        Example:
            >>> provider = MemoryProvider(words={"မြန်မာ": 50})
            >>> provider.remove_word("မြန်မာ")
            True
            >>> provider.is_valid_word("မြန်မာ")
            False
        """
        if word in self._words:
            del self._words[word]
            return True
        return False

    def remove_bigram(self, prev_word: str, current_word: str) -> bool:
        """
        Remove a bigram from the dictionary.

        Args:
            prev_word: Previous word in sequence.
            current_word: Current word in sequence.

        Returns:
            True if bigram was removed, False if it didn't exist.

        Example:
            >>> provider = MemoryProvider(bigrams={("သူ", "သွား"): 0.234})
            >>> provider.remove_bigram("သူ", "သွား")
            True
            >>> provider.get_bigram_probability("သူ", "သွား")
            0.0
        """
        key = (prev_word, current_word)
        if key in self._bigrams:
            del self._bigrams[key]
            # Update index
            if prev_word in self._bigram_index:
                self._bigram_index[prev_word] = [
                    (w, p) for w, p in self._bigram_index[prev_word] if w != current_word
                ]
                # Remove empty lists
                if not self._bigram_index[prev_word]:
                    del self._bigram_index[prev_word]
            return True
        return False

    def get_syllable_count(self) -> int:
        """
        Get total number of syllables in dictionary.

        Returns:
            Integer count of unique syllables.

        Example:
            >>> provider = MemoryProvider(syllables={"မြန်": 100, "မာ": 200})
            >>> provider.get_syllable_count()
            2
        """
        return len(self._syllables)

    def get_word_count(self) -> int:
        """
        Get total number of words in dictionary.

        Returns:
            Integer count of unique words.

        Example:
            >>> provider = MemoryProvider(words={"မြန်မာ": 50, "နိုင်ငံ": 30})
            >>> provider.get_word_count()
            2
        """
        return len(self._words)

    def get_bigram_count(self) -> int:
        """
        Get total number of bigrams in dictionary.

        Returns:
            Integer count of unique bigrams.

        Example:
            >>> provider = MemoryProvider(bigrams={("သူ", "သွား"): 0.2})
            >>> provider.get_bigram_count()
            1
        """
        return len(self._bigrams)

    def clear(self) -> None:
        """
        Remove all data from the provider.

        Example:
            >>> provider = MemoryProvider(syllables={"သူ": 100})
            >>> provider.clear()
            >>> provider.get_syllable_count()
            0
        """
        self._syllables.clear()
        self._words.clear()
        self._bigrams.clear()
        self._trigrams.clear()
        self._fourgrams.clear()
        self._fivegrams.clear()
        self._word_pos.clear()
        self._bigram_index.clear()  # Clear index

    def get_pos_unigram_probabilities(self) -> dict[str, float]:
        """
        Get POS tag unigram probabilities.

        Returns:
            Empty dict (MemoryProvider does not store POS probability tables).
        """
        return {}

    def get_pos_bigram_probabilities(self) -> dict[tuple[str, str], float]:
        """
        Get POS tag bigram probabilities.

        Returns:
            Empty dict (MemoryProvider does not store POS probability tables).
        """
        return {}

    def get_pos_trigram_probabilities(self) -> dict[tuple[str, str, str], float]:
        """
        Get POS tag trigram probabilities.

        Returns:
            Empty dict (MemoryProvider doesn't support POS tagging by default).
        """
        return {}

    def load_from_lists(
        self,
        syllable_list: list[tuple[str, int]] | None = None,
        word_list: list[tuple[str, int]] | None = None,
        bigram_list: list[tuple[str, str, float]] | None = None,
    ) -> None:
        """
        Bulk load data from lists (useful for initialization from files).

        Args:
            syllable_list: List of (syllable, frequency) tuples.
            word_list: List of (word, frequency) tuples.
            bigram_list: List of (prev_word, curr_word, probability) tuples.

        Example:
            >>> provider = MemoryProvider()
            >>> syllables = [("မြန်", 1500), ("မာ", 2300)]
            >>> words = [("မြန်မာ", 850)]
            >>> bigrams = [("သူ", "သွား", 0.234)]
            >>> provider.load_from_lists(syllables, words, bigrams)
            >>> provider.get_syllable_count()
            2
            >>> provider.get_word_count()
            1

        Notes:
            - This method extends existing data (doesn't clear first)
            - Duplicate entries will overwrite previous values
        """
        if syllable_list:
            for syllable, frequency in syllable_list:
                self.add_syllable(syllable, frequency)

        if word_list:
            for word, frequency in word_list:
                self.add_word(word, frequency)

        if bigram_list:
            # Bulk load bigrams directly to avoid per-item index updates
            for prev_word, curr_word, probability in bigram_list:
                if not prev_word or not curr_word:
                    raise ValueError("Words cannot be empty")
                if not 0.0 <= probability <= 1.0:
                    raise ValueError("Probability must be in range [0.0, 1.0]")
                self._bigrams[(prev_word, curr_word)] = probability
            # Rebuild index once after bulk load
            self._rebuild_bigram_index()

    # ==========================================================================
    # Enrichment table queries
    # ==========================================================================

    def add_confusable_pair(
        self,
        word1: str,
        word2: str,
        confusion_type: str,
        context_overlap: float = 0.0,
        freq_ratio: float = 0.0,
        suppress: int = 0,
    ) -> None:
        """Add a confusable pair (indexes by both words)."""
        entry = (word2, confusion_type, context_overlap, freq_ratio, suppress)
        self._confusable_pairs.setdefault(word1, []).append(entry)
        entry_rev = (word1, confusion_type, context_overlap, freq_ratio, suppress)
        self._confusable_pairs.setdefault(word2, []).append(entry_rev)

    def get_confusable_pairs(self, word: str) -> list[tuple[str, str, float, float, int]]:
        """Get confusable pairs for a word.

        Returns list of (variant, confusion_type, context_overlap, freq_ratio, suppress) tuples.
        """
        return self._confusable_pairs.get(word, [])

    def is_confusable_suppressed(self, word1: str, word2: str) -> bool:
        """Check if a confusable pair is suppressed."""
        for variant, _ctype, _overlap, _fratio, suppress in self._confusable_pairs.get(word1, []):
            if variant == word2 and suppress:
                return True
        return False

    def get_confusable_context_overlap(self, word1: str, word2: str) -> float | None:
        """Get context overlap score for a confusable pair."""
        for variant, _ctype, overlap, _fratio, _suppress in self._confusable_pairs.get(word1, []):
            if variant == word2:
                return overlap
        return None

    def add_compound_confusion(
        self,
        compound: str,
        part1: str,
        part2: str,
        compound_freq: int = 0,
        split_freq: int = 0,
        pmi: float = 0.0,
    ) -> None:
        """Add a compound confusion entry."""
        self._compound_confusions[compound] = (part1, part2, compound_freq, split_freq, pmi)

    def get_compound_confusion(self, compound: str) -> tuple[str, str, int, int, float] | None:
        """Get compound confusion data."""
        return self._compound_confusions.get(compound)

    def add_collocation(self, word1: str, word2: str, pmi: float) -> None:
        """Add a collocation with PMI score."""
        self._collocations[(word1, word2)] = pmi

    def get_collocation_pmi(self, word1: str, word2: str) -> float | None:
        """Get PMI score for a word pair."""
        return self._collocations.get((word1, word2))

    def add_register_tag(self, word: str, register: str) -> None:
        """Add a register tag for a word."""
        self._register_tags[word] = register

    def get_register_tag(self, word: str) -> str | None:
        """Get register tag for a word."""
        return self._register_tags.get(word)
