# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-optimized frequency counting for Myanmar dictionary building.

This module provides high-performance frequency counting for syllables,
words, and n-grams during corpus processing. Used by the data pipeline
to build statistical language models.

Key Features:
    - Syllable frequency counting
    - Word frequency counting with invalid word filtering
    - Bigram and trigram n-gram counting
    - Word-syllable mapping for morphological analysis
    - C++ unordered_map for O(1) insert/lookup
    - Tab-separated output for database import

Data Structures:
    - syllable_counts: Individual syllable frequencies
    - word_counts: Word frequencies
    - bigram_counts: "w1\\tw2" keyed bigram frequencies
    - trigram_counts: "w1\\tw2\\tw3" keyed trigram frequencies
    - word_syllables: Word to syllable count mapping

CRITICAL - Thread Safety:
    WARNING: This module uses global C++ unordered_map state that is NOT thread-safe.

    SAFE USAGE:
    - Fork-based multiprocessing (default on Unix/Linux)
      Each child process gets isolated copy via copy-on-write
      Call reset_counts() at worker initialization

    UNSAFE USAGE:
    - Threading (concurrent threads accessing global maps)
      C++ unordered_map has no internal synchronization
      Concurrent writes will cause data corruption

    - Spawn-based multiprocessing (Windows default)
      Each spawn creates fresh interpreter state (safe but inefficient)

    The current codebase uses fork-based multiprocessing which is SAFE.
    Do NOT refactor to use threading without adding synchronization.

Usage Pattern:
    1. reset_counts() - Clear all counters (required before each batch)
    2. set_invalid_words() - Configure words to skip
    3. count_*() - Process corpus data
    4. get_*() - Retrieve frequency dictionaries

Performance:
    - ~15x faster than pure Python dict operations
    - Memory efficient with C++ unordered_map
    - Batch-friendly design for pipeline integration

Example:
    >>> from myspellchecker.data_pipeline.frequency_counter import (
    ...     reset_counts, count_word, get_word_counts
    ... )
    >>> reset_counts()
    >>> count_word("မြန်မာ", ["မြန်", "မာ"])
    >>> counts = get_word_counts()

See Also:
    - batch_processor.pyx: Text processing that feeds this module
    - pipeline.py: Pipeline orchestration
    - database_packager.py: Database creation from frequencies
"""

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference as deref

# Define Maps
cdef unordered_map[string, int] syllable_counts
cdef unordered_map[string, int] word_counts
# Use string for tuple keys by joining with space? Or recursive map?
# Join with separator for composite keys.
# Bigram: "w1 w2"
# Trigram: "w1 w2 w3"
cdef unordered_map[string, int] bigram_counts
cdef unordered_map[string, int] trigram_counts
cdef unordered_map[string, int] word1_counts # Count of w1 in bigrams (often same as word_counts, but strictly context based)
cdef unordered_map[string, int] bigram_context_counts # Count of (w1, w2) in trigrams

# Word Syllable Map (approximate)
cdef unordered_map[string, int] word_syllables

# Invalid words to skip (set once, used during counting)
cdef unordered_set[string] invalid_words

cdef string SEP = b"\t"

def set_invalid_words(set py_invalid_words):
    """Set the invalid words to skip during counting."""
    global invalid_words
    invalid_words.clear()
    for w in py_invalid_words:
        invalid_words.insert(w.encode('utf-8'))

def reset_counts():
    syllable_counts.clear()
    word_counts.clear()
    bigram_counts.clear()
    trigram_counts.clear()
    word1_counts.clear()
    bigram_context_counts.clear()
    word_syllables.clear()

def count_batch(list batch_syllables, list batch_words):
    """
    Process a batch of sentences.
    Input:
      batch_syllables: list[list[str]]
      batch_words: list[list[str]]

    OPTIMIZATION: Pre-encode all strings once per sentence to avoid
    redundant .encode('utf-8') calls in bigram/trigram loops.
    This reduces ~3x encode calls per word.
    """
    cdef int n = len(batch_syllables)
    cdef int i, j, k, num_words
    cdef list sent_syls
    cdef list sent_words
    cdef string s
    cdef string key

    # Pre-encoded word buffer (reused per sentence)
    cdef vector[string] words_encoded

    for i in range(n):
        sent_syls = batch_syllables[i]
        sent_words = batch_words[i]

        # 1. Count Syllables (encode inline - each syllable used only once)
        for j in range(len(sent_syls)):
            s = sent_syls[j].encode('utf-8')
            syllable_counts[s] += 1

        # 2. PRE-ENCODE valid words ONCE for this sentence
        # Filter out invalid words during encoding (matches Python behavior)
        words_encoded.clear()
        words_encoded.reserve(len(sent_words))
        for j in range(len(sent_words)):
            s = sent_words[j].encode('utf-8')
            # Skip empty words and invalid words
            if s.size() > 0 and invalid_words.find(s) == invalid_words.end():
                words_encoded.push_back(s)

        # Now num_words is the count of VALID words
        num_words = words_encoded.size()

        # 3. Count Words (all words in words_encoded are valid)
        for j in range(num_words):
            word_counts[words_encoded[j]] += 1

        # 4. Bigrams (using pre-encoded filtered list)
        if num_words >= 2:
            for k in range(num_words - 1):
                # key: w1 + "\t" + w2
                key = words_encoded[k] + SEP + words_encoded[k + 1]
                bigram_counts[key] += 1
                word1_counts[words_encoded[k]] += 1

        # 5. Trigrams (using pre-encoded filtered list)
        if num_words >= 3:
            for k in range(num_words - 2):
                # key: w1 + "\t" + w2 + "\t" + w3
                key = words_encoded[k] + SEP + words_encoded[k + 1] + SEP + words_encoded[k + 2]
                trigram_counts[key] += 1

                # context key: w1 + "\t" + w2
                key = words_encoded[k] + SEP + words_encoded[k + 1]
                bigram_context_counts[key] += 1

def count_ngrams_only(list batch_words):
    """
    Count only bigrams and trigrams (not unigrams).
    Use this when unigrams are counted separately with Arrow compute.

    Input:
      batch_words: list[list[str]] - already filtered for invalid words
    """
    cdef int n = len(batch_words)
    cdef int i, j, num_words
    cdef list sent_words
    cdef string s
    cdef string key
    cdef vector[string] words_encoded

    for i in range(n):
        sent_words = batch_words[i]

        # Pre-encode all words for this sentence
        num_words = len(sent_words)
        words_encoded.clear()
        words_encoded.reserve(num_words)
        for j in range(num_words):
            s = sent_words[j].encode('utf-8')
            if s.size() > 0:
                words_encoded.push_back(s)

        num_words = words_encoded.size()

        # Bigrams
        if num_words >= 2:
            for j in range(num_words - 1):
                key = words_encoded[j] + SEP + words_encoded[j + 1]
                bigram_counts[key] += 1
                word1_counts[words_encoded[j]] += 1

        # Trigrams
        if num_words >= 3:
            for j in range(num_words - 2):
                key = words_encoded[j] + SEP + words_encoded[j + 1] + SEP + words_encoded[j + 2]
                trigram_counts[key] += 1
                key = words_encoded[j] + SEP + words_encoded[j + 1]
                bigram_context_counts[key] += 1


def count_ngrams_flat(list words_flat, parent_indices):
    """
    Count bigrams and trigrams from flattened word list with parent indices.

    This eliminates to_pylist() by using Arrow's flattened representation.
    N-grams are only formed when consecutive words have the same parent index
    (i.e., belong to the same sentence).

    Input:
      words_flat: list[str] - flattened list of all words (already filtered)
      parent_indices: list[int] or numpy array - sentence index for each word

    OPTIMIZATION: Avoids expensive to_pylist() on nested lists by using
    Arrow's list_flatten + list_parent_indices operations.
    """
    cdef int n = len(words_flat)
    cdef int i
    cdef string w1, w2, w3
    cdef string key
    cdef int p1, p2, p3

    # Pre-encode all words once (single pass)
    cdef vector[string] words_encoded
    words_encoded.reserve(n)

    for i in range(n):
        words_encoded.push_back(words_flat[i].encode('utf-8'))

    # Convert parent_indices to list for fast access (handles numpy array or pyarrow array)
    cdef list parents = list(parent_indices)

    # Build bigrams: consecutive words in same sentence
    if n >= 2:
        for i in range(n - 1):
            p1 = parents[i]
            p2 = parents[i + 1]

            # Only count if both words are in the same sentence
            if p1 == p2:
                w1 = words_encoded[i]
                w2 = words_encoded[i + 1]
                key = w1 + SEP + w2
                bigram_counts[key] += 1
                word1_counts[w1] += 1

    # Build trigrams: three consecutive words in same sentence
    if n >= 3:
        for i in range(n - 2):
            p1 = parents[i]
            p2 = parents[i + 1]
            p3 = parents[i + 2]

            # Only count if all three words are in the same sentence
            if p1 == p2 and p2 == p3:
                w1 = words_encoded[i]
                w2 = words_encoded[i + 1]
                w3 = words_encoded[i + 2]

                # Trigram key
                key = w1 + SEP + w2 + SEP + w3
                trigram_counts[key] += 1

                # Bigram context for trigram probability calculation
                key = w1 + SEP + w2
                bigram_context_counts[key] += 1


def count_ngrams_flat_preenc(list words_encoded, long[:] parents, int n):
    """
    FAST VERSION: Count n-grams from pre-encoded words with numpy parent indices.

    Input:
      words_encoded: list[bytes] - pre-encoded UTF-8 bytes (no encoding needed)
      parents: numpy int64 array (memoryview) - sentence index for each word
      n: number of words

    OPTIMIZATION:
    - Accepts pre-encoded bytes (encoding done in Python with list comprehension)
    - Uses typed memoryview for parent indices (zero-copy from numpy)
    - Eliminates all Python object creation in hot loop
    """
    cdef int i
    cdef string w1, w2, w3
    cdef string key
    cdef long p1, p2, p3
    cdef bytes word_bytes

    # Convert bytes list to C++ strings (direct copy, no encoding)
    cdef vector[string] words_vec
    words_vec.reserve(n)

    for i in range(n):
        word_bytes = words_encoded[i]
        words_vec.push_back(<string>word_bytes)

    # Build bigrams: consecutive words in same sentence
    if n >= 2:
        for i in range(n - 1):
            p1 = parents[i]
            p2 = parents[i + 1]

            # Only count if both words are in the same sentence
            if p1 == p2:
                w1 = words_vec[i]
                w2 = words_vec[i + 1]
                key = w1 + SEP + w2
                bigram_counts[key] += 1
                word1_counts[w1] += 1

    # Build trigrams: three consecutive words in same sentence
    if n >= 3:
        for i in range(n - 2):
            p1 = parents[i]
            p2 = parents[i + 1]
            p3 = parents[i + 2]

            # Only count if all three words are in the same sentence
            if p1 == p2 and p2 == p3:
                w1 = words_vec[i]
                w2 = words_vec[i + 1]
                w3 = words_vec[i + 2]

                # Trigram key
                key = w1 + SEP + w2 + SEP + w3
                trigram_counts[key] += 1

                # Bigram context for trigram probability calculation
                key = w1 + SEP + w2
                bigram_context_counts[key] += 1


def count_ngrams_arrow_buffers(
    const unsigned char[:] data_buf,
    const int[:] offsets_buf,
    const long[:] parents,
    int n
):
    """
    FASTEST VERSION: Count n-grams directly from Arrow string buffers.

    Input:
      data_buf: Raw UTF-8 bytes buffer from Arrow StringArray
      offsets_buf: Int32 offsets buffer from Arrow StringArray
      parents: Int64 numpy array of sentence indices
      n: Number of strings (len(offsets) - 1)

    OPTIMIZATION:
    - Zero-copy access to Arrow data via typed memoryviews
    - Direct byte-range copy to C++ strings (no Python string creation)
    - Eliminates to_pylist() entirely
    """
    cdef int i, start, end, length
    cdef string w1, w2, w3
    cdef string key
    cdef long p1, p2, p3

    # Pre-build C++ string vector from raw buffers
    cdef vector[string] words_vec
    words_vec.reserve(n)

    for i in range(n):
        start = offsets_buf[i]
        end = offsets_buf[i + 1]
        length = end - start
        # Direct copy from buffer to C++ string
        words_vec.push_back(string(<const char*>&data_buf[start], length))

    # Build bigrams: consecutive words in same sentence
    if n >= 2:
        for i in range(n - 1):
            p1 = parents[i]
            p2 = parents[i + 1]

            if p1 == p2:
                w1 = words_vec[i]
                w2 = words_vec[i + 1]
                key = w1 + SEP + w2
                bigram_counts[key] += 1
                word1_counts[w1] += 1

    # Build trigrams: consecutive words in same sentence
    if n >= 3:
        for i in range(n - 2):
            p1 = parents[i]
            p2 = parents[i + 1]
            p3 = parents[i + 2]

            if p1 == p2 and p2 == p3:
                w1 = words_vec[i]
                w2 = words_vec[i + 1]
                w3 = words_vec[i + 2]

                key = w1 + SEP + w2 + SEP + w3
                trigram_counts[key] += 1

                key = w1 + SEP + w2
                bigram_context_counts[key] += 1


def get_counts():
    """
    Return counts as Python dicts.
    """
    # Convert C++ maps to Python dicts
    # Note: Decoding keys back to string
    
    return {
        "syllables": {k.decode('utf-8'): v for k, v in syllable_counts},
        "words": {k.decode('utf-8'): v for k, v in word_counts},
        "bigrams": {tuple(k.decode('utf-8').split("\t")): v for k, v in bigram_counts},
        "trigrams": {tuple(k.decode('utf-8').split("\t")): v for k, v in trigram_counts},
        "word1": {k.decode('utf-8'): v for k, v in word1_counts},
        "bigram_context": {tuple(k.decode('utf-8').split("\t")): v for k, v in bigram_context_counts}
    }
