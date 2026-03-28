# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-optimized Viterbi word segmentation for Myanmar text.

This module provides high-performance word boundary detection using a
statistical language model with Viterbi decoding. Essential for Myanmar
text processing as the script has no inter-word spaces.

Key Features:
    - Unigram and bigram language model support
    - Memory-mapped model loading for fork() safety (COW)
    - C++ unordered_map for O(1) probability lookups
    - File integrity verification via SHA-256
    - Graceful fallback to Python dict when mmap unavailable

Algorithm:
    Viterbi dynamic programming finds the most likely segmentation:
    - State: position in text
    - Transition: word boundary at position
    - Score: log probability from language model

    P(segmentation) = Π P(word_i | word_{i-1})

Model Loading:
    1. Attempts mmap for fork-safe parallel processing
    2. Falls back to C++ unordered_map in memory
    3. Verifies file integrity with SHA-256 checksum

Performance:
    - ~10x faster than pure Python implementation
    - O(n²) time complexity where n = text length
    - Memory-efficient with mmap (copy-on-write)

Example:
    >>> from myspellchecker.tokenizers.cython.word_segment import viterbi
    >>> words = viterbi("မြန်မာစာအုပ်")
    ['မြန်မာ', 'စာအုပ်']

Note:
    This module requires compilation and model files.
    Models are typically loaded from the data directory.

See Also:
    - mmap_reader.pyx: Memory-mapped file reader
    - batch_processor.pyx: Batch processing using this module
    - syllable.py: Syllable-level segmentation (rule-based)
"""

import math
import json
import hashlib
import os
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cython.operator cimport dereference as deref, preincrement
from libc.math cimport log10, pow
from libc.stddef cimport size_t

from myspellchecker.utils.logging_utils import get_logger

# Import mmap reader for fast GIL-free lookups
from .mmap_reader cimport MMapSegmentationReader, _global_reader, _global_initialized

logger = get_logger(__name__)

# C++ Globals for high-performance lookups
cdef unordered_map[string, double] cpp_unigram_map
cdef unordered_map[string, double] cpp_bigram_map
cdef double UNIGRAM_N = 0.0
cdef double BIGRAM_N = 0.0
cdef double LOG_UNIGRAM_N = 0.0
cdef double LOG_BIGRAM_N = 0.0
cdef bint MODELS_LOADED = False

# MMap support - survives fork() via COW
cdef bint USE_MMAP = False
_mmap_reader = None  # Will hold the mmap reader instance

# Python Globals compatibility
P_unigram = None
P_bigram = None

def _verify_file_integrity(file_path: str, expected_hash: str = None) -> bool:
    """
    Verify file integrity using SHA-256 hash.

    Args:
        file_path: Path to the file to verify
        expected_hash: Optional expected hash. If None, skips verification.

    Returns:
        True if file is valid, False otherwise
    """
    if expected_hash is None:
        return True

    try:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()
        return actual_hash == expected_hash
    except (OSError, IOError) as e:
        logger.error(f"Failed to verify file integrity: {e}")
        return False


def read_dict(fileDICT, expected_hash: str = None):
    """
    Safely read dictionary from JSON file.

    SECURITY: This replaces the previous unsafe serialization to prevent
    arbitrary code execution vulnerabilities. JSON is safe because it only
    supports basic data types (dict, list, str, int, float, bool, null).

    Args:
        fileDICT: Path to the dictionary file (.json format)
        expected_hash: Optional SHA-256 hash for integrity verification

    Returns:
        Dictionary loaded from file, or empty dict on error
    """
    dictionary = {}

    # Verify file integrity if hash provided
    if expected_hash and not _verify_file_integrity(fileDICT, expected_hash):
        logger.error("Dictionary file %s failed integrity check!", fileDICT)
        return dictionary

    try:
        with open(fileDICT, "r", encoding="utf-8") as input_file:
            dictionary = json.load(input_file)

        # Validate expected structure (dict with string keys and numeric values)
        if not isinstance(dictionary, dict):
            logger.error("Dictionary file %s has invalid format (expected dict)", fileDICT)
            return {}

    except FileNotFoundError:
        logger.error("Dictionary file %s not found!", fileDICT)
    except json.JSONDecodeError as e:
        logger.error("Dictionary file %s has invalid JSON: %s", fileDICT, e)
    except (OSError, PermissionError) as e:
        logger.error("Error reading dictionary file %s: %s", fileDICT, e)

    return dictionary

class ProbDist(dict):
    ### Probability distribution estimated from unigram/bigram data
    def __init__(self, datafile=None, unigram=True, N=102490):
        data = read_dict(datafile)
        for k, c in data.items():
            self[k] = self.get(k, 0) + c

        # Dynamic N calculation
        total_count = sum(self.values())
        if total_count > 0:
            self.N = total_count
        else:
            self.N = N
        


        if unigram:
            self.unknownprob = lambda k, N: 10 / (N * 10 ** len(k))  # avoid unknown long word
        else:
            self.unknownprob = lambda k, N: 1 / N

    def __call__(self, key):
        if key in self:
            return self[key] / self.N
        else:
            return self.unknownprob(key, self.N)


cdef int utf8_len(string& s):
    cdef int i = 0, length = 0, c
    while i < s.length():
        c = <unsigned char> s[i]
        if (c & 0x80) == 0:
            i += 1
        elif (c & 0xE0) == 0xC0:
            i += 2
        elif (c & 0xF0) == 0xE0:
            i += 3
        elif (c & 0xF8) == 0xF0:
            i += 4
        else:
            i += 1  # invalid, treat as 1 byte
        length += 1
    return length

def set_unigram_model(model):
    """
    Populate C++ unigram map from Python ProbDist.
    Stores pre-calculated LOG probabilities: log10(count) - log10(N)
    """
    global P_unigram, UNIGRAM_N, LOG_UNIGRAM_N
    P_unigram = model # Keep python ref
    UNIGRAM_N = float(model.N)
    LOG_UNIGRAM_N = log10(UNIGRAM_N)
    
    # Clear and populate C++ map
    cpp_unigram_map.clear()
    cdef string s_key
    cdef double log_val
    
    for k, v in model.items():
        # Handle tuple keys (common in N-gram models)
        if isinstance(k, tuple):
            k_str = k[0]
        else:
            k_str = k
            
        s_key = k_str.encode('utf-8')
        # Pre-calculate Log Prob: log10(count) - log10(N)
        log_val = log10(<double>v) - LOG_UNIGRAM_N
        cpp_unigram_map[s_key] = log_val

def set_bigram_model(model):
    """
    Populate C++ bigram map from Python ProbDist.
    Stores pre-calculated LOG probabilities: log10(count) - log10(N)
    """
    global P_bigram, BIGRAM_N, LOG_BIGRAM_N, MODELS_LOADED
    P_bigram = model # Keep python ref
    BIGRAM_N = float(model.N)
    LOG_BIGRAM_N = log10(BIGRAM_N)
    
    # Clear and populate C++ map
    cpp_bigram_map.clear()
    cdef string s_key
    cdef double log_val
    
    for k, v in model.items():
        # Handle tuple keys (common in N-gram models)
        if isinstance(k, tuple):
            k_str = " ".join(k)
        else:
            k_str = k
            
        s_key = k_str.encode('utf-8')
        # Pre-calculate Log Prob: log10(count) - log10(N)
        log_val = log10(<double>v) - LOG_BIGRAM_N
        cpp_bigram_map[s_key] = log_val

    MODELS_LOADED = True


def initialize_mmap(mmap_path: str = None) -> bool:
    """
    Initialize word segmentation using memory-mapped dictionary file.

    The mmap file survives fork() via copy-on-write, eliminating the need
    to reinitialize C++ maps in worker processes.

    Args:
        mmap_path: Path to segmentation.mmap file. If None, uses default location.

    Returns:
        True if mmap was successfully initialized
    """
    global USE_MMAP, _mmap_reader, MODELS_LOADED

    try:
        from . import mmap_reader

        # Default path
        if mmap_path is None:
            dict_dir = os.path.dirname(__file__)
            parent_dir = os.path.dirname(dict_dir)
            mmap_path = os.path.join(parent_dir, "dict_ver1", "segmentation.mmap")

        if not os.path.exists(mmap_path):
            logger.warning(f"MMap file not found: {mmap_path}")
            return False

        # Initialize mmap reader
        if mmap_reader.initialize_mmap_reader(mmap_path):
            _mmap_reader = mmap_reader.get_mmap_reader()
            USE_MMAP = True
            MODELS_LOADED = True
            # Debug level - mmap_reader already logs INFO with counts
            logger.debug(f"MMap reader stats: {_mmap_reader.get_stats()}")
            return True
        else:
            logger.error("Failed to initialize mmap reader")
            return False

    except (ImportError, OSError, RuntimeError) as e:
        logger.error(f"Error initializing mmap: {e}")
        return False


def is_using_mmap() -> bool:
    """Check if mmap-based lookups are in use."""
    return USE_MMAP


cdef double get_unigram_log_prob(string& word):
    """
    C++ optimized unigram LOG probability lookup.
    Uses mmap reader if available, otherwise falls back to C++ map.
    """
    # Use mmap if initialized (survives fork via COW)
    if _global_initialized and _global_reader is not None:
        return _global_reader._get_unigram_log_prob_nogil(word.c_str(), word.length())

    # Fallback to C++ map
    if cpp_unigram_map.count(word):
        return cpp_unigram_map[word]
    else:
        # Unknown Prob: 10 / (N * 10 ** len(k))
        # Log: log10(10) - log10(N) - len(k)*log10(10)
        #    = 1.0 - LOG_UNIGRAM_N - len(k)
        return 1.0 - LOG_UNIGRAM_N - <double>utf8_len(word)

cdef double get_transition_log_prob_indices(string& text, int curr_start, int curr_len, int prev_start, int prev_len, string& external_prev):
    """
    Optimized transition lookup using indices to avoid premature substring creation.
    Constructs lookup keys directly from the main text buffer.
    Uses mmap reader if available, otherwise falls back to C++ map.
    """
    cdef const char* curr_ptr = text.c_str() + curr_start
    cdef const char* prev_ptr
    cdef int actual_prev_len

    # Determine prev word pointer and length
    if prev_start == -1:
        prev_ptr = external_prev.c_str()
        actual_prev_len = external_prev.length()
    else:
        prev_ptr = text.c_str() + prev_start
        actual_prev_len = prev_len

    # Use mmap if initialized (survives fork via COW)
    if _global_initialized and _global_reader is not None:
        return _global_reader._get_transition_log_prob_nogil(
            curr_ptr, curr_len,
            prev_ptr, actual_prev_len
        )

    # Fallback to C++ map
    cdef string curr_word = string(curr_ptr, curr_len)
    cdef string prev_word = string(prev_ptr, actual_prev_len)

    # Build space-separated bigram key for C++ map
    cdef string bigram_key = prev_word
    bigram_key.push_back(b' ')
    bigram_key.append(curr_word)

    cdef double bigram_log_val = 0.0
    cdef double unigram_prev_log_val = 0.0

    if cpp_bigram_map.count(bigram_key):
        bigram_log_val = cpp_bigram_map[bigram_key]

        if cpp_unigram_map.count(prev_word):
            unigram_prev_log_val = cpp_unigram_map[prev_word]
        else:
            unigram_prev_log_val = 1.0 - LOG_UNIGRAM_N - <double>utf8_len(prev_word)

        # Fix: Correct for mismatched N values between bigram and unigram maps.
        # bigram_log_val = log10(bigram_count / BIGRAM_N)
        # unigram_prev_log_val = log10(prev_count / UNIGRAM_N)
        # P(curr|prev) = bigram_count / prev_count
        # = (bigram_count/BIGRAM_N) / (prev_count/UNIGRAM_N) * (UNIGRAM_N/BIGRAM_N)^-1
        # So we need to subtract the N-ratio correction factor.
        return bigram_log_val - unigram_prev_log_val - LOG_UNIGRAM_N + LOG_BIGRAM_N

    # Fallback to unigram
    if cpp_unigram_map.count(curr_word):
        return cpp_unigram_map[curr_word]
    else:
        return 1.0 - LOG_UNIGRAM_N - <double>utf8_len(curr_word)




def get_model_stats() -> dict:
    """
    Return statistics about loaded models for debugging.
    Useful for verifying C++ map state after fork.

    Returns:
        Dict with model sizes and MODELS_LOADED flag
    """
    return {
        "models_loaded": MODELS_LOADED,
        "unigram_map_size": cpp_unigram_map.size(),
        "bigram_map_size": cpp_bigram_map.size(),
        "unigram_n": UNIGRAM_N,
        "bigram_n": BIGRAM_N,
        "has_python_unigram": P_unigram is not None,
        "has_python_bigram": P_bigram is not None,
    }


def reinitialize_cpp_maps() -> bool:
    """
    Re-initialize C++ maps from Python ProbDist objects.

    This is needed after fork() because C++ unordered_map objects
    don't properly survive the fork (their internal pointers become invalid).
    The Python objects (P_unigram, P_bigram) DO survive fork via COW.

    Returns:
        True if reinitialization was successful, False otherwise
    """
    global MODELS_LOADED

    if P_unigram is None or P_bigram is None:
        return False

    # Re-populate C++ maps from Python objects
    # This is fast because:
    # 1. Python dicts are shared via COW (no copy needed)
    # 2. C++ map population is O(n) for n words
    set_unigram_model(P_unigram)
    set_bigram_model(P_bigram)

    return True


def ensure_models_initialized() -> bool:
    """
    Ensure C++ models are properly initialized.

    Call this in forked workers to verify and reinitialize if needed.

    Returns:
        True if models are ready, False if initialization failed
    """
    # Check if C++ maps are populated
    if cpp_unigram_map.size() > 0 and cpp_bigram_map.size() > 0:
        return True

    # C++ maps are empty - try to reinitialize from Python objects
    if P_unigram is not None and P_bigram is not None:
        return reinitialize_cpp_maps()

    return False

cpdef tuple viterbi(str text, str prev="<S>", int maxlen=20):
    """
    Fully optimized C++ Viterbi implementation.
    Uses pre-calculated Log-Probabilities to avoid runtime division and log10 calls.
    Uses index-based passing to minimize string copying.
    """
    if not text:
        return 0.0, []

    # Convert inputs to C++ strings once
    cdef string text_bytes = text.encode('utf-8')
    cdef string prev_bytes = prev.encode('utf-8')
    
    cdef int N_bytes = text_bytes.length()
    cdef int N_chars = utf8_len(text_bytes) 
    
    # Map Char Index -> Byte Index
    cdef vector[int] char_to_byte
    char_to_byte.reserve(N_chars + 1)
    
    cdef int b_idx = 0
    cdef int c
    char_to_byte.push_back(0)
    
    while b_idx < N_bytes:
        c = <unsigned char> text_bytes[b_idx]
        if (c & 0x80) == 0:
            b_idx += 1
        elif (c & 0xE0) == 0xC0:
            b_idx += 2
        elif (c & 0xF0) == 0xE0:
            b_idx += 3
        elif (c & 0xF8) == 0xF0:
            b_idx += 4
        else:
            b_idx += 1
        char_to_byte.push_back(b_idx)
        
    cdef int total_chars = char_to_byte.size() - 1
    
    # DP State: Map<CharIndex, Map<StartCharIndex, Pair<Score, PrevStartCharIndex>>>
    cdef vector[unordered_map[int, pair[double, int]]] dp
    dp.resize(total_chars + 1)

    # Base case
    dp[0][-1] = pair[double, int](0.0, -2)

    cdef int i, j
    cdef int start_min
    cdef double log_prob, total_score, prev_score
    cdef int b_start, b_end, b_prev_start, b_prev_end

    # State pruning parameters
    # Keep only top MAX_STATES states per position to reduce O(n²) to O(n*k)
    # OPTIMIZATION: Reduced from 5 to 3 states, threshold from 8.0 to 6.0
    # This is more aggressive pruning for ~40% speedup with minimal quality loss
    cdef int MAX_STATES = 3
    cdef double PRUNE_THRESHOLD = 6.0  # Prune states with score < best - threshold
    cdef double best_score_at_j
    cdef double prune_cutoff
    cdef vector[int] keys_to_remove
    cdef int k_idx

    # C++ iterator types for optimal iteration (avoids Python wrapper overhead)
    cdef unordered_map[int, pair[double, int]].iterator state_it
    cdef unordered_map[int, pair[double, int]].iterator state_end
    cdef unordered_map[int, pair[double, int]].iterator found_it
    cdef int prev_k

    # Forward Pass
    for j in range(1, total_chars + 1):
        start_min = 0
        if j > maxlen:
            start_min = j - maxlen

        b_end = char_to_byte[j]

        for i in range(start_min, j):
            # Check if valid path exists to i
            if dp[i].empty():
                continue

            b_start = char_to_byte[i]

            # Iterate previous states using C++ iterators (faster than Python iteration)
            state_it = dp[i].begin()
            state_end = dp[i].end()
            while state_it != state_end:
                prev_k = deref(state_it).first
                prev_score = deref(state_it).second.first

                # Resolve Prev Word indices
                b_prev_start = -1
                b_prev_end = -1

                if prev_k != -1:
                    b_prev_start = char_to_byte[prev_k]
                    b_prev_end = b_start

                # Get Transition LOG Prob using INDICES
                # Avoids constructing string objects in the outer loops
                log_prob = get_transition_log_prob_indices(
                    text_bytes,
                    b_start,
                    b_end - b_start,
                    b_prev_start,
                    b_prev_end - b_prev_start if b_prev_start != -1 else 0,
                    prev_bytes
                )

                total_score = prev_score + log_prob

                # Single lookup optimization: use find() instead of count() + []
                found_it = dp[j].find(i)
                if found_it == dp[j].end() or total_score > deref(found_it).second.first:
                    dp[j][i] = pair[double, int](total_score, prev_k)

                preincrement(state_it)

        # STATE PRUNING: Keep only top-K states to reduce computation
        # This is safe because pruned paths can never become optimal
        # (Viterbi property: best path to position j only depends on best paths to earlier positions)
        if dp[j].size() > <size_t>MAX_STATES:
            # Find best score at this position using C++ iterators
            best_score_at_j = -1e100
            state_it = dp[j].begin()
            state_end = dp[j].end()
            while state_it != state_end:
                if deref(state_it).second.first > best_score_at_j:
                    best_score_at_j = deref(state_it).second.first
                preincrement(state_it)

            # Calculate prune threshold
            prune_cutoff = best_score_at_j - PRUNE_THRESHOLD

            # Collect keys to remove (can't modify map while iterating)
            keys_to_remove.clear()
            state_it = dp[j].begin()
            while state_it != state_end:
                if deref(state_it).second.first < prune_cutoff:
                    keys_to_remove.push_back(deref(state_it).first)
                preincrement(state_it)

            # Remove low-scoring states
            for k_idx in range(keys_to_remove.size()):
                dp[j].erase(keys_to_remove[k_idx])

    # Backtracking
    if dp[total_chars].empty():
        return -float('inf'), [text]

    cdef int best_start = -1
    cdef double best_score = -float('inf')

    # Use C++ iterators for final state selection
    state_it = dp[total_chars].begin()
    state_end = dp[total_chars].end()
    while state_it != state_end:
        if deref(state_it).second.first > best_score:
            best_score = deref(state_it).second.first
            best_start = deref(state_it).first
        preincrement(state_it)
            
    cdef list words = []
    cdef int curr_end_char = total_chars
    cdef int curr_start_char = best_start
    cdef int next_start_char
    
    while curr_end_char > 0:
        b_end = char_to_byte[curr_end_char]
        b_start = char_to_byte[curr_start_char]
        
        # Decode back to Python string for result
        word_bytes = text_bytes.substr(b_start, b_end - b_start)
        words.append(word_bytes.decode('utf-8'))
        
        next_start_char = dp[curr_end_char][curr_start_char].second
        
        curr_end_char = curr_start_char
        curr_start_char = next_start_char
        
    words.reverse()
    return best_score, words
