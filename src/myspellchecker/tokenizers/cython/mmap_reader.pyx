# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Memory-Mapped Segmentation Dictionary Reader.

Provides fast, GIL-free lookups into the memory-mapped dictionary file.
Automatically shares memory between forked workers via copy-on-write.
"""

from libc.stdint cimport uint64_t, uint32_t, uint16_t, int64_t
from libc.string cimport memcpy, memcmp
from libc.math cimport log10
from libc.stdlib cimport malloc, free
from cpython.bytes cimport PyBytes_AS_STRING

import mmap
import os
import struct

from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Header constants matching mmap_builder.py
DEF MAGIC_SIZE = 8
DEF HEADER_SIZE = 128
DEF UNIGRAM_ENTRY_SIZE = 24
DEF BIGRAM_ENTRY_SIZE = 32


cdef inline uint64_t fnv1a_hash(const char* data, int length) noexcept nogil:
    """FNV-1a 64-bit hash function."""
    cdef uint64_t h = <uint64_t>0xcbf29ce484222325  # FNV_OFFSET_BASIS
    cdef uint64_t prime = <uint64_t>0x100000001b3  # FNV_PRIME
    cdef int i
    for i in range(length):
        h ^= <uint64_t>(<unsigned char>data[i])
        h = h * prime  # uint64_t naturally wraps at 64 bits
    return h


cdef class MMapSegmentationReader:
    """
    Fast memory-mapped dictionary reader.

    Provides O(1) average lookups for unigrams and bigrams.
    All lookup methods are GIL-free for maximum parallelism.
    """

    # Note: All cdef attributes are declared in mmap_reader.pxd

    def __cinit__(self):
        self._initialized = False
        self._data = NULL

    def __dealloc__(self):
        self.close()

    cpdef bint open(self, str path):
        """
        Open and memory-map the segmentation file.

        Args:
            path: Path to the segmentation.mmap file

        Returns:
            True if successful, False otherwise
        """
        cdef int fd
        cdef object mv
        cdef const unsigned char[:] data_view

        if self._initialized:
            self.close()

        self._file_path = path

        try:
            # Open file and create mmap
            fd = os.open(path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size

            self._mmap_file = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            os.close(fd)

            # Get raw pointer to mmap data
            data_view = self._mmap_file
            self._data = <const char*>&data_view[0]
            self._data_size = len(self._mmap_file)

            # Validate magic bytes
            if memcmp(self._data, b"MYSEGV02", MAGIC_SIZE) != 0:
                logger.error(f"Invalid magic bytes in {path}")
                self._data = NULL
                data_view = None  # Release buffer view before closing mmap
                self._mmap_file.close()
                self._mmap_file = None
                return False

            # Validate file is large enough for header (prevents buffer overread)
            if self._data_size < HEADER_SIZE:
                logger.error(
                    f"File too small for header ({self._data_size} < {HEADER_SIZE}): {path}"
                )
                self._data = NULL
                data_view = None  # Release buffer view before closing mmap
                self._mmap_file.close()
                self._mmap_file = None
                return False

            # Parse header
            self._parse_header()

            self._initialized = True
            logger.debug(f"MMap reader initialized: {self._unigram_count} unigrams, {self._bigram_count} bigrams")
            return True

        except (OSError, ValueError, MemoryError) as e:
            logger.error(f"Failed to open mmap file {path}: {e}")
            return False

    cdef void _parse_header(self) noexcept:
        """Parse header fields from mapped data."""
        cdef const char* p = self._data + MAGIC_SIZE + 4 + 4  # Skip magic, version, checksum

        # Read header fields using memcpy (done once at init)
        memcpy(&self._unigram_offset, p, 8); p += 8
        memcpy(&self._unigram_count, p, 4); p += 4
        memcpy(&self._unigram_buckets, p, 4); p += 4
        memcpy(&self._bigram_offset, p, 8); p += 8
        memcpy(&self._bigram_count, p, 4); p += 4
        memcpy(&self._bigram_buckets, p, 4); p += 4
        memcpy(&self._string_pool_offset, p, 8); p += 8
        memcpy(&self._N_unigram, p, 8); p += 8
        memcpy(&self._N_bigram, p, 8); p += 8
        memcpy(&self._log_N_unigram, p, 8); p += 8
        memcpy(&self._log_N_bigram, p, 8)

    cpdef void close(self):
        """Close the memory-mapped file."""
        if self._mmap_file is not None:
            try:
                self._mmap_file.close()
            except OSError as e:
                # Log at debug level - close failures during cleanup are non-critical
                logger.debug(f"Could not close mmap file cleanly: {e}")
            self._mmap_file = None
        self._data = NULL
        self._initialized = False

    cpdef bint is_initialized(self):
        """Check if reader is ready for lookups."""
        return self._initialized

    cdef double _get_unigram_log_prob_nogil(self, const char* word, int word_len) noexcept nogil:
        """
        GIL-free unigram lookup.

        Returns the pre-computed log10(count/N) for the word,
        or the unknown word penalty if not found.
        """
        cdef:
            uint64_t h = fnv1a_hash(word, word_len)
            uint32_t bucket = h % self._unigram_buckets
            uint32_t probe = 0
            const char* entry_ptr
            uint64_t stored_hash
            uint32_t str_offset
            uint16_t str_len
            double log_prob

        # Linear probing
        while probe < self._unigram_buckets:
            entry_ptr = self._data + self._unigram_offset + (bucket * UNIGRAM_ENTRY_SIZE)

            # Read hash from entry
            memcpy(&stored_hash, entry_ptr, 8)

            if stored_hash == 0:  # Empty bucket - not found
                break

            if stored_hash == h:
                # Hash match - verify string
                memcpy(&str_offset, entry_ptr + 8, 4)
                memcpy(&str_len, entry_ptr + 12, 2)

                if str_len == word_len:
                    if memcmp(self._data + self._string_pool_offset + str_offset, word, word_len) == 0:
                        # Found! Return pre-computed log prob
                        memcpy(&log_prob, entry_ptr + 16, 8)
                        return log_prob

            # Continue probing
            probe += 1
            bucket = (h + probe) % self._unigram_buckets

        # Not found - return unknown word penalty
        # log10(10) - log10(N) - utf8_char_count
        # Simplified: 1.0 - log_N - len (for Myanmar, char_count ~= len/3)
        return 1.0 - self._log_N_unigram - <double>(word_len // 3 + 1)

    cdef double _get_bigram_log_prob_nogil(self,
                                           const char* w1, int len1,
                                           const char* w2, int len2) noexcept nogil:
        """
        GIL-free bigram lookup.

        Returns the pre-computed log10(count/N) for the bigram,
        or falls back to unigram probability if not found.
        """
        cdef:
            uint64_t h1 = fnv1a_hash(w1, len1)
            uint64_t h2 = fnv1a_hash(w2, len2)
            uint64_t golden = <uint64_t>0x9E3779B97F4A7C15
            uint64_t combined = h1 ^ (h2 * golden)  # uint64_t naturally wraps
            uint32_t bucket = combined % self._bigram_buckets
            uint32_t probe = 0
            const char* entry_ptr
            uint64_t stored_hash
            uint32_t o1, o2
            uint16_t l1, l2
            double log_prob

        # Linear probing
        while probe < self._bigram_buckets:
            entry_ptr = self._data + self._bigram_offset + (bucket * BIGRAM_ENTRY_SIZE)

            # Read hash from entry
            memcpy(&stored_hash, entry_ptr, 8)

            if stored_hash == 0:  # Empty bucket
                break

            if stored_hash == combined:
                # Hash match - verify strings
                memcpy(&o1, entry_ptr + 8, 4)
                memcpy(&l1, entry_ptr + 12, 2)
                memcpy(&o2, entry_ptr + 14, 4)
                memcpy(&l2, entry_ptr + 18, 2)

                if l1 == len1 and l2 == len2:
                    if (memcmp(self._data + self._string_pool_offset + o1, w1, len1) == 0 and
                        memcmp(self._data + self._string_pool_offset + o2, w2, len2) == 0):
                        # Found! Return pre-computed log prob
                        memcpy(&log_prob, entry_ptr + 24, 8)
                        return log_prob

            # Continue probing
            probe += 1
            bucket = (combined + probe) % self._bigram_buckets

        # Not found - fall back to unigram probability
        return self._get_unigram_log_prob_nogil(w2, len2)

    cdef double _get_transition_log_prob_nogil(self,
                                                const char* curr, int curr_len,
                                                const char* prev, int prev_len) noexcept nogil:
        """
        GIL-free transition probability lookup.

        Computes P(curr|prev) = P(curr,prev)/P(prev) when bigram exists,
        or falls back to P(curr) when bigram not found.

        This matches the behavior of get_transition_log_prob_indices in word_segment.pyx.
        """
        cdef:
            uint64_t h1 = fnv1a_hash(prev, prev_len)
            uint64_t h2 = fnv1a_hash(curr, curr_len)
            uint64_t golden = <uint64_t>0x9E3779B97F4A7C15
            uint64_t combined = h1 ^ (h2 * golden)
            uint32_t bucket = combined % self._bigram_buckets
            uint32_t probe = 0
            const char* entry_ptr
            uint64_t stored_hash
            uint32_t o1, o2
            uint16_t l1, l2
            double bigram_log_prob
            double unigram_prev_log_prob

        # Linear probing for bigram
        while probe < self._bigram_buckets:
            entry_ptr = self._data + self._bigram_offset + (bucket * BIGRAM_ENTRY_SIZE)

            # Read hash from entry
            memcpy(&stored_hash, entry_ptr, 8)

            if stored_hash == 0:  # Empty bucket - bigram not found
                break

            if stored_hash == combined:
                # Hash match - verify strings
                memcpy(&o1, entry_ptr + 8, 4)
                memcpy(&l1, entry_ptr + 12, 2)
                memcpy(&o2, entry_ptr + 14, 4)
                memcpy(&l2, entry_ptr + 18, 2)

                if l1 == prev_len and l2 == curr_len:
                    if (memcmp(self._data + self._string_pool_offset + o1, prev, prev_len) == 0 and
                        memcmp(self._data + self._string_pool_offset + o2, curr, curr_len) == 0):
                        # Bigram found! Compute conditional probability
                        memcpy(&bigram_log_prob, entry_ptr + 24, 8)
                        unigram_prev_log_prob = self._get_unigram_log_prob_nogil(prev, prev_len)
                        # Fix: Correct for mismatched N values.
                        # bigram uses log_N_bigram, unigram uses log_N_unigram.
                        # P(curr|prev) = count(prev,curr)/count(prev)
                        # Need to cancel the different N denominators.
                        return bigram_log_prob - unigram_prev_log_prob - self._log_N_unigram + self._log_N_bigram

            # Continue probing
            probe += 1
            bucket = (combined + probe) % self._bigram_buckets

        # Bigram not found - fall back to unigram probability for current word
        return self._get_unigram_log_prob_nogil(curr, curr_len)

    def get_unigram_log_prob(self, str word) -> float:
        """
        Python-callable unigram lookup.

        Args:
            word: The word to look up

        Returns:
            log10(count/N) for known words, or unknown penalty
        """
        if not self._initialized:
            raise RuntimeError("MMap reader not initialized")

        cdef bytes word_bytes = word.encode('utf-8')
        cdef const char* word_ptr = PyBytes_AS_STRING(word_bytes)
        cdef int word_len = len(word_bytes)

        return self._get_unigram_log_prob_nogil(word_ptr, word_len)

    def get_bigram_log_prob(self, str w1, str w2) -> float:
        """
        Python-callable bigram lookup.

        Args:
            w1: First word
            w2: Second word

        Returns:
            log10(count/N) for known bigrams, or fallback to unigram
        """
        if not self._initialized:
            raise RuntimeError("MMap reader not initialized")

        cdef bytes w1_bytes = w1.encode('utf-8')
        cdef bytes w2_bytes = w2.encode('utf-8')
        cdef const char* w1_ptr = PyBytes_AS_STRING(w1_bytes)
        cdef const char* w2_ptr = PyBytes_AS_STRING(w2_bytes)
        cdef int len1 = len(w1_bytes)
        cdef int len2 = len(w2_bytes)

        return self._get_bigram_log_prob_nogil(w1_ptr, len1, w2_ptr, len2)

    def get_stats(self) -> dict:
        """Get statistics about the loaded dictionary."""
        if not self._initialized:
            return {"initialized": False}

        return {
            "initialized": True,
            "file_path": self._file_path,
            "unigram_count": self._unigram_count,
            "unigram_buckets": self._unigram_buckets,
            "bigram_count": self._bigram_count,
            "bigram_buckets": self._bigram_buckets,
            "N_unigram": self._N_unigram,
            "N_bigram": self._N_bigram,
            "log_N_unigram": self._log_N_unigram,
            "log_N_bigram": self._log_N_bigram,
        }


# Global reader instance for module-level access
cdef MMapSegmentationReader _global_reader = None
cdef bint _global_initialized = False


def initialize_mmap_reader(str path) -> bool:
    """
    Initialize the global mmap reader.

    Call this once in the parent process before forking workers.
    The memory-mapped file will be automatically shared via COW.

    Args:
        path: Path to segmentation.mmap file

    Returns:
        True if successful
    """
    global _global_reader, _global_initialized

    _global_reader = MMapSegmentationReader()
    if _global_reader.open(path):
        _global_initialized = True
        return True
    return False


def ensure_mmap_initialized() -> bool:
    """
    Check if mmap is ready (safe to call after fork).

    Returns:
        True if initialized
    """
    return _global_initialized and _global_reader is not None and _global_reader.is_initialized()


def get_mmap_reader() -> MMapSegmentationReader:
    """Get the global mmap reader instance."""
    if not _global_initialized:
        raise RuntimeError("MMap reader not initialized. Call initialize_mmap_reader() first.")
    return _global_reader


def get_unigram_prob(str word) -> float:
    """Convenience function for unigram lookup."""
    return get_mmap_reader().get_unigram_log_prob(word)


def get_bigram_prob(str w1, str w2) -> float:
    """Convenience function for bigram lookup."""
    return get_mmap_reader().get_bigram_log_prob(w1, w2)
