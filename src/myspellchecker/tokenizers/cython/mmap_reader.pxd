# cython: language_level=3
"""
Header file for mmap_reader.pyx

Exposes the MMapSegmentationReader class and its cdef methods
for use by other Cython modules (e.g., word_segment_fast.pyx).
"""

from libc.stdint cimport uint64_t, uint32_t, uint16_t


cdef class MMapSegmentationReader:
    cdef:
        object _mmap_file
        const char* _data
        size_t _data_size

        uint64_t _unigram_offset
        uint32_t _unigram_count
        uint32_t _unigram_buckets
        uint64_t _bigram_offset
        uint32_t _bigram_count
        uint32_t _bigram_buckets
        uint64_t _string_pool_offset
        double _N_unigram
        double _N_bigram
        double _log_N_unigram
        double _log_N_bigram

        bint _initialized
        str _file_path

    cpdef bint open(self, str path)
    cpdef void close(self)
    cpdef bint is_initialized(self)

    # Internal methods
    cdef void _parse_header(self) noexcept

    # GIL-free lookup methods (for use in nogil blocks)
    cdef double _get_unigram_log_prob_nogil(self, const char* word, int word_len) noexcept nogil
    cdef double _get_bigram_log_prob_nogil(self, const char* w1, int len1, const char* w2, int len2) noexcept nogil
    cdef double _get_transition_log_prob_nogil(self, const char* curr, int curr_len, const char* prev, int prev_len) noexcept nogil


# Global reader access
cdef MMapSegmentationReader _global_reader
cdef bint _global_initialized
