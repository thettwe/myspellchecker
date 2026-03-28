"""
Cython-optimized modules for tokenization.

This package contains compiled Cython modules for:
- word_segment: Viterbi algorithm for word segmentation
- mmap_reader: Memory-mapped file reader for dictionary access
"""

from __future__ import annotations

# These modules are imported dynamically when needed
# to avoid compilation issues on systems without Cython

__all__ = ["word_segment", "mmap_reader"]
