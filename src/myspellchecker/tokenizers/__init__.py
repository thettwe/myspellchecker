"""
Tokenizers for Myanmar (Burmese) text processing.

This module provides tokenizers for segmenting Myanmar text at different
levels: syllables and words.

Classes:
    SyllableTokenizer: Regex-based syllable segmentation
    WordTokenizer: Word segmentation using CRF or myword (Viterbi) engines

Example:
    >>> from myspellchecker.tokenizers import WordTokenizer
    >>> tokenizer = WordTokenizer(engine="myword")
    >>> tokenizer.tokenize("မြန်မာနိုင်ငံ")
    ['မြန်မာ', 'နိုင်ငံ']
"""

from __future__ import annotations

from .syllable import SyllableTokenizer
from .word import WordTokenizer

__all__ = ["SyllableTokenizer", "WordTokenizer"]
