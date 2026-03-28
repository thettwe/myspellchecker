"""
Data Pipeline and Training Constants.

This module contains constants for:
- Pipeline processing (batch sizes, file names)
- Training defaults (tokenizer files)
- Data pipeline file naming conventions
"""

# =============================================================================
# Pipeline Processing Constants
# =============================================================================

# Optimal batch size for Arrow/database operations
DEFAULT_BATCH_SIZE = 10000

# =============================================================================
# Pipeline Default File Names
# =============================================================================

# Frequency files
DEFAULT_PIPELINE_SYLLABLE_FREQS_FILE = "syllable_frequencies.tsv"
DEFAULT_PIPELINE_WORD_FREQS_FILE = "word_frequencies.tsv"

# N-gram probability files
DEFAULT_PIPELINE_BIGRAM_PROBS_FILE = "bigram_probabilities.tsv"
DEFAULT_PIPELINE_TRIGRAM_PROBS_FILE = "trigram_probabilities.tsv"
DEFAULT_PIPELINE_FOURGRAM_PROBS_FILE = "fourgram_probabilities.tsv"
DEFAULT_PIPELINE_FIVEGRAM_PROBS_FILE = "fivegram_probabilities.tsv"

# POS probability files
DEFAULT_PIPELINE_POS_UNIGRAM_PROBS_FILE = "pos_unigram_probabilities.tsv"
DEFAULT_PIPELINE_POS_BIGRAM_PROBS_FILE = "pos_bigram_probabilities.tsv"
DEFAULT_PIPELINE_POS_TRIGRAM_PROBS_FILE = "pos_trigram_probabilities.tsv"

# =============================================================================
# Training Default File Names
# =============================================================================

# Tokenizer file for semantic models
DEFAULT_TOKENIZER_FILE = "tokenizer.json"

# =============================================================================
# Build-Mode SQLite PRAGMA Constants
# =============================================================================

# 1 GB page cache during build (negative = KiB)
BUILD_PRAGMA_CACHE_SIZE = -1048576

# 2 GB memory-mapped I/O during build
BUILD_PRAGMA_MMAP_SIZE = 2147483648

# =============================================================================
# Pipeline Quality Thresholds
# =============================================================================

# Maximum percentage of empty sentences before warning
QUALITY_MAX_EMPTY_PCT = 20

# Min/max average words per sentence before warning
QUALITY_MIN_AVG_WORDS = 3
QUALITY_MAX_AVG_WORDS = 30

# =============================================================================
# Lite-Mode Database Limits
# =============================================================================

LITE_MODE_SYLLABLES_LIMIT = 2000
LITE_MODE_WORDS_LIMIT = 10000
LITE_MODE_BIGRAMS_LIMIT = 50000
LITE_MODE_TRIGRAMS_LIMIT = 50000
