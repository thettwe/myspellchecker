# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-optimized batch processing for Myanmar text corpus ingestion.

This module provides high-performance batch text processing for the
dictionary building pipeline, with optional OpenMP parallelization.

Key Features:
    - Parallel text segmentation using OpenMP (when available)
    - Efficient Viterbi-based word segmentation with chunk optimization
    - Myanmar text detection and normalization
    - Memory-efficient batch processing
    - Graceful degradation without OpenMP

Processing Pipeline:
    1. Text normalization (zero-width removal, diacritic reordering)
    2. Myanmar text detection and filtering
    3. Sentence splitting at ။ (period) and ၊ (comma) boundaries
    4. Word segmentation (CRF O(n) or Viterbi O(n²) per chunk)
    5. Result aggregation and batch output

Performance Optimizations:
    - Chunk size limited to 100 chars (2.25x faster than 150)
    - Sentence boundary splitting for O(n²) Viterbi efficiency
    - Optional OpenMP parallelization for multi-core systems
    - Requires: `brew install libomp` on macOS

Example:
    >>> from myspellchecker.data_pipeline.batch_processor import process_batch
    >>> texts = ["မြန်မာစာ", "ပြည်ထောင်စု"]
    >>> results = process_batch(texts, num_threads=4)

Note:
    OpenMP support is automatically detected at import time.
    Falls back to single-threaded processing if unavailable.

See Also:
    - frequency_counter.pyx: Frequency counting companion module
    - word_segment.pyx: Viterbi word segmentation algorithm
    - normalize_c.pyx: Text normalization functions
"""

from libcpp.string cimport string
from libcpp.vector cimport vector
from myspellchecker.tokenizers.cython.word_segment cimport viterbi
from myspellchecker.text.normalize_c cimport (
    segment_syllables_c,
    is_myanmar_string,
    is_myanmar_string_scoped,
    reorder_myanmar_diacritics,
    remove_zero_width_chars,
    clean_text_for_segmentation
)

# OpenMP support - conditional import
cdef bint HAS_OPENMP = False
try:
    from cython.parallel cimport prange, parallel
    HAS_OPENMP = True
except ImportError:
    pass

# Define regex pattern for English token (simulated)
cdef string ENG_TOKEN = b"<ENG>"

# Myanmar sentence/phrase separators for splitting long texts
# Viterbi is O(n²), so we split by sentence boundaries for efficiency
cdef str SENTENCE_SEP = "။"
cdef str PHRASE_SEP = "၊"

# Maximum chunk size for Viterbi (characters)
# OPTIMIZATION: Reduced from 150 to 100 chars
# Viterbi is O(n²): 100 chars is ~2.25x faster than 150 chars per chunk
# Trade-off: More chunks but each is much faster
cdef int MAX_VITERBI_CHUNK = 100

# Memory management configuration
# Default batch chunk size for streaming processing
cdef int DEFAULT_STREAMING_CHUNK_SIZE = 1000
# Maximum sentences to hold in memory before yielding
cdef int MAX_MEMORY_BATCH_SIZE = 10000

# Myanmar character sets for post-processing filters
# These fix segmentation artifacts from the mmap dictionary
# NOTE: Use frozenset/set at module level to avoid Cython type issues
# IMPORTANT: These constants MUST match core/constants/myanmar_constants.py
# to ensure consistent behavior between Python and Cython code paths.
# See core/constants/myanmar_constants.py for canonical definitions.
_MYANMAR_CONSONANTS = frozenset("ကခဂဃငစဆဇဈဉညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအ")
_ASAT = "်"  # Myanmar asat (killer mark)
_VIRAMA = "္"  # Myanmar virama (for stacking consonants)
# Dependent vowel signs U+102B-U+1032 (matches VOWEL_SIGNS in core/constants)
# Note: ံ (U+1036 anusvara) is a tone/mark, NOT a vowel sign
_MYANMAR_VOWELS = frozenset("ါာိီုူေဲ")
_MYANMAR_NUMERALS = frozenset("၀၁၂၃၄၅၆၇၈၉")
_ZERO_NUMERAL = "၀"  # Myanmar numeral zero (U+1040)
_WA_LETTER = "ဝ"  # Myanmar letter wa (U+101D)

# Extended Myanmar scope flag
# When True: includes non-Burmese Myanmar scripts (Mon, Shan, Karen)
# When False: strict Burmese only (excludes non-standard core chars)
cdef bint _allow_extended_myanmar = False

# CRF engine support — module-level flag + tagger (same pattern as _allow_extended_myanmar)
cdef bint _use_crf_engine = False
_crf_tagger = None  # pycrfsuite.Tagger instance (Python object)

# Pre-segmented input detection — auto-detect if input is already space-delimited
# When True: checks if input text is pre-segmented before CRF/Viterbi dispatch
# When False: always runs CRF/Viterbi segmentation (legacy behavior)
cdef bint _auto_detect_pre_segmented = True


def set_allow_extended_myanmar(bint value):
    """Set the extended Myanmar scope flag for batch processing.

    When True: includes non-Burmese Myanmar scripts (Mon, Shan, Karen, etc.)
    When False: strict Burmese only (default)

    Enables scope-aware batch processing.
    """
    global _allow_extended_myanmar
    _allow_extended_myanmar = value


def get_allow_extended_myanmar() -> bool:
    """Get the current extended Myanmar scope flag value."""
    return _allow_extended_myanmar


def set_crf_engine(bint value):
    """Enable or disable CRF engine for word segmentation in batch processing."""
    global _use_crf_engine
    _use_crf_engine = value


def get_crf_engine() -> bool:
    """Get the current CRF engine flag value."""
    return _use_crf_engine


def set_crf_tagger(tagger):
    """Set the pycrfsuite.Tagger instance for CRF word segmentation."""
    global _crf_tagger
    _crf_tagger = tagger


def get_crf_tagger():
    """Get the current pycrfsuite.Tagger instance."""
    return _crf_tagger


def set_auto_detect_pre_segmented(bint enabled):
    """Enable or disable auto-detection of pre-segmented input.

    When enabled, the batch processor checks if input text is already
    space-delimited (pre-segmented) before running CRF/Viterbi word
    segmentation. Pre-segmented tokens are used directly, avoiding
    re-segmentation artifacts.

    Args:
        enabled: True to enable auto-detection (default), False to disable.
    """
    global _auto_detect_pre_segmented
    _auto_detect_pre_segmented = enabled


def get_auto_detect_pre_segmented() -> bool:
    """Get the current pre-segmented auto-detection flag value."""
    return _auto_detect_pre_segmented


cpdef list crf_features(str sent):
    """
    Extract CRF features for all positions in a sentence.

    Cython-optimized version of WordTokenizer._word2features() from tokenizers/word.py.
    Each character gets a feature dict with context window features.

    Args:
        sent: Input sentence (string, each character is a position).

    Returns:
        List of feature dicts, one per character.
    """
    cdef int sent_len = len(sent)
    cdef list result = []
    cdef int i
    cdef str word, prev_char, nxt
    cdef dict feats

    for i in range(sent_len):
        word = sent[i]
        feats = {"number": word.isdigit()}

        if i > 0:
            prev_char = sent[i - 1]
            feats["prev_word.lower()"] = prev_char.lower()
            feats["prev_number"] = prev_char.isdigit()
            feats["bigram"] = prev_char.lower() + "_" + word.lower()
        else:
            feats["BOS"] = True

        if i < sent_len - 1:
            nxt = sent[i + 1]
            feats["next_word.lower()"] = nxt.lower()
            feats["next_number"] = nxt.isdigit()
        else:
            feats["EOS"] = True

        if i > 1:
            feats["trigram_1"] = (
                sent[i - 2].lower() + "_" + sent[i - 1].lower() + "_" + word.lower()
            )

        if i < sent_len - 2:
            feats["trigram_2"] = (
                word.lower() + "_" + sent[i + 1].lower() + "_" + sent[i + 2].lower()
            )

        result.append(feats)

    return result


cpdef list crf_segment_words(str text):
    """
    Segment Myanmar text into words using the CRF engine.

    Uses the module-level _crf_tagger to tag characters and splits at '|' boundaries.
    O(n) complexity — no chunking needed unlike Viterbi O(n²).

    Args:
        text: Input Myanmar text.

    Returns:
        List of segmented words.
    """
    cdef str sent = text.replace(" ", "")
    if not sent:
        return []

    cdef list features = crf_features(sent)
    cdef list preds = _crf_tagger.tag(features)

    cdef int sent_len = len(sent)
    cdef int preds_len = len(preds)

    # Safety check: CRF tagger must return exactly one tag per character.
    # If lengths mismatch, return the whole text as a single word to avoid
    # out-of-bounds access (boundscheck is disabled at module level).
    if preds_len != sent_len:
        return [sent]

    # Build words: '|' tag marks end of word boundary
    cdef list words = []
    cdef list current_word = []
    cdef int i
    cdef str char, tag

    for i in range(sent_len):
        char = sent[i]
        tag = preds[i]
        current_word.append(char)
        if tag == "|":
            words.append("".join(current_word))
            current_word = []

    # Don't drop trailing characters (unlike the original split("_")[:-1] pattern)
    # The original pattern drops the last segment because every word ends with "_",
    # but if CRF doesn't tag the last char as "|", we should keep it
    if current_word:
        words.append("".join(current_word))

    return words


cpdef str normalize_zero_to_wa(str text):
    """
    Normalize Myanmar numeral zero (၀) to letter wa (ဝ) when not in numeric context.

    This fixes a common encoding issue where numeral zero (U+1040) is incorrectly
    used instead of letter wa (U+101D).
    """
    if not text or _ZERO_NUMERAL not in text:
        return text

    cdef list result = []
    cdef int i
    cdef str char
    cdef bint prev_is_numeral, next_is_numeral
    cdef int text_len = len(text)

    for i in range(text_len):
        char = text[i]
        if char == _ZERO_NUMERAL:
            prev_is_numeral = i > 0 and text[i - 1] in _MYANMAR_NUMERALS
            next_is_numeral = i < text_len - 1 and text[i + 1] in _MYANMAR_NUMERALS
            if prev_is_numeral or next_is_numeral:
                result.append(char)
            else:
                result.append(_WA_LETTER)
        else:
            result.append(char)
    return "".join(result)


cpdef bint is_invalid_fragment(str word):
    """
    Check if a word is an invalid segmentation fragment.

    Invalid patterns include:
    1. Consonant + asat only (e.g., က်, င်)
    2. Consonant + tone + asat (e.g., င့်)
    3. Words starting with consonant + asat (e.g., င်ငံ)
    4. Incomplete stacking: virama followed by vowel
    """
    if not word:
        return False

    cdef int word_len = len(word)
    cdef int i

    # Pattern 1: consonant + asat only
    if word_len == 2 and word[0] in _MYANMAR_CONSONANTS and word[1] == _ASAT:
        return True

    # Pattern 2: consonant + tone + asat
    if word_len == 3 and word[0] in _MYANMAR_CONSONANTS and word[2] == _ASAT:
        if word[1] in "့း":
            return True

    # Pattern 3: starts with consonant + asat (fragments like င်ငံ)
    if word_len > 2 and word[0] in _MYANMAR_CONSONANTS and word[1] == _ASAT:
        return True

    # Pattern 4: incomplete stacking (virama followed by vowel)
    if _VIRAMA in word:
        for i in range(word_len - 1):
            if word[i] == _VIRAMA and word[i + 1] in _MYANMAR_VOWELS:
                return True

    return False


cpdef list merge_invalid_fragments(list tokens):
    """
    Merge invalid fragments with adjacent words.

    FIX: Do NOT modify input list in-place (caused SIGBUS with boundscheck=False).
    FIX: Do NOT use result[-1] with wraparound=False (undefined behavior).
    """
    if not tokens:
        return tokens

    cdef list result = []
    cdef int i = 0
    cdef str token
    cdef int tokens_len = len(tokens)
    cdef str pending = ""  # Hold fragments that need to prepend to next word
    cdef int result_len = 0  # Track result length to avoid negative indexing

    while i < tokens_len:
        token = tokens[i]

        # Prepend any pending fragment from previous iteration
        if pending:
            token = pending + token
            pending = ""

        if is_invalid_fragment(token):
            if result_len > 0:
                # Append to previous word in result (avoid result[-1] with wraparound=False)
                result[result_len - 1] = result[result_len - 1] + token
            else:
                # No previous word yet - save to prepend to next word
                pending = token
        else:
            result.append(token)
            result_len += 1

        i += 1

    # Handle any remaining pending fragment at end
    if pending:
        if result_len > 0:
            result[result_len - 1] = result[result_len - 1] + pending
        else:
            result.append(pending)

    return result


cpdef list split_word_numeral_tokens(list tokens):
    """
    Split tokens that contain both Myanmar letters and numerals.

    Examples: လ၁ → ['လ', '၁'], ကို၂၀၁၄ → ['ကို', '၂၀၁၄']
    """
    if not tokens:
        return tokens

    cdef list result = []
    cdef str token
    cdef bint has_letters, has_numerals
    cdef str c

    for token in tokens:
        if not token:
            continue

        has_letters = False
        has_numerals = False

        for c in token:
            if c in _MYANMAR_NUMERALS:
                has_numerals = True
            else:
                has_letters = True
            if has_letters and has_numerals:
                break

        if has_letters and has_numerals:
            result.extend(split_at_numeral_boundary(token))
        else:
            result.append(token)

    return result


cpdef list split_at_numeral_boundary(str token):
    """
    Split a token at boundaries between letters and numerals.
    """
    if not token:
        return []

    cdef list parts = []
    cdef list current_part = []
    cdef bint prev_is_numeral = False
    cdef bint is_numeral
    cdef bint first_char = True
    cdef str char

    for char in token:
        is_numeral = char in _MYANMAR_NUMERALS

        if not first_char and is_numeral != prev_is_numeral:
            if current_part:
                parts.append("".join(current_part))
            current_part = [char]
        else:
            current_part.append(char)

        prev_is_numeral = is_numeral
        first_char = False

    if current_part:
        parts.append("".join(current_part))

    return parts


cpdef list post_process_tokens(list tokens):
    """
    Apply all post-processing filters to viterbi output.

    1. Merge invalid fragments (consonant+asat patterns)
    2. Split word+numeral concatenations
    3. Merge fragments AGAIN (splitting can recreate fragments)

    Example of why step 3 is needed:
    - Input: ['မှု၃၂၅', 'န့်'] (fragment 'န့်')
    - After step 1: ['မှု၃၂၅န့်'] (merged)
    - After step 2: ['မှု', '၃၂၅', 'န့်'] (fragment recreated!)
    - After step 3: ['မှု', '၃၂၅န့်'] (merged again)
    """
    tokens = merge_invalid_fragments(tokens)
    tokens = split_word_numeral_tokens(tokens)
    # CRITICAL: Split can recreate fragments, so merge again
    tokens = merge_invalid_fragments(tokens)
    return tokens


cpdef list split_for_viterbi(str text):
    """
    Split long text into smaller chunks for efficient Viterbi processing.

    Strategy:
    1. Split by sentence separator (။) first
    2. For still-long pieces, split by phrase separator (၊)
    3. For very long pieces, split at MAX_VITERBI_CHUNK boundaries

    Returns list of (chunk, is_sentence_end) tuples to preserve boundaries.
    """
    cdef list result = []
    cdef list sentences
    cdef list phrases
    cdef str sent, phrase, chunk
    cdef int start

    # First split by sentence separator
    sentences = text.split(SENTENCE_SEP)

    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue

        # If sentence is short enough, use it directly
        if len(sent) <= MAX_VITERBI_CHUNK:
            result.append(sent)
            continue

        # Split long sentences by phrase separator
        phrases = sent.split(PHRASE_SEP)

        for phrase in phrases:
            phrase = phrase.strip()
            if not phrase:
                continue

            # If phrase is short enough, use it directly
            if len(phrase) <= MAX_VITERBI_CHUNK:
                result.append(phrase)
                continue

            # Very long phrase - split into fixed-size chunks
            # This is a last resort for texts without proper punctuation
            start = 0
            while start < len(phrase):
                chunk = phrase[start:start + MAX_VITERBI_CHUNK]
                result.append(chunk)
                start += MAX_VITERBI_CHUNK

    return result

def process_batch(list text_list, list source_list):
    """
    Process a batch of sentences entirely in Cython to avoid Python loop overhead.
    Returns a dict of lists suitable for Arrow.
    """
    cdef int n = len(text_list)
    cdef int i, j
    cdef str sentence_py
    cdef string sentence
    cdef str src
    cdef vector[string] parts
    cdef string part
    
    # Output buffers
    cdef list out_text = []
    cdef list out_source = []
    cdef list out_syllables = []
    cdef list out_words = []
    cdef list out_syl_count = []
    cdef list out_word_count = []
    
    # Temp vars
    cdef list s_part_py
    cdef list w_part_py
    cdef list current_syllables
    cdef list current_words
    cdef tuple viterbi_res
    cdef str part_clean
    cdef list viterbi_chunks
    cdef str viterbi_chunk
    
    for i in range(n):
        sentence_py = text_list[i]
        src = source_list[i]
        
        if not sentence_py:
            continue
            
        # Normalize (C++ optimized)
        sentence_py = remove_zero_width_chars(sentence_py)
        sentence_py = reorder_myanmar_diacritics(sentence_py)
        
        # Split by ENG_TOKEN
        parts_py = sentence_py.split("<ENG>")
        
        current_syllables = []
        current_words = []
        
        for j, part_py in enumerate(parts_py):
            if part_py.strip():
                # Filter Logic inside Cython
                # segment_syllables_c returns list[str]
                s_part_py = segment_syllables_c(part_py, _allow_extended_myanmar)
                
                # Filter: [s for s in s_part if is_myanmar_string_scoped(s)]
                s_part_filtered = []
                for s in s_part_py:
                    if is_myanmar_string_scoped(s, _allow_extended_myanmar):
                        s_part_filtered.append(s)
                current_syllables.extend(s_part_filtered)

                # Word Segmentation — check for pre-segmented input first
                # If input is already space-delimited (e.g., from curated corpus),
                # use the tokens directly instead of re-segmenting with CRF/Viterbi.
                # This avoids ~60% of segmentation artifacts from re-segmentation.
                _skip_segmentation = False
                if _auto_detect_pre_segmented:
                    _pre_seg_tokens = part_py.split()
                    if len(_pre_seg_tokens) >= 2:
                        _mm_count = 0
                        for _t in _pre_seg_tokens:
                            if is_myanmar_string_scoped(_t, _allow_extended_myanmar):
                                _mm_count += 1
                        if <double>_mm_count / <double>len(_pre_seg_tokens) >= 0.7:
                            # Pre-segmented: use space-split tokens directly
                            for _w in _pre_seg_tokens:
                                _w_stripped = _w.strip()
                                if _w_stripped and is_myanmar_string_scoped(
                                    _w_stripped, _allow_extended_myanmar
                                ):
                                    current_words.append(_w_stripped)
                            _skip_segmentation = True

                if not _skip_segmentation:
                    # Word Segmentation — dispatch by engine
                    if _use_crf_engine and _crf_tagger is not None:
                        # CRF path — O(n), no chunking needed
                        part_clean = clean_text_for_segmentation(part_py, _allow_extended_myanmar)
                        part_clean = normalize_zero_to_wa(part_clean)
                        if part_clean and is_myanmar_string_scoped(part_clean, _allow_extended_myanmar):
                            w_part_py = crf_segment_words(part_clean)
                            for w in w_part_py:
                                if is_myanmar_string_scoped(w, _allow_extended_myanmar):
                                    current_words.append(w)
                    else:
                        # Viterbi path — O(n²), needs split_for_viterbi chunking
                        # Split BEFORE cleaning to preserve sentence boundaries
                        viterbi_chunks = split_for_viterbi(part_py)

                        for viterbi_chunk in viterbi_chunks:
                            part_clean = clean_text_for_segmentation(
                                viterbi_chunk, _allow_extended_myanmar
                            )
                            part_clean = normalize_zero_to_wa(part_clean)
                            if part_clean and is_myanmar_string_scoped(
                                part_clean, _allow_extended_myanmar
                            ):
                                viterbi_res = viterbi(part_clean)
                                w_part_py = viterbi_res[1]
                                for w in w_part_py:
                                    if is_myanmar_string_scoped(w, _allow_extended_myanmar):
                                        current_words.append(w)

            # Restore ENG_TOKEN
            if j < len(parts_py) - 1:
                current_words.append("<ENG>")

        # Post-process: apply filters to fix segmentation artifacts
        # 1. Merge invalid fragments (consonant+asat patterns)
        # 2. Split word+numeral concatenations
        current_words = post_process_tokens(current_words)

        # Append to outputs
        out_text.append(sentence_py)
        out_source.append(src)
        out_syllables.append(current_syllables)
        out_words.append(current_words)
        out_syl_count.append(len(current_syllables))
        out_word_count.append(len(current_words))
        
    return {
        "text": out_text,
        "source": out_source,
        "syllables": out_syllables,
        "words": out_words,
        "syllable_count": out_syl_count,
        "word_count": out_word_count
    }


cdef tuple process_single_sentence(str sentence_py):
    """
    Process a single sentence and return (syllables, words) tuple.
    Extracted for use in parallel processing.
    """
    cdef list s_part_py
    cdef list w_part_py
    cdef list current_syllables = []
    cdef list current_words = []
    cdef tuple viterbi_res
    cdef str part_clean
    cdef list viterbi_chunks
    cdef str viterbi_chunk
    cdef int j

    if not sentence_py:
        return ([], [])

    # Normalize (C++ optimized)
    sentence_py = remove_zero_width_chars(sentence_py)
    sentence_py = reorder_myanmar_diacritics(sentence_py)

    # Split by ENG_TOKEN
    parts_py = sentence_py.split("<ENG>")

    for j, part_py in enumerate(parts_py):
        if part_py.strip():
            # Syllable segmentation
            s_part_py = segment_syllables_c(part_py, _allow_extended_myanmar)

            # Filter Myanmar syllables
            for s in s_part_py:
                if is_myanmar_string_scoped(s, _allow_extended_myanmar):
                    current_syllables.append(s)

            # Word Segmentation — check for pre-segmented input first
            _skip_segmentation = False
            if _auto_detect_pre_segmented:
                _pre_seg_tokens = part_py.split()
                if len(_pre_seg_tokens) >= 2:
                    _mm_count = 0
                    for _t in _pre_seg_tokens:
                        if is_myanmar_string_scoped(_t, _allow_extended_myanmar):
                            _mm_count += 1
                    if <double>_mm_count / <double>len(_pre_seg_tokens) >= 0.7:
                        # Pre-segmented: use space-split tokens directly
                        for _w in _pre_seg_tokens:
                            _w_stripped = _w.strip()
                            if _w_stripped and is_myanmar_string_scoped(
                                _w_stripped, _allow_extended_myanmar
                            ):
                                current_words.append(_w_stripped)
                        _skip_segmentation = True

            if not _skip_segmentation:
                # Word Segmentation — dispatch by engine
                if _use_crf_engine and _crf_tagger is not None:
                    # CRF path — O(n), no chunking needed
                    part_clean = clean_text_for_segmentation(part_py, _allow_extended_myanmar)
                    part_clean = normalize_zero_to_wa(part_clean)
                    if part_clean and is_myanmar_string_scoped(part_clean, _allow_extended_myanmar):
                        w_part_py = crf_segment_words(part_clean)
                        for w in w_part_py:
                            if is_myanmar_string_scoped(w, _allow_extended_myanmar):
                                current_words.append(w)
                else:
                    # Viterbi path — O(n²), needs split_for_viterbi chunking
                    viterbi_chunks = split_for_viterbi(part_py)

                    for viterbi_chunk in viterbi_chunks:
                        part_clean = clean_text_for_segmentation(
                            viterbi_chunk, _allow_extended_myanmar
                        )
                        part_clean = normalize_zero_to_wa(part_clean)
                        if part_clean and is_myanmar_string_scoped(
                            part_clean, _allow_extended_myanmar
                        ):
                            viterbi_res = viterbi(part_clean)
                            w_part_py = viterbi_res[1]
                            for w in w_part_py:
                                if is_myanmar_string_scoped(w, _allow_extended_myanmar):
                                    current_words.append(w)

        # Restore ENG_TOKEN
        if j < len(parts_py) - 1:
            current_words.append("<ENG>")

    # Post-process: apply filters to fix segmentation artifacts
    current_words = post_process_tokens(current_words)

    return (current_syllables, current_words)


def process_batch_parallel(list text_list, list source_list, int num_threads=4):
    """
    Process a batch of sentences with OpenMP parallelization.

    Uses prange with GIL-release pattern for parallel processing.
    Each thread processes different sentences, writing to pre-allocated slots.

    Args:
        text_list: List of input sentences
        source_list: List of source identifiers
        num_threads: Number of OpenMP threads (default: 4)

    Returns:
        Dict with processed results (same format as process_batch)
    """
    cdef int n = len(text_list)
    cdef int i

    # Pre-allocate result lists (one slot per sentence)
    # Each slot will be filled by a different thread
    cdef list results = [None] * n  # Will hold (syllables, words) tuples
    cdef list normalized = [None] * n  # Will hold normalized text

    # Phase 1: Process sentences in parallel
    # Using prange with GIL acquisition for Python operations
    if HAS_OPENMP:
        with nogil, parallel(num_threads=num_threads):
            for i in prange(n, schedule='dynamic', chunksize=10):
                with gil:
                    # Process single sentence
                    sentence_py = text_list[i]
                    if sentence_py:
                        # Normalize
                        norm_text = remove_zero_width_chars(sentence_py)
                        norm_text = reorder_myanmar_diacritics(norm_text)
                        normalized[i] = norm_text

                        # Process normalized text (not original)
                        results[i] = process_single_sentence(norm_text)
                    else:
                        results[i] = ([], [])
                        normalized[i] = sentence_py
    else:
        # Fallback to sequential processing if OpenMP not available
        for i in range(n):
            sentence_py = text_list[i]
            if sentence_py:
                norm_text = remove_zero_width_chars(sentence_py)
                norm_text = reorder_myanmar_diacritics(norm_text)
                normalized[i] = norm_text
                # Process normalized text (not original)
                results[i] = process_single_sentence(norm_text)
            else:
                results[i] = ([], [])
                normalized[i] = sentence_py

    # Phase 2: Collect results into output lists (sequential - fast)
    cdef list out_text = []
    cdef list out_source = []
    cdef list out_syllables = []
    cdef list out_words = []
    cdef list out_syl_count = []
    cdef list out_word_count = []

    for i in range(n):
        if results[i] is not None and results[i][0] is not None:
            syllables, words = results[i]
            if syllables or words:  # Only include non-empty results
                out_text.append(normalized[i] if normalized[i] else text_list[i])
                out_source.append(source_list[i])
                out_syllables.append(syllables)
                out_words.append(words)
                out_syl_count.append(len(syllables))
                out_word_count.append(len(words))

    return {
        "text": out_text,
        "source": out_source,
        "syllables": out_syllables,
        "words": out_words,
        "syllable_count": out_syl_count,
        "word_count": out_word_count
    }


def has_openmp():
    """Check if OpenMP is available."""
    return HAS_OPENMP


def process_batch_streaming(list text_list, list source_list, int chunk_size=0, int num_threads=4):
    """
    Process a batch of sentences with streaming/chunked processing.

    This function yields results in chunks to reduce memory footprint for
    large files. Instead of holding all intermediate results in memory,
    it processes and yields chunks of results.

    Args:
        text_list: List of input sentences
        source_list: List of source identifiers
        chunk_size: Number of sentences to process per chunk (default: DEFAULT_STREAMING_CHUNK_SIZE)
        num_threads: Number of OpenMP threads (default: 4)

    Yields:
        Dict with processed results for each chunk (same format as process_batch)

    Example:
        >>> for chunk_result in process_batch_streaming(texts, sources, chunk_size=500):
        ...     # Process chunk_result
        ...     save_to_database(chunk_result)
        ...     # Memory is released after each chunk
    """
    if chunk_size <= 0:
        chunk_size = DEFAULT_STREAMING_CHUNK_SIZE

    cdef int n = len(text_list)
    cdef int start = 0
    cdef int end

    while start < n:
        end = min(start + chunk_size, n)

        # Process this chunk
        chunk_texts = text_list[start:end]
        chunk_sources = source_list[start:end]

        if HAS_OPENMP:
            result = process_batch_parallel(chunk_texts, chunk_sources, num_threads)
        else:
            result = process_batch(chunk_texts, chunk_sources)

        yield result

        # Move to next chunk (previous chunk's memory can be garbage collected)
        start = end


def get_streaming_chunk_size():
    """Get the default streaming chunk size for batch processing."""
    return DEFAULT_STREAMING_CHUNK_SIZE


def get_max_memory_batch_size():
    """Get the maximum batch size to hold in memory."""
    return MAX_MEMORY_BATCH_SIZE
