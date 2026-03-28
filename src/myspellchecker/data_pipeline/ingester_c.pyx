# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from myspellchecker.text.normalize import (
    normalize_with_zawgyi_conversion,
    is_myanmar_text,
    remove_word_segmentation_markers,
)
from myspellchecker.text.validator import validate_word
from myspellchecker.core.config.text_configs import ZawgyiConfig

# Global config to avoid recreation (cached at module level)
cdef object _ingester_zawgyi_config = ZawgyiConfig(myanmar_text_threshold=0.1)

# Extended Myanmar configuration (module-level for multiprocessing compatibility)
cdef bint _allow_extended_myanmar = False

# Segmentation marker removal flag (module-level for multiprocessing compatibility)
cdef bint _remove_segmentation_markers_c = True


def set_allow_extended_myanmar_c(bint allow):
    """Configure Extended Myanmar character handling for Cython ingester."""
    global _allow_extended_myanmar
    _allow_extended_myanmar = allow


def get_allow_extended_myanmar_c():
    """Get current Extended Myanmar character handling setting."""
    return _allow_extended_myanmar


def set_remove_segmentation_markers_c(bint remove):
    """Configure segmentation marker removal for Cython ingester."""
    global _remove_segmentation_markers_c
    _remove_segmentation_markers_c = remove


def get_remove_segmentation_markers_c():
    """Get current segmentation marker removal setting."""
    return _remove_segmentation_markers_c

def normalize_batch_c(list batch):
    """
    Cython implementation of _normalize_batch.
    Processes a list of raw text lines:
    1. Zawgyi -> Unicode
    2. Unicode Normalization
    3. Myanmar Text Check
    4. Structure Validation
    """
    cdef list results = []
    cdef str line
    cdef str cleaned
    cdef list words
    cdef str w
    cdef list valid_words
    cdef bint is_myanmar
    cdef Py_ssize_t i
    cdef Py_ssize_t n = len(batch)
    
    # Pre-allocate if possible? List append is efficient enough.
    
    for i in range(n):
        line = batch[i]
        
        # Step 1-2: Zawgyi conversion + normalization
        # This calls Python function (heavy)
        cleaned = normalize_with_zawgyi_conversion(line)

        # Step 2.5: Remove word segmentation markers (underscores/spaces)
        if cleaned and _remove_segmentation_markers_c:
            cleaned = remove_word_segmentation_markers(cleaned)

        if cleaned:
            # Step 3: Check Myanmar text ratio (scope-aware)
            # This calls Python function that calls C function
            is_myanmar = is_myanmar_text(
                cleaned, config=_ingester_zawgyi_config, allow_extended=_allow_extended_myanmar
            )
            
            if is_myanmar:
                # Step 4: Validate text structure
                # Split by whitespace
                words = cleaned.split()
                valid_words = []
                
                # Check each word
                for w in words:
                    # validate_word calls regexes in Python
                    # Uses module-level _allow_extended_myanmar config
                    if validate_word(w, allow_extended_myanmar=_allow_extended_myanmar):
                        valid_words.append(w)
                
                if valid_words:
                    # Reconstruct line
                    cleaned = " ".join(valid_words)
                    results.append((cleaned, True))
                else:
                    results.append(("", False))
            else:
                results.append(("", False))
        else:
            results.append(("", False))
            
    return results
