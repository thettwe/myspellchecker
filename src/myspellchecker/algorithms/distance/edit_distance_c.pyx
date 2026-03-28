# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-optimized edit distance algorithms for Myanmar spell checking.

This module provides high-performance implementations of edit distance algorithms:
- Levenshtein distance (insertion, deletion, substitution)
- Damerau-Levenshtein distance (includes transposition)
- Weighted distance with Myanmar-specific substitution costs

Key Features:
    - UTF-8 aware codepoint handling for Myanmar Unicode
    - Visual similarity weighting for commonly confused characters
    - Keyboard adjacency awareness for typo detection
    - Myanmar-specific substitution costs for linguistic accuracy

Performance:
    - ~10x faster than pure Python implementation
    - Optimized for Myanmar Unicode range (U+1000-U+109F)
    - Memory-efficient row-based dynamic programming

Example:
    >>> from myspellchecker.algorithms.distance import edit_distance_c
    >>> edit_distance_c.c_levenshtein("မြန်", "မြမ်")
    1

Note:
    This module requires compilation. Use `python setup.py build_ext --inplace`
    to compile after modifications.

See Also:
    - edit_distance.py: Python wrapper with fallback implementation
    - phonetic_data.py: Visual similarity and substitution cost data
"""

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set

# Helper to convert UTF-8 string to a vector of Unicode code points
cdef vector[int] utf8_to_codepoints(string s):
    cdef vector[int] res
    cdef int i = 0
    cdef int n = s.length()
    cdef unsigned char c1, c2, c3, c4
    cdef int cp
    
    while i < n:
        c1 = <unsigned char>s[i]
        if c1 < 0x80:
            res.push_back(c1)
            i += 1
        elif (c1 & 0xE0) == 0xC0:
            if i + 1 < n:
                c2 = <unsigned char>s[i+1]
                cp = ((c1 & 0x1F) << 6) | (c2 & 0x3F)
                res.push_back(cp)
                i += 2
            else: i += 1
        elif (c1 & 0xF0) == 0xE0:
            if i + 2 < n:
                c2 = <unsigned char>s[i+1]
                c3 = <unsigned char>s[i+2]
                cp = ((c1 & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F)
                res.push_back(cp)
                i += 3
            else: i += 1
        elif (c1 & 0xF8) == 0xF0:
            if i + 3 < n:
                c2 = <unsigned char>s[i+1]
                c3 = <unsigned char>s[i+2]
                c4 = <unsigned char>s[i+3]
                cp = ((c1 & 0x07) << 18) | ((c2 & 0x3F) << 12) | ((c3 & 0x3F) << 6) | (c4 & 0x3F)
                res.push_back(cp)
                i += 4
            else: i += 1
        else:
            i += 1
    return res

cdef int c_levenshtein_distance(vector[int] s1, vector[int] s2):
    cdef int len1 = s1.size()
    cdef int len2 = s2.size()
    
    if len1 == 0: return len2
    if len2 == 0: return len1
    
    cdef vector[int] prev_row
    cdef vector[int] curr_row
    prev_row.resize(len2 + 1)
    curr_row.resize(len2 + 1)
    
    cdef int i, j, cost
    
    for j in range(len2 + 1):
        prev_row[j] = j
        
    for i in range(1, len1 + 1):
        curr_row[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,
                min(curr_row[j-1] + 1, prev_row[j-1] + cost)
            )
        prev_row = curr_row
        
    return prev_row[len2]

def levenshtein_distance(str s1_py, str s2_py) -> int:
    return c_levenshtein_distance(utf8_to_codepoints(s1_py.encode('utf-8')), 
                                  utf8_to_codepoints(s2_py.encode('utf-8')))

cdef int c_damerau_levenshtein_distance(vector[int] s1, vector[int] s2):
    cdef int len1 = s1.size()
    cdef int len2 = s2.size()
    
    if len1 == 0: return len2
    if len2 == 0: return len1
    
    cdef vector[vector[int]] d
    d.resize(len1 + 1)
    cdef int i, j, cost
    for i in range(len1 + 1):
        d[i].resize(len2 + 1)
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j
        
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,
                min(d[i][j-1] + 1, d[i-1][j-1] + cost)
            )
            if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + 1)
                
    return d[len1][len2]

def damerau_levenshtein_distance(str s1_py, str s2_py) -> int:
    return c_damerau_levenshtein_distance(utf8_to_codepoints(s1_py.encode('utf-8')), 
                                          utf8_to_codepoints(s2_py.encode('utf-8')))

# Weighted Damerau Levenshtein
# We need to pass adjacency and similarity data to C++
cdef unordered_map[int, unordered_set[int]] cpp_visual_similar
cdef unordered_map[int, unordered_set[int]] cpp_keyboard_adjacent
# Myanmar-specific substitution costs
cdef unordered_map[int, unordered_map[int, double]] cpp_myanmar_substitution_costs
# Guard against redundant re-initialization (these should only be set once at import time)
_weighted_data_initialized = False
_substitution_costs_initialized = False

def set_weighted_data(dict visual_sim, dict keyboard_adj):
    """Initialize visual similarity and keyboard adjacency maps.

    WARNING: This function should only be called once at import time.
    Calling it at runtime while other threads are computing edit distances
    is unsafe (module-level C++ maps are not protected by locks).
    """
    global cpp_visual_similar, cpp_keyboard_adjacent, _weighted_data_initialized
    if _weighted_data_initialized:
        return
    cpp_visual_similar.clear()
    cpp_keyboard_adjacent.clear()

    for k, v_set in visual_sim.items():
        k_cp = ord(k)
        for v in v_set:
            cpp_visual_similar[k_cp].insert(ord(v))

    for k, v_set in keyboard_adj.items():
        k_cp = ord(k)
        for v in v_set:
            cpp_keyboard_adjacent[k_cp].insert(ord(v))

    _weighted_data_initialized = True

def set_myanmar_substitution_costs(dict substitution_costs):
    """Set Myanmar-specific substitution costs for weighted edit distance.

    WARNING: This function should only be called once at import time.
    Calling it at runtime while other threads are computing edit distances
    is unsafe (module-level C++ maps are not protected by locks).

    Args:
        substitution_costs: Dict mapping char -> Dict[char, float cost].
            Lower costs indicate more likely character confusions.

    Example:
        >>> set_myanmar_substitution_costs({
        ...     "ည": {"ဉ": 0.1},  # NYA variants - very low cost
        ...     "ရ": {"ယ": 0.3},  # RA/YA confusion
        ... })
    """
    global cpp_myanmar_substitution_costs, _substitution_costs_initialized
    if _substitution_costs_initialized:
        return
    cpp_myanmar_substitution_costs.clear()

    for k, v_dict in substitution_costs.items():
        k_cp = ord(k)
        for v, cost in v_dict.items():
            cpp_myanmar_substitution_costs[k_cp][ord(v)] = cost

    _substitution_costs_initialized = True

cdef inline bint _is_myanmar_diacritic(int cp) noexcept nogil:
    """Check if codepoint is a Myanmar diacritical mark (reduced indel cost).
    NOTE: Asat (0x103A) intentionally excluded — it changes word identity."""
    return (cp == 0x1036 or cp == 0x1037 or cp == 0x1038
            or (0x103B <= cp <= 0x103E))

# Reduced insertion/deletion cost for diacritics (matching Python fallback)
cdef double DIACRITIC_INDEL_COST = 0.5

cdef double c_weighted_damerau_levenshtein_distance(vector[int] s1, vector[int] s2, double keyboard_weight, double visual_weight, double transposition_weight):
    cdef int len1 = s1.size()
    cdef int len2 = s2.size()

    if len1 == 0: return <double>len2
    if len2 == 0: return <double>len1

    cdef vector[vector[double]] d
    d.resize(len1 + 1)
    cdef int i, j
    cdef double cost, del_cost, ins_cost
    for i in range(len1 + 1):
        d[i].resize(len2 + 1)
        d[i][0] = <double>i
    for j in range(len2 + 1):
        d[0][j] = <double>j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i-1] == s2[j-1]:
                cost = 0.0
            else:
                # Calculate substitution cost with Myanmar-specific weights
                cost = 1.0
                # 1. Check Myanmar-specific substitution costs first (most precise)
                if cpp_myanmar_substitution_costs.count(s1[i-1]) and cpp_myanmar_substitution_costs[s1[i-1]].count(s2[j-1]):
                    cost = min(cost, cpp_myanmar_substitution_costs[s1[i-1]][s2[j-1]])
                # 2. Check visual similarity (bidirectional lookup)
                elif cpp_visual_similar.count(s1[i-1]) and cpp_visual_similar[s1[i-1]].count(s2[j-1]):
                    cost = min(cost, visual_weight)
                # 3. Check keyboard adjacency
                elif cpp_keyboard_adjacent.count(s1[i-1]) and cpp_keyboard_adjacent[s1[i-1]].count(s2[j-1]):
                    cost = min(cost, keyboard_weight)

            # Diacritical marks use reduced insertion/deletion cost
            del_cost = DIACRITIC_INDEL_COST if _is_myanmar_diacritic(s1[i-1]) else 1.0
            ins_cost = DIACRITIC_INDEL_COST if _is_myanmar_diacritic(s2[j-1]) else 1.0
            d[i][j] = min(
                d[i-1][j] + del_cost,
                min(d[i][j-1] + ins_cost, d[i-1][j-1] + cost)
            )
            # Transposition - use transposition_weight instead of hardcoded 1.0
            # Common in Myanmar: "ာေ" vs "ော" (vowel ordering confusion)
            if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + transposition_weight)

    return d[len1][len2]

def weighted_damerau_levenshtein_distance(str s1_py, str s2_py, double keyboard_weight=0.5, double visual_weight=0.5, double transposition_weight=0.7) -> float:
    return c_weighted_damerau_levenshtein_distance(utf8_to_codepoints(s1_py.encode('utf-8')),
                                                   utf8_to_codepoints(s2_py.encode('utf-8')),
                                                   keyboard_weight, visual_weight, transposition_weight)

# Myanmar Syllable Aware Units
# Kinzi sequence: NGA (0x1004) + ASAT (0x103A) + VIRAMA (0x1039)
cdef vector[vector[int]] c_tokenize_myanmar_units(vector[int] cps):
    cdef vector[vector[int]] units
    cdef int i = 0
    cdef int n = cps.size()
    cdef int cp
    cdef vector[int] unit

    while i < n:
        cp = cps[i]
        unit.clear()

        # Check for Kinzi sequence: NGA (0x1004) + ASAT (0x103A) + VIRAMA (0x1039)
        # Kinzi modifies the following consonant, so group them together
        if (cp == 0x1004 and i + 2 < n
                and cps[i + 1] == 0x103A and cps[i + 2] == 0x1039):
            # Check if there's a consonant after Kinzi
            if i + 3 < n and 0x1000 <= cps[i + 3] <= 0x1021:
                unit.push_back(cps[i])      # NGA
                unit.push_back(cps[i + 1])  # ASAT
                unit.push_back(cps[i + 2])  # VIRAMA
                unit.push_back(cps[i + 3])  # Host consonant
                i += 4
                # Collect any following medials
                while i < n and 0x103B <= cps[i] <= 0x103E:
                    unit.push_back(cps[i])
                    i += 1
                units.push_back(unit)
                continue
            # If no consonant follows, fall through to normal handling

        # Myanmar Consonants: 0x1000 to 0x1021
        if 0x1000 <= cp <= 0x1021:
            unit.push_back(cp)
            i += 1
            # Check for medials: 0x103B to 0x103E
            while i < n and 0x103B <= cps[i] <= 0x103E:
                unit.push_back(cps[i])
                i += 1
            units.push_back(unit)
        else:
            unit.push_back(cp)
            units.push_back(unit)
            i += 1

    return units

cdef bint _is_confusable_pair(int m1, int m2) noexcept nogil:
    """Check if two medials are a known confusion pair."""
    if (m1 == 0x103B and m2 == 0x103C) or (m1 == 0x103C and m2 == 0x103B):
        return True
    if (m1 == 0x103D and m2 == 0x103E) or (m1 == 0x103E and m2 == 0x103D):
        return True
    return False

cdef bint c_is_medial_confusion(vector[int] u1, vector[int] u2):
    """Check if two units differ only by a commonly confused medial.

    Handles both single-medial (size 2) and multi-medial (size 3+) units
    by computing set difference of medials, matching the Python fallback.
    """
    if u1.size() < 2 or u2.size() < 2: return False
    if u1[0] != u2[0]: return False

    cdef int i, j, m1, m2
    cdef vector[int] only_in_u1, only_in_u2
    cdef bint found

    # Extract medials unique to each unit via set difference
    for i in range(1, <int>u1.size()):
        found = False
        for j in range(1, <int>u2.size()):
            if u1[i] == u2[j]:
                found = True
                break
        if not found:
            only_in_u1.push_back(u1[i])

    for i in range(1, <int>u2.size()):
        found = False
        for j in range(1, <int>u1.size()):
            if u2[i] == u1[j]:
                found = True
                break
        if not found:
            only_in_u2.push_back(u2[i])

    # Exactly one differing medial in each direction
    if only_in_u1.size() == 1 and only_in_u2.size() == 1:
        return _is_confusable_pair(only_in_u1[0], only_in_u2[0])

    return False

def myanmar_syllable_edit_distance(str s1_py, str s2_py) -> tuple:
    cdef vector[int] cps1 = utf8_to_codepoints(s1_py.encode('utf-8'))
    cdef vector[int] cps2 = utf8_to_codepoints(s2_py.encode('utf-8'))
    
    cdef vector[vector[int]] units1 = c_tokenize_myanmar_units(cps1)
    cdef vector[vector[int]] units2 = c_tokenize_myanmar_units(cps2)
    
    cdef int n1 = units1.size()
    cdef int n2 = units2.size()
    
    if n1 == 0: return (n2, float(n2))
    if n2 == 0: return (n1, float(n1))
    
    cdef vector[vector[int]] d_int
    cdef vector[vector[double]] d_weighted
    d_int.resize(n1 + 1)
    d_weighted.resize(n1 + 1)
    
    cdef int i, j, cost_int
    cdef double cost_weighted
    
    for i in range(n1 + 1):
        d_int[i].resize(n2 + 1)
        d_weighted[i].resize(n2 + 1)
        d_int[i][0] = i
        d_weighted[i][0] = float(i)
        
    for j in range(n2 + 1):
        d_int[0][j] = j
        d_weighted[0][j] = float(j)
        
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if units1[i-1] == units2[j-1]:
                cost_int = 0
                cost_weighted = 0.0
            elif c_is_medial_confusion(units1[i-1], units2[j-1]):
                cost_int = 1
                cost_weighted = 0.5
            else:
                cost_int = 1
                cost_weighted = 1.0
                
            d_int[i][j] = min(
                d_int[i-1][j] + 1,
                min(d_int[i][j-1] + 1, d_int[i-1][j-1] + cost_int)
            )
            d_weighted[i][j] = min(
                d_weighted[i-1][j] + 1.0,
                min(d_weighted[i][j-1] + 1.0, d_weighted[i-1][j-1] + cost_weighted)
            )
            
            if i > 1 and j > 1 and units1[i-1] == units2[j-2] and units1[i-2] == units2[j-1]:
                d_int[i][j] = min(d_int[i][j], d_int[i-2][j-2] + 1)
                d_weighted[i][j] = min(d_weighted[i][j], d_weighted[i-2][j-2] + 1.0)
                
    return (d_int[n1][n2], d_weighted[n1][n2])
