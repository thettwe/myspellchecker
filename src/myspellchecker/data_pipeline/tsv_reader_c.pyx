# distutils: language=c++
# cython: language_level=3

from libc.string cimport strchr
import os

cdef extern from "<fstream>" namespace "std":
    cdef cppclass ifstream:
        ifstream()
        ifstream(const char* filename)
        void open(const char* filename)
        void close()
        bint is_open()
        bint eof()
        bint fail()
        void getline(char* s, int n, char delim)

# Constants
cdef int MAX_LINE_LEN = 4096
cdef char NEWLINE = b'\n'
cdef int TAB = 9  # ASCII for \t

def read_syllables_tsv(str filepath):
    """
    Read syllables TSV and yield (syllable, frequency).
    Input format: syllable\tfrequency
    """
    cdef ifstream file
    cdef char line[4096]
    cdef char* p_line
    cdef char* p_tab
    cdef bytes b_syl, b_freq
    cdef int freq
    cdef int len_syl
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")
        
    file.open(filepath.encode('utf-8'))
    if not file.is_open():
        raise IOError(f"Could not open {filepath}")
        
    # Skip header
    file.getline(line, MAX_LINE_LEN, NEWLINE)
    
    while not file.eof():
        file.getline(line, MAX_LINE_LEN, NEWLINE)
        if file.fail():
            break
            
        p_line = line
        if p_line[0] == 0:
            continue
            
        p_tab = strchr(p_line, TAB)
        if p_tab == NULL:
            continue
            
        len_syl = p_tab - p_line
        b_syl = p_line[:len_syl]
        b_freq = p_tab + 1
        
        try:
            freq = int(b_freq)
            yield (b_syl.decode('utf-8'), freq)
        except ValueError:
            continue

    file.close()

def read_words_tsv(str filepath):
    """
    Read words TSV and yield (word, syllable_count, frequency, pos_tag).
    Input format: word\tsyllable_count\tfrequency\tpos_tag

    Note: pos_tag may be empty string if not available.
    For backward compatibility, also handles 3-column format (without pos_tag).
    """
    cdef ifstream file
    cdef char line[4096]
    cdef char* p_line
    cdef char* p_tab1
    cdef char* p_tab2
    cdef char* p_tab3
    cdef bytes b_word, b_sc, b_freq, b_pos
    cdef int len_word, len_sc, len_freq
    cdef str pos_tag

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")

    file.open(filepath.encode('utf-8'))
    if not file.is_open():
        raise IOError(f"Could not open {filepath}")

    file.getline(line, MAX_LINE_LEN, NEWLINE)

    while not file.eof():
        file.getline(line, MAX_LINE_LEN, NEWLINE)
        if file.fail(): break

        p_line = line
        if p_line[0] == 0: continue

        p_tab1 = strchr(p_line, TAB)
        if p_tab1 == NULL: continue

        p_tab2 = strchr(p_tab1 + 1, TAB)
        if p_tab2 == NULL: continue

        len_word = p_tab1 - p_line
        len_sc = p_tab2 - (p_tab1 + 1)

        b_word = p_line[:len_word]
        b_sc = (p_tab1 + 1)[:len_sc]

        # Check for 4th column (pos_tag)
        p_tab3 = strchr(p_tab2 + 1, TAB)
        if p_tab3 != NULL:
            # 4-column format: word\tsyllable_count\tfrequency\tpos_tag
            len_freq = p_tab3 - (p_tab2 + 1)
            b_freq = (p_tab2 + 1)[:len_freq]
            b_pos = p_tab3 + 1
            # Strip newline/carriage return from pos tag
            pos_tag = b_pos.decode('utf-8').rstrip('\r\n')
        else:
            # 3-column format (backward compatible): word\tsyllable_count\tfrequency
            b_freq = p_tab2 + 1
            pos_tag = ""

        try:
            yield (b_word.decode('utf-8'), int(b_sc), int(b_freq.rstrip()), pos_tag)
        except ValueError:
            continue

    file.close()

def read_bigrams_tsv(str filepath, dict word_to_id):
    """
    Read bigrams TSV and yield (id1, id2, prob, count).
    Input format: w1\tw2\tprob\tcount
    """
    cdef ifstream file
    cdef char line[4096]
    cdef char* p_line
    cdef char* p_t1
    cdef char* p_t2
    cdef char* p_t3
    cdef bytes b_w1, b_w2, b_prob, b_count
    cdef int len_w1, len_w2, len_prob
    cdef object w1_obj, w2_obj
    cdef object id1_obj, id2_obj
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")
        
    file.open(filepath.encode('utf-8'))
    if not file.is_open():
        raise IOError(f"Could not open {filepath}")
        
    file.getline(line, MAX_LINE_LEN, NEWLINE)
    
    while not file.eof():
        file.getline(line, MAX_LINE_LEN, NEWLINE)
        if file.fail(): break
        p_line = line
        if p_line[0] == 0: continue
        
        p_t1 = strchr(p_line, TAB)
        if p_t1 == NULL: continue
        
        p_t2 = strchr(p_t1 + 1, TAB)
        if p_t2 == NULL: continue
        
        p_t3 = strchr(p_t2 + 1, TAB)
        
        len_w1 = p_t1 - p_line
        len_w2 = p_t2 - (p_t1 + 1)
        
        b_w1 = p_line[:len_w1]
        b_w2 = (p_t1 + 1)[:len_w2]
        
        if p_t3 != NULL:
            len_prob = p_t3 - (p_t2 + 1)
            b_prob = (p_t2 + 1)[:len_prob]
            b_count = p_t3 + 1
        else:
            b_prob = p_t2 + 1
            b_count = b"0"
            
        w1_obj = b_w1.decode('utf-8')
        w2_obj = b_w2.decode('utf-8')
        
        id1_obj = word_to_id.get(w1_obj)
        id2_obj = word_to_id.get(w2_obj)
        
        if id1_obj is None or id2_obj is None:
            continue
            
        try:
            yield (id1_obj, id2_obj, float(b_prob), int(b_count))
        except ValueError:
            continue
            
    file.close()

def read_trigrams_tsv(str filepath, dict word_to_id):
    """
    Read trigrams TSV and yield (id1, id2, id3, prob, count).
    """
    cdef ifstream file
    cdef char line[4096]
    cdef char* p_line
    cdef char* p_t1
    cdef char* p_t2
    cdef char* p_t3
    cdef char* p_t4
    cdef bytes b_w1, b_w2, b_w3, b_prob, b_count
    cdef int len_w1, len_w2, len_w3, len_prob
    cdef object id1_obj, id2_obj, id3_obj
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")
        
    file.open(filepath.encode('utf-8'))
    if not file.is_open():
        raise IOError(f"Could not open {filepath}")
        
    file.getline(line, MAX_LINE_LEN, NEWLINE)
    
    while not file.eof():
        file.getline(line, MAX_LINE_LEN, NEWLINE)
        if file.fail(): break
        p_line = line
        if p_line[0] == 0: continue
        
        p_t1 = strchr(p_line, TAB)
        if p_t1 == NULL: continue
        p_t2 = strchr(p_t1 + 1, TAB)
        if p_t2 == NULL: continue
        p_t3 = strchr(p_t2 + 1, TAB)
        if p_t3 == NULL: continue
        p_t4 = strchr(p_t3 + 1, TAB)
        
        len_w1 = p_t1 - p_line
        len_w2 = p_t2 - (p_t1 + 1)
        len_w3 = p_t3 - (p_t2 + 1)
        
        b_w1 = p_line[:len_w1]
        b_w2 = (p_t1 + 1)[:len_w2]
        b_w3 = (p_t2 + 1)[:len_w3]
        
        if p_t4 != NULL:
            len_prob = p_t4 - (p_t3 + 1)
            b_prob = (p_t3 + 1)[:len_prob]
            b_count = p_t4 + 1
        else:
            b_prob = p_t3 + 1
            b_count = b"0"
            
        id1_obj = word_to_id.get(b_w1.decode('utf-8'))
        id2_obj = word_to_id.get(b_w2.decode('utf-8'))
        id3_obj = word_to_id.get(b_w3.decode('utf-8'))
        
        if id1_obj is None or id2_obj is None or id3_obj is None:
            continue
            
        try:
            yield (id1_obj, id2_obj, id3_obj, float(b_prob), int(b_count))
        except ValueError:
            continue
            
    file.close()