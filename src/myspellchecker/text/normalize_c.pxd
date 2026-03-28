# distutils: language=c++
from libcpp.string cimport string
from libcpp.vector cimport vector

cpdef list segment_syllables_c(str text, bint allow_extended=*)
cpdef bint is_myanmar_string(str text)
cpdef bint is_myanmar_string_scoped(str text, bint allow_extended=*)
cpdef str reorder_myanmar_diacritics(str text)
cpdef str remove_zero_width_chars(str text)
cpdef str clean_text_for_segmentation(str text, bint allow_extended=*)
