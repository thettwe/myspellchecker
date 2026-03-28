# distutils: language=c++
from libcpp.string cimport string
from libcpp.vector cimport vector

cpdef tuple viterbi(str text, str prev=*, int maxlen=*)
