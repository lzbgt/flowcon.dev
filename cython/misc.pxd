
from common cimport *

cdef logger
cdef object showflow(const ipfix_flow_tuple* ftup)
cdef object showattr(const ipfix_attributes* attr)
cdef void checkbytes(const unsigned char* buf, const char* pref, const char* nm, uint32_t nbytes)