
from common cimport *

cdef uint32_t minsize
cdef float growthrate
cdef float shrinkrate

cdef logger
cdef object showflow(const ipfix_flow_tuple* ftup)
cdef object showattr(const ipfix_attributes* attr)
cdef void checkbytes(const unsigned char* buf, const char* pref, const char* nm, uint32_t nbytes)
cdef object showapp(const ipfix_app_tuple* atup)
cdef getreqval(req, nm, default)
cdef debentries(req, n, flows, nms, chks)
