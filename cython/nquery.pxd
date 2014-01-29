
from common cimport *

ctypedef int (*fchecktype)(const ipfix_flow* flow) nogil
ctypedef void (*freptype)(const ipfix_flow* flow, char* buf, size_t size) nogil

cdef class RawQuery(object):
	cdef bytes qid
	cdef RawQuery next
	cdef RawQuery prev
	cdef void* mod
	cdef fchecktype checker
	cdef freptype reporter
	cdef callback
	
	cdef void onflow(self, const ipfix_flow* flow)