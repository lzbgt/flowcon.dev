
from common cimport *

ctypedef int (*fchecktype)(const ipfix_flow* flow) nogil

cdef class RawQuery(object):
	cdef RawQuery next
	cdef RawQuery prev
	cdef void* mod
	cdef fchecktype checker
	
	cdef int onflow(self, const ipfix_flow* flow)