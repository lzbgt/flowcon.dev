
from common cimport *

ctypedef int (*fcheckrawtype)(const ipfix_flow* flow) nogil
ctypedef void (*freprawtype)(const ipfix_flow* flow, char* buf, size_t size) nogil
ctypedef int (*fexpchecktype)(uint32_t ip) nogil

cdef class Query(object):
	cdef bytes qid
	cdef void* mod
	cdef void* checker
	cdef void* reporter
	
cdef class RawQuery(Query):
	cdef RawQuery next
	cdef RawQuery prev
	cdef callback
	
	cdef void onflow(self, const ipfix_flow* flow)

cdef class QueryBuffer(object):
	cdef _entries
	cdef _poses
	cdef uint32_t _width
	cdef uint32_t _sizehint
	cdef ipfix_query_buf _buf
	cdef ipfix_query_pos _positions
	
	cdef const ipfix_query_buf* init(self, uint32_t width, uint32_t sizehint)
	cdef void grow(self)
	cdef ipfix_query_pos* getposes(self) nogil
	cdef void release(self, const ipfix_query_buf* buf)
	
cdef class PeriodicQuery(Query):
	cdef fexpchecktype expchecker
	cdef uint32_t _width
	cdef uint32_t _sizehint
	
	cdef void collect(self, QueryBuffer bufinfo, const ipfix_query_info* info, uint32_t expip, void* data) nogil
