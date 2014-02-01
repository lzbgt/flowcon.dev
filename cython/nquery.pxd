
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
	cdef uint32_t _width
	cdef ipfix_query_buf _buf
	cdef ipfix_query_pos _positions
	
	cdef void init(self, uint32_t width)
	cdef void grow(self)
	cdef const ipfix_query_buf* getbuf(self) nogil
	cdef ipfix_query_pos* getposes(self) nogil
	
cdef class PeriodicQuery(Query):
	cdef fexpchecktype expchecker
	cdef uint32_t _width
	
	cdef void collect(self, QueryBuffer bufinfo, const ipfix_query_info* info, uint32_t expip) nogil
