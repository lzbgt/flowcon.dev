
from common cimport *
from collectors cimport Collector

cdef class Apps(Collector):
	cdef _pcobj
	cdef uint64_t _zeroportcount
	cdef uint64_t _totalcount 
	cdef uint32_t* _portcounts
	cdef float _portrate
	cdef uint32_t _minthreshold

	cdef uint32_t getflowapp(self, const ipfix_flow_tuple* flow, int* ingress) nogil
	cdef void collect(self, const ipfix_store_flow* flows, const ipfix_store_counts* start, uint32_t count) nogil
	cdef void remove(self, const ipfix_store_flow* flows, const ipfix_store_counts* start, uint32_t count) nogil
	cdef void reduce(self) nogil