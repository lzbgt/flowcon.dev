
from common cimport *
from collectors cimport Collector

cdef class Apps(Collector):
	cdef _pcobj
	cdef uint64_t _zeroportcount
	cdef uint64_t _totalcount 
	cdef uint32_t* _portcounts
	cdef float _portrate
	cdef uint32_t _minthreshold
	cdef uint32_t _minactivity

	cdef int _evalports(self, ipfix_apps_ports* ports, uint16_t src, uint16_t dst) nogil
	cdef uint32_t getflowapp(self, const ipfix_flow_tuple* flow, int* ingress) nogil
	cdef void _removeapp(self, ipfix_apps* apprec, uint32_t index) nogil
	cdef void removeapp(self, uint32_t index) nogil
	cdef void collectports(self, const ipfix_store_flow* flows, const ipfix_store_counts* start, uint32_t count) nogil
	cdef void removeports(self, const ipfix_store_flow* flows, const ipfix_store_counts* start, uint32_t count) nogil
	cdef void checkactivity(self) nogil
	cdef void countapp(self, uint32_t index) nogil
	cdef ipfix_apps* _getapprec(self, uint32_t index) nogil
