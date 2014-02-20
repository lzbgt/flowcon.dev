
from common cimport *
from collectors cimport FlowCollector
from nquery cimport FlowQuery, QueryBuffer
from napps cimport Apps

cdef class SecondsCollector(object):
    cdef _name
    cdef uint32_t _ip
    cdef _counters
    cdef ipfix_store_counts* _counterset
    cdef ipfix_store_counts* _last
    cdef ipfix_store_counts* _first
    cdef ipfix_store_counts* _end
    cdef uint32_t _maxcount
    cdef uint32_t _count
    cdef uint32_t _depth
    cdef uint32_t [:] _seconds
    cdef uint64_t [:] _stamps
    cdef uint32_t _currentsec
    cdef FlowCollector _flows
	
    cdef void _add(self, uint32_t bytes, uint32_t packets, uint32_t flowindex)
    cdef void _alloc(self, uint32_t size)
    cdef void _grow(self)
    cdef _fixseconds(self, ipfix_store_counts* start, uint32_t offset, uint32_t startsz)
    cdef void _removeold(self, Apps apps, uint32_t lastpos, uint32_t nextpos)
    cdef void _rmold(self, Apps apps, const ipfix_store_counts* start, uint32_t count)
    cdef uint32_t _lookup(self, uint64_t oldeststamp) nogil
    cdef void collect(self, FlowQuery q, QueryBuffer bufinfo, 
    				  uint64_t neweststamp, uint64_t oldeststamp, uint32_t step) nogil
    cdef void _collect(self, FlowQuery q, QueryBuffer bufinfo, ipfix_query_info* qinfo,
                       uint32_t oldestpos, uint32_t lastpos) nogil
    cdef void _initqinfo(self, ipfix_query_info qinfo) nogil
    cdef uint32_t currentpos(self) nogil

cdef class MinutesCollector(object):
    cdef _name
    cdef uint32_t _ip
    cdef uint32_t _prevsecpos
    cdef FlowQuery _query

    cdef int _onapp(self, const ipfix_flow_tuple* flow, ipfix_app_tuple* vals) nogil