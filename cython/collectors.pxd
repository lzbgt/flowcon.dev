
from common cimport *
from nquery cimport FlowQuery, QueryBuffer

cdef class Collector(object):
    cdef unsigned char* entryset
    cdef uint32_t adler
    cdef uint32_t end
    cdef uint32_t* indexset
    cdef uint32_t mask, maxentries, freepos, freecount
    cdef int _width
    cdef _entries
    cdef _indices
    cdef dtypes
    cdef _name

    cdef int _resz(self, int size)
    cdef uint32_t _add(self, const void* ptr, uint32_t index, int dsize)
    cdef uint32_t _findnewpos(self, uint32_t sz)
    cdef void _grow(self)
    cdef void _resize(self, uint32_t size)
    cdef void _removepos(self, ipfix_store_entry* entryrec, uint32_t pos, int sz)
    cdef ipfix_store_entry* _get(self, int pos)
    cdef void _onindex(self, ipfix_store_entry* entry, uint32_t index)

cdef class FlowCollector(Collector):
    cdef AttrCollector _attributes
    
    cdef void _onindex(self, ipfix_store_entry* entry, uint32_t index)
    cdef void remove(self, const ipfix_store_counts* counts, uint32_t num)
    cdef void _shrink(self, uint32_t maxpos)

cdef class AttrCollector(Collector):
    pass

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
    cdef void _removeold(self, uint32_t lastpos, uint32_t nextpos)
    cdef uint32_t _lookup(self, uint64_t oldeststamp) nogil
    cdef void collect(self, FlowQuery q, QueryBuffer bufinfo, 
    				  uint64_t neweststamp, uint64_t oldeststamp, uint32_t step, void* data) nogil
    cdef void _collect(self, FlowQuery q, QueryBuffer bufinfo, void* data, ipfix_query_info* qinfo,
                       uint32_t oldestpos, uint32_t lastpos) nogil
