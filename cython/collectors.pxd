
from common cimport *

cdef class SecondsCollector(object):
    cdef _name
    cdef _counters
    cdef ipfix_store_counts* _counterset
    cdef ipfix_store_counts* _last
    cdef ipfix_store_counts* _first
    cdef ipfix_store_counts* _end
    cdef uint32_t _maxcount
    cdef uint32_t _count
	
    cdef void _add(self, uint32_t bytes, uint32_t packets, uint32_t flowindex)
    cdef void _alloc(self, uint32_t size)
    cdef void _grow(self)

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
    cdef void remove(self, uint32_t pos)
    cdef void _shrink(self)

cdef class AttrCollector(Collector):
    pass
