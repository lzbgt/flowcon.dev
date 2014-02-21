
from common cimport *

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
    cdef uint32_t _add(self, const void* ptr, uint32_t index, int dsize) nogil
    cdef ipfix_store_entry* _findentry(self, const void* ptr, uint32_t* outpos, int dsize) nogil    
    cdef uint32_t _findnewpos(self, uint32_t sz) nogil
    cdef void _grow(self)
    cdef void _resize(self, uint32_t size)
    cdef void _removepos(self, ipfix_store_entry* entryrec, uint32_t pos, int sz) nogil
    cdef ipfix_store_entry* _get(self, int pos)
    cdef void _onindex(self, ipfix_store_entry* entry, uint32_t index) nogil

cdef class FlowCollector(Collector):
    cdef AttrCollector _attributes
    
    cdef void _onindex(self, ipfix_store_entry* entry, uint32_t index) nogil
    cdef const ipfix_store_flow* getflows(self) nogil
    cdef void remove(self, const ipfix_store_counts* counts, uint32_t num) nogil
    cdef void _shrink(self, uint32_t maxpos)

cdef class AttrCollector(Collector):
    pass

cdef class AppFlowCollector(Collector):
    cdef AttrCollector _attributes

    cdef void _oningress(self, ipfix_store_entry* entry, uint32_t index) nogil
    cdef void _onegress(self, ipfix_store_entry* entry, uint32_t index) nogil