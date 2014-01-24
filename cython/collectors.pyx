# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z

#### distutils: library_dirs = 
#### distutils: depends = 

import numpy as np
cimport cython
cimport numpy as np

from common cimport *
from misc cimport logger, showflow, showattr

cdef uint32_t minsize = 16
cdef float growthrate = 2.0
cdef float shrinkrate = 2.0

def _dummy():
    "exists only to get rid of compile warnings"
    cdef int tmp = 0
    if tmp:
        _import_umath()    
        _import_array()

cdef class SecondsCollector(object):

    def __init__(self, nm):
        self._name = nm

        self._alloc(minsize)

        self._count = 0

        self._first = self._counterset
        self._last = self._counterset

    cdef void _alloc(self, uint32_t size):
        self._counters = np.zeros((size, sizeof(ipfix_store_counts)), dtype=np.uint8)
        cdef np.ndarray[np.uint8_t, ndim=2] arr
        arr = self._counters
        self._counterset = <ipfix_store_counts*>arr.data
        self._maxcount = self._counters.size
        self._end = self._counterset + self._maxcount
        
    cdef void _add(self, uint32_t bytes, uint32_t packets, uint32_t flowindex):
        cdef ipfix_store_counts* last
        
        if self._count >= self._maxcount:    # need resize
            self._grow()

        last = self._last
        last.bytes = bytes
        last.packets = packets
        last.flowindex = flowindex

        self._last += 1
        if self._last >= self._end:
            self._last = self. _counterset

        self._count += 1

    cdef void _grow(self):
        cdef uint32_t newsize = <uint32_t>(self._maxcount*growthrate)
        cdef ipfix_store_counts* start = self._counterset
        cdef ipfix_store_counts* end = self._end
        cdef oldcounters = self._counters

        self._alloc(newsize)
        
        # move old data to new location; self._first should be equal to self._last 
        memcpy(self._counterset, self._first)

        self._first = self._counterset
        self._last = self._counterset + self._count
        del oldcounters     # not needed; just in case

cdef class Collector(object):

    def __init__(self, nm, int width, int size = minsize):
        self._name = nm
        self._width = width
        self.freepos = 0
        self.freecount = 0
        self.end = 1
        self.adler = adler32(0, <unsigned char*>0, 0)
        
        self._entries = np.zeros((1, width), dtype=np.uint8)
        self._indices = np.zeros(1, dtype=np.dtype('u4'))
        self._resz(size)
        
    cdef int _resz(self, int size):
        cdef int bits = int(np.math.log(2*size-1, 2))
        cdef int indsize = 2**bits
        cdef np.ndarray[np.uint8_t, ndim=2] arr
        cdef np.ndarray[np.uint32_t, ndim=1] inds
        
        self.maxentries = size
        self.mask = 2**bits-1

        self._entries.resize((size, self._width), refcheck=False)
        arr = self._entries
        self.entryset = <unsigned char*>arr.data

        if indsize != self._indices.size:
            # brand new space for indices
            self._indices = np.zeros(indsize, dtype=self._indices.dtype)
            inds = self._indices
            self.indexset = <uint32_t*>inds.data
            return True
        return False

    cdef uint32_t _add(self, const void* ptr, uint32_t index, int dsize):
        cdef uint32_t crc, pos, ind, lastpos
        cdef ipfix_store_entry* entryrec
        cdef int sz = self._width
        
        #logger("adding to %s"%(self._name))
        crc = adler32(self.adler, <unsigned char*>ptr, sz)
        ind = crc & self.mask
        pos = self.indexset[ind]
        #logger("  crc:%08x ind:%d pos:%d addr:%08x"%(crc, ind, pos, <uint64_t>cython.address(self.indexset[ind])))
        if pos > 0:
            while pos > 0:
                entryrec = <ipfix_store_entry*>(self.entryset+pos*sz)
                if memcmp(entryrec.data, ptr, dsize) == 0:
                    # found identical flow
                    self._onindex(entryrec, index)  # commit index value
                    return pos
                pos = entryrec.next
            # need new
            pos = self._findnewpos(sz)
            #logger("  found:%d"%(pos))
            if pos == 0:    # need resize
                self._grow()
                return self._add(ptr, index, dsize)    # repeat on bigger array

            entryrec.next = pos # link to previous
        else:
            pos = self._findnewpos(sz)
            #logger("  found:%d"%(pos))
            if pos == 0:    # need resize
                self._grow()
                return self._add(ptr, index, dsize)    # repeat on bigger array
            self.indexset[ind] = pos
        
        entryrec = <ipfix_store_entry*>(self.entryset+pos*sz)
        
        entryrec.next = 0 
        entryrec.crc = crc
        memcpy(entryrec.data, ptr, dsize)
        self._onindex(entryrec, index)
        
        return pos

    cdef uint32_t _findnewpos(self, uint32_t sz):
        cdef uint32_t pos = self.freepos
        cdef ipfix_store_entry* entryrec
        
        if pos > 0:
            entryrec = <ipfix_store_entry*>(self.entryset+pos*sz)
            self.freepos = entryrec.next
            self.freecount -= 1
            return pos

        if self.end >= self.maxentries:    # need resize
            return 0
        pos = self.end
        self.end += 1
        return pos
    
    cdef void _grow(self):
        cdef uint32_t size = <uint32_t>(self.maxentries*growthrate)
        self._resize(size)

    cdef void _resize(self, uint32_t size):
        cdef uint32_t count = self.end
        cdef int sz = self._width
        cdef uint32_t mask, ind, indpos, pos
        cdef unsigned char* eset
        cdef uint32_t* iset
        cdef ipfix_store_entry* entry

        logger('resizing %s %d->%d'%(self._name, self.maxentries, size))
        if not self._resz(size): return
        # lets fix indices and links
        mask = self.mask
        eset = self.entryset
        iset = self.indexset
        entry = <ipfix_store_entry*>(eset+1)

        for pos in range(count):
            entry = <ipfix_store_entry*>(eset+pos*sz)
            ind = entry.crc & mask
            entry.next = iset[ind]
            iset[ind] = pos
    
    cdef void _removepos(self, ipfix_store_entry* entryrec, uint32_t pos, int sz):
        cdef ipfix_store_entry* prevrec
        cdef uint32_t ind, prevpos
        cdef unsigned char* eset = self.entryset
        
        ind = entryrec.crc & self.mask
        prevpos = self.indexset[ind]
        if prevpos == pos:
            self.indexset[ind] = entryrec.next
        else:
            prevrec = <ipfix_store_entry*>(eset+prevpos*sz)
            prevpos = prevrec.next
            while prevpos != pos:
                if prevpos == 0:   # this should never happen
                    logger('unexpected value for prevrec.next; prevpos:%d ind:%d pos:%d'%(prevpos, ind, pos))
                    return
                prevrec = <ipfix_store_entry*>(eset+prevpos*sz)
                prevpos = prevrec.next
            prevrec.next = entryrec.next

        entryrec.next = self.freepos
        self.freepos = pos
        self.freecount += 1
    
    cdef ipfix_store_entry* _get(self, int pos):
        return <ipfix_store_entry*>(self.entryset+pos*self._width)

    def entries(self):
        return self._entries.view(dtype=np.dtype(self.dtypes))
    
    def indices(self):
        return self._indices.view(dtype='u4')
    
    cdef void _onindex(self, ipfix_store_entry* entry, uint32_t index):
        pass
    
cdef class FlowCollector(Collector):

    def __init__(self, nm, attribs):
        super(FlowCollector, self).__init__(nm, sizeof(ipfix_store_flow))
        cdef ipfix_store_flow flow
        self._attributes = attribs
        
        self.dtypes = [('next',     'u%d'%sizeof(flow.next)),
                       ('crc',      'u%d'%sizeof(flow.crc)),
                       ('flow',     'a%d'%sizeof(flow.flow)),
                       ('attrindex','u%d'%sizeof(flow.attrindex))]

    cdef void _onindex(self, ipfix_store_entry* entry, uint32_t index):
        cdef ipfix_store_attributes* prev, *curr
        cdef ipfix_store_flow* flowrec = <ipfix_store_flow*>entry
        cdef int report = False # disable reporting for now

        flowrec.refcount += 1

        if flowrec.attrindex != index:
            if flowrec.attrindex != 0 and report:
                prev = <ipfix_store_attributes*>self._attributes._get(flowrec.attrindex)
                curr = <ipfix_store_attributes*>self._attributes._get(index)

                logger('%s changed for flow (%s)\n  <- %s\n  -> %s'%(self._name, showflow(cython.address(flowrec.flow)), 
                                                                     showattr(cython.address(prev.attributes)),
                                                                     showattr(cython.address(curr.attributes))))
            flowrec.attrindex = index

    cdef void remove(self, uint32_t pos):
        cdef unsigned char* eset = self.entryset
        cdef int sz = self._width
        cdef ipfix_store_flow* flow = <ipfix_store_flow*>(eset+pos*sz)
        cdef ipfix_store_flow* nextflow, *prevflow

        if flow.refcount == 0: return # already deleted
        if flow.refcount > 1:
            flow.refcount -= 1
            return # still referenced
        flow.refcount = 0
        
        self._removepos(<ipfix_store_entry*>flow, pos, sz) # delete entry
        
        # let's link in reverse direction;
        # assuming flow is first in free list
        if flow.next != 0:
            nextflow = <ipfix_store_flow*>(eset+flow.next*sz)
            flow.attrindex = nextflow.attrindex
            nextflow.attrindex = pos
        else:
            flow.attrindex = 0
         
        cdef uint32_t last = self.end-1
        if pos == last:  # let's shrink inventory until first non-free record
            while flow.refcount == 0:
                if flow.next > 0:   # fix next
                    nextflow = <ipfix_store_flow*>(eset+flow.next*sz)
                    nextflow.attrindex = flow.attrindex
                if flow.attrindex > 0: # fix previous
                    prevflow = <ipfix_store_flow*>(eset+flow.attrindex*sz)
                    prevflow.next = flow.next
                else:  # it's the first one in the list 
                    self.freepos = flow.next
                self.freecount -= 1
                last -= 1
                if last == 0: break
                flow = <ipfix_store_flow*>(eset+last*sz)
            self.end = last+1
            if last*3 < self.maxentries: # need to shrink
                self._shrink()
        
    cdef void _shrink(self):
        cdef uint32_t newsize = <uint32_t>(self.maxentries/shrinkrate)
        if newsize < minsize: return
        self._resize(newsize)
        
#    cdef _compact(self):
#        cdef uint32_t freepos
#        cdef unsigned char* eset
#        cdef ipfix_store_flow* freerec, *rec
#        cdef int sz = self._width
#        cdef uint32_t last = self.end-1
#        
#        eset = self.entryset
#                
#        freepos = self.freepos
#        while freepos > 0:
#            freerec = <ipfix_store_flow*>(eset+freepos*sz)
#            freepos = freerec.next
#            if last > 0:
#                while last > 0:
#                    rec = <ipfix_store_flow*>(eset+last*sz)
#                    last -= 1
#                    if rec.refcount != 0: break
#            
#        self.freepos = 0
#        self.freecount = 0

cdef class AttrCollector(Collector):

    def __init__(self, nm):
        super(AttrCollector, self).__init__(nm, sizeof(ipfix_store_attributes))
        cdef ipfix_store_attributes attr

        self.dtypes = [('next',      'u%d'%sizeof(attr.next)),
                       ('crc',       'u%d'%sizeof(attr.crc)),
                       ('attributes','a%d'%sizeof(attr.attributes))]        