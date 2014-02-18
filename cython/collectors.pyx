# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z

#### distutils: library_dirs = 
#### distutils: depends = 


#TMP
#import sys
#
import numpy as np
cimport cython
cimport numpy as np

from common cimport *
from misc cimport logger, showflow, showattr
from nquery cimport FlowQuery

cdef uint32_t INVALID = (<uint32_t>(-1))

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

    def __init__(self, nm, uint32_t ip, FlowCollector flows, uint32_t secondsdepth, uint32_t stamp):
        cdef uint32_t pos
        self._name = nm
        self._ip = ip
        self._flows = flows
        self._depth = secondsdepth
        self._seconds = np.zeros(secondsdepth, dtype=np.uint32)
        self._stamps = np.zeros(secondsdepth, dtype=np.uint64)
        
        for pos in range(secondsdepth):
            self._seconds[pos] = INVALID
        
        self._currentsec = 0
        self._seconds[self._currentsec] = 0
        self._stamps[self._currentsec] = stamp

        self._alloc(minsize)

        self._count = 0

        self._first = self._counterset
        self._last = self._counterset

    @cython.boundscheck(False)
    cdef void _alloc(self, uint32_t size):
        self._counters = np.zeros((size, sizeof(ipfix_store_counts)), dtype=np.uint8)
        cdef np.ndarray[np.uint8_t, ndim=2] arr
        arr = self._counters
        self._counterset = <ipfix_store_counts*>arr.data
        self._maxcount = size
        self._end = self._counterset + self._maxcount
        
    @cython.boundscheck(False)    
    cdef void _add(self, uint32_t bytes, uint32_t packets, uint32_t flowindex):
        cdef ipfix_store_counts* last
        
        if self._count >= self._maxcount:    # need resize
            self._grow()
            if self._count >= self._maxcount:
                logger("can not grow time counters for %s"%(self._name))
                return

        last = self._last
        last.bytes = bytes
        last.packets = packets
        last.flowindex = flowindex

        self._last += 1
        if self._last >= self._end:
            self._last = self. _counterset

        self._count += 1

    @cython.boundscheck(False)
    cdef void _grow(self):
        cdef uint32_t pos, secpos, offset, startsz, newsize = <uint32_t>(self._maxcount*growthrate)
        cdef ipfix_store_counts* start = self._counterset
        cdef ipfix_store_counts* end = self._end
        cdef oldcounters = self._counters
        cdef uint64_t startbytes, endbytes

#        logger("%s: growing time counters %d->%d"%(self._name, self._maxcount, newsize))
        
        self._alloc(newsize)
        
        # move old data to new location
        offset = <uint32_t>(((<uint64_t>self._first) - (<uint64_t>start))/sizeof(ipfix_store_counts));

        if self._first >= self._last:
            startbytes = (<uint64_t>end) - (<uint64_t>self._first);
            startsz = <uint32_t>(startbytes/sizeof(ipfix_store_counts))
            if startbytes > 0:
                memcpy(self._counterset, self._first, startbytes)
            endbytes = (<uint64_t>self._last) - (<uint64_t>start);
            if endbytes > 0:
                memcpy((<unsigned char*>self._counterset)+startbytes, start, endbytes)
            # update pointers to where relocated data is sitting now
            self._fixseconds(start, offset, startsz)
        else:
            startbytes = (<uint64_t>self._last) - (<uint64_t>self._first);
            memcpy(self._counterset, self._first, startbytes)
            # update pointers to where relocated data is sitting now
            for pos in range(self._depth):
                secpos = self._seconds[pos]
                if secpos == INVALID: continue
                self._seconds[pos] = secpos-offset
        
        self._first = self._counterset
        self._last = self._counterset + self._count
        
        del oldcounters     # not needed; just in case
        
    @cython.boundscheck(False)
    cdef _fixseconds(self, ipfix_store_counts* start, uint32_t offset, uint32_t startsz):
        cdef uint32_t pos, secpos, prevpos = 0
        
        pos = self._currentsec+1    # start from oldest second
        if pos >= self._depth: pos = 0
        for _ in range(self._depth):
            secpos = self._seconds[pos]
            if secpos != INVALID: 
                if (start+secpos) >= self._first:   # it is in upper chunk
                    secpos -= offset                # shift down
                    if secpos == 0 and prevpos > 0:
                        secpos += offset + startsz    # revert and shift up because it is last
                    self._seconds[pos] = secpos
                else:                               # it is in lower chunk
                    secpos += startsz               # shift up
                    self._seconds[pos] = secpos
                prevpos = secpos
            pos += 1
            if pos >= self._depth: pos = 0

    @cython.boundscheck(False)
    def onsecond(self, uint64_t stamp):
        cdef uint32_t newloc, next
        cdef uint32_t current = self._currentsec
        
        current += 1        # oldest seconds position
        if current >= self._depth:
            current = 0

        next = current + 1  # next to oldest; this is where oldest ends
        if next >= self._depth:
            next = 0
        
        self._removeold(self._seconds[current], self._seconds[next])
        
        newloc = <uint32_t>(((<uint64_t>self._last) - (<uint64_t>self._counterset))/sizeof(ipfix_store_counts))
        
        self._seconds[current] = newloc
        self._stamps[current] = stamp
        self._currentsec = current
    
    @cython.boundscheck(False)
    cdef uint32_t _lookup(self, uint64_t oldeststamp) nogil:
        # seconds may not be evenly distributed; lets jump to approximate location and then lookup
        cdef uint32_t prevpos, secpos = self._currentsec
        cdef uint64_t stamp = self._stamps[secpos]
        cdef uint64_t prevstamp, seconds = stamp - oldeststamp

        if secpos < seconds:
            if seconds >= self._depth:
                secpos += 1 # just get oldest we have
                if secpos >= self._depth: secpos = 0
            else:
                secpos = self._depth + secpos - seconds
        else:
            secpos -= seconds

        stamp = self._stamps[secpos]
        if stamp <= oldeststamp:
            # look forward; we will find something >= oldeststamp for sure since current stamp is bigger
            while stamp < oldeststamp:
                secpos += 1
                if secpos >= self._depth: secpos = 0
                stamp = self._stamps[secpos]
        else:
            # look backward; stamp is too big let's find something smaller then oldeststamp
            prevpos = secpos   # to avoid compiler warning
            while stamp >= oldeststamp:
                prevpos = secpos
                if secpos == 0: secpos = self._depth
                secpos -= 1
                prevstamp = stamp
                stamp = self._stamps[secpos]
                # go back only until stamps are decreasing
                # need this in case if oldeststamp is older than anything we have
                if stamp >= prevstamp: break
            secpos = prevpos               # this is last known good position

        return secpos

    @cython.boundscheck(False)
    cdef void collect(self, FlowQuery q, QueryBuffer bufinfo, 
                      uint64_t neweststamp, uint64_t oldeststamp, uint32_t step, void* data) nogil:

        cdef uint32_t curpos, oldestpos, lastpos = self._currentsec
        cdef uint64_t nwstamp, currentstamp = self._stamps[lastpos]

        if currentstamp <= oldeststamp: return      # current or future seconds are not available yet

        if neweststamp > currentstamp: 
            pass                                    # can not go into future
        elif neweststamp > oldeststamp:             # there is something to collect
            lastpos = self._lookup(neweststamp)     # newest is also back in history somewhere
        else:
            return                                  # nothing to collect
            
        oldestpos = self._lookup(oldeststamp)
        
        cdef ipfix_query_info qinfo
        
        qinfo.flows = <ipfix_store_flow*>self._flows.entryset
        qinfo.attrs = <ipfix_store_attributes*>self._flows._attributes.entryset
        
        qinfo.exporter = self._ip
        
        if step == 0:
            qinfo.stamp = neweststamp
            self._collect(q, bufinfo, data, cython.address(qinfo), oldestpos, lastpos)
        else:
            nwstamp = self._stamps[lastpos]
            currentstamp = oldeststamp+step
            while currentstamp < nwstamp:
                curpos = self._lookup(currentstamp)
                if self._stamps[curpos] >= nwstamp: break
                qinfo.stamp = currentstamp
                self._collect(q, bufinfo, data, cython.address(qinfo), oldestpos, curpos)
                oldestpos = curpos
                currentstamp = currentstamp+step

            qinfo.stamp = neweststamp
            self._collect(q, bufinfo, data, cython.address(qinfo), oldestpos, lastpos)

    @cython.boundscheck(False)
    cdef void _collect(self, FlowQuery q, QueryBuffer bufinfo, void* data, ipfix_query_info* qinfo,
                       uint32_t oldestpos, uint32_t lastpos) nogil:
        cdef uint32_t oldestloc, lastloc

        lastloc = self._seconds[lastpos]
        oldestloc = self._seconds[oldestpos]

        # here we have secpos pointing to proper position where we need to start collection

        if lastloc >= oldestloc:  # one chunk of data
            qinfo.first = self._counterset+oldestloc
            qinfo.count = lastloc-oldestloc
            q.collect(bufinfo, qinfo, data)
        else:
            # two chunks of data
            # collect older
            qinfo.first = self._counterset+oldestloc
            qinfo.count = self._maxcount-oldestloc
            q.collect(bufinfo, qinfo, data)
            # collect newer
            qinfo.first = self._counterset
            qinfo.count = lastloc
            q.collect(bufinfo, qinfo, data)
        
    cdef void _removeold(self, uint32_t lastloc, uint32_t nextloc):
        cdef ipfix_store_counts* start
        cdef uint32_t count
        # need to remove oldest seconds if we already have valid points
        if lastloc == INVALID: return

        start = self._counterset+lastloc
        if nextloc >= lastloc:
            count = nextloc - lastloc
            if count > 0:
                self._flows.remove(start, count)
        else:
            # it's split into two chunks; remove one then another 
            count = self._maxcount-lastloc
            if count > 0:
                self._flows.remove(start, count)
            # count = nextloc - 0
            if nextloc > 0:
                self._flows.remove(self._counterset, nextloc) # from very beginning of buffer to nextloc
            count += nextloc
        self._first = self._counterset+nextloc
        self._count -= count

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
        
    @cython.boundscheck(False)
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

    @cython.boundscheck(False)
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

    @cython.boundscheck(False)
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
    
    @cython.boundscheck(False)
    cdef void _grow(self):
        cdef uint32_t size = <uint32_t>(self.maxentries*growthrate)
        self._resize(size)

    @cython.boundscheck(False)
    cdef void _resize(self, uint32_t size):
        cdef uint32_t count = self.end
        cdef int sz = self._width
        cdef uint32_t mask, ind, indpos, pos
        cdef unsigned char* eset
        cdef uint32_t* iset
        cdef ipfix_store_entry* entry

        #logger('resizing %s %d->%d'%(self._name, self.maxentries, size))
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
    
    @cython.boundscheck(False)
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
    
    @cython.boundscheck(False)
    cdef ipfix_store_entry* _get(self, int pos):
        return <ipfix_store_entry*>(self.entryset+pos*self._width)

    def entries(self):
        return self._entries.view(dtype=np.dtype(self.dtypes))
    
    def indices(self):
        return self._indices.view(dtype='u4')
    
    @cython.boundscheck(False)
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

    @cython.boundscheck(False)
    cdef void _onindex(self, ipfix_store_entry* entry, uint32_t index):
        cdef ipfix_store_attributes* prev
        cdef ipfix_store_attributes* curr
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

    @cython.boundscheck(False)
    cdef void remove(self, const ipfix_store_counts* counts, uint32_t num):
        cdef unsigned char* eset = self.entryset
        cdef int sz = self._width
        cdef ipfix_store_flow* flow
        cdef ipfix_store_flow* nextflow
        cdef uint32_t pos, maxindex = 0, index

        for pos in range(num):
            index = counts[pos].flowindex

            flow = <ipfix_store_flow*>(eset+index*sz)

            if flow.refcount == 0:  # already deleted 
                continue
            if flow.refcount > 1:   # still referenced
                flow.refcount -= 1
                continue

            flow.refcount = 0
            if maxindex < index: maxindex = index
            
            self._removepos(<ipfix_store_entry*>flow, index, sz) # delete entry
            # let's link removed flow in reverse direction;
            # assuming flow is first in free list
            if flow.next != 0:
                nextflow = <ipfix_store_flow*>(eset+flow.next*sz)
                flow.attrindex = nextflow.attrindex
                nextflow.attrindex = index
            else:
                flow.attrindex = 0
        
        if maxindex > 0:            # something was actually deleted
            self._shrink(maxindex)    # try to shrink
        
    @cython.boundscheck(False)
    cdef void _shrink(self, uint32_t maxpos):
        cdef uint32_t newsize
        cdef uint32_t last = self.end-1

        if maxpos != last: return

        cdef unsigned char* eset = self.entryset        
        cdef uint32_t sz = self._width
        cdef ipfix_store_flow* flow = <ipfix_store_flow*>(eset+maxpos*sz)
        cdef ipfix_store_flow* nextflow
        cdef ipfix_store_flow* prevflow

        # let's shrink inventory until first non-free record
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
        if last*3 < self.maxentries: # need to shrink whole buffer to release some unused memory
            newsize = <uint32_t>(self.maxentries/shrinkrate)
            if newsize < minsize: return
            self._resize(newsize)
        
cdef class AttrCollector(Collector):

    def __init__(self, nm):
        super(AttrCollector, self).__init__(nm, sizeof(ipfix_store_attributes))
        cdef ipfix_store_attributes attr

        self.dtypes = [('next',      'u%d'%sizeof(attr.next)),
                       ('crc',       'u%d'%sizeof(attr.crc)),
                       ('attributes','a%d'%sizeof(attr.attributes))]        
