# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z

import numpy as np
cimport cython
cimport numpy as np

from common cimport *

#from collectors
from misc cimport logger, minsize, growthrate, shrinkrate
from napps cimport Apps

cdef uint32_t INVALID = (<uint32_t>(-1))

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
    def onsecond(self, Apps apps, uint64_t stamp):
        cdef uint32_t prevloc, newloc, next
        cdef uint32_t current = self._currentsec
        
        prevloc = self._seconds[current]
        current += 1        # oldest seconds position
        if current >= self._depth:
            current = 0

        next = current + 1  # next to oldest; this is where oldest ends
        if next >= self._depth:
            next = 0
        
        self._removeold(apps, self._seconds[current], self._seconds[next])
        
        newloc = <uint32_t>(((<uint64_t>self._last) - (<uint64_t>self._counterset))/sizeof(ipfix_store_counts))
        
        self._seconds[current] = newloc
        self._stamps[current] = stamp
        self._currentsec = current
        
        if prevloc == INVALID: prevloc = 0

        cdef const ipfix_store_flow* flows = self._flows.getflows()

        if newloc >= prevloc:  # one chunk of data
            apps.collect(flows, self._counterset+prevloc, newloc-prevloc)
        else:
            # two chunks of data
            # collect older
            apps.collect(flows, self._counterset+prevloc, self._maxcount-prevloc)
            # collect newer
            apps.collect(flows, self._counterset, newloc)
    
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
                      uint64_t neweststamp, uint64_t oldeststamp, uint32_t step) nogil:

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
        
        self._initqinfo(cython.address(qinfo))

        if step == 0:
            qinfo.stamp = neweststamp
            self._collect(q, bufinfo, cython.address(qinfo), oldestpos, lastpos)
        else:
            nwstamp = self._stamps[lastpos]
            currentstamp = oldeststamp+step
            while currentstamp < nwstamp:
                curpos = self._lookup(currentstamp)
                if self._stamps[curpos] >= nwstamp: break
                qinfo.stamp = currentstamp
                self._collect(q, bufinfo, cython.address(qinfo), oldestpos, curpos)
                oldestpos = curpos
                currentstamp = currentstamp+step

            qinfo.stamp = neweststamp
            self._collect(q, bufinfo, cython.address(qinfo), oldestpos, lastpos)

    @cython.boundscheck(False)
    cdef uint32_t currentpos(self) nogil:
        return self._currentsec

    @cython.boundscheck(False)
    cdef void _initqinfo(self, ipfix_query_info* qinfo) nogil:
        qinfo.flows = <ipfix_store_flow*>self._flows.entryset
        qinfo.attrs = <ipfix_store_attributes*>self._flows._attributes.entryset
        
        qinfo.exporter = self._ip

    @cython.boundscheck(False)
    cdef void _collect(self, FlowQuery q, QueryBuffer bufinfo, ipfix_query_info* qinfo,
                       uint32_t oldestpos, uint32_t lastpos) nogil:
        cdef uint32_t oldestloc, lastloc

        lastloc = self._seconds[lastpos]
        oldestloc = self._seconds[oldestpos]

        # here we have secpos pointing to proper position where we need to start collection

        if lastloc >= oldestloc:  # one chunk of data
            qinfo.first = self._counterset+oldestloc
            qinfo.count = lastloc-oldestloc
            q.collect(bufinfo, qinfo)
        else:
            # two chunks of data
            # collect older
            qinfo.first = self._counterset+oldestloc
            qinfo.count = self._maxcount-oldestloc
            q.collect(bufinfo, qinfo)
            # collect newer
            qinfo.first = self._counterset
            qinfo.count = lastloc
            q.collect(bufinfo, qinfo)
    
    @cython.boundscheck(False)    
    cdef void _removeold(self, Apps apps, uint32_t lastloc, uint32_t nextloc):
        cdef ipfix_store_counts* start
        cdef uint32_t count
        # need to remove oldest seconds if we already have valid points
        if lastloc == INVALID: return

        start = self._counterset+lastloc
        if nextloc >= lastloc:
            count = nextloc - lastloc
            if count > 0:
                self._rmold(apps, start, count)
        else:
            # it's split into two chunks; remove one then another 
            count = self._maxcount-lastloc
            if count > 0:
                self._rmold(apps, start, count)
            # count = nextloc - 0
            if nextloc > 0:
                self._rmold(apps, self._counterset, nextloc) # from very beginning of buffer to nextloc
            count += nextloc
        self._first = self._counterset+nextloc
        self._count -= count
    
    @cython.boundscheck(False)
    cdef void _rmold(self, Apps apps, const ipfix_store_counts* start, uint32_t count):
        apps.remove(self._flows.getflows(), start, count)
        self._flows.remove(start, count)


cdef int onappcallback(void* obj, const ipfix_store_flow* flowentry, AppFlowValues* vals) nogil:
    cdef AppFlowObjects* objects = <AppFlowObjects*>obj
    
    return (<MinutesCollector>objects.minutes)._onapp(<Apps>objects.apps, 
                                                      <AppFlowCollector>objects.flows, 
                                                      flowentry, vals)

cdef class MinutesCollector(object):
 
    def __init__(self, nm, uint32_t ip, libname, uint32_t minutesdepth, uint32_t stamp):
        self._name = nm
        self._ip = ip
        self._prevsecpos = INVALID
        self._query = FlowQuery(libname, 'minutes')
        
    @cython.boundscheck(False)
    def onminute(self, QueryBuffer qbuf, Apps apps, AppFlowCollector flows, SecondsCollector seccoll, uint64_t stamp):
        cdef uint32_t appidx, cursecpos, count, pos

        cursecpos = seccoll.currentpos()
        
        if self._prevsecpos == INVALID:
            self._prevsecpos = cursecpos
            return

        self._query.initbuf(qbuf)
        
        cdef ipfix_query_info qinfo
        
        seccoll._initqinfo(cython.address(qinfo))
        
        cdef AppFlowObjects objects
        
        objects.minutes = <void*>self
        objects.apps = <void*>apps
        objects.flows = <void*>flows
        
        qinfo.callback = <FlowAppCallback>onappcallback
        qinfo.callobj = cython.address(objects)
        
        seccoll._collect(self._query, qbuf, cython.address(qinfo), self._prevsecpos, cursecpos)
        
        cdef AppsCollection* acollection = <AppsCollection*>qbuf.release(cython.address(count))
        cdef AppsCollection* entry
        for pos in range(count):
            entry = acollection + pos
            
            
        self._prevsecpos = cursecpos
        
    @cython.boundscheck(False)
    cdef int _onapp(self, Apps apps, AppFlowCollector flows, const ipfix_store_flow* flowentry, AppFlowValues* vals) nogil:
        cdef int ingress
        cdef ipfix_app_tuple atup
        cdef uint32_t pos
        cdef ipfix_store_entry* entryrec
        cdef const ipfix_flow_tuple* flow = cython.address(flowentry.flow)
        
        atup.application = apps.getflowapp(flow, cython.address(ingress))
        
        if ingress == 0:
            atup.srcaddr = flow.dstaddr
            atup.dstaddr = flow.srcaddr
            entryrec = flows._findentry(cython.address(atup), cython.address(pos), sizeof(atup))
            
            flows._onegress(entryrec, flowentry.attrindex)
        else:
            atup.srcaddr = flow.srcaddr
            atup.dstaddr = flow.dstaddr
            entryrec = flows._findentry(cython.address(atup), cython.address(pos), sizeof(atup))

            flows._oningress(entryrec, flowentry.attrindex)

        vals.crc = entryrec.crc
        vals.pos = pos

        return ingress
