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

cdef class TimeCollector(object):

    def __init__(self, nm, uint32_t ip, uint32_t width, uint32_t timedepth, uint32_t stamp):
        cdef uint32_t pos
        self._name = nm
        self._ip = ip
        self._width = width
        self._depth = timedepth
        self._ticks = np.zeros(timedepth, dtype=np.uint32)
        self._stamps = np.zeros(timedepth, dtype=np.uint64)
        
        for pos in range(timedepth):
            self._ticks[pos] = INVALID
        
        self._currenttick = 0
        self._ticks[self._currenttick] = 0
        self._stamps[self._currenttick] = stamp

        self._alloc(minsize)

        self._count = 0

        self._first = self._counterset
        self._last = self._counterset
        
    @cython.boundscheck(False)
    cdef void _alloc(self, uint32_t size):
        cdef uint32_t sz = self._width
        self._counters = np.zeros((size, sz), dtype=np.uint8)
        cdef np.ndarray[np.uint8_t, ndim=2] arr
        arr = self._counters
        self._counterset = <void*>arr.data
        self._maxcount = size
        self._end = (<char*>self._counterset) + self._maxcount*sz

    @cython.boundscheck(False)    
    cdef void* _addentry(self) nogil:
        if self._count >= self._maxcount:    # need resize
            with gil:
                self._grow()
                if self._count >= self._maxcount:
                    logger("can not grow time counters for %s"%(self._name))
                    return NULL

        cdef void* last = self._last

        self._last = <void*>((<char*>self._last)+self._width)
        if self._last >= self._end:
            self._last = self._counterset

        self._count += 1
        
        return last
    
    @cython.boundscheck(False)
    cdef void _grow(self):
        cdef uint32_t pos, tickpos, offset, startsz, newsize
        cdef void* start = self._counterset
        cdef void* end = self._end
        cdef oldcounters = self._counters
        cdef uint64_t startbytes, endbytes
        cdef uint32_t sz = self._width

        newsize = <uint32_t>(self._maxcount*growthrate)

#        logger("%s: growing time counters %d->%d"%(self._name, self._maxcount, newsize))
        
        self._alloc(newsize)
        
        # move old data to new location
        offset = <uint32_t>(((<uint64_t>self._first) - (<uint64_t>start))/sz);

        if self._first >= self._last:
            startbytes = (<uint64_t>end) - (<uint64_t>self._first);
            startsz = <uint32_t>(startbytes/sz)
            if startbytes > 0:
                memcpy(self._counterset, self._first, startbytes)
            endbytes = (<uint64_t>self._last) - (<uint64_t>start);
            if endbytes > 0:
                memcpy((<unsigned char*>self._counterset)+startbytes, start, endbytes)
            # update pointers to where relocated data is sitting now
            self._fixticks(start, sz, offset, startsz)
        else:
            startbytes = (<uint64_t>self._last) - (<uint64_t>self._first);
            memcpy(self._counterset, self._first, startbytes)
            # update pointers to where relocated data is sitting now
            for pos in range(self._depth):
                tickpos = self._ticks[pos]
                if tickpos == INVALID: continue
                self._ticks[pos] = tickpos-offset
        
        self._first = self._counterset
        self._last = (<char*>self._counterset) + self._count*sz
        
        del oldcounters     # not needed; just in case
        
    @cython.boundscheck(False)
    cdef void _fixticks(self, void* start, uint32_t sz, uint32_t offset, uint32_t startsz) nogil:
        cdef uint32_t pos, tickpos, prevpos = 0
        
        pos = self._currenttick+1    # start from oldest second
        if pos >= self._depth: pos = 0
        for _ in range(self._depth):
            tickpos = self._ticks[pos]
            if tickpos != INVALID: 
                if <void*>((<char*>start)+tickpos*sz) >= self._first:   # it is in upper chunk
                    tickpos -= offset                # shift down
                    if tickpos == 0 and prevpos > 0:
                        tickpos += offset + startsz    # revert and shift up because it is last
                    self._ticks[pos] = tickpos
                else:                               # it is in lower chunk
                    tickpos += startsz               # shift up
                    self._ticks[pos] = tickpos
                prevpos = tickpos
            pos += 1
            if pos >= self._depth: pos = 0

    @cython.boundscheck(False)
    cdef uint32_t ontick(self, uint64_t stamp) nogil:
        cdef uint32_t prevloc, newloc, next, sz = self._width
        cdef uint32_t current = self._currenttick
        
        prevloc = self._ticks[current]
        current += 1        # oldest seconds position
        if current >= self._depth:
            current = 0

        next = current + 1  # next to oldest; this is where oldest ends
        if next >= self._depth:
            next = 0
        
        self._removeold(self._ticks[current], self._ticks[next])
        
        newloc = <uint32_t>(((<uint64_t>self._last) - (<uint64_t>self._counterset))/sz)
        
        self._ticks[current] = newloc
        self._stamps[current] = stamp
        self._currenttick = current

        return prevloc
    
    @cython.boundscheck(False)
    cdef uint32_t _lookup(self, uint64_t oldeststamp) nogil:
        # ticks may not be evenly distributed; lets jump to approximate location and then lookup
        cdef uint32_t prevpos, tickpos = self._currenttick
        cdef uint64_t stamp = self._stamps[tickpos]
        cdef uint64_t prevstamp, ticks = stamp - oldeststamp

        if tickpos < ticks:
            if ticks >= self._depth:
                tickpos += 1 # just get oldest we have
                if tickpos >= self._depth: tickpos = 0
            else:
                tickpos = self._depth + tickpos - ticks
        else:
            tickpos -= ticks

        stamp = self._stamps[tickpos]
        if stamp <= oldeststamp:
            # look forward; we will find something >= oldeststamp for sure since current stamp is bigger
            while stamp < oldeststamp:
                tickpos += 1
                if tickpos >= self._depth: tickpos = 0
                stamp = self._stamps[tickpos]
        else:
            # look backward; stamp is too big let's find something smaller then oldeststamp
            prevpos = tickpos   # to avoid compiler warning
            while stamp >= oldeststamp:
                prevpos = tickpos
                if tickpos == 0: tickpos = self._depth
                tickpos -= 1
                prevstamp = stamp
                stamp = self._stamps[tickpos]
                # go back only until stamps are decreasing
                # need this in case if oldeststamp is older than anything we have
                if stamp >= prevstamp: break
            tickpos = prevpos               # this is last known good position

        return tickpos
    
    @cython.boundscheck(False)
    cdef void _initqinfo(self, ipfix_query_info* qinfo) nogil:
        pass
    
    @cython.boundscheck(False)
    cdef uint32_t currentpos(self) nogil:
        return self._currenttick

    @cython.boundscheck(False)
    cdef void collect(self, FlowQuery q, QueryBuffer bufinfo, 
                      uint64_t neweststamp, uint64_t oldeststamp, uint32_t step) nogil:

        cdef uint32_t curpos, oldestpos, lastpos = self._currenttick
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
    cdef void _collect(self, FlowQuery q, QueryBuffer bufinfo, ipfix_query_info* qinfo,
                       uint32_t oldestpos, uint32_t lastpos) nogil:
        cdef uint32_t oldestloc, lastloc, sz = self._width

        lastloc = self._ticks[lastpos]
        oldestloc = self._ticks[oldestpos]

        # here we have secpos pointing to proper position where we need to start collection

        if lastloc >= oldestloc:  # one chunk of data
            qinfo.entries = <void*>((<char*>self._counterset)+oldestloc*sz)
            qinfo.count = lastloc-oldestloc
            q.collect(bufinfo, qinfo)
        else:
            # two chunks of data
            # collect older
            qinfo.entries = <void*>((<char*>self._counterset)+oldestloc*sz)
            qinfo.count = self._maxcount-oldestloc
            q.collect(bufinfo, qinfo)
            # collect newer
            qinfo.entries = self._counterset
            qinfo.count = lastloc
            q.collect(bufinfo, qinfo)

    @cython.boundscheck(False)    
    cdef void _removeold(self, uint32_t lastloc, uint32_t nextloc) nogil:
        cdef uint32_t count, sz = self._width
        # need to remove oldest seconds if we already have valid points
        if lastloc == INVALID: return

        cdef void* start = <void*>((<char*>self._counterset)+lastloc*sz)
        if nextloc >= lastloc:
            count = nextloc - lastloc
            if count > 0:
                self._rmold(start, count)
        else:
            # it's split into two chunks; remove one then another 
            count = self._maxcount-lastloc
            if count > 0:
                self._rmold(start, count)
            # count = nextloc - 0
            if nextloc > 0:
                self._rmold(self._counterset, nextloc) # from very beginning of buffer to nextloc
            count += nextloc
        self._first = <void*>((<char*>self._counterset)+nextloc*sz)
        self._count -= count
        
    cdef void _rmold(self, const void* start, uint32_t count) nogil:
        pass
                

cdef class SecondsCollector(TimeCollector):

    def __init__(self, nm, uint32_t ip, FlowCollector flows, uint32_t secondsdepth, uint32_t stamp):
        super(SecondsCollector, self).__init__(nm, ip, sizeof(ipfix_store_counts), secondsdepth, stamp)

        self._flows = flows

    @cython.boundscheck(False)    
    cdef void _add(self, uint32_t bytes, uint32_t packets, uint32_t flowindex) nogil:
        cdef ipfix_store_counts* entry = <ipfix_store_counts*>self._addentry()
        if entry == NULL: return

        entry.bytes = bytes
        entry.packets = packets
        entry.flowindex = flowindex

    @cython.boundscheck(False)
    def onsecond(self, Apps apps, uint64_t stamp):
        cdef uint32_t prevloc, newloc
        
        self._apps = apps
        
        prevloc = self.ontick(stamp)
        
        newloc = self._ticks[self._currenttick]

        if prevloc == INVALID: prevloc = 0

        cdef const ipfix_store_flow* flows = self._flows.getflows()
        cdef ipfix_store_counts* counts = <ipfix_store_counts*>self._counterset

        if newloc >= prevloc:  # one chunk of data
            apps.collect(flows, counts+prevloc, newloc-prevloc)
        else:
            # two chunks of data
            # collect older
            apps.collect(flows, counts+prevloc, self._maxcount-prevloc)
            # collect newer
            apps.collect(flows, counts, newloc)

    @cython.boundscheck(False)
    cdef void _initqinfo(self, ipfix_query_info* qinfo) nogil:
        qinfo.flows = <ipfix_store_flow*>self._flows.entryset
        qinfo.attrs = <ipfix_store_attributes*>self._flows._attributes.entryset
        
        qinfo.exporter = self._ip

    @cython.boundscheck(False)
    cdef void _rmold(self, const void* start, uint32_t count) nogil:
        cdef const ipfix_store_counts* countstart = <ipfix_store_counts*>start
        self._apps.remove(self._flows.getflows(), countstart, count)
        self._flows.remove(countstart, count)


cdef int onappcallback(void* obj, const ipfix_store_flow* flowentry, AppFlowValues* vals) nogil:
    cdef AppFlowObjects* objects = <AppFlowObjects*>obj
    
    return (<MinutesCollector>objects.minutes)._onapp(<Apps>objects.apps, 
                                                      <AppFlowCollector>objects.flows, 
                                                      flowentry, vals)

cdef class MinutesCollector(TimeCollector):
 
    def __init__(self, nm, uint32_t ip, libname, uint32_t minutesdepth, uint32_t stamp):
        super(MinutesCollector, self).__init__(nm, ip, sizeof(ipfix_app_counts), minutesdepth, stamp)        

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

        self._prevsecpos = cursecpos
        
        cdef AppsCollection* acollection = <AppsCollection*>qbuf.release(cython.address(count))
        cdef AppsCollection* colentry
        cdef ipfix_app_counts* minentry
        
        for pos in range(count):
            colentry = acollection + pos
            minentry = <ipfix_app_counts*>self._addentry()
            if minentry == NULL: return

            minentry.appindex = colentry.values.pos
            minentry.inbytes = colentry.inbytes
            minentry.inpackets = colentry.inpackets
            minentry.outbytes = colentry.outbytes
            minentry.outpackets = colentry.outpackets

        self.ontick(stamp)
            
        #TMP
        print "collected %d entries"%(count)
        #
            
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


def _dummy():
    "exists only to get rid of compile warnings"
    cdef int tmp = 0
    if tmp:
        _import_umath()    
        _import_array()
