# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z
# distutils: define_macros = NPY_NO_DEPRECATION_WARNING=

import numpy as np
import datetime
import dateutil.tz
cimport cython
cimport numpy as np

from common cimport *

#from collectors
from misc cimport logger, minsize, growthrate, shrinkrate
from misc import backtable, backparm, backval, resparm, resval
from napps cimport Apps

cdef uint32_t INVALID = (<uint32_t>(-1))
cdef tzutc = dateutil.tz.tzutc()

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
        
    def backup(self, fileh, grp):
        backtable(fileh, grp, 'ticks', self._ticks)
        backtable(fileh, grp, 'stamps', self._stamps)
        backtable(fileh, grp, 'counters', self.counters())

        backparm(self, grp, '_currenttick')
        backparm(self, grp, '_count')
        backval(grp, 'first', (<uint64_t>self._first)-(<uint64_t>self._counterset))
        backval(grp, 'last', (<uint64_t>self._last)-(<uint64_t>self._counterset))

    def restore(self, fileh, grp):
        counters = fileh.get_node(grp, 'counters')
        if self._width != counters.rowsize:
            raise Exception("%s width (%d) does not match stored width (%d)"%(self._name, 
                                                                              self._width, 
                                                                              counters.rowsize))
        self._resize(len(counters))
        counters.read(out=self._counters)
        
        cdef uint64_t first = resval(grp, 'first')
        self._first = <void*>((<char*>self._counterset)+first)
        cdef uint64_t last = resval(grp, 'last')
        self._last = <void*>((<char*>self._counterset)+last)
        
        resparm(self, grp, '_count')
        
        ticks = fileh.get_node(grp, 'ticks')
        if self._depth != len(ticks):
            raise Exception("%s change in ticks depth can not be handled yet: %d != %d"%(self._name, 
                                                                                         self._depth,
                                                                                         len(ticks)))
        ticks.read(out=self._ticks)
        
        stamps = fileh.get_node(grp, 'stamps')
        stamps.read(out=self._stamps)

        resparm(self, grp, '_currenttick')
        
    @cython.boundscheck(False)
    cdef void _alloc(self, uint32_t size):
        cdef uint32_t sz = self._width
        self._counters = np.zeros((size, sz), dtype=np.uint8)
        cdef np.ndarray[np.uint8_t, ndim=2] arr
        arr = self._counters
        self._counterset = <void*>arr.data
        self._maxcount = size
        self._end = (<char*>self._counterset) + size*sz

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
    @cython.boundscheck(False)
    cdef void _grow(self):
        self._resize(<uint32_t>(self._maxcount*growthrate))
        
    cdef void _resize(self, uint32_t newsize):
        cdef uint32_t pos, tickpos, offset, startsz
        cdef void* start = self._counterset
        cdef void* end = self._end
        cdef oldcounters = self._counters
        cdef uint64_t startbytes, endbytes
        cdef uint32_t sz = self._width

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
        #TMP
#        with gil:
#            print "  %s: oldeststamp:%d -> %s[%d]  %d[0]"%(self._name, oldeststamp, self._stamps[oldestpos], oldestpos, self._stamps[0])
        #
        
        cdef ipfix_query_info qinfo
        
        self._initqinfo(cython.address(qinfo))

        cdef ipfix_query_pos* poses = bufinfo.getposes()

        if poses.oldest == 0 or poses.oldest > self._stamps[oldestpos]:
            poses.oldest = self._stamps[oldestpos]

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
    
    def _stamptostr(self, uint32_t pos):
        cdef d
        try:
            d = datetime.datetime.utcfromtimestamp(self._stamps[pos]).replace(tzinfo=tzutc)
        except Exception, e:
            print "got an exception in _stamptostr(%d):"%(pos),str(e)
            return 'unknown'
        
        return str(d)

    def status(self):
        cdef uint32_t sz = self._width

        return {'entries':{'size':len(self._counters),
                           'bytes':int(self._counters.nbytes),
                           'count':int(self._count),
                           'first':int((((<uint64_t>self._first) - (<uint64_t>self._counterset))/sz)),
                           'last':int((((<uint64_t>self._last) - (<uint64_t>self._counterset))/sz))},
                'ticks':{'size':int(self._depth),
                         'position':int(self._currenttick),
                         'stamp':self._stamptostr(self._currenttick)}}
                
    def counters(self):
        return self._counters.view(dtype=self.dtypes)[:,0]
                

cdef class SecondsCollector(TimeCollector):

    def __init__(self, nm, uint32_t ip, FlowCollector flows, uint32_t secondsdepth, uint32_t stamp):
        super(SecondsCollector, self).__init__(nm, ip, sizeof(ipfix_store_counts), secondsdepth, stamp)
        
        cdef ipfix_store_counts fcount
        
        self.dtypes = [('flowindex',    'u%d'%sizeof(fcount.flowindex)),
                       ('bytes',        'u%d'%sizeof(fcount.bytes)),
                       ('packets',      'u%d'%sizeof(fcount.packets))]

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
            apps.collectports(flows, counts+prevloc, newloc-prevloc)
        else:
            # two chunks of data
            # collect older
            apps.collectports(flows, counts+prevloc, self._maxcount-prevloc)
            # collect newer
            apps.collectports(flows, counts, newloc)

    @cython.boundscheck(False)
    cdef void _initqinfo(self, ipfix_query_info* qinfo) nogil:
        qinfo.flows = <ipfix_store_flow*>self._flows.entryset
        qinfo.appflows = NULL
        qinfo.apps = NULL
        qinfo.attrs = <ipfix_store_attributes*>self._flows._attributes.entryset
        
        qinfo.exporter = self._ip

    @cython.boundscheck(False)
    cdef void _rmold(self, const void* start, uint32_t count) nogil:
        cdef const ipfix_store_counts* countstart = <ipfix_store_counts*>start
        self._apps.removeports(self._flows.getflows(), countstart, count)
        self._flows.remove(countstart, count)


cdef int onminutescallback(void* obj, const void* entry, AppFlowValues* vals) nogil:
    cdef AppFlowObjects* objects = <AppFlowObjects*>obj
    
    return (<MinutesCollector>objects.ticks)._onminapp(<Apps>objects.apps, 
                                                       <AppFlowCollector>objects.flows,
                                                       <ipfix_store_flow*>entry, vals)
    
cdef int onhourscallback(void* obj, const void* entry, AppFlowValues* vals) nogil:
    cdef AppFlowObjects* objects = <AppFlowObjects*>obj
    
    return (<HoursCollector>objects.ticks)._onhoursapp(<Apps>objects.apps,
                                                       <AppFlowCollector>objects.flows,
                                                       <ipfix_app_flow*>entry, vals)
    

cdef class LongCollector(TimeCollector):
 
    def __init__(self, nm, uint32_t ip, libname, callname, AppFlowCollector appflows, uint32_t ticksdepth, uint32_t stamp):
        super(LongCollector, self).__init__(nm, ip, sizeof(ipfix_app_counts), ticksdepth, stamp)        

        cdef ipfix_app_counts acount
        
        self.dtypes = [('appindex',    'u%d'%sizeof(acount.appindex)),
                       ('inbytes',     'u%d'%sizeof(acount.inbytes)),
                       ('inpackets',   'u%d'%sizeof(acount.inpackets)),
                       ('outbytes',    'u%d'%sizeof(acount.outbytes)),
                       ('outpackets',  'u%d'%sizeof(acount.outpackets))]

        self._apps = appflows._apps
        self._appflows = appflows
        self._prevtickpos = INVALID
        self._query = SimpleQuery(libname, callname)

    @cython.boundscheck(False)
    cdef const ipfix_app_flow* _getappflows(self) nogil:
        return <ipfix_app_flow*>self._appflows.entryset 

    @cython.boundscheck(False)
    cdef void _initqinfo(self, ipfix_query_info* qinfo) nogil:
        qinfo.flows = NULL
        qinfo.appflows = self._getappflows()
        qinfo.apps = <ipfix_apps*>self._appflows._apps.entryset
        qinfo.attrs = <ipfix_store_attributes*>self._appflows._attributes.entryset
        
        qinfo.exporter = self._ip

    @cython.boundscheck(False)
    cdef void _onlongtick(self, QueryBuffer qbuf, Apps apps, AppFlowCollector flows, 
                          TimeCollector timecoll, uint64_t stamp, 
                          FlowAppCallback callback):
        cdef uint32_t appidx, curpos, count, pos

        #TMP
        #print "%s: %d"%(self._name, stamp)
        #

        curpos = timecoll.currentpos()
        
        if self._prevtickpos == INVALID:
            self._prevtickpos = curpos
            self.ontick(stamp)
            return

        self._query.initbuf(qbuf)
        
        cdef ipfix_query_info qinfo
        
        timecoll._initqinfo(cython.address(qinfo))
        
        cdef AppFlowObjects objects
        
        objects.ticks = <void*>self
        objects.apps = <void*>apps
        objects.flows = <void*>flows
        
        qinfo.callback = callback
        qinfo.callobj = cython.address(objects)
        
        timecoll._collect(self._query, qbuf, cython.address(qinfo), self._prevtickpos, curpos)
        
        self._prevtickpos = curpos
        
        cdef AppsCollection* acollection = <AppsCollection*>qbuf.release(cython.address(count))
        cdef AppsCollection* colentry
        cdef ipfix_app_counts* tickentry
        
        cdef const ipfix_app_flow* aflows = self._getappflows()
        cdef const ipfix_app_flow* aflow
        
        for pos in range(count):
            colentry = acollection + pos
            tickentry = <ipfix_app_counts*>self._addentry()
            if tickentry == NULL: break 

            tickentry.appindex = colentry.values.pos
            tickentry.inbytes = colentry.inbytes
            tickentry.inpackets = colentry.inpackets
            tickentry.outbytes = colentry.outbytes
            tickentry.outpackets = colentry.outpackets
            
            aflow = aflows+tickentry.appindex
            
            flows.countflowapp(<ipfix_app_flow*>aflow)
            
        self.ontick(stamp)

    @cython.boundscheck(False)
    cdef void _rmold(self, const void* start, uint32_t count) nogil:
        cdef const ipfix_app_counts* countstart = <ipfix_app_counts*>start
        self._appflows.removeapps(countstart, count)

    def status(self, TimeCollector prevcoll):
        cdef prev
        
        if self._prevtickpos == INVALID:
            prev = ""
        else:
            prev = prevcoll._stamptostr(self._prevtickpos)
        res = super(LongCollector, self).status()
        res['ticks']['prevpos'] = int(self._prevtickpos)
        res['ticks']['prevstamp'] = prev

        return res

    def backup(self, fileh, grp):
        super(LongCollector, self).backup(fileh, grp)

        backparm(self, grp, '_prevtickpos')

    def restore(self, fileh, grp):
        super(LongCollector, self).restore(fileh, grp)
        
        resparm(self, grp, '_prevtickpos')


cdef class MinutesCollector(LongCollector):
 
    def __init__(self, nm, uint32_t ip, libname, AppFlowCollector appflows, uint32_t minutesdepth, 
                uint32_t stamp, uint32_t checkperiod):
        super(MinutesCollector, self).__init__(nm, ip, libname, 'minutes', appflows, minutesdepth, stamp)
        self._checkcount = 0
        self._checkperiod = checkperiod
        
    @cython.boundscheck(False)
    def onminute(self, QueryBuffer qbuf, Apps apps, AppFlowCollector flows, SecondsCollector seccoll, uint64_t stamp):
        self._onlongtick(qbuf, apps, flows, seccoll, stamp, onminutescallback)
        
        self._checkcount += 1
        if self._checkcount >= self._checkperiod:
            self._checkcount = 0
            apps.checkactivity()

    @cython.boundscheck(False)
    cdef int _onminapp(self, Apps apps, AppFlowCollector flows, const ipfix_store_flow* flowentry, AppFlowValues* vals) nogil:
        cdef int ingress
        cdef ipfix_app_tuple atup
        cdef const ipfix_flow_tuple* flow = cython.address(flowentry.flow)
        
        atup.application = apps.getflowapp(flow, cython.address(ingress))
        
        if ingress == 0:
            atup.srcaddr = flow.dstaddr
            atup.dstaddr = flow.srcaddr
        else:
            atup.srcaddr = flow.srcaddr
            atup.dstaddr = flow.dstaddr

        flows.findapp(cython.address(atup), flowentry.attrindex, vals, ingress)

        return ingress
    

cdef class HoursCollector(LongCollector):
    def __init__(self, nm, uint32_t ip, libname, AppFlowCollector appflows, uint32_t hoursdepth, uint32_t stamp):
        super(HoursCollector, self).__init__(nm, ip, libname, 'hours', appflows, hoursdepth, stamp)

    @cython.boundscheck(False)
    def onhour(self, QueryBuffer qbuf, Apps apps, AppFlowCollector flows, MinutesCollector mincoll, uint64_t stamp):
        self._onlongtick(qbuf, apps, flows, mincoll, stamp, onhourscallback)

    @cython.boundscheck(False)
    cdef int _onhoursapp(self, Apps apps, AppFlowCollector flows, const ipfix_app_flow* flowentry, AppFlowValues* vals) nogil:
        #let's find more generic app if possible
            
        vals.crc = flowentry.crc

        return 0
    
cdef class DaysCollector(LongCollector):
    def __init__(self, nm, uint32_t ip, libname, AppFlowCollector appflows, uint32_t daysdepth, uint32_t stamp):
        super(DaysCollector, self).__init__(nm, ip, libname, 'days', appflows, daysdepth, stamp)
        
    @cython.boundscheck(False)
    def onday(self, QueryBuffer qbuf, Apps apps, AppFlowCollector flows, HoursCollector hourcoll, uint64_t stamp):
        self._onlongtick(qbuf, apps, flows, hourcoll, stamp, NULL)


def _dummy():
    "exists only to get rid of compile warnings"
    cdef int tmp = 0
    if tmp:
        _import_umath()    
        _import_array()
