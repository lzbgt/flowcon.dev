
from common cimport *
from collectors cimport FlowCollector, AppFlowCollector
from nquery cimport FlowQuery, SimpleQuery, QueryBuffer
from napps cimport Apps

cdef class TimeCollector(object):
    cdef _name
    cdef uint32_t _ip
    cdef _counters
    cdef void* _counterset
    cdef void* _last
    cdef void* _first
    cdef void* _end
    cdef uint32_t _width
    cdef uint32_t _maxcount
    cdef uint32_t _count
    cdef uint32_t _depth
    cdef uint32_t [:] _ticks
    cdef uint64_t [:] _stamps
    cdef uint32_t _currenttick

    cdef void _alloc(self, uint32_t size)
    cdef void* _addentry(self) nogil
    cdef void _grow(self)
    cdef void _fixticks(self, void* start, uint32_t sz, uint32_t offset, uint32_t startsz) nogil
    cdef uint32_t ontick(self, uint64_t stamp) nogil
    cdef uint32_t _lookup(self, uint64_t oldeststamp) nogil
    cdef uint32_t currentpos(self) nogil
    cdef void collect(self, FlowQuery q, QueryBuffer bufinfo, 
                       uint64_t neweststamp, uint64_t oldeststamp, uint32_t step) nogil
    cdef void _initqinfo(self, ipfix_query_info* qinfo) nogil
    cdef void _collect(self, FlowQuery q, QueryBuffer bufinfo, ipfix_query_info* qinfo,
                       uint32_t oldestpos, uint32_t lastpos) nogil
    cdef void _removeold(self, uint32_t lastpos, uint32_t nextpos) nogil
    cdef void _rmold(self, const void* start, uint32_t count) nogil

cdef class SecondsCollector(TimeCollector):
    cdef FlowCollector _flows
    cdef Apps _apps 

    cdef void _add(self, uint32_t bytes, uint32_t packets, uint32_t flowindex) nogil

    cdef void _rmold(self, const void* start, uint32_t count) nogil

    cdef void _initqinfo(self, ipfix_query_info* qinfo) nogil


cdef class LongCollector(TimeCollector):
    cdef uint32_t _prevtickpos
    cdef SimpleQuery _query
    cdef AppFlowCollector _appflows
    cdef Apps _apps
    
    cdef void _onlongtick(self, QueryBuffer qbuf, Apps apps, AppFlowCollector flows, 
                          TimeCollector timecoll, uint64_t stamp, FlowAppCallback callback)


cdef class MinutesCollector(LongCollector):

    cdef int _onminapp(self, Apps apps, AppFlowCollector flows, 
                       const ipfix_store_flow* flowentry, AppFlowValues* vals) nogil
    

cdef class HoursCollector(LongCollector):

    cdef int _onhoursapp(self, AppFlowCollector flows, const ipfix_app_flow* flowentry, AppFlowValues* vals) nogil
    
cdef class DaysCollector(LongCollector):
    pass

