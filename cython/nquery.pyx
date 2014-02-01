# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = dl

#### distutils: library_dirs = 
#### distutils: depends = 

cdef extern from "dlfcn.h":
    void *dlopen (const char* fname, int mode) nogil
    int dlclose (void * handle) nogil
    void *dlsym (void * handle, const char * name) nogil
    char *dlerror() nogil
    int RTLD_GLOBAL
    int RTLD_LAZY

import numpy as np
cimport cython
cimport numpy as np

from common cimport *
from collectors cimport SecondsCollector, FlowCollector, AttrCollector
from misc cimport logger 

cdef uint32_t minbufsize = 256

cdef class Query(object):

    def __init__(self, const char* modname, const char* qid):
        cdef const char* err
        self.qid = qid

        self.mod = dlopen(modname, RTLD_LAZY)
        if self.mod is NULL:
            raise Exception("Can not load %s: '%s'"%(modname, dlerror()))
        dlerror() # clear existing error
        
        cdef bytes checkername = <bytes>"fcheck_%s"%(qid)
        self.checker = dlsym(self.mod, checkername)
        err = dlerror()
        if err != NULL:
            raise Exception("Can not load symbol %s from %s: %s"%(checkername, modname, err))

        cdef bytes reportername = <bytes>"freport_%s"%(qid)
        self.reporter = dlsym(self.mod, reportername)
        err = dlerror()
        if err != NULL:
            raise Exception("Can not load symbol %s from %s: %s"%(reportername, modname, err))

        if self.checker is NULL or self.reporter is NULL:
            raise Exception("Can not load function calls")
        
    def __dealloc__(self):
        if not (self.mod is NULL):
            dlclose(self.mod)

    def id(self):
        return self.qid

cdef class RawQuery(Query):

    def __init__(self, const char* modname, const char* qid):
        super(RawQuery, self).__init__(modname, qid)
        self.next = None
        self.prev = None
        self.callback = None
    
    @cython.boundscheck(False)
    def setcallback(self, onmsg):
        self.callback = onmsg
        
    @cython.boundscheck(False)
    cdef void onflow(self, const ipfix_flow* flow):
        if (<fcheckrawtype>self.checker)(flow) == 0: return
        cdef char buffer[512]
        (<freprawtype>self.reporter)(flow, buffer, sizeof(buffer))
        self.callback(buffer)
        
    @cython.boundscheck(False)
    def testflow(self, long val):
        cdef ipfix_flow flow
        cdef char buffer[512]
        
        flow.exporter = val
        if (<fcheckrawtype>self.checker)(cython.address(flow)) == 1:
            (<freprawtype>self.reporter)(cython.address(flow), buffer, sizeof(buffer))
            print "result: '%s'"%(buffer)
        else:
            print "none"

cdef class QueryBuffer(object):
    def __cinit__(self):
        self._width = None
        self._entries = np.zeros(minbufsize, dtype=np.uint8)
        cdef np.ndarray[np.uint8_t, ndim=1] arr = self._entries
        self._buf.data = <void*>arr.data

    @cython.boundscheck(False)
    cdef void init(self, uint32_t width):
        cdef uint32_t size = self._entries.size
        self._width = width
        self._buf.count = size/width

        self._positions.pos = 0
        self._positions.countpos = 0
        
    @cython.boundscheck(False)
    cdef const ipfix_query_buf* getbuf(self) nogil:
        return cython.address(self._buf)
    
    @cython.boundscheck(False)
    cdef ipfix_query_pos* getposes(self) nogil:
        return cython.address(self._positions)

    @cython.boundscheck(False)
    cdef void grow(self):
        cdef uint32_t size = self._entries.size*2

        logger("growing Q buf %d->%d"%(self._entries.size, size))

        self._entries.resize(size, refcheck=False)

        cdef np.ndarray[np.uint8_t, ndim=1] arr = self._entries
        self._buf.data = <void*>arr.data
        self._buf.count = size/self._width
        
cdef class PeriodicQuery(Query):

    def __init__(self, const char* modname, const char* qid):
        super(PeriodicQuery, self).__init__(modname, qid)
        cdef bytes expname = <bytes>"fexporter_%s"%(qid)
        self.expchecker = <fexpchecktype>dlsym(self.mod, expname)
        err = dlerror()
        if err != NULL:
            raise Exception("Can not load symbol %s from %s: %s"%(expname, modname, err))        
        
        cdef bytes widthname = <bytes>"fwidth_%s"%(qid)
        cdef uint32_t* wptr = <uint32_t*>dlsym(self.mod, widthname)
        
        if self.expchecker is NULL or wptr is NULL:
            raise Exception("Can not load source checker function call")

        self._width = cython.operator.dereference(wptr)
        if self._width == 0 or self._width > 1000:
            raise Exception("Unexpected width for query results: %d"%(self._width))

    @cython.boundscheck(False)
    def matchsource(self, long ip):
        if self.expchecker(ip) != 0: return True
        return False
        
    def runseconds(self, QueryBuffer qbuf, secset, uint64_t stamp):
        cdef SecondsCollector sec
        
        qbuf.init(self._width)
        
        for sec in secset:
            sec.collect(self, qbuf, stamp)
            
        cdef const ipfix_query_buf* buf = qbuf.getbuf()

        (<ipfix_collector_report_t>self.reporter)(buf)
            
    @cython.boundscheck(False)
    cdef void collect(self, QueryBuffer bufinfo, const ipfix_query_info* info, uint32_t expip) nogil:
        cdef const ipfix_query_buf* buf = bufinfo.getbuf()
        cdef ipfix_query_pos* poses = bufinfo.getposes()
            
        while True:
            (<ipfix_collector_call_t>self.checker)(buf, info, poses, expip)
            if poses.countpos >= info.count: return # checker ran through all entries

            with gil:        
                # not all entries are collected yet
                if poses.bufpos < buf.count: # some unknown error
                    raise Exception("bufpos %d < total %d"%(poses.bufpos, buf.count))
                # run out of space ; need to grow
                bufinfo.grow()
            
#===================================================

def _dummy():
    "exists only to get rid of compile warnings"
    cdef int tmp = 0
    if tmp:
        _import_umath()    
        _import_array()