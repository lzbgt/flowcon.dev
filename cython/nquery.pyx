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
        self._width = 0
        self._entries = np.zeros(minbufsize, dtype=np.uint8)
        cdef np.ndarray[np.uint8_t, ndim=1] arr = self._entries
        self._buf.data = <void*>arr.data
        self._extras = np.zeros(1, dtype=np.uint8)
        self._extraresize(minbufsize)

    @cython.boundscheck(False)
    cdef void _extraresize(self, uint32_t nbytes):
        self._extras.resize(nbytes, refcheck=False)
        cdef np.ndarray[np.uint8_t, ndim=1] arr = self._extras
        self._extradata = <char*>arr.data

    @cython.boundscheck(False)
    cdef const ipfix_query_buf* init(self, uint32_t width, uint32_t sizehint):
        cdef uint32_t size = self._entries.size
        self._width = width
        self._buf.count = size/width

        self._positions.bufpos = 1  # position 0 is illegal, let's make sure it's not taken and not used 
        self._positions.countpos = 0
    
        cdef int bits = int(np.math.log(2*sizehint-1, 2))
        cdef int indsize = 2**bits
        
        if (sizeof(uint32_t)*indsize) > self._extras.size:
            self._extraresize(sizeof(uint32_t)*indsize)
        
        self._buf.poses = <uint32_t*>self._extradata
        self._buf.mask = 2**bits-1
        
        return cython.address(self._buf)

    @cython.boundscheck(False)
    cdef char* repcallback(self, size_t* size_p):
        if size_p == NULL: return NULL
        cdef size_t size = self._extras.size*2
        if size < size_p[0]: size = size_p[0]

        logger("growing Q extras %d->%d"%(self._extras.size, size))
        
        self._extraresize(size)
        size_p[0] = size
        
        return self._extradata

    @cython.boundscheck(False)
    cdef bytes onreport(self, const ipfix_query_buf* buf, ipfix_collector_report_t reporter):
        cdef bytes result
        cdef size_t printed
        cdef uint32_t count, size = self._extras.size
        cdef char* input

        self._buf.poses = NULL
        self._buf.mask = 0

        count = self._positions.bufpos
        if count <= 1: return <bytes>''
        count -= 1
        input = <char*>self._buf.data+self._width # skip first entry; it is never used

        printed = reporter(input, count, self._extradata, size, repcallback, <void*>self)
        if printed <= 0:
            return <bytes>('{"error":"can not print %d collected entries"}'%(count))

        return <bytes>(self._extradata)[:printed]
                
    @cython.boundscheck(False)
    cdef ipfix_query_pos* getposes(self) nogil:
        return cython.address(self._positions)

    @cython.boundscheck(False)
    cdef void grow(self):
        cdef uint32_t size = self._entries.size*2

        logger("growing Q entries %d->%d"%(self._entries.size, size))

        self._entries.resize(size, refcheck=False)

        cdef np.ndarray[np.uint8_t, ndim=1] arr = self._entries
        self._buf.data = <void*>arr.data
        self._buf.count = size/self._width
        
cdef char* repcallback(void* data, size_t* size_p):
    cdef QueryBuffer qbuf = <QueryBuffer>data
    return qbuf.repcallback(size_p)
        
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
        
        self._sizehint = minbufsize

    @cython.boundscheck(False)
    def matchsource(self, long ip):
        if self.expchecker(ip) != 0: return True
        return False
        
    def runseconds(self, QueryBuffer qbuf, secset, uint64_t stamp):
        cdef SecondsCollector sec
        
        cdef const ipfix_query_buf* buf = qbuf.init(self._width, self._sizehint)
        
        for sec in secset:
            sec.collect(self, qbuf, stamp, <void*>buf)

        cdef bytes result = qbuf.onreport(buf, <ipfix_collector_report_t>self.reporter)
        
        return results

    @cython.boundscheck(False)
    cdef void collect(self, QueryBuffer bufinfo, const ipfix_query_info* info, uint32_t expip, void* data) nogil:
        cdef const ipfix_query_buf* buf = <ipfix_query_buf*>data
        cdef ipfix_query_pos* poses = bufinfo.getposes()
            
        poses.countpos = 0    # reset to start from first flow in info

        while True:
            # TMP
            #with gil:
            #    print "data: 0x%08x count: %d 0x%08x 0x%08x"%(<uint64_t>buf.data, buf.count, <uint64_t>buf.poses, buf.mask)
            #    print "qinfo: count:%d"%(info.count)
            #    print "poses before: %d, %d"%(poses.bufpos, poses.countpos)
            #
            (<ipfix_collector_call_t>self.checker)(buf, info, poses, expip)
            # TMP
            #with gil:
            #    print "poses after: %d, %d"%(poses.bufpos, poses.countpos)
            #            
            
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