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
from collectors cimport FlowCollector, AttrCollector
from timecollect cimport SecondsCollector, MinutesCollector 
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

        self._flowchecker = self._loadsymbol(modname, <bytes>"fcheck_%s"%(qid))
        self.reporter = self._loadsymbol(modname, <bytes>"freport_%s"%(qid))
        
    @cython.boundscheck(False)
    cdef void* _loadsymbol(self, const char* modname, bytes nm) except NULL:
        cdef void* sym = dlsym(self.mod, nm)
        cdef const char* err = dlerror()

        if err != NULL or sym == NULL:
            raise Exception("Can not load symbol %s from %s: %s"%(nm, modname, err))
        
        return sym
        
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
        if (<fcheckrawtype>self._flowchecker)(flow) == 0: return
        cdef char buffer[512]
        (<freprawtype>self.reporter)(flow, buffer, sizeof(buffer))
        self.callback(buffer)
        
    @cython.boundscheck(False)
    def testflow(self, long val):
        cdef ipfix_flow flow
        cdef char buffer[512]
        
        flow.exporter = val
        if (<fcheckrawtype>self._flowchecker)(cython.address(flow)) == 1:
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
    cdef const ipfix_query_buf* init(self, uint32_t width, uint32_t offset, uint32_t sizehint) except NULL:
        cdef uint32_t size = self._entries.size
        self._width = width
        self._offset = offset
        
        if (self._offset+sizeof(uint64_t)*2) > self._width:
            raise Exception("invalid offset provided: offset:%d width:%d"%(self._offset, self._width))        
        
        self._buf.count = size/width

        self._positions.bufpos = 1  # position 0 is illegal, let's make sure it's not taken and not used
        self._positions.curpos = 0 
        self._positions.countpos = 0
        self._positions.totbytes = 0
        self._positions.totpackets = 0
    
        cdef int bits = int(np.math.log(2*sizehint-1, 2))
        cdef int indsize = 2**bits
        cdef int bytesize = sizeof(uint32_t)*indsize
        
        if bytesize > self._extras.size:
            self._extraresize(bytesize)
        
        self._extras[:bytesize].fill(0)
        
        self._buf.poses = <uint32_t*>self._extradata
        self._buf.mask = 2**bits-1
        
        return cython.address(self._buf)

    @cython.boundscheck(False)
    cdef const ipfix_query_buf* getbuf(self) nogil:
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
    cdef char* release(self, uint32_t* pcount) nogil:
        self._buf.poses = NULL
        self._buf.mask = 0
        if self._positions.bufpos <= 1:
            pcount[0] = 0
        else:
            pcount[0] = self._positions.bufpos-1
        return (<char*>self._buf.data) + self._width

    @cython.boundscheck(False)
    cdef bytes onreport(self, const ipfix_query_buf* buf, ipfix_collector_report_t reporter, int field, int slice):
        cdef bytes result
        cdef size_t printed
        cdef uint32_t count, size = self._extras.size, skip
        cdef char* input
        cdef char* data
        cdef eview
        cdef uint32_t suffix
        cdef dtype
        cdef int accending = 1

        data = self.release(cython.address(count))

        if count <= 0: return <bytes>''
        skip = 0
        
        if field > 0:
            suffix = self._width - (self._offset+sizeof(uint64_t)*2)
            dtype = [('cont',    'a%d'%(self._offset)),
                     ('bytes',   'u8'),
                     ('packets', 'u8')]

            fnm  = dtype[field][0]

            if suffix > 0: dtype.append(('suffix',  'a%d'%(suffix)))
            eview = self._entries[self._width:((count+1)*self._width)].view(dtype=dtype)    # skip first entry
            
            if slice != 0:
                if slice > 0:   # max elements needed
                    if slice < eview.size:
                        skip = eview.size-slice    # first in eview array
                        eview.partition(skip, order=[fnm])
                        eview[skip:].sort(order=[fnm])
                    else:
                        slice = eview.size
                        eview.sort(order=[fnm])
                    accending = 0       # should be reported in descending order
                else:           # min elements needed
                    slice = -slice
                    if slice < eview.size:
                        eview.partition(slice-1, order=[fnm])
                        eview[:slice].sort(order=[fnm])
                    else:
                        slice = eview.size
                        eview.sort(order=[fnm])
            else:
                slice = count # all available
        else:
            slice = count # all available
                
        input = <char*>data + skip*self._width # skip few entries if needed

        printed = reporter(cython.address(self._positions), accending, 
                           input, slice, 
                           self._extradata, size, 
                           repcallback, <void*>self)
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
        
    def status(self):
        return {"entrybuf":self._entries.size,
                "extrabuf":self._extras.size,
                "counts":{"collected":self._positions.bufpos-1,
                          "bytes":self._positions.totbytes,
                          "packets":self._positions.totpackets}}
        
cdef char* repcallback(void* data, size_t* size_p):
    cdef QueryBuffer qbuf = <QueryBuffer>data
    return qbuf.repcallback(size_p)
        
cdef class FlowQuery(Query):

    def __init__(self, const char* modname, const char* qid):
        super(FlowQuery, self).__init__(modname, qid)
        
        self.expchecker = <fexpchecktype>self._loadsymbol(modname, <bytes>"fexporter_%s"%(qid))
        
        self._width = self._getvalue(modname, 'width', qid)
        self._offset = self._getvalue(modname, 'offset', qid)
        
        self._sizehint = minbufsize
        
        self._appchecker = self._loadsymbol(modname, <bytes>"acheck_%s"%(qid))

        self._checker = NULL

    @cython.boundscheck(False)
    cdef uint32_t _getvalue(self, const char* modname, const char* nm, const char* qid) except 0:
        cdef uint32_t* wptr = <uint32_t*>self._loadsymbol(modname, <bytes>("f%s_%s"%(nm, qid)))
        
        cdef uint32_t val = cython.operator.dereference(wptr)
        if val == 0 or val > 1000:
            raise Exception("Unexpected %s for query results: %d"%(nm, val))
        
        return val

    @cython.boundscheck(False)
    def matchsource(self, long ip):
        if self.expchecker(ip) != 0: return True
        return False
        
    @cython.boundscheck(False)
    def initbuf(self, QueryBuffer qbuf):
        qbuf.init(self._width, self._offset, self._sizehint)

    @cython.boundscheck(False)
    def runseconds(self, QueryBuffer qbuf, secset, uint64_t newstamp, uint64_t oldstamp, uint32_t step):
        cdef SecondsCollector sec
        
        self._checker = self._flowchecker
        
        for sec in secset:
            sec.collect(self, qbuf, newstamp, oldstamp, step)
            
    @cython.boundscheck(False)
    def runminutes(self, QueryBuffer qbuf, minset, uint64_t newstamp, uint64_t oldstamp, uint32_t step):
        cdef MinutesCollector mint
        
        self._checker = self._appchecker
        
        for mint in minset:
            mint.collect(self, qbuf, newstamp, oldstamp, step)

    @cython.boundscheck(False)
    def report(self, QueryBuffer qbuf, field, dir, uint32_t count):
        cdef const ipfix_query_buf* buf = qbuf.getbuf()
        cdef int fld, slice
        
        if field == 'bytes':
            fld = 1
        elif field == 'packets':
            fld = 2
        else:
            fld = 0

        if dir == 'min':
            slice = -count
        elif dir == 'max':
            slice = count
        else:
            slice = 0

        cdef bytes result = qbuf.onreport(buf, <ipfix_collector_report_t>self.reporter, fld, slice)

        return result

    @cython.boundscheck(False)
    cdef void collect(self, QueryBuffer bufinfo, const ipfix_query_info* info) nogil:
        cdef ipfix_query_pos* poses = bufinfo.getposes()
        cdef const ipfix_query_buf* buf = bufinfo.getbuf()
            
        poses.countpos = 0    # reset to start from first flow in info

        while True:
            #TMP
#            with gil:
#                import sys
#                print "data: 0x%08x count: %d 0x%08x 0x%08x"%(<uint64_t>buf.data, buf.count, <uint64_t>buf.poses, buf.mask)
#                print "qinfo: count:%d"%(info.count)
#                print "poses before: %d, %d"%(poses.bufpos, poses.countpos)
#                sys.stdout.flush()
            
            (<ipfix_collector_call_t>self._checker)(buf, info, poses)
            #TMP
#            with gil:
#                print "poses after: %d, %d"%(poses.bufpos, poses.countpos)
#                sys.stdout.flush()
            
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