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

cimport cython

from common cimport *

cdef class RawQuery(object):

    def __init__(self, const char* modname, const char* qid):
        cdef const char* err
        self.next = None
        self.prev = None
        self.qid = qid
        self.callback = None
        
        self.mod = dlopen(modname, RTLD_LAZY)
        if self.mod is NULL:
            raise Exception("Can not load %s: '%s'"%(modname, dlerror()))
        dlerror() # clear existing error
        
        cdef bytes checkername = <bytes>"fcheck_%s"%(qid)
        self.checker = <fchecktype>dlsym(self.mod, checkername)
        err = dlerror()
        if err != NULL:
            raise Exception("Can not load symbol %s from %s: %s"%(checkername, modname, err))

        cdef bytes reportername = <bytes>"freport_%s"%(qid)
        self.reporter = <freptype>dlsym(self.mod, reportername)
        err = dlerror()
        if err != NULL:
            raise Exception("Can not load symbol %s from %s: %s"%(reportername, modname, err))

        if self.checker is NULL:
            raise Exception("Can not load checker call")
        
    def __dealloc__(self):
        if not (self.mod is NULL):
            dlclose(self.mod)
        
    def id(self):
        return self.qid
        
    def setcallback(self, onmsg):
        self.callback = onmsg
        
    cdef void onflow(self, const ipfix_flow* flow):
        if self.checker(flow) == 0: return
        cdef char buffer[512]
        self.reporter(flow, buffer, sizeof(buffer))
        self.callback(buffer)
        
    def testflow(self, long val):
        cdef ipfix_flow flow
        cdef char buffer[512]
        
        flow.exporter = val
        if self.checker(cython.address(flow)) == 1:
            self.reporter(cython.address(flow), buffer, sizeof(buffer))
            print "result: '%s'"%(buffer)
        else:
            print "none"
