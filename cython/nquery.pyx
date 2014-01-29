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
        
        self.mod = dlopen(modname, RTLD_LAZY)
        if self.mod is NULL:
            raise Exception("Can not load %s: '%s'"%(modname, dlerror()))
        dlerror() # clear existing error
        
        cdef bytes callname = <bytes>"fcheck_%s"%(qid)
        self.checker = <fchecktype>dlsym(self.mod, callname)
        err = dlerror()
        if err != NULL:
            raise Exception("Can not load symbol %s from %s: %s"%(callname, modname, err))

        if self.checker is NULL:
            raise Exception("Can not load checker call")
        
    def __dealloc__(self):
        if not (self.mod is NULL):
            dlclose(self.mod)
        
    cdef int onflow(self, const ipfix_flow* flow):
        #flow.
        return True
        
    def testflow(self, long val):
        cdef ipfix_flow flow
        
        flow.exporter = val

        print "result: %d"%(self.checker(cython.address(flow)))
    