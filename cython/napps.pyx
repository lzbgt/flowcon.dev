# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z

import numpy as np
cimport cython
cimport numpy as np

from common cimport *
from misc cimport logger
from collectors cimport Collector

def _dummy():
    "exists only to get rid of compile warnings"
    cdef int tmp = 0
    if tmp:
        _import_umath()    
        _import_array()

cdef class Apps(Collector):

    def __init__(self, float portrate, uint32_t minthreshold):
        super(Apps, self).__init__('apps', sizeof(ipfix_apps))
        
        cdef ipfix_apps apps

        self.dtypes = [('next',      'u%d'%sizeof(apps.next)),
                       ('crc',       'u%d'%sizeof(apps.crc)),
                       ('p1',        'u%d'%sizeof(apps.ports.p1)),
                       ('p2',        'u%d'%sizeof(apps.ports.p2))]                
        
        cdef uint32_t size
        
        self._totalcount = 0
        self._zeroportcount = 0
        self._portrate = portrate
        self._minthreshold = minthreshold

        size = 2**16
        self._pcobj = np.zeros(size, dtype=np.uint32)
        
        cdef np.ndarray[np.uint32_t, ndim=1] arr = self._pcobj
        
        self._portcounts = <uint32_t*>arr.data
    
    @cython.boundscheck(False)
    cdef void collect(self, const ipfix_store_flow* flows, const ipfix_store_counts* counts, uint32_t num) nogil:
        cdef uint32_t pos, index
        cdef const ipfix_flow_tuple* flow
        cdef uint32_t* portcounts = self._portcounts

        for pos in range(num):
            index = counts[pos].flowindex
            flow = cython.address(flows[index].flow)
            portcounts[flow.srcport] += 1
            portcounts[flow.dstport] += 1
            self._totalcount += 2
            
    @cython.boundscheck(False)
    cdef uint32_t getflowapp(self, const ipfix_flow_tuple* flow) nogil:
        cdef uint32_t* portcounts = self._portcounts
        cdef float rt = self._portrate
        cdef uint16_t p1, p2
        cdef ipfix_apps_ports ports
        
        if flow.srcport < flow.dstport:
            p1 = flow.srcport
            p2 = flow.dstport
        else:
            if flow.dstport == 0:
                self._zeroportcount += 1 
                return 0  # invalid app
            p1 = flow.dstport
            p2 = flow.srcport
        
        cdef uint32_t src = portcounts[p1]
        cdef uint32_t dst = portcounts[p2]

        ports.protocol = flow.protocol

        if src > dst:
            if src >= self._minthreshold and src >= dst*rt:
                ports.p1 = 0    # ignore p2 port
                ports.p2 = p1   
            else:
                ports.p1 = p1
                ports.p2 = p2
        else:
            if dst >= self._minthreshold and src*rt <= dst:
                ports.p1 = 0    # ignore p1 port
                ports.p2 = p2
            else:
                ports.p1 = p1
                ports.p2 = p2

        return self._add(cython.address(ports), 0, sizeof(ipfix_apps_ports))

    @cython.boundscheck(False)
    cdef void remove(self, const ipfix_store_flow* flows, const ipfix_store_counts* counts, uint32_t num) nogil:
        cdef uint32_t pos, index
        cdef const ipfix_flow_tuple* flow
        cdef uint32_t* portcounts = self._portcounts

        for pos in range(num):
            index = counts[pos].flowindex
            flow = cython.address(flows[index].flow)
            portcounts[flow.srcport] -= 1
            portcounts[flow.dstport] -= 1
            self._totalcount -= 2

    def report(self):
        cdef uint32_t* portcounts = self._portcounts
        cdef uint32_t pos, size = self._pcobj.size
        
        cdef ports = {}
        
        for pos in range(size):
            if portcounts[pos] == 0: continue
            ports[pos] = portcounts[pos]
            
        cdef apps = []
        cdef appset = self.entries()
        
        cdef int p1idx = appset.dtype.names.index('p1')
        cdef int p2idx = appset.dtype.names.index('p2')
        
        for app in appset[1:self.end,0]:
            apps.append((int(app[p1idx]), int(app[p2idx])))
        
        res = {'ports':{'counts':ports, 'total':self._totalcount},
                'apps':{'ports':apps, 'total':self.end-1, 'zeros':self._zeroportcount}}

        return res
