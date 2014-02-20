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
                       ('protocol',  'u%d'%sizeof(apps.ports.protocol)),
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
    cdef uint32_t getflowapp(self, const ipfix_flow_tuple* flow, int* ingress) nogil:
        cdef uint32_t* portcounts = self._portcounts
        cdef float rt = self._portrate
        cdef uint16_t dst, src
        cdef ipfix_apps_ports ports
        cdef uint32_t srccnt, dstcnt
        
        dst = flow.dstport
        src = flow.srcport
        if dst == 0:
            ingress[0] = 1 
            return 0
        
        ports.protocol = flow.protocol

        if src == 0:
            self._zeroportcount += 1
            ingress[0] = 1
            ports.p1 = 0
            ports.p2 = dst
        else:   
            srccnt = portcounts[src]
            dstcnt = portcounts[dst]
            
            if srccnt > dstcnt:
                if srccnt >= self._minthreshold and srccnt >= dstcnt*rt:
                    ports.p1 = 0    # ignore dst port
                    ports.p2 = src
                    ingress[0] = 0
                else:   # unknown application; sort the ports; consider smaller port as application   
                    if src > dst:
                        ingress[0] = 1
                        ports.p1 = dst
                        ports.p2 = src
                    else:
                        ingress[0] = 0
                        ports.p1 = src
                        ports.p2 = dst
            else:
                if dstcnt >= self._minthreshold and srccnt*rt <= dstcnt:
                    ports.p1 = 0    # ignore src port
                    ports.p2 = dst
                    ingress[0] = 1
                else:   # unknown application; sort the ports; consider smaller port as application   
                    if src > dst:
                        ingress[0] = 1
                        ports.p1 = dst
                        ports.p2 = src
                    else:
                        ingress[0] = 0
                        ports.p1 = src
                        ports.p2 = dst
        
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
