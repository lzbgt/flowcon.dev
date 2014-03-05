# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z
# distutils: define_macros = NPY_NO_DEPRECATION_WARNING=

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

    def __init__(self, float portrate, uint32_t minthreshold, uint32_t minactivity):
        super(Apps, self).__init__('apps', sizeof(ipfix_apps))
        
        cdef ipfix_apps apps

        self.dtypes = [('next',      'u%d'%sizeof(apps.next)),
                       ('crc',       'u%d'%sizeof(apps.crc)),
                       ('protocol',  'u%d'%sizeof(apps.ports.protocol)),
                       ('src',        'u%d'%sizeof(apps.ports.src)),
                       ('dst',        'u%d'%sizeof(apps.ports.dst))]                
        
        cdef uint32_t size
        
        self._totalcount = 0
        self._zeroportcount = 0
        self._portrate = portrate
        self._minthreshold = minthreshold
        self._minactivity = minactivity

        size = 2**16
        self._pcobj = np.zeros(size, dtype=np.uint32)
        
        cdef np.ndarray[np.uint32_t, ndim=1] arr = self._pcobj
        
        self._portcounts = <uint32_t*>arr.data
    
    @cython.boundscheck(False)
    cdef void collectports(self, const ipfix_store_flow* flows, const ipfix_store_counts* counts, uint32_t num) nogil:
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
    cdef int _evalports(self, ipfix_apps_ports* ports, uint16_t src, uint16_t dst) nogil:
        cdef float rt = self._portrate
        cdef uint32_t srccnt, dstcnt

        srccnt = self._portcounts[src]
        dstcnt = self._portcounts[dst]
        
        if srccnt > dstcnt:
            if srccnt >= self._minthreshold and srccnt >= dstcnt*rt:
                ports.src = 0    # ignore dst port
                ports.dst = src
                return 0
            # unknown application; sort the ports; consider smaller port as an application   
            if src > dst:
                ports.src = src
                ports.dst = dst
                return 1

            ports.src = dst
            ports.dst = src
            return 0

        if dstcnt >= self._minthreshold and srccnt*rt <= dstcnt:
            ports.src = 0    # ignore src port
            ports.dst = dst
            return 1
        # unknown application; sort the ports; consider smaller port as an application   
        if src > dst:
            ports.src = src
            ports.dst = dst
            return 1

        ports.src = dst
        ports.dst = src
        return 0

    @cython.boundscheck(False)
    cdef uint32_t getflowapp(self, const ipfix_flow_tuple* flow, int* ingress) nogil:
        cdef uint16_t dst, src
        cdef ipfix_apps_ports ports
        
        dst = flow.dstport
        src = flow.srcport
        if dst == 0:
            ingress[0] = 1 
            return 0
        
        ports.protocol = flow.protocol

        if src == 0:
            self._zeroportcount += 1
            ingress[0] = 1
            ports.src = 0
            ports.dst = dst
        else:   
            ingress[0] = self._evalports(cython.address(ports), src, dst)
            
        cdef uint32_t apppos
        
        cdef ipfix_apps* apprec = <ipfix_apps*>self._findentry(cython.address(ports), 
                                                               cython.address(apppos), sizeof(ipfix_apps_ports))
        apprec.ticks += 1

        if ports.src != 0:
            if apprec.activity < self._minactivity:
                # declare this app as unknown
                ports.src = 0
                ports.dst = 0
                apprec = <ipfix_apps*>self._findentry(cython.address(ports),
                                                      cython.address(apppos), sizeof(ipfix_apps_ports))
                apprec.ticks += 1
        
        return apppos        

    @cython.boundscheck(False)
    cdef void countapp(self, uint32_t index) nogil:
        cdef ipfix_apps* apprec = self._getapprec(index)
        apprec.refcount += 1

    @cython.boundscheck(False)
    cdef void checkactivity(self) nogil:
        cdef uint32_t pos
        cdef ipfix_apps* appset = self._getapprec(0)
        cdef ipfix_apps* apprec
        
        for pos in range(self.maxentries):
            apprec = appset+pos
            if apprec.ticks == 0 and apprec.refcount == 0:
                if apprec.activity != 0:    # check if it's already deleted
                    self._removeapp(apprec, pos)
                continue

            apprec.activity = apprec.ticks
            apprec.ticks = 0

    @cython.boundscheck(False)
    cdef ipfix_apps* _getapprec(self, uint32_t index) nogil:
        return (<ipfix_apps*>self.entryset)+index

    @cython.boundscheck(False)
    cdef void removeapp(self, uint32_t index) nogil:
        cdef ipfix_apps* apprec = self._getapprec(index)

        if apprec.refcount == 0:  # already deleted 
            with gil:
                logger("%s: deleting unreferenced entry %08x->%08x:%d[%d] activity:%d, ticks:%d"%(self._name, 
                                apprec.ports.src, apprec.ports.dst, apprec.ports.procotol, index,
                                apprec.ticks, apprec.activity))
            return
        if apprec.refcount > 1:   # still referenced
            apprec.refcount -= 1
            return

        if apprec.ticks != 0 or apprec.activity != 0: # still active, can not be deleted
            apprec.refcount = 0                       # however, it's no longer referenced
            return

        self._removeapp(apprec, index)

    @cython.boundscheck(False)
    cdef void _removeapp(self, ipfix_apps* apprec, uint32_t index) nogil:
        apprec.refcount = 0
        apprec.activity = 0
        apprec.ticks = 0

        self._removepos(<ipfix_store_entry*>apprec, index, self._width) # delete entry

    @cython.boundscheck(False)
    cdef void removeports(self, const ipfix_store_flow* flows, const ipfix_store_counts* counts, uint32_t num) nogil:
        cdef uint32_t pos, index
        cdef const ipfix_flow_tuple* flow
        cdef uint32_t* portcounts = self._portcounts

        for pos in range(num):
            index = counts[pos].flowindex
            flow = cython.address(flows[index].flow)
            portcounts[flow.srcport] -= 1
            portcounts[flow.dstport] -= 1
            self._totalcount -= 2

    def status(self):
        cdef baserep = super(Apps, self).status()
        
        res = {'ports':{'bytes':int(self._pcobj.nbytes), 'total':int(self._totalcount), 
                        'settings':{'rate':float(self._portrate), 
                                    'thres':int(self._minthreshold),
                                    'activity':self._minactivity}},
                'apps':{'collector':baserep, 'zeros':int(self._zeroportcount)}}

        return res

    def extstatus(self):
        cdef uint32_t* portcounts = self._portcounts
        cdef uint32_t pos, size = self._pcobj.size
        
        cdef ports = {}
        
        for pos in range(size):
            if portcounts[pos] == 0: continue
            ports[pos] = portcounts[pos]
            
        cdef apps = []
        cdef appset = self.entries()
        
        cdef int p1idx = appset.dtype.names.index('src')
        cdef int p2idx = appset.dtype.names.index('dst')
        
        for app in appset[1:self.end,0]:
            apps.append((int(app[p1idx]), int(app[p2idx])))
       
