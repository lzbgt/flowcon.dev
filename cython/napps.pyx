# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z
# distutils: define_macros = NPY_NO_DEPRECATION_WARNING=

import numpy as np
cimport cython
cimport numpy as np

from common cimport *
from misc cimport logger, getreqval, debentries
from misc import backtable, backval, resval
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

        self.dtypes = [('next',       'u%d'%sizeof(apps.next)),
                       ('crc',        'u%d'%sizeof(apps.crc)),
                       ('protocol',   'u%d'%sizeof(apps.ports.protocol)),
                       ('src',        'u%d'%sizeof(apps.ports.src)),
                       ('dst',        'u%d'%sizeof(apps.ports.dst)),
                       ('ticks',      'u%d'%sizeof(apps.ticks)),
                       ('activity',   'u%d'%sizeof(apps.activity)),
                       ('refcount',   'u%d'%sizeof(apps.refcount))]
        
        cdef uint32_t pos, nports
        
        self._totalcount = 0
        self._zeroportcount = 0
        self._portrate = portrate
        self._minthreshold = minthreshold
        self._minactivity = minactivity

        nports = 2**8

        self._protobjs = np.zeros(nports, dtype=object)
        self._protarr = np.zeros(nports, dtype=np.uint64)
        cdef np.ndarray[np.uint64_t, ndim=1] parr = self._protarr
        
        self._protset = <uint64_t*>parr.data
        for pos in range(nports):
            self._protset[pos] = <uint64_t>NULL
            self._protobjs[pos] = None
            
    
    def backup(self, fileh, grp):
        super(Apps, self).backup(fileh, grp)
        
        backval(grp, 'totalcount', self._totalcount)
        backval(grp, 'zeroportcount', self._zeroportcount)

        pgrp = fileh.create_group(grp, 'protocols')
        for pos in range(len(self._protobjs)):
            pobj = self._protobjs[pos]
            if pobj is None: continue
            backtable(fileh, pgrp, 'p%d'%(pos), pobj.view(dtype=[('count','u4')]))
    
    def restore(self, fileh, grp):
        cdef int protocol
        pgrp = fileh.get_node(grp, 'protocols')

        for tbl in fileh.iter_nodes(pgrp, 'Table'):
            nm = tbl.name
            if nm[0] != 'p':
                logger('Ignoring protocol %s. Unexpected name.'%(nm)) 
                continue
            try:
                protocol = int(nm[1:])
            except:
                logger('Ignoring protocol %s. Unexpected name.'%(nm))
                continue
            if protocol <= 0 or protocol > 255: 
                logger('Ignoring protocol %s. Unexpected value %d.'%(nm, protocol))
                continue

            expsize = 2**16
            pobj = tbl.read()
            if len(pobj) != expsize:
                raise Exception("Unexpected table size: %d != %d"%(len(pobj), expsize))
            self._regprotocol(pobj, protocol)
        
        self._totalcount = <uint64_t>resval(grp, 'totalcount')    
        self._zeroportcount = <uint64_t>resval(grp, 'zeroportcount')
        
        super(Apps, self).restore(fileh, grp)
    
    @cython.boundscheck(False)
    cdef uint32_t* _regprotocol(self, countobj, uint8_t protocol):
        self._protobjs[protocol] = countobj
        
        cdef np.ndarray[np.uint32_t, ndim=1] arr = countobj
        
        cdef uint32_t* counts = <uint32_t*>arr.data
        self._protset[protocol] = <uint64_t>counts

        return counts
    
    @cython.boundscheck(False)
    cdef uint32_t* _addprotocol(self, uint8_t protocol):
        cdef uint32_t size = 2**16

        cdef countsobj = np.zeros(size, dtype=np.uint32)
        
        return self._regprotocol(countsobj, protocol)
    
    @cython.boundscheck(False)
    cdef void collectports(self, const ipfix_store_flow* flows, const ipfix_store_counts* counts, uint32_t num) nogil:
        cdef uint32_t pos, index
        cdef const ipfix_flow_tuple* flow
        cdef uint32_t* portcounts

        for pos in range(num):
            index = counts[pos].flowindex
            
            flow = cython.address(flows[index].flow)
            portcounts = <uint32_t*>(self._protset[flow.protocol])
            if portcounts == NULL:
                with gil:
                    portcounts = self._addprotocol(flow.protocol)
            
            portcounts[flow.srcport] += 1
            portcounts[flow.dstport] += 1
            self._totalcount += 2
            
    @cython.boundscheck(False)
    cdef int _evalports(self, ipfix_apps_ports* ports, uint16_t src, uint16_t dst) nogil:
        cdef float rt = self._portrate
        cdef uint32_t srccnt, dstcnt
        cdef uint32_t* portcounts
        
        portcounts = <uint32_t*>(self._protset[ports.protocol])
        if portcounts == NULL:
            with gil:
                logger("%s: attempt account for %d->%d ports from missing %d protocol"%(self._name, 
                                                                                      src,
                                                                                      dst,
                                                                                      ports.protocol))
            return 0

        srccnt = portcounts[src]
        dstcnt = portcounts[dst]
        
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
        cdef uint32_t* portcounts

        for pos in range(num):
            index = counts[pos].flowindex
            flow = cython.address(flows[index].flow)
            portcounts = <uint32_t*>(self._protset[flow.protocol])
            if portcounts == NULL:
                with gil:
                    logger("%s: attempt to remove %d->%d ports from missing %d protocol"%(self._name, 
                                                                                          flow.srcport,
                                                                                          flow.dstport,
                                                                                          flow.protocol))
                return
            
            portcounts[flow.srcport] -= 1
            portcounts[flow.dstport] -= 1
            self._totalcount -= 2

    def status(self):
        cdef baserep = super(Apps, self).status()
        cdef arr
        
        cdef int protbytes = 0
        cdef int protcount = 0
        for pos in range(len(self._protobjs)):
            arr = self._protobjs[pos]
            if arr is None: continue
            protbytes += arr.nbytes
            protcount += 1
        
        res = {'ports':{'bytes':int(protbytes), 'total':int(self._totalcount),
                        'protocols':{'count':int(protcount), 'bytes':int(self._protobjs.nbytes)}, 
                        'settings':{'rate':float(self._portrate), 
                                    'thres':int(self._minthreshold),
                                    'activity':self._minactivity}},
                'apps':{'collector':baserep, 'zeros':int(self._zeroportcount)}}

        return res

    def debprotocol(self, req):
            protnum = req.get('num', None)
            if protnum is None:
                prots = []
                for num in range(len(self._protobjs)):
                    if self._protobjs[num] is None: continue
                    prots.append(num)
                return {'protocols':prots}
            try:
                protnum = int(protnum)
            except:
                return {'error':'invalid protocol number %s'%(protnum)}

            if protnum >= len(self._protobjs):
                return {'error':'unexpected protocol number %d'%(protnum)}

            counters = self._protobjs[protnum]
            if counters is None:
                return {'error':'no records for protocol %d'%(protnum)}

            hist = req.get('hist', None)
            if hist is not None:
                try:
                    hist = int(hist)
                except:
                    return {'error':'unexpected histogram bins %s'%(hist)}
                hh, hb = np.histogram(counters, bins=hist)
                countrecs = []
                for num in range(len(hh)):
                    countrecs.append(("%.1f"%(hb[num]), hh[num]))
                
                return {'protocol':{'num':protnum, 'histogram':countrecs}}
            
            tres = getreqval(req, 'treshold', 0)

            countrecs = []
            for num in range(len(counters)):
                if counters[num] <= tres: continue
                countrecs.append((num, int(counters[num])))
                
            return {'protocol':{'num':protnum, 'counters':countrecs, 'total':int(self._totalcount)}}

    def debapps(self, req):
        apps = self.entries()
        return debentries(req, 'apps', self.entries(),
                   ('protocol', 'src', 'dst', 'ticks', 'activity', 'refcount'),
                   ('ticks', 'activity', 'refcount'))

    def debug(self, req):
        preq = req.get('protocol', None)
        if preq is not None:
            return self.debprotocol(preq)
        
        areq = req.get('apps', None)    
        if areq is not None:
            return self.debapps(areq)

        return None

