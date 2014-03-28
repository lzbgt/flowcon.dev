# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z
# distutils: define_macros = NPY_NO_DEPRECATION_WARNING=

#### distutils: library_dirs = 
#### distutils: depends = 

import numpy as np
cimport cython
cimport numpy as np

from common cimport *
from napps cimport Apps

from misc cimport logger, showapp, showflow, showattr, minsize, growthrate, shrinkrate, debentries 
from misc import backtable, backval, resval

def _dummy():
    "exists only to get rid of compile warnings"
    cdef int tmp = 0
    if tmp:
        _import_umath()    
        _import_array()

cdef class Collector(object):

    def __init__(self, nm, int width, int size = minsize):
        self._name = nm
        self._width = width
        self.freepos = 0
        self.freecount = 0
        self.end = 1            # 0 is invalid position
        self.adler = adler32(0, <unsigned char*>0, 0)
        
        self._entries = np.zeros((1, width), dtype=np.uint8)
        self._indices = np.zeros(1, dtype=np.dtype('u4'))
        self._resz(size)
        
    def backup(self, fileh, grp):
        backtable(fileh, grp, 'entries', self.entries())
        backtable(fileh, grp, 'indices', self.indices())
        backval(grp, 'freepos', self.freepos)
        backval(grp, 'freecount', self.freecount)
        backval(grp, 'end', self.end)

    def restore(self, fileh, grp):
        self.freepos = <uint32_t>resval(grp, 'freepos')
        self.freecount = <uint32_t>resval(grp, 'freecount')
        self.end = <uint32_t>resval(grp, 'end')
        
        ents = fileh.get_node(grp, 'entries')
        if self._width != ents.rowsize:
            raise Exception("%s width (%d) does not match stored width (%d)"%(self._name, 
                                                                              self._width, 
                                                                              ents.rowsize))
            
        #TMP
        #print "%s restore pos:%d count:%d end:%d len:%d"%(self._name, self.freepos, self.freecount, self.end, len(ents))
        #
        self._resz(len(ents))
        ents.read(out=self.entries())
        
        inds = fileh.get_node(grp, 'indices')
        inds.read(out=self.indices())
        
    @cython.boundscheck(False)
    cdef int _resz(self, int size):
        cdef int bits = int(np.math.log(2*size-1, 2))
        cdef int indsize = 2**bits
        cdef np.ndarray[np.uint8_t, ndim=2] arr
        cdef np.ndarray[np.uint32_t, ndim=1] inds
        
        self.maxentries = size
        self.mask = 2**bits-1

        self._entries.resize((size, self._width), refcheck=False)
        arr = self._entries
        self.entryset = <unsigned char*>arr.data

        if indsize != self._indices.size:
            # brand new space for indices
            self._indices = np.zeros(indsize, dtype=self._indices.dtype)
            inds = self._indices
            self.indexset = <uint32_t*>inds.data
            return True
        return False

    @cython.boundscheck(False)
    cdef uint32_t _add(self, const void* ptr, uint32_t index, int dsize) nogil:
        cdef uint32_t pos
        
        self._findentry(ptr, cython.address(pos), dsize)
        
        return pos
    
    @cython.boundscheck(False)
    cdef ipfix_store_entry* _findentry(self, const void* ptr, uint32_t* outpos, int dsize) nogil:    
        cdef uint32_t crc, pos, ind, lastpos
        cdef ipfix_store_entry* entryrec
        cdef int sz = self._width
        
        #TMP
#        with gil:
#            print "%s: find 0x%08x, 0x%08x, %d"%(self._name, <uint64_t>ptr, <uint64_t>outpos, dsize)
        #
        
        #logger("adding to %s"%(self._name))
        crc = adler32(self.adler, <unsigned char*>ptr, dsize)
        ind = crc & self.mask
        pos = self.indexset[ind]
        #TMP
#        with gil:
#            logger("  crc:%08x ind:%d pos:%d addr:%08x"%(crc, ind, pos, <uint64_t>cython.address(self.indexset[ind])))
        #
        if pos > 0:
            while pos > 0:
                entryrec = <ipfix_store_entry*>(self.entryset+pos*sz)
                if memcmp(entryrec.data, ptr, dsize) == 0:
                    # found identical flow
                    outpos[0] = pos
                    #TMP
#                    with gil:
#                        print "%s: done 1"%(self._name)
                    #
                    return entryrec
                pos = entryrec.next
            # need new
            pos = self._findnewpos(sz)
            #logger("  found:%d"%(pos))
            if pos == 0:    # need resize
                with gil:
                    self._grow()
                return self._findentry(ptr, outpos, dsize)    # repeat on bigger array

            entryrec.next = pos # link to previous
        else:
            #TMP
#            with gil:
#                logger("%s prefound: pos %d, count %d, end: %d max:%d"%(self._name, self.freepos, self.freecount, self.end, self.maxentries))
            #
            pos = self._findnewpos(sz)
            #TMP
#            with gil:
#                logger("  found:%d"%(pos))
            #
            if pos == 0:    # need resize
                with gil:
                    self._grow()
                return self._findentry(ptr, outpos, dsize)    # repeat on bigger array
            self.indexset[ind] = pos
        
        entryrec = <ipfix_store_entry*>(self.entryset+pos*sz)
        
        entryrec.next = 0 
        entryrec.crc = crc
        memcpy(entryrec.data, ptr, dsize)

        outpos[0] = pos
        
        #TMP
#        with gil:
#            print "%s: done 2"%(self._name)
        #        
        return entryrec

    @cython.boundscheck(False)
    cdef uint32_t _findnewpos(self, uint32_t sz) nogil:
        cdef uint32_t pos = self.freepos
        cdef ipfix_store_entry* entryrec
        
        if pos > 0:
            entryrec = <ipfix_store_entry*>(self.entryset+pos*sz)
            self.freepos = entryrec.next
            self.freecount -= 1
            return pos

        if self.end >= self.maxentries:    # need resize
            return 0
        pos = self.end
        self.end += 1
        return pos
    
    @cython.boundscheck(False)
    cdef void _grow(self):
        cdef uint32_t size = <uint32_t>(self.maxentries*growthrate)
        self._resize(size)

    @cython.boundscheck(False)
    cdef void _resize(self, uint32_t size):
        cdef uint32_t count = self.end
        cdef int sz = self._width
        cdef uint32_t mask, ind, indpos, pos
        cdef unsigned char* eset
        cdef uint32_t* iset
        cdef ipfix_store_entry* entry

        #logger('resizing %s %d->%d'%(self._name, self.maxentries, size))
        if not self._resz(size): return
        # lets fix indices and links
        mask = self.mask
        eset = self.entryset
        iset = self.indexset

        for pos in range(1, count):
            entry = <ipfix_store_entry*>(eset+pos*sz)
            ind = entry.crc & mask
            entry.next = iset[ind]
            iset[ind] = pos
    
    @cython.boundscheck(False)
    cdef void _removepos(self, ipfix_store_entry* entryrec, uint32_t pos, int sz) nogil:
        cdef ipfix_store_entry* prevrec
        cdef uint32_t ind, prevpos
        cdef unsigned char* eset = self.entryset
        
        ind = entryrec.crc & self.mask
        prevpos = self.indexset[ind]
        if prevpos == pos:
            self.indexset[ind] = entryrec.next
        else:
            prevrec = <ipfix_store_entry*>(eset+prevpos*sz)
            prevpos = prevrec.next
            while prevpos != pos:
                if prevpos == 0:   # this should never happen
                    with gil:
                        logger('%s: unexpected value for prevrec.next; prevpos:%d ind:%d pos:%d'%(self._name, 
                                                                                                  prevpos, ind, pos))
                    return
                prevrec = <ipfix_store_entry*>(eset+prevpos*sz)
                prevpos = prevrec.next
            prevrec.next = entryrec.next

        entryrec.next = self.freepos
        self.freepos = pos
        self.freecount += 1
    
    @cython.boundscheck(False)
    cdef ipfix_store_entry* _get(self, int pos):
        return <ipfix_store_entry*>(self.entryset+pos*self._width)

    def entries(self):
        return self._entries.view(dtype=self.dtypes)[:,0]
    
    def indices(self):
        return self._indices.view(dtype=[('index', 'u4')])
    
    def status(self):
        return {'entries':{'mask':('0x%08x'%(self.mask)), 
                           'maxentries':int(self.maxentries),
                           'bytes':int(self._entries.nbytes),
                           'count':int(self.end-1-self.freecount),
                           'free':int(self.freecount)}, 
                'indices':{'bytes':int(self._indices.nbytes),
                           'size':len(self._indices)}}

    
cdef class FlowCollector(Collector):

    def __init__(self, nm, attribs):
        super(FlowCollector, self).__init__(nm, sizeof(ipfix_store_flow))
        cdef ipfix_store_flow flow
        self._attributes = attribs
        
        self.dtypes = [('next',     'u%d'%sizeof(flow.next)),
                       ('crc',      'u%d'%sizeof(flow.crc)),
                       ('protocol', 'u%d'%sizeof(flow.flow.protocol)),
                       ('srcport', 'u%d'%sizeof(flow.flow.srcport)),
                       ('srcaddr', 'u%d'%sizeof(flow.flow.srcaddr)),
                       ('dstport', 'u%d'%sizeof(flow.flow.dstport)),
                       ('dstaddr', 'u%d'%sizeof(flow.flow.dstaddr)),
                       ('attrindex','u%d'%sizeof(flow.attrindex)),
                       ('refcount','u%d'%sizeof(flow.refcount))]

    @cython.boundscheck(False)
    cdef uint32_t _add(self, const void* ptr, uint32_t index, int dsize) nogil:
        cdef uint32_t pos
        
        cdef ipfix_store_flow* flowrec = <ipfix_store_flow*>self._findentry(ptr, cython.address(pos), dsize)
        
        if flowrec.refcount == 0:
            flowrec.refcount = 1
            flowrec.attrindex = index
        else:
            flowrec.refcount += 1
            if flowrec.attrindex != index:
                #self._report_attr(flowrec, index)
                flowrec.attrindex = index

        return pos

    @cython.boundscheck(False)
    cdef void _report_attr(self, const ipfix_store_flow* flowrec, uint32_t index) nogil:
        cdef ipfix_store_attributes* prev
        cdef ipfix_store_attributes* curr

        with gil:
            prev = <ipfix_store_attributes*>self._attributes._get(flowrec.attrindex)
            curr = <ipfix_store_attributes*>self._attributes._get(index)
            
            logger('%s changed for flow (%s)\n  <- %s\n  -> %s'%(self._name, showflow(cython.address(flowrec.flow)), 
                                                                 showattr(cython.address(prev.attributes)),
                                                                 showattr(cython.address(curr.attributes))))

    @cython.boundscheck(False)
    cdef const ipfix_store_flow* getflows(self) nogil:
        return <ipfix_store_flow*>self.entryset

    @cython.boundscheck(False)
    cdef void remove(self, const ipfix_store_counts* counts, uint32_t num) nogil:
        cdef unsigned char* eset = self.entryset
        cdef int sz = self._width
        cdef ipfix_store_flow* flow
        cdef ipfix_store_flow* nextflow
        cdef uint32_t pos, maxindex = 0, index
        
        for pos in range(num):
            index = counts[pos].flowindex

            flow = <ipfix_store_flow*>(eset+index*sz)

            if flow.refcount == 0:  # already deleted
                with gil:
                    logger("%s: deleting unreferenced flow %s[%d]"%(self._name, 
                                                showflow(cython.address(flow.flow)), index))
                continue
            if flow.refcount > 1:   # still referenced
                flow.refcount -= 1
                continue

            flow.refcount = 0
            if maxindex < index: maxindex = index

            self._removepos(<ipfix_store_entry*>flow, index, sz) # delete entry
            # let's link removed flow in reverse direction;
            # assuming flow is first in free list
            if flow.next != 0:
                nextflow = <ipfix_store_flow*>(eset+flow.next*sz)
                flow.attrindex = nextflow.attrindex
                nextflow.attrindex = index
            else:
                flow.attrindex = 0

        if maxindex > 0:            # something was actually deleted
            with gil:
                self._shrink(maxindex)    # try to shrink
        
    @cython.boundscheck(False)
    cdef void _shrink(self, uint32_t maxpos):
        cdef uint32_t newsize
        cdef uint32_t last = self.end-1

        if maxpos != last: return

        logger("%s: shrinking %d"%(self._name, maxpos))
        
        cdef unsigned char* eset = self.entryset        
        cdef uint32_t sz = self._width
        cdef ipfix_store_flow* flow = <ipfix_store_flow*>(eset+maxpos*sz)
        cdef ipfix_store_flow* nextflow
        cdef ipfix_store_flow* prevflow

        # let's shrink inventory until first non-free record
        while flow.refcount == 0:
            if flow.next > 0:   # fix next
                nextflow = <ipfix_store_flow*>(eset+flow.next*sz)
                nextflow.attrindex = flow.attrindex
            if flow.attrindex > 0: # fix previous
                prevflow = <ipfix_store_flow*>(eset+flow.attrindex*sz)
                prevflow.next = flow.next
            else:  # it's the first one in the list 
                self.freepos = flow.next
            self.freecount -= 1
            last -= 1
            if last == 0: break
            flow = <ipfix_store_flow*>(eset+last*sz)

        self.end = last+1
        if last*3 < self.maxentries: # need to shrink whole buffer to release some unused memory
            newsize = <uint32_t>(self.maxentries/shrinkrate)
            if newsize < minsize: return
            self._resize(newsize)

    def debug(self, req):
        return debentries(req, 'flows', self.entries(), 
                          ('protocol', 'srcport', ('srcaddr', 'ip'), 'dstport', 
                                                  ('dstaddr', 'ip'), 'attrindex', 'refcount'), 
                          ('refcount',))


cdef class AttrCollector(Collector):

    def __init__(self, nm):
        super(AttrCollector, self).__init__(nm, sizeof(ipfix_store_attributes))
        cdef ipfix_store_attributes attr

        self.dtypes = [('next',      'u%d'%sizeof(attr.next)),
                       ('crc',       'u%d'%sizeof(attr.crc)),
                       ('tos',       'u%d'%sizeof(attr.attributes.tos)),
                       ('tcpflags',  'u%d'%sizeof(attr.attributes.tcpflags)),
                       ('srcmask',   'u%d'%sizeof(attr.attributes.srcmask)),
                       ('inpsnmp',   'u%d'%sizeof(attr.attributes.inpsnmp)),
                       ('dstmask',   'u%d'%sizeof(attr.attributes.dstmask)),
                       ('outsnmp',   'u%d'%sizeof(attr.attributes.outsnmp)),
                       ('nexthop',   'u%d'%sizeof(attr.attributes.nexthop)),
                       ('srcas',     'u%d'%sizeof(attr.attributes.srcas)),
                       ('dstas',     'u%d'%sizeof(attr.attributes.dstas))]
        
    def debug(self, req):
        return debentries(req, 'attributes', self.entries(), 
                          ('tos', 'tcpflags', 'srcmask', 'inpsnmp', 'dstmask', 'outsnmp', 
                           ('nexthop', 'ip'), 'srcas', 'dstas'), ())



cdef class AppFlowCollector(Collector):

    def __init__(self, nm, Apps apps, AttrCollector attribs):
        super(AppFlowCollector, self).__init__(nm, sizeof(ipfix_app_flow))
        cdef ipfix_app_flow flow
        
        self._apps = apps
        self._attributes = attribs
        
        self.dtypes = [('next',         'u%d'%sizeof(flow.next)),
                       ('crc',          'u%d'%sizeof(flow.crc)),
                       ('application',  'u%d'%sizeof(flow.app.application)),
                       ('srcaddr',      'u%d'%sizeof(flow.app.srcaddr)),
                       ('dstaddr',      'u%d'%sizeof(flow.app.dstaddr)),
                       ('inattrindex',  'u%d'%sizeof(flow.inattrindex)),
                       ('outattrindex', 'u%d'%sizeof(flow.outattrindex)),
                       ('refcount',     'u%d'%sizeof(flow.refcount))]

    @cython.boundscheck(False)
    cdef void findapp(self, const ipfix_app_tuple* atup, uint32_t attrindex, AppFlowValues* vals, int ingress) nogil:
        cdef uint32_t pos
        cdef ipfix_app_flow* aflow

        aflow = <ipfix_app_flow*>self._findentry(atup, cython.address(pos), sizeof(ipfix_app_tuple))
        
        if ingress != 0:
            aflow.inattrindex = attrindex
        else:
            aflow.outattrindex = attrindex
        
        vals.crc = aflow.crc
        vals.pos = pos
        
    @cython.boundscheck(False)    
    cdef void countflowapp(self, ipfix_app_flow* aflow) nogil:
        if aflow.refcount == 0:
            # newly created flow app; let's account for it in global apps
            (<Apps>self._apps).countapp(aflow.app.application)

        aflow.refcount += 1
        
    @cython.boundscheck(False)    
    cdef void removeapps(self, const ipfix_app_counts* counts, uint32_t num) nogil:
        cdef int sz = self._width
        cdef uint32_t pos, index
        cdef unsigned char* eset = self.entryset
        cdef ipfix_app_flow* aflow

        for pos in range(num):
            index = counts[pos].appindex

            aflow = <ipfix_app_flow*>(eset+index*sz)

            if aflow.refcount == 0:  # already deleted
                with gil:
                    logger("%s: deleting unreferenced app flow %s[%d]"%(self._name, 
                                                showapp(cython.address(aflow.app)), index))
                continue
            if aflow.refcount > 1:   # still referenced
                aflow.refcount -= 1
                continue

            aflow.refcount = 0

            (<Apps>self._apps).removeapp(aflow.app.application)

            self._removepos(<ipfix_store_entry*>aflow, index, sz) # delete entry
            
    def debug(self, req):
        return debentries(req, 'appflows', self.entries(), 
                           ('application', ('srcaddr', 'ip'), ('dstaddr', 'ip'), 
                            'inattrindex', 'outattrindex', 'refcount'), 
                           ('refcount',))

            
