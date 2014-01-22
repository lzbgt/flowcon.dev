# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z

#### distutils: library_dirs = 
#### distutils: depends = 

import numpy as np
import sys
cimport cython
cimport numpy as np

def simplelogger(msg):
    print msg
    sys.stdout.flush()

logger = simplelogger

def setlogger(lgr):
    global logger
    logger = lgr

cdef extern from "stdint.h":
    ctypedef long uint32_t
    ctypedef long uint64_t

cdef extern from "netinet/in.h":
    int ntohs (int __netshort)
    long ntohl (long __netlong)

cdef extern from "string.h" nogil:
    int memcmp (const void *A1, const void *A2, size_t SIZE)
    void *memcpy(void *restrict, const void *restrict, size_t)

cdef extern from "zlib.h":
    long adler32(long crc, const unsigned char * buf, int len)

cdef extern:
    int _import_array()
    int _import_umath()

def _dummy():
    "exists only to get rid of compile warnings"
    cdef int tmp = 0
    if tmp:
        _import_umath()    
        _import_array()

cdef extern from "ipfix.h":
    cdef int MINDATA_SET_ID

    cdef struct ipfix_header:
        int version
        int length
        int exportTime
        int sequenceNumber
        int observationDomainId

    cdef struct ipfix_template_set_header:
        int id
        int length
        
    cdef struct ipfix_flow:
        long bytes
        long packets
        int  protocol
        int  tos
        int  tcpflags
        int  srcport
        long srcaddr
        int  srcmask
        long inpsnmp
        int  dstport
        long dstaddr
        int  dstmask
        long outsnmp
        long nexthop
        long srcas
        long dstas
        long last
        long first
        long exporter

    cdef struct ipfix_flow_tuple:
        int  protocol
        int  srcport
        long srcaddr
        int  dstport
        long dstaddr

    cdef struct ipfix_attributes:
        int  tos
        int  tcpflags
        int  srcmask
        long inpsnmp
        int  dstmask
        long outsnmp
        long nexthop
        long srcas
        long dstas
        
    cdef struct ipfix_store_flow:
        long                next
        long                crc
        ipfix_flow_tuple    flow
        long                attrindex

    cdef struct ipfix_store_attributes:
        long                next
        long                crc
        ipfix_attributes    attributes
        
    cdef struct ipfix_store_entry:
        long    next
        long    crc
        char    data[0]

cdef class Receiver(object):
    cdef sourceset
    cdef uint32_t exporter
    cdef Collector flowcollect
    cdef Collector attrcollect

    def __cinit__(self, sourceset):
        print "header size",sizeof(ipfix_header)
        print "set size",sizeof(ipfix_template_set_header)
        print "flow size",sizeof(ipfix_flow)
        print "ulong",sizeof(uint32_t)
        self.sourceset = sourceset
        self.exporter = 0
        self.flowcollect = None
        self.attrcollect = None
        
    def __dealloc__(self):
        pass
        
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    def receive(self, const char* buffer, int size):
        cdef ipfix_template_set_header* header
        cdef ipfix_header* buf = <ipfix_header*>buffer
        cdef int pos = sizeof(ipfix_header)
        cdef unsigned short id
        cdef unsigned short hlen
        cdef int end
        cdef const ipfix_flow* flows

        cdef unsigned short buflen = ntohs(buf.length)
        if buflen > size:  # broken packet 
            return
        end = buflen - sizeof(ipfix_template_set_header)
        while pos <= end:
            header = <ipfix_template_set_header*>(buffer + pos)
            id = ntohs(header.id)
            hlen = ntohs(header.length)
            
            flows = <ipfix_flow*>(buffer+pos+sizeof(ipfix_template_set_header))
            pos += hlen
            
            if pos > buflen: # broken packet
                return
            if id < MINDATA_SET_ID: # ignore all non data buffers
                continue
            self._onflows(flows, hlen-sizeof(ipfix_template_set_header))

        return
    
    cdef void _onflows(self, const ipfix_flow* flow, int bytes):
        cdef int count = bytes/sizeof(ipfix_flow)
        cdef ipfix_flow_tuple ftup
        cdef ipfix_flow_tuple* ftup_p
        cdef unsigned char* ptr
        cdef ipfix_attributes atup
        cdef ipfix_attributes* atup_p
        cdef uint32_t (*fadd)(Collector slf, const void* ptr, uint32_t index)
        cdef uint32_t (*aadd)(Collector slf, const void* ptr, uint32_t index)
        cdef uint32_t index, pos, exporter
        cdef Collector fcol, acol
        
        exporter = self.exporter
        if exporter != flow.exporter:
            exporter = flow.exporter
            self.exporter = exporter
            self.flowcollect, self.attrcollect = self.sourceset.find(ntohl(exporter))

        fcol = self.flowcollect
        acol = self.attrcollect
        fadd = fcol._add
        aadd = acol._add

        #logger("%d"%count)
        while count > 0:
            if exporter != flow.exporter:
                exporter = flow.exporter
                # pull matching collectors and save them for later use
                self.flowcollect, self.attrcollect = self.sourceset.find(ntohl(exporter))
                self.exporter = exporter
                # reinit everything for new collectors
                fcol = self.flowcollect
                acol = self.attrcollect
                fadd = fcol._add
                aadd = acol._add
                    
            ftup_p = cython.address(ftup)
            atup_p = cython.address(atup)
            copyflow(flow, ftup_p)
            copyattr(flow, atup_p)

            index = aadd(acol, atup_p, 0)
            fadd(fcol, ftup_p, index)

            #logger("  (%s)"%(showflow(cython.address(ftup))))
            flow += 1
            count -= 1

cdef object showflow(const ipfix_flow_tuple* ftup):
    return "%2d, %5d, %08x, %5d, %08x"%(ftup.protocol, ftup.srcport, ftup.srcaddr,
                                        ftup.dstport, ftup.dstaddr)

cdef void copyflow(const ipfix_flow* flow, ipfix_flow_tuple* ftup):
    ftup.protocol = flow.protocol
    ftup.srcport = ntohs(flow.srcport)
    ftup.srcaddr = ntohl(flow.srcaddr)
    ftup.dstport = ntohs(flow.dstport)
    ftup.dstaddr = ntohl(flow.dstaddr)
    
cdef void copyattr(const ipfix_flow* flow, ipfix_attributes* atup):
    atup.tos = flow.tos
    atup.tcpflags = flow.tcpflags
    atup.srcmask = flow.srcmask
    atup.inpsnmp = flow.inpsnmp
    atup.dstmask = flow.dstmask
    atup.outsnmp = flow.outsnmp
    atup.nexthop = flow.nexthop
    atup.srcas = flow.srcas
    atup.dstas = flow.dstas

cdef class Collector(object):
    cdef unsigned char* entryset
    cdef uint32_t adler
    cdef uint32_t last
    cdef uint32_t* indexset
    cdef uint32_t mask, maxentries, freepos
    cdef int _width
    cdef _entries
    cdef _indices
    cdef dtypes
    cdef _name

    def __init__(self, nm, int width, int size = 16):
        self._name = nm
        self._width = width
        self.freepos = 0
        self.last = 1
        self.adler = adler32(0, <unsigned char*>0, 0)
        
        self._entries = np.zeros((1, width), dtype=np.uint8)
        self._indices = np.zeros(1, dtype=np.dtype('u4'))
        self._resize(size)
        
    cdef void _resize(self, int size):
        cdef int bits = int(np.math.log(2*size-1, 2))
        cdef int indsize = 2**bits
        self.maxentries = size
        self.mask = 2**bits-1

        self._entries.resize((size, self._width), refcheck=False)
        cdef np.ndarray[np.uint8_t, ndim=2] arr = self._entries

        self._indices.resize(indsize, refcheck=False)
        cdef np.ndarray[np.uint32_t, ndim=1] inds = self._indices

        self.entryset = <unsigned char*>arr.data
        self.indexset = <uint32_t*>inds.data

    cdef uint32_t _add(self, const void* ptr, uint32_t index):
        cdef uint32_t crc, pos, ind, lastpos
        cdef ipfix_store_entry* entryrec
        cdef int sz = self._width
        
        #logger("adding to %s"%(self._name))
        crc = adler32(self.adler, <unsigned char*>ptr, sz)
        ind = crc & self.mask
        pos = self.indexset[ind]
        #logger("  crc:%08x ind:%d pos:%d addr:%08x"%(crc, ind, pos, <uint64_t>cython.address(self.indexset[ind])))
        if pos > 0:
            while pos > 0:
                entryrec = <ipfix_store_entry*>(self.entryset+pos*sz)
                if memcmp(cython.address(entryrec.data[0]), ptr, sz) == 0:
                    # found identical flow
                    self._onindex(entryrec, index)  # commit index value
                    return pos
                pos = entryrec.next
            # need new
            pos = self._findnewpos(sz)
            #logger("  found:%d"%(pos))
            if pos == 0:    # need resize
                self._grow()
                return self._add(ptr, index)    # repeat on bigger array

            entryrec.next = pos # link to previous
        else:
            pos = self._findnewpos(sz)
            #logger("  found:%d"%(pos))
            if pos == 0:    # need resize
                self._grow()
                return self._add(ptr, index)    # repeat on bigger array
            self.indexset[ind] = pos
        
        entryrec = <ipfix_store_entry*>(self.entryset+pos*sz)
        
        entryrec.next = 0 
        entryrec.crc = crc
        memcpy(cython.address(entryrec.data[0]), ptr, sz)
        self._onindex(entryrec, index)
        
        return pos

    cdef uint32_t _findnewpos(self, uint32_t sz):
        cdef uint32_t pos = self.freepos
        cdef ipfix_store_entry* entryrec
        
        if pos > 0:
            entryrec = <ipfix_store_entry*>(self.entryset+pos*sz)
            self.freepos = entryrec.next
            return pos

        if self.last >= self.maxentries:    # need resize
            return 0
        pos = self.last
        self.last += 1
        return pos
    
    cdef void _grow(self):
        cdef uint32_t size = self.maxentries*2
        logger('resizing %s to %d'%(self._name, size))
        print "before",self.entries()
        self._resize(size)
        print "after",self.entries()
    
    cdef void _removepos(self, uint32_t pos):
        cdef ipfix_store_entry* entryrec
        cdef ipfix_store_entry* prevrec
        cdef uint32_t ind, prevpos
        cdef int sz = self._width
        
        entryrec = <ipfix_store_entry*>(self.entryset+pos*sz)
        ind = entryrec.crc & self.mask
        prevpos = self.indexset[ind]
        if prevpos == pos:
            self.indexset[ind] = entryrec.next
        else:
            prevrec = <ipfix_store_entry*>(self.entryset+prevpos*sz)
            prevpos = prevrec.next
            while prevpos != pos:
                if prevpos == 0:   # this should never happen
                    logger('unexpected value for prevrec.next; prevpos:%d ind:%d pos:%d'%(prevpos, ind, pos))
                    return
                prevrec = <ipfix_store_entry*>(self.entryset+prevpos*sz)
                prevpos = prevrec.next
            prevrec.next = entryrec.next
            
        entryrec.next = self.freepos
        self.freepos = pos

    def entries(self):
        return self._entries.view(dtype=np.dtype(self.dtypes))
    
    def indices(self):
        return self._indices.view(dtype='u4')
    
    cdef void _onindex(self, ipfix_store_entry* entry, uint32_t index):
        pass
    
cdef class FlowCollector(Collector):

    def __init__(self, nm):
        super(FlowCollector, self).__init__(nm, sizeof(ipfix_store_flow))
        cdef ipfix_store_flow flow
        
        self.dtypes = [('next',     'u%d'%sizeof(flow.next)),
                       ('crc',      'u%d'%sizeof(flow.crc)),
                       ('flow',     'a%d'%sizeof(flow.flow)),
                       ('attrindex','u%d'%sizeof(flow.attrindex))]

    cdef void _onindex(self, ipfix_store_entry* entry, uint32_t index):
        cdef ipfix_store_flow* flowrec = <ipfix_store_flow*>entry
        if flowrec.attrindex != index:
            if flowrec.attrindex != 0:
                logger('%s changed for flow (%s) %08x -> %d'%(self._name, showflow(cython.address(flowrec.flow)), 
                                                              flowrec.attrindex, index))
            flowrec.attrindex = index

cdef class AttrCollector(Collector):

    def __init__(self, nm):
        super(AttrCollector, self).__init__(nm, sizeof(ipfix_store_attributes))
        cdef ipfix_store_attributes attr

        self.dtypes = [('next',      'u%d'%sizeof(attr.next)),
                       ('crc',       'u%d'%sizeof(attr.crc)),
                       ('attributes','a%d'%sizeof(attr.attributes))]        


#cdef void checkbytes(const unsigned char* buf, const char* pref, const char* nm, uint32_t nbytes):
#    logger(" %s"%pref)
#    logger("  %s: %08x[%d]"%(nm, <uint64_t>buf, nbytes))
#    for i in range(nbytes):
#        if buf[i] != 0:
#            logger("   %02x[%d] != 0"%(buf[i], i))
#    logger("  --------")
