# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z

#### distutils: library_dirs = 
#### distutils: depends = 

cimport cython

from common cimport *
from misc cimport logger, showflow
from nquery cimport RawQuery
from collectors cimport Collector
from timecollect cimport SecondsCollector

cdef class Receiver(object):
    cdef sourceset
    cdef ExporterSet eset

    cdef RawQuery first

    def __cinit__(self, sourceset):
        self.sourceset = sourceset
        self.eset.exporter = 0

        self.first = None
        
    def __dealloc__(self):
        pass
        
    @cython.boundscheck(False)
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

    @cython.boundscheck(False)
    cdef void _onflows(self, const ipfix_flow* inflow, int bytes) nogil:
        cdef int count
        cdef ipfix_flow outflow
        cdef uint32_t index
        cdef ipfix_flow_tuple ftup
        cdef ipfix_attributes atup
        cdef ExporterSet* eset

        eset = cython.address(self.eset)

        count = bytes/sizeof(ipfix_flow)
        
        if self.first is None:
            while count > 0:
                convertflow(inflow, cython.address(outflow))
                
                if eset.exporter != outflow.exporter:
                    with gil:
                        self._setupexporter(eset, cython.address(outflow))
                
                copyflow(cython.address(outflow), cython.address(ftup))
                copyattr(cython.address(outflow), cython.address(atup))
                
                # register attributes
                index = eset.aadd(eset.aobj, cython.address(atup), 0, sizeof(ipfix_attributes))
                # register flow with attributes 
                index = eset.fadd(eset.fobj, cython.address(ftup), index, sizeof(ipfix_flow_tuple))
                # register counters with flow
                eset.tadd(eset.tobj, outflow.bytes, outflow.packets, index)
    
                inflow += 1
                count -= 1
        else:
            while count > 0:
                convertflow(inflow, cython.address(outflow))
                
                with gil:
                    self.onqueries(cython.address(outflow))

                if eset.exporter != outflow.exporter:
                    with gil:
                        self._setupexporter(eset, cython.address(outflow))
                
                copyflow(cython.address(outflow), cython.address(ftup))
                copyattr(cython.address(outflow), cython.address(atup))
                
                # register attributes
                index = eset.aadd(eset.aobj, cython.address(atup), 0, sizeof(ipfix_attributes))
                # register flow with attributes 
                index = eset.fadd(eset.fobj, cython.address(ftup), index, sizeof(ipfix_flow_tuple))
                # register counters with flow
                eset.tadd(eset.tobj, outflow.bytes, outflow.packets, index)
    
                inflow += 1
                count -= 1

    @cython.boundscheck(False)
    cdef void _setupexporter(self, ExporterSet* eset, const ipfix_flow* flow):
        cdef SecondsCollector seccollect
        cdef Collector flowcollect
        cdef Collector attrcollect

        eset.exporter = flow.exporter

        flowcollect, attrcollect, seccollect = self.sourceset.find(flow.exporter)
        
        eset.fobj = <void*>flowcollect
        eset.aobj = <void*>attrcollect
        eset.tobj = <void*>seccollect
        
        eset.fadd = <FlowAdd>flowcollect._add
        eset.aadd = <AppAdd>attrcollect._add
        eset.tadd = <TimeAdd>seccollect._add

    @cython.boundscheck(False)
    cdef void onqueries(self, const ipfix_flow* flow):
        cdef RawQuery next
        cdef RawQuery q = self.first

        while q is not None:
            q.onflow(flow)
            q = q.next
        
    @cython.boundscheck(False)
    def register(self, RawQuery q):
        q.next = self.first
        q.prev = None
        self.first = q
        if q.next is not None:
            q.next.prev = q
    
    @cython.boundscheck(False)
    def unregister(self, RawQuery q):
        cdef RawQuery next = q.next
        if next is not None:
            next.prev = q.prev
            q.next = None
        if q.prev is not None:
            q.prev.next = next
            q.prev = None
        else:
            self.first = next

@cython.boundscheck(False)
cdef void convertflow(const ipfix_flow* inflow, ipfix_flow* outflow) nogil:
    outflow.bytes = ntohl(inflow.bytes)
    outflow.packets = ntohl(inflow.packets)
    outflow.protocol = inflow.protocol
    outflow.tos = inflow.tos
    outflow.tcpflags = inflow.tcpflags
    outflow.srcport = ntohs(inflow.srcport)
    outflow.srcaddr = ntohl(inflow.srcaddr)
    outflow.srcmask = inflow.srcmask
    outflow.inpsnmp = ntohl(inflow.inpsnmp)
    outflow.dstport = ntohs(inflow.dstport)
    outflow.dstaddr = ntohl(inflow.dstaddr)
    outflow.dstmask = inflow.dstmask
    outflow.outsnmp = ntohl(inflow.outsnmp)
    outflow.nexthop = ntohl(inflow.nexthop)
    outflow.srcas = ntohl(inflow.srcas)
    outflow.dstas = ntohl(inflow.dstas)
    outflow.last = ntohl(inflow.last)
    outflow.first = ntohl(inflow.first)
    outflow.exporter = ntohl(inflow.exporter)
    
@cython.boundscheck(False)
cdef void copyflow(const ipfix_flow* flow, ipfix_flow_tuple* ftup) nogil:
    ftup.protocol = flow.protocol
    ftup.srcport = flow.srcport
    ftup.srcaddr = flow.srcaddr
    ftup.dstport = flow.dstport
    ftup.dstaddr = flow.dstaddr
    
@cython.boundscheck(False)
cdef void copyattr(const ipfix_flow* flow, ipfix_attributes* atup) nogil:
    atup.tos = flow.tos
    atup.tcpflags = flow.tcpflags
    atup.srcmask = flow.srcmask
    atup.inpsnmp = flow.inpsnmp
    atup.dstmask = flow.dstmask
    atup.outsnmp = flow.outsnmp
    atup.nexthop = flow.nexthop
    atup.srcas = flow.srcas
    atup.dstas = flow.dstas
