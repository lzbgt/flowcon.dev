# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z

#### distutils: library_dirs = 
#### distutils: depends = 

import numpy as np
import sys
cimport cython
cimport numpy as np

from common cimport *
from misc cimport logger
from collectors cimport SecondsCollector, Collector

def _dummy():
    "exists only to get rid of compile warnings"
    cdef int tmp = 0
    if tmp:
        _import_umath()    
        _import_array()

cdef class Receiver(object):
    cdef sourceset
    cdef uint32_t exporter
    cdef SecondsCollector   seccollect
    cdef Collector flowcollect
    cdef Collector attrcollect

    def __cinit__(self, sourceset):
        print "header size",sizeof(ipfix_header)
        print "set size",sizeof(ipfix_template_set_header)
        print "flow size",sizeof(ipfix_flow)
        print "ulong",sizeof(uint32_t)
        self.sourceset = sourceset
        self.exporter = 0
        self.seccollect = None
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
        cdef uint32_t (*fadd)(Collector slf, const void* ptr, uint32_t index, int dsize)
        cdef uint32_t (*aadd)(Collector slf, const void* ptr, uint32_t index, int dsize)
        cdef void     (*tadd)(SecondsCollector slf, uint32_t bts, uint32_t packets, uint32_t flowindex)
        cdef uint32_t index, pos, exporter
        cdef Collector fcol, acol
        cdef SecondsCollector scol
        
        exporter = self.exporter
        if exporter != flow.exporter:
            exporter = flow.exporter
            self.exporter = exporter
            self.flowcollect, self.attrcollect, self.seccollect = self.sourceset.find(ntohl(exporter))

        fcol = self.flowcollect
        acol = self.attrcollect
        scol = self.seccollect
        fadd = fcol._add
        aadd = acol._add
        tadd = scol._add

        #logger("%d"%count)
        while count > 0:
            if exporter != flow.exporter:
                exporter = flow.exporter
                # pull matching collectors and save them for later use
                self.flowcollect, self.attrcollect, self.seccollect = self.sourceset.find(ntohl(exporter))
                self.exporter = exporter
                # reinit everything for new collectors
                fcol = self.flowcollect
                acol = self.attrcollect
                scol = self.seccollect
                fadd = fcol._add
                aadd = acol._add
                tadd = scol._add
                    
            ftup_p = cython.address(ftup)
            atup_p = cython.address(atup)
            copyflow(flow, ftup_p)
            copyattr(flow, atup_p)
            
            # register attributes
            index = aadd(acol, atup_p, 0, sizeof(ipfix_attributes))
            # register flow with attributes 
            index = fadd(fcol, ftup_p, index, sizeof(ipfix_flow_tuple))
            # register counters with flow
            tadd(scol, flow.bytes, flow.packets, index)

            #logger("  (%s)"%(showflow(cython.address(ftup))))
            flow += 1
            count -= 1

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
