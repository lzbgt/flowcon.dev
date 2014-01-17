# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z



#### distutils: libraries = 
#### distutils: library_dirs = 
#### distutils: depends = 

import numpy as np
cimport cython
cimport numpy as np

cdef extern from "netinet/in.h":
    int ntohs (int __netshort)
    long ntohl (long __netlong)

cdef extern from "string.h" nogil:
    int memcmp (const void *A1, const void *A2, size_t SIZE)

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

cdef class Receiver:
    cdef unsigned long adler
    
    def __cinit__(self):
        self.adler = adler32(0, <unsigned char*>0, 0)
        print "header size",sizeof(ipfix_header)
        print "set size",sizeof(ipfix_template_set_header)
        print "flow size",sizeof(ipfix_flow)
#        self.thisptr = new System()
#        self.thisptr.parsefile(fname)
#        self.thisptr.process()
        
    def __dealloc__(self):
        #del self.thisptr
        pass
        
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    def receive(self, const char* buffer, int size):
        cdef ipfix_template_set_header* header
        cdef ipfix_header* buf = <ipfix_header*>buffer
        cdef int pos = sizeof(ipfix_header)
        cdef unsigned short id
        cdef unsigned short hlen
        cdef int end
        cdef const char* flows

        cdef unsigned short buflen = ntohs(buf.length)
        if buflen > size:  # broken packet 
            return
        end = buflen - sizeof(ipfix_template_set_header)
        while pos <= end:
            header = <ipfix_template_set_header*>(buffer + pos)
            id = ntohs(header.id)
            hlen = ntohs(header.length)
            
            flows = buffer+pos+sizeof(ipfix_template_set_header)
            pos += hlen
            
            if pos > buflen: # broken packet
                return
            if id < MINDATA_SET_ID: # ignore all non data buffers
                continue
            self._onflows(flows, hlen-sizeof(ipfix_template_set_header))

        return
    
    cdef void _onflows(self, const char* fl, int bytes):
        cdef ipfix_flow* flow = <ipfix_flow*> fl
        cdef int count = bytes/sizeof(ipfix_flow)
        cdef ipfix_flow_tuple ftup
        cdef unsigned char* ptr
        cdef unsigned long crcval
         
        print count
        while count > 0:
            ftup.protocol = flow.protocol
            ftup.srcport = ntohs(flow.srcport)
            ftup.srcaddr = ntohl(flow.srcaddr)
            ftup.dstport = ntohs(flow.dstport)
            ftup.dstaddr = ntohl(flow.dstaddr)
        
            crcval = adler32(self.adler, <unsigned char*>cython.address(ftup), sizeof(ftup))

            print "  %08x (%d, %d, %08x, %d, %08x) %s"%(crcval, 
                                                        ftup.protocol, ftup.srcport, ftup.srcaddr,
                                                        ftup.dstport, ftup.dstaddr)
            flow+=1
            count -= 1

