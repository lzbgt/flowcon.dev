
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
        int                 refcount

    cdef struct ipfix_store_attributes:
        long                next
        long                crc
        ipfix_attributes    attributes
        
    cdef struct ipfix_store_entry:
        long    next
        long    crc
        char    data[0]
        
    cdef struct ipfix_store_counts:
        long    flowindex
        long    bytes
        long    packets
