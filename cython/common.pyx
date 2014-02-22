
cdef extern from "stdint.h":
    ctypedef long uint16_t
    ctypedef long uint32_t
    ctypedef long uint64_t

cdef extern from "netinet/in.h":
    int ntohs (int __netshort) nogil
    long ntohl (long __netlong) nogil

cdef extern from "string.h" nogil:
    int memcmp (const void *A1, const void *A2, size_t SIZE) nogil
    void *memcpy(void *restrict, const void *restrict, size_t) nogil

cdef extern from "zlib.h":
    long adler32(long crc, const unsigned char * buf, int len) nogil

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

    cdef struct ipfix_app_tuple:
        long    application
        long    srcaddr
        long    dstaddr
        
    cdef struct AppFlowValues:
        long    crc
        long    pos
        
    ctypedef int (*FlowAppCallback)(void* obj, const ipfix_store_flow* flow, AppFlowValues* vals) nogil
        
    cdef struct ipfix_app_flow:
        long                next
        long                crc
        ipfix_app_tuple     app
        long                inattrindex
        long                outattrindex

    cdef struct ipfix_apps_ports:
        int     protocol
        int     src
        int     dst
        
    cdef struct ipfix_apps:
        long                next
        long                crc
        ipfix_apps_ports    ports
        
    cdef struct ipfix_query_info:
        const void*                    entries
        long                           count
        const ipfix_store_flow*        flows
        const ipfix_app_flow*          appflows
        const ipfix_apps*              apps
        const ipfix_store_attributes*  attrs
        long                           stamp
        long                           exporter
        FlowAppCallback                callback
        void*                          callobj

    cdef struct ipfix_query_buf:
        void*   data
        long    count
        long*   poses
        long    mask

    cdef struct ipfix_query_pos:
        long    bufpos
        long    curpos
        long    countpos
        long    totbytes
        long    totpackets
                
    cdef struct AppFlowObjects:
        void*   minutes
        void*   apps
        void*   flows

    cdef struct AppsCollection:
        long            next
    
        AppFlowValues   values
    
        long            inbytes
        long            inpackets
        long            outbytes
        long            outpackets

    cdef struct ipfix_app_counts:
        long            appindex
        long            inbytes
        long            inpackets
        long            outbytes
        long            outpackets

    ctypedef uint32_t (*FlowAdd)(void* slf, const void* ptr, uint32_t index, int dsize) nogil
    ctypedef uint32_t (*AppAdd)(void* slf, const void* ptr, uint32_t index, int dsize) nogil
    ctypedef void     (*TimeAdd)(void* slf, uint32_t bts, uint32_t packets, uint32_t flowindex) nogil
    
    cdef struct ExporterSet:
        long        exporter
        FlowAdd     fadd
        AppAdd      aadd
        TimeAdd     tadd
        void*       fobj
        void*       aobj
        void*       tobj

    ctypedef void (*ipfix_collector_call_t)(const ipfix_query_buf* buf, 
                                            const ipfix_query_info* info,
                                            ipfix_query_pos* poses) nogil

    ctypedef char* (*rep_callback_t)(void* data, size_t* size)
    ctypedef size_t (*ipfix_collector_report_t)(const ipfix_query_pos* totals, int accending, 
                                                const void* buf, uint32_t count, 
                                                char* out, size_t maxsize, 
                                                rep_callback_t callback, void* obj) nogil
