# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z

import sys

from common cimport *
from misc cimport logger

cdef uint32_t minsize = 16
cdef float growthrate = 2.0
cdef float shrinkrate = 2.0

def simplelogger(msg):
    print msg
    sys.stdout.flush()

logger = simplelogger

def setlogger(lgr):
    global logger
    logger = lgr

cdef object showapp(const ipfix_app_tuple* atup):
    return "%08x,%08x"%(atup.srcaddr, ftup.dstaddr)
    
cdef object showflow(const ipfix_flow_tuple* ftup):
    return "%2d, %5d, %08x, %5d, %08x"%(ftup.protocol, ftup.srcport, ftup.srcaddr,
                                        ftup.dstport, ftup.dstaddr)
    
cdef object showattr(const ipfix_attributes* attr):
    return "%2d, %02x, %2d, %2d, %02x, %2d, %08x, %d, %d"%(attr.tos, attr.tcpflags, attr.srcmask, 
                                                           attr.inpsnmp, attr.dstmask, attr.outsnmp, 
                                                           attr.nexthop, attr.srcas, attr.dstas)    

cdef void checkbytes(const unsigned char* buf, const char* pref, const char* nm, uint32_t nbytes):
    logger(" %s"%pref)
    logger("  %s: %08x[%d]"%(nm, <uint64_t>buf, nbytes))
    for i in range(nbytes):
        if buf[i] != 0:
            logger("   %02x[%d] != 0"%(buf[i], i))
    logger("  --------")
