# distutils: language = c
# distutils: include_dirs = ../includes
# distutils: libraries = z
# distutils: define_macros = NPY_NO_DEPRECATION_WARNING=

import sys

import numpy as np
cimport cython
cimport numpy as np

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
    return "%08x,%08x"%(atup.srcaddr, atup.dstaddr)
    
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

cdef getreqval(req, nm, default):
    val = req.get(nm, None)
    if val is not None:
        try:
            val = int(val)
        except:
            val = default
    else:
        val = default
    return val

cdef debentries(req, n, flows, nms, chks):
    size = len(flows)
    offset = getreqval(req, 'offset', 0)
    count = getreqval(req, 'count', 0)
    res = {'size':size}
    if count > 0:
        flows = flows[offset: offset+count] 
        srt = req.get('sort', None)
        if srt is not None:
            flowsargs = np.argsort(flows, order=srt)[::-1]
        else:
            flowsargs = np.arange(len(flows))
        
        convtypes = {'ip':toip}
        convs = []
        nset = []
        for nm in nms:
            if hasattr(nm, '__iter__'):
                convs.append(convtypes[nm[1]])
                nset.append(nm[0])
            else:
                convs.append(int)
                nset.append(nm)
            
        hdr = ['pos']
        hdr.extend(nset)
        ares = [tuple(hdr)]
        res[n] = ares
        chkset = []
        for chk in chks:
            chkset.append(flows[chk])

        valset = []
        for nm in nset:
            valset.append(flows[nm])
            
        cnt = 0
        for idx in range(len(flowsargs)):
            pos = flowsargs[idx]
            if chkset:
                for chk in chkset:
                    if chk[pos] != 0: break
                else:
                    continue

            vv = [int(pos+offset)]
            for num in range(len(valset)):
                vals = valset[num]
                conv = convs[num]
                vv.append(conv(vals[pos]))
            ares.append(tuple(vv))
            cnt += 1
        res['count'] = cnt

    return res

def toip(uint32_t ip):
    cdef nm = ''
    for _ in range(4):
        nm = ('%d.'%(ip & 0xFF))+nm 
        ip >>= 8
    return nm[:-1]

def backtable(fileh, grp, nm, ents):
    tbl = fileh.create_table(grp, nm, ents.dtype, expectedrows=len(ents))
    tbl.append(ents)
    tbl.flush()
    
def backval(grp, nm, val):
    setattr(grp._v_attrs, nm, val)
    
def backparm(obj, grp, nm):
    backval(grp, nm, getattr(obj, nm))

def resval(grp, nm):
    return getattr(grp._v_attrs, nm)

def resparm(obj, grp, nm):
    setattr(obj, nm, resval(grp, nm))


def _dummy():
    "exists only to get rid of compile warnings"
    cdef int tmp = 0
    if tmp:
        _import_umath()    
        _import_array()
