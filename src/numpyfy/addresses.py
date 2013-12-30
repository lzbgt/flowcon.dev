'''
Created on Dec 18, 2013

@author: schernikov
'''

import re, numpy as np, sys

import misc 

ipmaskre = re.compile('(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$')

class IPCollection(object):
    idxtype = np.uint32
    noidx = np.array([-1], dtype=idxtype)[0]
    incsize = 10000
    valtype = np.uint32

    def __init__(self):
        self._acc = misc.Accountable()
        self._indexmap = {}
        self._maxidx = 0
        self._ipvals = None
        self._incipvals()

    def _incipvals(self):
        inc = np.zeros(self.incsize, dtype=self.valtype)
        if self._ipvals is None:
            self._ipvals = inc
        else:
            self._ipvals = np.append(self._ipvals, inc)
        self._acc.acc_reset()
        self._acc.acc_add(self._ipvals)
    
    def add(self, ipstr):
        m = ipmaskre.match(ipstr)
        if not m: return None
        bvs = m.groups()
        ipval = self._onip(bvs)
        return self._indexmap.lookup(*bvs)

    def _onip(self, bvs):
        res = 0
        for bv in bvs:
            res = res << 8 + bv
        return res

    def _onnew(self, *bytevals):
        if self._maxidx >= self._ipvals.size:   # need to extend array
            self._incipvals()
            
        idx = self._maxidx
        self._maxidx += 1
        res = 0
        for bv in bytevals:
            res = res << 8 + bv
        self._ipvals[idx] = res
        return idx
    
    def report(self):
        rep = sys.getsizeof(self._indexmap)
        my = self._acc.report()
        my['bytes'] += sys.getsizeof(self)
        return {'map':rep, 'set':my, 'count':self._maxidx}
