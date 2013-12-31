'''
Created on Dec 19, 2013

@author: schernikov
'''

import numpy as np, sys, hashlib

#import flowtools.logger

class Accountable(object):
    def __init__(self):
        self.acc_reset()
    
    def acc_reset(self):
        self._anum = 0
        self._asize = 0
        self._abytes = 0

    def acc_add(self, a):
        self._anum += 1
        self._asize += a.size
        self._abytes += a.nbytes+sys.getsizeof(a)
        
    def report(self):
        return {'count':self._anum,
                'size':self._asize,
                'bytes':self._abytes}

class BytesMap(Accountable):
    bytehashsize = 256
    zeros = np.array([None]*bytehashsize, dtype=object)
    
    def __init__(self, dtype, default, onnew):
        self.acc_reset()
        self._root = self._clone()
        self._valzeros = np.array([default]*self.bytehashsize, dtype=dtype)
        self._onnew = onnew
        self._def = default

    def _clone(self):
        a = self.zeros.copy()
        self.acc_add(a)
        return a

    def _clonevals(self):
        a = self._valzeros.copy()
        self.acc_add(a)
        return a
                
    def lookup(self, b0, b1, b2, b3):
        b0 = int(b0)
        b1 = int(b1)
        b2 = int(b2)
        b3 = int(b3)
        arr1 = self._root[b0]
        if arr1 is None:
            arr1 = self._clone()
            self._root[b0] = arr1
        arr2 = arr1[b1]
        if arr2 is None:
            arr2 = self._clone()
            arr1[b1] = arr2
        arrvals = arr2[b2]
        if arrvals is None:
            arrvals = self._clonevals()
            arr2[b2] = arrvals

        val = arrvals[b3]
        if val == self._def:
            val = self._onnew(b0, b1, b2, b3)
            arrvals[b3] = val
        return val
    
    def report(self):
        rep = Accountable.report(self)
        my = sys.getsizeof(self) + sys.getsizeof(self._valzeros) + self._valzeros.nbytes
        rep['bytes'] += my
        return rep

def _mkhash(vals):
    sh = hashlib.sha1() 
    for v in vals:
        sh.update(str(v))
    return sh.digest()
