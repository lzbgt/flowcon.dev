'''
Created on Jan 7, 2014

@author: schernikov
'''

import numpy as np, hashlib

dt = np.dtype([('digest','a20'),('mid','u4'),('last','u8'),('field1','u4'),('field2','u4')])

class Flow(object):
    def __init__(self, row):
        self._row = row
        self._pos = 0
        
    def add(self, val):
        pos = self._pos
        self._row[pos] = val
        self._pos += 1
        
    def done(self, bcnt, pcnt):
        sha = hashlib.sha1()
        sha.update(s)
        self._row[0] = sha.digest()
        
class FlowSet(object):

    def __init__(self):
        self._a = np.zeros(size, dtype=dt)
        self._pos = 0
        
    def newflow(self):
        pos = self._pos
        self._pos += 1
        return Flow(self._a[pos])
