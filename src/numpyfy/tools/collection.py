'''
Created on Jan 10, 2014

@author: schernikov
'''

import numpy as np
import numpyfy.tools.hash as hashtool

import flowtools.logger as logger

class Collector(object):
    def __init__(self, nm, types, startsize=16):
        self._nm = nm
        self._map = hashtool.HashMap(self._ondigs, self._onnew)
        self._pos = 1;
        self._a = np.zeros(startsize, dtype=types.maintype)
        self._view = self._a.view(dtype=types.viewtype)
        
    def __len__(self):
        return self._pos-1
        
    def add(self, digs, ents):
        "return indices for provided digs and ents"
        indices = self._map.lookup(digs, ents)
        return indices
    
    def _ondigs(self, indices):
        "return digests from given indices"
        return hashtool.Digs.fromentries(self._a, subset=indices)
    
    def _onnew(self, entries):
        "return new indices for given entries; indices must be strictly > 0"
        pos = self._pos
        end = pos+len(entries)
        while end > self._a.size:
            # need to grow
            size = self._a.size*2
            logger.dump("resizing %s to %d"%(self._nm, size))
            self._a.resize(size, refcheck=False)
            self._view = self._a.view(dtype=self._view.dtype)

        self._view[pos:end] = entries
        self._pos = end
        return np.arange(pos, end, dtype=self._map.dtype)

    def report(self):
        return {'count':len(self), 'size':self._a.size,'map':self._map.report()}

