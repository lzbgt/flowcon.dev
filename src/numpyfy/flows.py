'''
Created on Jan 7, 2014

@author: schernikov
'''

import numpy as np, hashlib

import flowtools.logger as logger

class Type(object):
    all = {}
    
    def __init__(self, nm, fid, sz):
        self._nm = nm
        self._id = fid
        self._size = sz
        self.all[fid] = self
        
    @property
    def id(self):
        return self._id
    
    @property
    def size(self):
        return self._size
    
    @property
    def name(self):
        return self._nm

class Flow(object):
    def __init__(self, row, dig):
        self._row = row
        self._dig = dig
        self._pos = 0
        
    def add(self, val):
        pos = self._pos
        self._row[pos] = val
        self._pos += 1
        
    def done(self, bcnt, pcnt):
        sha = hashlib.sha1()
        sha.update(self._dig[0])
        self._dig[2] = sha.digest()
        sha = hashlib.sha1()
        sha.update(self._dig[1])
        self._dig[3] = sha.digest()

class FlowSet(object):
    dtype = None
    digesttype = None

    @classmethod
    def setup(cls, flowtypes, attrtypes):
        typetups = []
        tsize = 0
        for ft in flowtypes:
            typetups.append(('%s'%(ft.name), 'u%d'%(ft.size)))
            tsize += ft.size
        asize = 0
        for at in attrtypes:
            typetups.append(('%s'%(at.name), 'u%d'%(at.size)))
            asize += at.size
        sha = hashlib.sha1()
        typetups.append(('tdigest', 'a%d'%(sha.digest_size)))
        typetups.append(('adigest', 'a%d'%(sha.digest_size)))
        
        cls.digesttype = np.dtype([('flow', tsize), ('attr', asize), 
                                   ('flowdig', 'a%d'%(sha.digest_size)), 
                                   ('attrdig', 'a%d'%(sha.digest_size))])
        cls.dtype = np.dtype(typetups)

    def __init__(self, size, flowfields, attrfields):
        self._a = np.zeros(size, dtype=self.dtype)
        self._digview = self._a.view(dtype=self.digesttype)
        self._pos = 0
        
    def newflow(self):
        if self._pos >= self._a.size:
            size = self._a.size*2
            logger.dump("resizing to %d"%(size))
            self._a.resize(size, refcheck=False)
            self._digview = self._a.view(dtype=self.digesttype)
        pos = self._pos
        self._pos += 1
        return Flow(self._a[pos], self._digview[pos])

    def __len__(self):
        return self._pos
    
    @property
    def size(self):
        return self._a.size
