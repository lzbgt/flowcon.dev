'''
Created on Jan 7, 2014

@author: schernikov
'''

import numpy as np, hashlib

import flowtools.logger as logger, numpyfy.tools.hash as hashtool

class Type(object):
    all = {}
    
    def __init__(self, nm, fid, sz):
        self._nm = nm
        self._id = fid
        self._size = sz
        self.all[fid] = self
        
    def convert(self, val):
        raise Exception("no conversion defined for %s"%(self.__class__.__name__))
        
    @property
    def id(self):
        return self._id
    
    @property
    def size(self):
        return self._size
    
    @property
    def name(self):
        return self._nm

class IntType(Type):
    def convert(self, val):
        return val

class IPType(Type):
    def convert(self, val):
        res = 0
        for v in val.split('.'):
            res <<= 8
            res += int(v)
        return res

class TimeType(Type):
    pass

class Flow(object):
    def __init__(self, types, row, dig):
        self._row = row
        self._dig = dig
        self._pos = 0
        self._types = types
        
    def add(self, val):
        pos = self._pos
        self._row[pos] = self._types[pos].convert(val)
        self._pos += 1
        
    def done(self, bcnt, pcnt):
        sha = hashlib.sha1()
        sha.update(self._dig[0])
        self._dig[1] = sha.digest()
        sha = hashlib.sha1()
        sha.update(self._dig[2])
        self._dig[3] = sha.digest()
        self._dig[4] = bcnt
        self._dig[5] = pcnt

class SubTypes(object):
    def __init__(self, digtups, types):
        self.size = 0
        self.tups = []
        for t in types:
            self.tups.append(('%s'%(t.name), 'u%d'%(t.size)))
            self.size += t.size
        
        sha = hashlib.sha1()
        alltups = []
        alltups.extend(self.tups)
        alltups.extend(digtups)
        self.viewtype = np.dtype([('vals', 'a%d'%(self.size+sha.digest_size))])
        self.maintype = np.dtype(alltups)

class FlowTypes(object):
    def __init__(self, flowtypes, attrtypes):
        digtups = [('first', 'u8'), ('mid', 'u4'), ('last', 'u8')]
        bsize = 4
        psize = 4
        sha = hashlib.sha1()

        typetups = []        
        self.ftypes = SubTypes(digtups, flowtypes)
        typetups.extend(self.ftypes.tups)
        typetups.append(('flowdig', 'a%d'%(sha.digest_size)))
        self.atypes = SubTypes(digtups, attrtypes)
        typetups.extend(self.atypes.tups)
        typetups.append(('attrdig', 'a%d'%(sha.digest_size)))

        typetups.append(('bytes', 'u%d'%(bsize)))
        typetups.append(('packets', 'u%d'%(psize)))
        
        self.digesttype = np.dtype([('flow', 'a%d'%(self.ftypes.size)),
                                    ('flowdig', 'a%d'%(sha.digest_size)),
                                    ('attr', 'a%d'%(self.atypes.size)), 
                                    ('attrdig', 'a%d'%(sha.digest_size)),
                                    ('bytes', 'u%d'%(bsize)), ('packets', 'u%d'%(psize))])
        self.copytype = np.dtype([('flow', 'a%d'%(self.ftypes.size+sha.digest_size)),
                                  ('attr', 'a%d'%(self.atypes.size+sha.digest_size)), 
                                  ('counters', 'u%d'%(bsize+psize))])

        self.digparts = np.dtype(digtups)
        self.originaltype = np.dtype(typetups)
        valtypes = []
        valtypes.extend(flowtypes)
        valtypes.extend(attrtypes)
        self.valtypes = tuple(valtypes)

class FlowSet(object):
    def __init__(self, nm, size, ftypes):
        self._ftypes = ftypes
        self._a = np.zeros(size, dtype=ftypes.originaltype)
        self._digview = self._a.view(dtype=ftypes.digesttype)
        self._copyview = self._a.view(dtype=ftypes.copytype)
        self._pos = 0
        self._name = nm
        
    def newflow(self):
        if self._pos >= self._a.size:
            size = self._a.size*2
            logger.dump("resizing %s to %d"%(self.name, size))
            self._a.resize(size, refcheck=False)
            self._digview = self._a.view(dtype=self._digview.dtype)
            self._copyview = self._a.view(dtype=self._copyview.dtype)
        pos = self._pos
        self._pos += 1
        return Flow(self._ftypes.valtypes, self._a[pos], self._digview[pos])

    def __len__(self):
        return self._pos
    
    def _collected(self, dignm, colnm):
        "return collected attribute digests"
        dv = self._digview[dignm].view(dtype=self._ftypes.digparts)
        return hashtool.Digs(dv['first'][:self._pos], dv['mid'][:self._pos], dv['last'][:self._pos]), self._copyview[colnm][:self._pos]
    
    def attrs(self):
        return self._collected('attrdig', 'attr')
    
    def flows(self):
        return self._collected('flowdig', 'flow')
    
    @property
    def size(self):
        return self._a.size
    
    @property
    def ftypes(self):
        return self._ftypes
    
    @property
    def name(self):
        return self._name
    