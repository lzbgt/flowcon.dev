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

class Digs(object):
    """Should be refactored for non-copying fancy indexing with use of numba or numexpr"""
    def __init__(self, first=None, mid=None, last=None, subset=None):
        self._subset = subset
        self._first = first
        self._mid = mid
        self._last = last
    
    def diffs(self, other):
        "consider stored indexing subset"
        mismatches = ((other._first != self._first)+
                      (other._mid   != self._mid)+
                      (other._last  != self._last))
        return mismatches.nonzero()[0]

    def mask(self, msk):
        return np.bitwise_and(self._last, self._mask)

    @property
    def keys(self):
        return self._subset
    
    def __getitem__(self, key):
        "consider stored indexing subset"
        if self._subset:
            subset = self._subset[key]
        else:
            subset = np.arange(len(self._last))[key]
        return Digs(first=self._first, mid=self._mid, last=self._last, subset=subset)
    
    def __len__(self):
        return len(self._subset) if self._subset else len(self._last)


class HashCompact(object):
    def __init__(self, positions):
        self
    
    def __len__(self):
        "number of active positions in this compact"
        return self._count
    
    def move(self, subset, digs, indices):
        """accept brand new entry and return it's new position 
             pos > 0 if success else <= 0
             """
        return
    
    def lookup(self, poses, subset, digs, onnew):
        """look for matching entry in existing position
        may create new entry; digs[subset] should be looked up 
        should return index"""
        return
    
    def getindices(self):
        "pull all indices"
        return 
        
    def remove(self, positions, digs, shift):
        """remove digs from positions, shift dig bits by 'shift' to get proper values
        return removed indices of positions"""
        return 

class HashSub(object):
    def __init__(self, bits, parent, copy=None):
        self._bits = bits
        size = 2**bits
        self._mask = size-1
        self._indexes = np.zeros(size, dtype='i4')
        self._parent = parent
        self._upper = int(parent._upper*size)
        self._lower = int(parent._lower*size)
        self._getdigs = parent._getdigs
        self._compact = HashCompact(self._upper)
        self._count = 0
        if copy: self._copy(copy)
        
    def __len__(self):
        return self._count

    def _copy(self, other):
        indices = other._indexes
        positives = (indices > 0).nonzero()[0]
        
        if len(positives) > 0:
            posindices = indices[positives]
            def onnew(dgs):
                return posindices[dgs.keys]
            self._lookup(self._getdigs(posindices), onnew)

        negindices = self._compact.getindices()
        self._lookup(self._getdigs(negindices), onnew)
        
    def _locations(self, digs):
        return digs.mask(self._mask)

    def lookup(self, entries):
        "lookup collection of digs and values"
        digs = Digs(entries['first'], entries['mid'], entries['last'])
        
        def onnew(dgs):
            self._count += len(dgs)
            return self._parent._onnew(entries[dgs.keys])

        return self._lookup(self, digs, onnew)
        
    def _lookup(self, digs, onnew):
        locations = self._locations(digs)       # may have non-unique values there
        ulocs, uinds = np.unique(locations, return_index=True)
        if len(ulocs) >= len(locations):
            # they are all unique 
            return self._ulookup(digs, locations, onnew)
        # look only in unique subset        
        indices = np.zeros(len(digs), dtype=self._indexes.dtype)
        indices[uinds] = self._ulookup(digs[uinds], ulocs, onnew)
        
        remains = np.setdiff1d(np.arange(len(digs)), uinds, assume_unique=True)
        while True:
            ulocs, uinds = np.unique(locations[remains], return_index=True)
            indices[uinds] = self._ulookup(digs[uinds], ulocs, onnew)
            if len(uinds) >= len(remains): break
            remains = np.setdiff1d(remains, uinds, assume_unique=True)
        
        return indices
        
    def _ulookup(self, digs, locations, onnew):
        "lookup digs in unique locations"
        indices = self._indexes[locations]

        positives = (indices > 0).nonzero()[0]
        zeros = (indices == 0).nonzero()[0]
        negatives = (indices < 0).nonzero()[0]
        
        if len(positives) > 0:
            otherdigs = self._getdigs(indices[positives])
            badmatches = digs[positives].diffs(otherdigs)

            if len(badmatches) > 0:
                # those are collisions
                bads = positives[badmatches]
                newposes = self._compact.move(otherdigs[bads], indices[bads])
                self._indexes[locations[bads]] = (-1)*newposes
                # handle collided entries collision
                indices[bads] = self._compact.lookup(newposes, digs[bads], onnew)
                if len(self._compact) > self._upper: 
                    self._parent._grow(self._bits)

        if len(zeros) > 0:      # never seen these entries before
            newindex = onnew(digs[zeros])
            self._indexes[locations[zeros]] = newindex

        if len(negatives) > 0:  # it's already collided before; now it is one more collision
            poses = (-1)*indices[negatives]
            indices[negatives] = self._compact.lookup(poses, digs[negatives], onnew)
        return indices
    
    def remove(self, digs):
        """remove all digs
        digs is a numpy array with records containing (('first','u8'),('mid','u4) ,('last','u8')) fields"""
        locations = self._locations(digs)                       # locations may not be unique
        indices = self._indexes[locations]
        positives = (indices > 0).nonzero()[0]                  # all positives should be unique
        if len(positives) > 0:
            self._count -= len(positives)
            self._indexes[locations[positives]] = 0
        negatives = (indices < 0).nonzero()[0]                  # negatives may not be unique 
        if len(negatives) > 0:
            self._count -= len(negatives)
            checklocks = locations[negatives]
            positions = (-1)*self._indexes[checklocks]
            removed = self._compact.remove(self._bits, positions, digs[negatives])
            if removed is not None and len(removed) > 0:
                self._indexes[checklocks[removed]] = 0            # cleanup subset of checklocks indicated as removed   
        if self._count <= self._lower:
            self._parent._shrink(self._bits)
    
class HashMap(Accountable):
    maxbits = 32
    minbits = 8
    
    def __init__(self, getdigs, onnewindex, startbits=minbits, upper=0.25, lower=0.25):
        if startbits > self.maxbits or startbits < self.minbits: raise Exception("invalid hashmap order: %d"%startbits)
        self._upper = upper
        self._lower = lower
        self._getdigs = getdigs
        self._onnew = onnewindex

        self._sub = HashSub(startbits, self)

    def lookup(self, entries):
        return self._sub.lookup(entries)

    def _grow(self, bits):
        """grow hash size twice (by one bit), plug new value there and return new index"""
        if bits >= self.maxbits: return    # can not grow any more
        self._sub = HashSub(bits+1, self, copy=self._sub)   # grow one bit - make it twice as big
    
    def _shrink(self, bits):
        if bits <= self.minbits: return
        self._sub = HashSub(bits-1, self, copy=self._sub)   # shrink one bit - make it twice as small
    
    def remove(self, digs):
        return self._sub.remove(digs)

    def report(self):
        self
