'''
Created on Dec 30, 2013

@author: schernikov
'''

import numpy as np

class Digs(object):
    """Should be refactored for non-copying fancy indexing with use of numba or numexpr"""

    @classmethod
    def fromentries(cls, entries):
        return Digs(entries['first'], entries['mid'], entries['last'])
    
    def __init__(self, first=None, mid=None, last=None, subset=None):
        self._subset = subset
        self._first = first
        self._firstbits = self._first.dtype.itemsize*8
        self._mid = mid
        self._midbits = self._mid.dtype.itemsize*8
        self._last = last
        self._lastbits = self._last.dtype.itemsize*8
    
    def diffs(self, o):
        "consider stored indexing subset"
        mismatches = o._first[o._subset] != self._first[self._subset]
        mismatches += o._mid[o._subset] != self._mid[self._subset]
        mismatches += o._last[o._subset] != self._last[self._subset]

        return mismatches.nonzero()[0]

    def mask(self, msk):
        bts = self._last if self._subset is None else self._last[self._subset]
        return np.bitwise_and(bts, msk)

    def _applyone(self, field, offset, out, num):
        bts = field if self._subset is None else field[self._subset]

        np.right_shift(bts, offset, out)
        mask = 2**num-1
        np.bitwise_and(out, mask, out)

    def _applytwo(self, f1, f2, offset, out, num, rem):
        bts1 = f1 if self._subset is None else f1[self._subset]
        bts2 = f2 if self._subset is None else f2[self._subset]
        mask = 2**(num-rem)-1
        np.bitwise_and(bts2, mask, out)
        out <<= rem
        out += np.right_shift(bts1, offset)

    def applybits(self, offset, num, out):
        "return num bits offset bits from LSB side"
        if (offset+num) > self._lastbits:
            if offset >= self._lastbits:
                offset -= self._lastbits
                if (offset+num) > self._midbits:
                    if offset >= self._midbits:
                        offset -= self._midbits
                        if (offset+num) > self._firstbits:
                            if offset >= self._firstbits:
                                pass # this should never happen; all digest bits are skipped  
                            else:
                                # use only remaining available bits
                                self._applyone(self._first, offset, out, self._firstbits-offset)
                        else:
                            self._applyone(self._first, offset, out, num)
                    else:
                        self._applytwo(self._mid, self._first, offset, out, num, self._midbits-offset)
                else:
                    self._applyone(self._mid, offset, out, num)
            else:
                self._applytwo(self._last, self._mid, offset, out, num, self._lastbits-offset)
        else:
            self._applyone(self._last, offset, out, num)

    def select(self, vals):
        return vals[self._subset] if self._subset is not None else vals
    
    def __getitem__(self, key):
        "consider stored indexing subset"
        if key is not None:
            if self._subset is not None:
                subset = self._subset[key]
            else:
                subset = np.arange(len(self._last))[key]
        return Digs(first=self._first, mid=self._mid, last=self._last, subset=subset)
    
    def __len__(self):
        return len(self._subset) if self._subset is not None else len(self._last)

class HashDict(object):
    def __init__(self):
        self

class HashLookup(object):
    indextype = np.dtype('i4')
    
    def __init__(self, ondigs, bits, indarray):
        self._indexes = indarray
        self._getdigs = ondigs
        self.__comp = None
        self._bits = bits

    @property
    def _compact(self):
        if self.__comp: return self.__comp
        self.__comp = HashCompact(self._bits, self._indexes.dtype, self._getdigs)
        return self.__comp

    def _anylookup(self, digs, addresses, onnew):
        uaddrs, uinds = np.unique(addresses, return_index=True)
        if len(uaddrs) >= len(addresses):
            # they are all unique 
            return self._uniquelookup(digs, addresses, onnew)
        # look only in unique subset        
        indices = np.zeros(len(digs), dtype=self._indexes.dtype)
        indices[uinds], bits = self._uniquelookup(digs[uinds], uaddrs, onnew)
        
        remains = np.setdiff1d(np.arange(len(digs)), uinds, assume_unique=True)
        while True:
            uaddrs, uinds = np.unique(addresses[remains], return_index=True)
            indices[uinds], bts = self._uniquelookup(digs[uinds], uaddrs, onnew)
            if not bits: bits = bts
            if len(uinds) >= len(remains): break
            remains = np.setdiff1d(remains, uinds, assume_unique=True)
        
        return indices, bits
        
    def _uniquelookup(self, digs, addresses, onnew):
        "lookup digs in unique addresses"
        indices = self._indexes[addresses]
        bits = None
        
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
                self._indexes[addresses[bads]] = (-1)*newposes
                # handle collided entries collision
                indices[bads] = self._compact.lookup(newposes, digs[bads], onnew)
                if len(self._compact) >= self._upper: bits = self._bits

        if len(zeros) > 0:      # never seen these entries before
            newindex = onnew(digs[zeros])
            self._indexes[addresses[zeros]] = newindex
            indices[zeros] = newindex

        if len(negatives) > 0:  # it's already collided before; now it is one more collision
            poses = (-1)*indices[negatives]
            indices[negatives] = self._compact.lookup(poses, digs[negatives], onnew)

        return indices, bits
    
    def _remove(self, addresses, digs):
        indices = self._indexes[addresses]
        positives = (indices > 0).nonzero()[0]                  # all positives should be unique
        if len(positives) > 0:
            self._indexes[addresses[positives]] = 0
        negatives = (indices < 0).nonzero()[0]                  # negatives may not be unique 
        if len(negatives) > 0:
            checklocks = addresses[negatives]
            positions = (-1)*self._indexes[checklocks]
            removed = self._compact.remove(positions, digs[negatives])
            if removed is not None and len(removed) > 0:
                self._indexes[checklocks[removed]] = 0            # cleanup subset of checklocks indicated as removed


class HashCompact(HashLookup):
    growthrate = 2.0
    scopebits = 2
    chunkbits = 16
    
    def __init__(self, offset, dtype, ondigs):
        super(HashCompact, self).__init__(ondigs, offset+self.scopebits, None)

        self._offset = offset
        self._dtype = dtype
        
        self._first = 0
        self._available = 0
        self._positions = None

        self._count = 0
    
    def __len__(self):
        "number of active positions in this compact"
        return self._count
    
    def move(self, digs, indices):
        "one dig - one new position"
        self._count += len(digs)
        # first initialization
        if not self._indexes: return self._setup(digs, indices)

        poses = self._findposes(len(digs))
        self._assign(poses, digs, indices)

        return poses
    
    def _growsizes(self, init):
        chunksize = 2**self.chunkbits
        # grow with growthrate and roundup to chunksize        
        size = int(((init*self.growthrate)+chunksize-1)//chunksize)*chunksize
        return size, size >> self.scopebits
    
    def _setup(self, digs, indices):
        poscount = len(digs)

        size, posnum = self._growsizes(poscount << self.scopebits)
        
        self._indexes = np.zeros(size, dtype=self._dtype)
        self._positions = np.zeros(posnum, dtype=self._dtype)
        self._first = poscount
        self._initposes(poscount)
       
        poses = np.arange(0, poscount, dtype=self._dtype)
        self._assign(poses, digs, indices)
        return poses

    def _mkaddresses(self, poses, digs):
        addresses = np.left_shift(poses, self.scopebits)
        digs.applybits(self._offset, self.scopebits, addresses)
        return addresses

    def _initposes(self, start):
        posnum = self._positions.size
        self._positions[start:] = np.arange(start+1, posnum+1)    # list of free positions
        self._available += posnum-start

    def _assign(self, poses, digs, indices):
        addresses = self._mkaddresses(poses, digs)
        self._indexes[addresses] = indices

    def _findposes(self, count):
        "return array with free poses; grow chunk if needed"
        poses = np.zeros(count, dtype=self._dtype)
        if self._available < count:
            # need to grow
            poscount = self._positions.size
            size, posnum = self._growsizes(self._indexes.size+count-self._available)
            self._indexes.resize(size, refcheck=False)
            self._positions.resize(posnum, refcheck=False)
            self._initposes(poscount)
        self._available -= count
        
        current = self._first
        for i in xrange(count):
            poses[i] = current
            current = self._positions[current]
        self._first = current
        self._positions[poses] = 0
        return poses
            
    def lookup(self, positions, digs, onnew):
        """look for matching entry in existing position may create new entry 
        should return index"""
        addresses = self._mkaddresses(positions, digs)
        def oncompnew(dgs):
            self._count += len(dgs)
            return onnew(dgs)
        return self._anylookup(digs, addresses, oncompnew)
    
    def getindices(self):
        "pull all indices"
        results = np.zeros(self._count, dtype=self._dtype)
        indices = self._indexes
        if indices:
            positives = (indices > 0).nonzero()[0]
            posnum = len(positives)
            if posnum > 0:
                results[:posnum] = indices[positives]
        else:
            posnum = 0

        if self._count > posnum:
            results[posnum:] = self._compact.getinidices()

        return results
        
    def remove(self, positions, digs):
        """remove digs from positions, shift dig bits by 'bits' to get proper values
        return removed indices of positions"""
        addresses = self._mkaddresses(positions, digs)
        self._remove(addresses, digs)
        self._count -= len(digs)


class HashSub(HashLookup):
    def __init__(self, bits, parent, copy=None):
        size = 2**bits
        super(HashSub, self).__init__(parent._getdigs, bits, np.zeros(size, dtype=self.indextype))

        self._onnew = parent._onnew
        self._mask = size-1
        self._upper = int(parent._upper*size)
        self._lower = int(parent._lower*size)
        self._count = 0
        if copy: self._copy(copy)
        
    def __len__(self):
        return self._count

    def _copy(self, other):
        indices = other._indexes
        positives = (indices > 0).nonzero()[0]
        self._count = len(other)

        if len(positives) > 0:
            posindices = indices[positives]
            def onposnew(dgs):
                return dgs.select(posindices)
            self._lookup(self._getdigs(posindices), onposnew)

        negindices = self._compact.getindices()
        def onnegnew(dgs):
            return dgs.select(negindices)
        self._lookup(self._getdigs(negindices), onnegnew)
        
    def _locations(self, digs):
        return digs.mask(self._mask)

    def lookup(self, entries):
        "lookup collection of digs and values"
        digs = Digs.fromentries(entries)
        
        def onnew(dgs):
            self._count += len(dgs)
            return self._onnew(dgs.select(entries))

        return self._lookup(digs, onnew)
        
    def _lookup(self, digs, onnew):
        addresses = self._locations(digs)       # may have non-unique values there
        return self._anylookup(digs, addresses, onnew)
        
    def remove(self, digs):
        """remove all digs
        digs is a numpy array with records containing (('first','u8'),('mid','u4) ,('last','u8')) fields"""
        addresses = self._locations(digs)                       # addresses may not be unique
        self._remove(addresses, digs)
        self._count -= len(digs)
        return self._bits if self._count <= self._lower else None

class HashMap(object):
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
        indices, bits = self._sub.lookup(entries)
        if bits: self._grow(bits)
        return indices

    def _grow(self, bits):
        """grow hash size twice (by one bit), plug new value there and return new index"""
        if bits >= self.maxbits: return    # can not grow any more
        self._sub = HashSub(bits+1, self, copy=self._sub)   # grow one bit - make it twice as big
    
    def _shrink(self, bits):
        if bits <= self.minbits: return
        self._sub = HashSub(bits-1, self, copy=self._sub)   # shrink one bit - make it twice as small
    
    def remove(self, digs):
        bits = self._sub.remove(digs)
        if bits: self._shrink(bits)

    def report(self):
        self
