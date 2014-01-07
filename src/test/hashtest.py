'''
Created on Dec 30, 2013

@author: schernikov
'''

import numpy as np

import numpyfy.tools.hash as hashmap

dt = np.dtype([('first','u8'),('mid','u4'),('last','u8'),('field1','u4'),('field2','u4')])

def main():
    #t1()
    #t2()
    #t3()
    #t4()
    #t5()
    #t6()
    #t7()
    #t8()
    t9()
    
def t1():
    entries = [(0, 0, 1, 1,2),
               (0, 0, 2, 3,4),
               (0, 0, 3, 5,6)]
    indices = [10,11,12]

    tes_all_new(entries, indices)
    
def t2():
    size = 256
    ents = np.zeros((size, 5))
    ents[:, 2] = range(size)
    ents[:, 3] = np.random.randint(size, size=size)
    ents[:, 4] = np.random.randint(size, size=size)
    entries = []
    for i in range(size):
        entries.append(tuple(ents[i]))
    indices = range(size)
    tes_all_new(entries, indices)
    
def t3():
    size = 1000
    rng = 256
    nums = np.unique(np.random.randint(rng, size=size))
    size = len(nums)

    ents = np.zeros((size, 5))
    ents[:, 2] = nums
    ents[:, 3] = np.random.randint(rng, size=size)
    ents[:, 4] = np.random.randint(rng, size=size)
    entries = []
    for i in range(size):
        entries.append(tuple(ents[i]))
    indices = range(size)
    tes_all_new(entries, indices)

def t4():
    entries = [(0, 0, 1, 1,2),
               (0, 0, 2, 3,4),
               (0, 0, 257, 5,6)]    # same as #1
    indices = [10,11,12]

    tes_all_new(entries, indices)

def t5():
    entries = [(0, 0, 1, 1,2),
               (0, 0, 2, 3,4),
               (0, 0, 258, 5,6)]    # same as # 2
    indices = [10,11,12]

    tes_all_new(entries, indices)
    
def t7():
    entries = [(0, 0, 1, 1,2),
               (0, 0, 2, 3,4),
               (0, 0, 258, 5,6)]    # same as # 2
    indices = [10,11,12]

    ents = np.array(entries, dtype=dt)
    hm = _tes_all_new(ents, indices)
    entries = [(0, 0, 2, 3,4),
               (0, 0, 258, 5,6),
               (0, 0, 1, 1,2),]
    ents = np.array(entries, dtype=dt)
    newindices = hm.lookup(ents)

    validate([11,12,10], newindices)
    
def t6():
    entries = [(0, 0, 1, 1,2),
               (0, 0, 2, 3,4),
               (0, 0, 258, 5,6)]    # same as # 2
    indices = [10,11,12]

    ents = np.array(entries, dtype=dt)
    hm = _tes_all_new(ents, indices)
    newindices = hm.lookup(ents)

    validate(indices, newindices)
    
def t8():
    entries = [(0, 0, 1, 1,2),
               (0, 0, 2, 3,4),
               (0, 0, 1, 5,6)]
    indices = [10,11,10]

    tes_all_new(entries, indices)
    
def t9():
    size = 256
    newsize = 32
    sz = size + newsize
    ents = np.zeros(sz, dtype=dt)
    
    ents['last'] = np.arange(sz)+1
    ents['field1'] = np.random.randint(size, size=sz)
    ents['field2'] = np.random.randint(size, size=sz)
    indices = np.arange(sz)+10

    hm = _setup_new(ents, indices)
    
    newindices = hm.lookup(ents[:size])
    validate(indices[:size], newindices)
    print "============================"
    newindices = hm.lookup(ents[size:])

    print "============================"
    print hm.report()

    validate(indices[size:], newindices)
    
# =======================================================================
    
def tes_all_new(entries, indices):
    ents = np.array(entries, dtype=dt)
    return _tes_all_new(ents, indices)

def _tes_all_new(entries, indices):
    hm = _setup_new(entries, indices)
    newindices = hm.lookup(entries)
    validate(indices, newindices)
    return hm

def _setup_new(entries, indices):    
    emap = {}
    imap = {}
    pos = 0
    for e in entries:
        emap[tuple(e)] = indices[pos]
        imap[indices[pos]] = pos
        pos += 1
        
    def getdigs(idxs):
        "this should return digs"
        poses = []
        for idx in idxs:
            poses.append(imap[idx])
        return hashmap.Digs.fromentries(entries, np.array(poses))

    def onnewindex(ents):
        "this should return indices"
        idxs = []
        for e in ents:
            idxs.append(emap[tuple(e)])
        return np.array(idxs, dtype=hashmap.HashLookup.indextype)

    return hashmap.HashMap(getdigs, onnewindex)

def validate(indices, newindices):
    try:
        assert np.array_equal(indices, newindices)
    except Exception, e:
        print "not equal:",indices, newindices
        raise e

if __name__ == '__main__':
    main()
