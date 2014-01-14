'''
Created on Dec 30, 2013

@author: schernikov
'''

import numpy as np

import numpyfy.tools.hash as hashmap

dt = np.dtype([('first','u8'),('mid','u4'),('last','u8'),('field1','u4'),('field2','u4')])

def main():
    #hashmap.enabledebug()
    #t1()
    #t2()
    #t3()
    #t4()
    #t5()
    #t6()
    #t7()
    #t8()
    #t9()
    #t10()
    t11()
    
def t1():
    "check basic functionality"
    entries = [(0, 0, 1, 1,2),
               (0, 0, 2, 3,4),
               (0, 0, 3, 5,6)]
    indices = [10,11,12]

    tes_all_new(entries, indices)
    
def t2():
    "check full range with unique addresses"
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
    "check random addresses"
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
    "check conflicting addresses (with first)"
    entries = [(0, 0, 1, 1,2),
               (0, 0, 2, 3,4),
               (0, 0, 257, 5,6)]    # same as #1
    indices = [10,11,12]

    tes_all_new(entries, indices)

def t5():
    "check conflicting addresses (with second)"
    entries = [(0, 0, 1, 1,2),
               (0, 0, 2, 3,4),
               (0, 0, 258, 5,6)]    # same as # 2
    indices = [10,11,12]

    tes_all_new(entries, indices)

def t6():
    "check conflicting addresses (with second)"
    entries = [(0, 0, 1, 1,2),
               (0, 0, 2, 3,4),
               (0, 0, 258, 5,6)]    # same as # 2
    indices = [10,11,12]

    ents = np.array(entries, dtype=dt)
    hm = _tes_all_new(ents, indices)
    newindices = lookup(hm, ents)

    validate(indices, newindices)
    
def t7():
    "check repeating queries"
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
    newindices = lookup(hm, ents)

    validate([11,12,10], newindices)
    
def t8():
    "check non-unique input"
    entries = [(0, 0, 1, 1,2),
               (0, 0, 2, 3,4),
               (0, 0, 1, 5,6)]
    indices = [10,11,10]

    tes_all_new(entries, indices)
    
def t9():
    "check growth"
    size = 256
    newsize = 32
    sz = size + newsize
    ents = np.zeros(sz, dtype=dt)
    
    ents['last'] = np.arange(sz)+1
    ents['field1'] = np.random.randint(size, size=sz)
    ents['field2'] = np.random.randint(size, size=sz)
    indices = np.arange(sz)+10

    hm = _setup_new(ents, indices)
    
    print "============================"
    newindices = lookup(hm, ents[:size])
    validate(indices[:size], newindices)
    rep = hm.report()
    assert rep['bits'] == 8
    print rep
    print "============================"
    newindices = lookup(hm, ents[size:])
    validate(indices[size:], newindices)
    rep = hm.report()
    assert rep['bits'] == 9
    print rep
    print "============================"
    newindices = lookup(hm, ents)
    validate(indices, newindices)
    rep = hm.report()
    assert rep['count'] == len(ents)
    print rep
    
def t10():
    "check remove"
    sz = 256
    rm = 5
    ents = np.zeros(sz, dtype=dt)
    rng = np.arange(sz)
    ents['last'] = rng+1
    ents['field1'] = np.random.randint(sz, size=sz)
    ents['field2'] = np.random.randint(sz, size=sz)
    indices = np.arange(sz)+10
    
    hm = _setup_new(ents, indices)
    
    newindices = lookup(hm, ents)
    validate(indices, newindices)
    rep = hm.report()
    assert rep['count'] == sz
    rems = np.array(np.random.random_sample(rm)*sz, dtype=int)
    hm.remove(ents[rems])
    rep = hm.report()
    assert rep['count'] == (sz-rm)
    remains = np.setdiff1d(rng, rems, assume_unique=True)
    newindices = lookup(hm, ents[remains])
    rep = hm.report()
    validate(indices[remains], newindices)
    assert rep['count'] == (sz-rm)
    print rep
    
def t11():
    "check shrink"
    sz = 300
    rm = 200
    ents = np.zeros(sz, dtype=dt)
    rng = np.arange(sz)
    ents['last'] = rng+1
    ents['field1'] = np.random.randint(sz, size=sz)
    ents['field2'] = np.random.randint(sz, size=sz)
    indices = np.arange(sz)+10
    
    hm = _setup_new(ents, indices)
    
    newindices = lookup(hm, ents)
    validate(indices, newindices)
    rep = hm.report()
    assert rep['count'] == sz
    rems = np.array(np.random.random_sample(rm)*sz, dtype=int)
    hm.remove(ents[rems])
    rep = hm.report()
    assert rep['count'] == (sz-rm)
    remains = np.setdiff1d(rng, rems, assume_unique=True)
    newindices = lookup(hm, ents[remains])
    rep = hm.report()
    validate(indices[remains], newindices)
    assert rep['count'] == (sz-rm)
    print rep
    
# =======================================================================
    
def lookup(hm, ents):
    digs = hashmap.Digs.fromentries(ents)
    return hm.lookup(digs, ents)
    
def tes_all_new(entries, indices):
    ents = np.array(entries, dtype=dt)
    return _tes_all_new(ents, indices)

def _tes_all_new(entries, indices):
    hm = _setup_new(entries, indices)
    newindices = lookup(hm, entries)
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
        print "+++",len(idxs)
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
