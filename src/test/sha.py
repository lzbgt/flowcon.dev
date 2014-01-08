'''
Created on Jan 7, 2014

@author: schernikov
'''

import numpy as np, hashlib

def main():
    size = 150000 #1000000
    bits = 18
    dt = np.dtype([('first','u8'),('mid','u4'),('last','u8')])    
    
    a = np.arange(size)
    dd = np.zeros(size, dtype='a20')
    ss = a.view(dtype='a%s'%(a.dtype.itemsize))
    pos = 0
    for s in ss:
        sha = hashlib.sha1()
        sha.update(s)
        dd[pos] = sha.digest()
        pos += 1
    digs = dd.view(dtype=dt)
    
    msk = 2**bits-1
    addresses = np.bitwise_and(digs['last'], msk)
    ua, uinds = np.unique(addresses, return_index=True)
    remains = np.setdiff1d(np.arange(size), uinds, assume_unique=True)
    uaddr = np.setdiff1d(ua, np.unique(addresses[remains]), assume_unique=True)
    ln = 2**bits
    fr = ln-len(ua)
    collides = size-len(uaddr)
    print "      size: %d"%(ln)
    print "   samples: %d"%(size)
    print "    unique: %d"%(len(uaddr))
    print "      free: %d/%d = %.1f%%"%(fr, ln, 100.0*fr/ln)
    print "collisions: %d/%d = %.1f%%"%(collides, ln, 100.0*collides/ln)

if __name__ == '__main__':
    main()