'''
Created on Dec 30, 2013

@author: schernikov
'''

import numpy as np

import numpyfy.tools.hash as hashmap

def main():
    dt = np.dtype([('first','u8'),('mid','u4'),('last','u8'),('field1','u4'),('field2','u4')])

    entries = np.array([(0, 0, 1, 1,2),
                        (0, 0, 2, 3,4),
                        (0, 0, 3, 5,6)], dtype=dt)
    genindices = np.array([10,11,12], dtype=hashmap.HashLookup.indextype)
    def getdigs(indices):
        "this should return digs"
        return hashmap.Digs.fromentries(entries[indices])
    
    def onnewindex(entries):
        "this should return indices"
        return genindices

    hm = hashmap.HashMap(getdigs, onnewindex)
    
    indices = hm.lookup(entries)
    
    print indices
    assert np.array_equal(indices, genindices)

if __name__ == '__main__':
    main()