'''
Created on Dec 30, 2013

@author: schernikov
'''

import numpy as np

import numpyfy.tools.hash as hashmap

def getdigs(indices):
    "this should return digs"
    return

def onnewindex(entries):
    "this should return indices"
    return 

def main():
    hm = hashmap.HashMap(getdigs, onnewindex)
    
    dt = np.dtype([('first','u8'),('mid','u4'),('last','u8'),('field1','u4'),('field2','u4')])

    entries = np.array((), dtype=dt)
    
    indices = hm.lookup(entries)
    
    indices

if __name__ == '__main__':
    main()