'''
Created on Jan 1, 2014

@author: schernikov
'''
from numpy import * #@UnusedWildImport

import numpy as np, sys

def prepare(names, pref):
    module = sys.modules[__name__]
    for nm in names:
        setattr(module, nm, genfunc(nm, pref))

def genfunc(name, pref):
    def func(*args, **kargs):
        call = getattr(np, name)
        print "%s%s"%(pref, name),args, kargs
        return call(*args, **kargs)
    return func

names2 = ['bitwise_and',
         'right_shift',
         'arange',
         'dtype',
         'unique',
         'setdiff1d',
         'eft_shift']

names1 = ['zeros']

prepare(names1, '')
prepare(names2, '  ')
