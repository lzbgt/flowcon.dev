import os, sys

libloc = os.path.join(os.path.dirname(__file__), '..', '..', 'cython')
sys.path.insert(0, libloc)

def loadmod(mname):
    return __import__(mname)
