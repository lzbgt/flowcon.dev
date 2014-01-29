'''
Created on Jan 28, 2014

@author: schernikov
'''

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
    iptypes = {}
    
    def __init__(self, *args):
        super(IPType, self).__init__(*args)
        self.iptypes[self.id] = self
    
    def convert(self, val):
        res = 0
        for v in val.split('.'):
            res <<= 8
            res += int(v)
        return res

class TimeType(Type):
    timetypes = {}
    
    def __init__(self, *args):
        super(TimeType, self).__init__(*args)
        self.timetypes[self.id] = self
