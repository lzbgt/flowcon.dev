'''
Created on Jan 15, 2014

@author: schernikov
'''

import socket, os, sys

libloc = os.path.join(os.path.dirname(__file__), '..', '..', 'cython')
sys.path.insert(0, libloc)

recmod = __import__('receiver')
colmod = __import__('collectors')

def main():
    myaddr = "10.215.1.6"
    myport = 2059
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((myaddr, myport))

    receiver = recmod.Receiver(Sources)
    while True:
        data, addr = sock.recvfrom(2048); addr
        receiver.receive(data, len(data))

class Sources(object):
    allsources = {}

    @classmethod
    def find(cls, ip):
        src = cls.allsources.get(ip, None)
        if src is None:
            src = cls(ip)
            cls.allsources[ip] = src
            print "created %s"%(src.name)

        return src._flows, src._attrs, src._seconds
    
    def __init__(self, ip):
        nm = ''
        for _ in range(4):
            nm = ('%d.'%(ip & 0xFF))+nm 
            ip >>= 8
        self._name = nm[:-1]
        self._attrs = colmod.AttrCollector("A:"+self._name)
        self._seconds = colmod.SecondsCollector("S:"+self._name)
        self._flows = colmod.FlowCollector("F:"+self._name, self._attrs)
        
    @property
    def name(self):
        return self._name

if __name__ == '__main__':
    main()