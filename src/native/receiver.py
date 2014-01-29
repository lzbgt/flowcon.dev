'''
Created on Jan 24, 2014

@author: schernikov
'''

import socket, urlparse

import native

recmod = native.loadmod('nreceiver')
colmod = native.loadmod('collectors')

class Receiver(object):

    def __init__(self, addr, ioloop):
        p = urlparse.urlsplit(addr)
        if not p.scheme or p.scheme.lower() != 'udp':
            raise Exception("Only udp scheme is supported for flow reception. Got '%s'"%(addr))
        if not p.port:
            raise Exception("Please provide port to receive flows on. Got '%s'"%(addr))

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setblocking(0)
        sock.bind((p.hostname, p.port))
        self._sock = sock

        self._receiver = recmod.Receiver(Sources)
        
        ioloop.add_handler(sock.fileno(), self._recv, ioloop.READ)
        
    def _recv(self, fd, events):
        data, addr = self._sock.recvfrom(2048); addr
        self._receiver.receive(data, len(data))
        
    def sources(self):
        return Sources.allsources.values()

class Sources(object):
    allsources = {}
    maxseconds = 3600

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
        self._flows = colmod.FlowCollector("F:"+self._name, self._attrs)
        self._seconds = colmod.SecondsCollector("S:"+self._name, self._flows, self.maxseconds)
        
    @property
    def name(self):
        return self._name

    def address(self):
        return self.name
    
    def stats(self):
        return {}

    def on_time(self, now):
        self._seconds.onsecond()
