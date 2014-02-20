'''
Created on Jan 24, 2014

@author: schernikov
'''

import socket, urlparse, datetime

import native, flowtools.settings

recmod = native.loadmod('nreceiver')
colmod = native.loadmod('collectors')
timecolmod = native.loadmod('timecollect')
appsmod = native.loadmod('napps')

class Receiver(object):

    def __init__(self, addr, ioloop):
        self.allsources = {}
        self._onsource = None
        
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

        self._nreceiver = recmod.Receiver(self)
        self._queries = {}
        
        ioloop.add_handler(sock.fileno(), self._recv, ioloop.READ)
        
    def _recv(self, fd, events):
        data, addr = self._sock.recvfrom(2048); addr
        self._nreceiver.receive(data, len(data))
        
    def find(self, ip):
        src = self.allsources.get(ip, None)
        if src is None:
            src = Sources(ip)
            self.allsources[ip] = src
            if self._onsource: self._onsource(src)

        return src.getcollectors()
        
    def sourcecallback(self, callback):
        self._onsource = callback
        
    def sources(self):
        return self.allsources.values()
    
    def registerraw(self, q, onmsg):
        qnat = self._queries.get(q.id, None)
        if qnat: # old query?
            # re-register in case onmsg is changed
            self.unregisterraw(q.id)
        qnat = q.native
        self._queries[q.id] = qnat
        qnat.setcallback(onmsg)
        self._nreceiver.register(qnat)
        
    def unregisterraw(self, qid):
        qnat = self._queries.get(qid, None)
        if qnat:
            self._nreceiver.unregister(qnat)
            
class Sources(object):

    def __init__(self, ip):
        self._ip = ip
        nm = ''
        for _ in range(4):
            nm = ('%d.'%(ip & 0xFF))+nm 
            ip >>= 8
        stamp = native.query.mkstamp(datetime.datetime.utcnow())
        self._name = nm[:-1]
        self._attrs = colmod.AttrCollector("A:"+self._name)
        self._flows = colmod.FlowCollector("F:"+self._name, self._attrs)
        self._appflows = colmod.AppFlowCollector("C"+self._name, self._attrs)
        self._seconds = timecolmod.SecondsCollector("S:"+self._name, self._ip, self._flows, flowtools.settings.maxseconds, stamp)
        self._minutes = timecolmod.MinutesCollector("M:"+self._name, self._ip, flowtools.settings.maxminutes, stamp)
        
    def getcollectors(self):
        return self._flows, self._attrs, self._seconds
        
    def address(self):
        return self.name
    
    def stats(self):
        return {'address':self._ip}

    def on_time(self, qbuf, apps, secs, mins, hours, days):
        self._seconds.onsecond(apps._nativeapps, secs)
        if mins:
            self._minutes.onminute(qbuf._native, apps._nativeapps, self._appflows, self._seconds, mins)
        
    @property
    def name(self):
        return self._name

    @property
    def ip(self):
        return self._ip
    
class Apps(object):
    
    def __init__(self):
        self._nativeapps = appsmod.Apps(flowtools.settings.portrate, flowtools.settings.minthreshold)

    def report(self):
        return self._nativeapps.report()
