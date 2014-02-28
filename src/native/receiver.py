'''
Created on Jan 24, 2014

@author: schernikov
'''

import socket, urlparse, datetime, os

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

        self._apps = None
        
        ioloop.add_handler(sock.fileno(), self._recv, ioloop.READ)
       
    def _recv(self, fd, events):
        data, addr = self._sock.recvfrom(2048); addr
        self._nreceiver.receive(data, len(data))
        
    def find(self, ip):
        src = self.allsources.get(ip, None)
        if src is None:
            src = Sources(self._apps, ip)
            self.allsources[ip] = src
            if self._onsource: self._onsource(src)

        return src.getcollectors()
        
    def sourcecallback(self, apps, callback):
        self._apps = apps
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

    def __init__(self, apps, ip):
        self._ip = ip
        self._apps = apps
        nm = ''
        for _ in range(4):
            nm = ('%d.'%(ip & 0xFF))+nm 
            ip >>= 8
        stamp = native.query.mkstamp(datetime.datetime.utcnow())
        self._name = nm[:-1]
        self._attrs = colmod.AttrCollector("A:"+self._name)
        self._flows = colmod.FlowCollector("F:"+self._name, self._attrs)
        self._appflows = colmod.AppFlowCollector("C"+self._name, apps._nativeapps, self._attrs)
        self._seconds = timecolmod.SecondsCollector("S:"+self._name, self._ip, self._flows, flowtools.settings.maxseconds, stamp)
        libname = os.path.join(native.libloc, 'minutescoll.so')
        self._minutes = timecolmod.MinutesCollector("M:"+self._name, self._ip, libname, 
                                                    self._appflows, flowtools.settings.maxminutes, stamp)
        libname = os.path.join(native.libloc, 'hourscoll.so')
        self._hours = timecolmod.HoursCollector("H:"+self._name, self._ip, libname, 
                                                self._appflows, flowtools.settings.maxhours, 
                                                stamp, flowtools.settings.hoursrefs)
        libname = os.path.join(native.libloc, 'dayscoll.so')
        self._days = timecolmod.DaysCollector("D:"+self._name, self._ip, libname, 
                                              self._appflows, flowtools.settings.maxdays, 
                                              stamp, flowtools.settings.daysrefs)
        
    def getcollectors(self):
        return self._flows, self._attrs, self._seconds
    
    def getseconds(self):
        return self._seconds
    
    def getminutes(self):
        return self._minutes
    
    def gethours(self):
        return self._hours
        
    def getdays(self):
        return self._days
        
    def address(self):
        return self.name
    
    def stats(self):
        return {'address':self._name,
                'flows':{'raw':self._flows.status(), 
                         'attributes':self._attrs.status(),
                         'apps':self._appflows.status()},
                'time':{'seconds':self._seconds.status(),
                        'minutes':self._minutes.status(self._seconds),
                        'hours':self._hours.status(self._minutes),
                        'days':self._days.status(self._hours)}}

    def on_time(self, qbuf, secs, mins, hours, days):
        self._seconds.onsecond(self._apps._nativeapps, secs)
        if mins:
            self._minutes.onminute(qbuf._native, self._apps._nativeapps, self._appflows, self._seconds, mins)
            if hours:
                self._hours.onhour(qbuf._native, self._apps._nativeapps, self._appflows,  self._minutes, hours)
                if days:
                    self._days.onday(qbuf._native, self._apps._nativeapps, self._appflows,  self._hours, days)
        
    @property
    def name(self):
        return self._name

    @property
    def ip(self):
        return self._ip
    
class Apps(object):
    
    def __init__(self):
        self._nativeapps = appsmod.Apps(flowtools.settings.portrate, flowtools.settings.minthreshold)

    def status(self):
        return self._nativeapps.status()
