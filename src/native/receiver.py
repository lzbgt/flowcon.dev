'''
Created on Jan 24, 2014

@author: schernikov
'''

import socket, urlparse, datetime, os, re

import native, flowtools.settings

recmod = native.loadmod('nreceiver')
colmod = native.loadmod('collectors')
timecolmod = native.loadmod('timecollect')
appsmod = native.loadmod('napps')

snamere = re.compile('S(\d{1,3})_(\d{1,3})_(\d{1,3})_(\d{1,3})')

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
        
    def _newsource(self, ip):
        src = Sources(self._apps, ip)
        self.allsources[ip] = src
        return src
            
    def find(self, ip):
        src = self.allsources.get(ip, None)
        if src is None:
            src = self._newsource(ip)
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
            
    def backup(self, fileh, grp):
        for src in self.sources():
            nm = 'S'+src.name.replace('.', '_')
            sgrp = fileh.create_group(grp, nm)
            src.backup(fileh, sgrp)
            fileh.flush()
        
    def restore(self, fileh, grp):
        for sgrp in fileh.iter_nodes(grp):
            m = snamere.match(sgrp._v_name)
            if not m: continue
            ip = 0
            for p in m.groups():
                ip <<= 8
                ip += int(p)
            src = self._newsource(ip)
            src.restore(fileh, sgrp)
            if self._onsource: self._onsource(src)
            
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
        self._minutes = timecolmod.MinutesCollector("M:"+self._name, self._ip, libname, self._appflows, 
                                                    flowtools.settings.maxminutes, stamp, 
                                                    flowtools.settings.checkminutes)
        libname = os.path.join(native.libloc, 'hourscoll.so')
        self._hours = timecolmod.HoursCollector("H:"+self._name, self._ip, libname, 
                                                self._appflows, flowtools.settings.maxhours, stamp)
        libname = os.path.join(native.libloc, 'dayscoll.so')
        self._days = timecolmod.DaysCollector("D:"+self._name, self._ip, libname, 
                                              self._appflows, flowtools.settings.maxdays, stamp)
        self._collmap = {'attrs':self._attrs,
                         'flows':self._flows,
                         'aflows':self._appflows,
                         'seconds':self._seconds,
                         'minutes':self._minutes,
                         'hours':self._hours,
                         'days':self._days}
        
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
    
    def debug(self, req):
        freq = req.get('flows', None)
        if freq is not None:
            res = self._flows.debug(freq)
            if res is None:
                return {"error":"don't know what to do with request", "request":req}
            return res
        
        freq = req.get('attributes', None)
        if freq is not None:
            res = self._attrs.debug(freq)
            if res is None:
                return {"error":"don't know what to do with request", "request":req}
            return res
        
        freq = req.get('appflows', None)
        if freq is not None:
            res = self._appflows.debug(freq)
            if res is None:
                return {"error":"don't know what to do with request", "request":req}
            return res
        
    def backup(self, fileh, grp):
        for nm, obj in self._collmap.items():
            ogrp = fileh.create_group(grp, nm)
            obj.backup(fileh, ogrp)

    def restore(self, fileh, grp):
        for nm, obj in self._collmap.items():
            ogrp = fileh.get_node(grp, nm)
            obj.restore(fileh, ogrp)


class Apps(object):
    
    def __init__(self):
        actseconds = int(flowtools.settings.checkminutes*60*flowtools.settings.activerate)
        self._nativeapps = appsmod.Apps(flowtools.settings.portrate, 
                                        flowtools.settings.minthreshold,
                                        actseconds)

    def status(self):
        return self._nativeapps.status()

    def debug(self, req):
        res = self._nativeapps.debug(req)
        if res is None:
            return {"error":"don't know what to do with request", "request":req}
        return res
    
    def backup(self, fileh, grp):
        self._nativeapps.backup(fileh, grp)
        
    def restore(self, fileh, grp):
        self._nativeapps.restore(fileh, grp)
