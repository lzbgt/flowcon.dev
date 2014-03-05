'''
Created on Nov 25, 2013

@author: schernikov
'''

import zmq, datetime, pprint
import dateutil.tz

from zmq.eventloop import ioloop
ioloop.install()

import connector, flowtools.logger as logger , flowtools.settings, native.query as querymod
#import numpyfy.source as querymod
import native.receiver

tzutc = dateutil.tz.tzutc()

class HUnit(object):
    
    def __init__(self, nm, seconds, count, stamp):
        self._name = nm
        self._one = seconds
        self._first = None
        self._now = self.fromstamp(stamp)
        self._first = self._now + self._one
        self._width = (count-1)*seconds
        
    @property
    def one(self):
        return self._one

    def fromstamp(self, stamp):
        return int(stamp/self._one)*self._one
    
    @property
    def now(self):
        return self._now
    
    @property
    def oldest(self):
        return self._now-self._width
    
    @property
    def first(self):
        return self._first
    
    def tick(self, now):
        self._now = now
        
    def name(self):
        return self._name

class History(object):
    onesecond = 1
    oneminute = 60
    onehour = 3600
    oneday = 86400

    def __init__(self):
        now = datetime.datetime.utcnow()
        self._oldest = now
        stamp = native.query.mkstamp(now)
        self._now = datetime.datetime.utcfromtimestamp(stamp)
        self._second = HUnit('seconds', self.onesecond, flowtools.settings.maxseconds, stamp)
        self._minute = HUnit('minutes', self.oneminute, flowtools.settings.maxminutes, stamp)
        self._hour = HUnit('hours', self.onehour, flowtools.settings.maxhours, stamp)
        self._day = HUnit('days', self.oneday, flowtools.settings.maxdays, stamp)

    def oldest(self):
        return self._oldest
            
    def tick(self):
        now = datetime.datetime.utcnow()
        stamp = native.query.mkstamp(now)
        self._now = datetime.datetime.utcfromtimestamp(stamp)
        mins = None
        hours = None
        days = None
        if self._second.now != stamp:
            self._second.tick(stamp)
            minute = self._minute.fromstamp(stamp)
            if self._minute.now != minute:
                self._minute.tick(minute)
                mins = minute
                hour = self._hour.fromstamp(stamp)
                if self._hour.now != hour:
                    self._hour.tick(hour)
                    hours = hour
                    day = self._day.fromstamp(stamp)
                    if self._day.now != day:
                        self._day.tick(day)
                        days = day
        return now, stamp, mins, hours, days
    
    def now(self):
        return self._now
    
    def seconds(self):
        return self._second
    
    def minutes(self):
        return self._minute
    
    def hours(self):
        return self._hour
    
    def days(self):
        return self._day
    
class FlowProc(connector.Connection):
    def __init__(self, receiver):
        self._addresses = {}
        self._long_queries = {}
        self._periodic = {}
        self._nreceiver = receiver
        self._nbuf = native.query.QueryBuffer()
        self._history = History()
        self._apps = native.receiver.Apps()
        
        receiver.sourcecallback(self._apps, self.onnewsource)

    def _send(self, addr, res):
        res = zmq.utils.jsonapi.dumps(res)
        self.send_multipart([addr, res])
                    
    def _on_status(self, addr, msg):
        stats = []
#        fields = {}
        for s in self._nreceiver.sources():
#            fields[s.address()] = sorted([int(f) for f in s.fields()])
            stats.append(s.stats())
#        res = {'fields':fields, 'stats':stats, 'apps':self._apps.report()}
        d = datetime.datetime.utcnow().replace(tzinfo=tzutc)
        res = {'oldest':str(self._history.oldest().replace(tzinfo=tzutc)),
               'now':str(d),
               'stats':stats, 
               'apps':self._apps.status(),
               'querybuf':self._nbuf.status()}
        self.send_multipart([addr, zmq.utils.jsonapi.dumps(res)])
             
    def on_time(self):
        # check for expired peers
        now, secs, mins, hours, days = self._history.tick()

        self._check_peers(now)
        
        for source in self._nreceiver.sources():
            source.on_time(self._nbuf, secs, mins, hours, days)
            
        for per in self._periodic.values():
            res = per.on_time(self._nbuf, now, secs)
            if res:
                qrec = self._long_queries.get(per.id, None)
                if qrec:
                    for addr in qrec.aset:
                        self.send_multipart([addr, res])

    def onnewsource(self, src):
        print "created %s"%(src.name)
        for per in self._periodic.values():
            per.addsource(src)

    def on_msg(self, msg):
        req = msg[1]
        addr = msg[0]
        if req == self.hbmessage:
            arec = self._addresses.get(addr, None)
            if not arec:
                logger.dump("unexpected heartbeat from %s"%([addr]))
                return
            arec.stamp()
            return
        try:
            qry = zmq.utils.jsonapi.loads(req)
            hb = int(qry.get('heartbeat', self.heartbeat))
        except Exception, e:
            logger.dump("invalid query: %s"%(req))
            err = {'error':str(e), 'badmsg':req}
            self._send(addr, err)
            return
        arec = self._addresses.get(addr, None)
        if not arec:
            arec = ARecord(addr, hb)
            self._addresses[addr] = arec
        
        query = qry.get('query', None)
        if query:
            try:
                q = querymod.Query.create(query, self._history)
            except Exception, e:
                import traceback
                traceback.print_exc()
                logger.dump("bad query: %s from %s (hb:%ds): %s"%(query, [addr], hb, str(e)))
                err = {'error':str(e), 'badmsg':req}
                self._send(addr, err)
                return

            logger.dump("got query %s from %s (hb:%ds):\n%s\n"%(q.id, [addr], hb, pprint.pformat(query)))
            
            if q.is_live():
                self._register(q, addr)
                arec.add(q.id)
            else:
                res = q.collect(self._nbuf, self._nreceiver.sources())
                self.send_multipart([addr, res])

            return
        stat = qry.get('status', None)
        if not (stat is None):
            self._on_status(addr, stat)
            return
        logger.dump("unknown request: %s"%req)
        return

    def _register(self, q, addr):
        qrec = self._long_queries.get(q.id, None)
        if qrec is None:
            qrec = QRecord(q)
            self._long_queries[q.id] = qrec
            if type(q) == querymod.RawQuery:
                # only raw queries need to be registered with receiver                
                def onmsg(msg):
                    for addr in qrec.aset:
                        self.send_multipart([addr, msg])
                self._nreceiver.registerraw(q, onmsg)
                
            if type(q) == querymod.PeriodicQuery:
                self._periodic[q.id] = q
                for src in self._nreceiver.sources():
                    q.addsource(src)
                
        qrec.aset.add(addr)
        
    def _check_peers(self, now):
        for addr, rec in self._addresses.items():
            qids = rec.check(now)
            if qids is not None:
                del self._addresses[addr]
                logger.dump("dropping peer from %s"%([addr]))
                for qid in qids:
                    qrec = self._long_queries.get(qid, None)
                    if qrec is None: continue
                    qrec.aset.discard(addr)
                    if not qrec.aset:   # if all addresses are gone for this query 
                        del self._long_queries[qid]
                        q = qrec.query
                        logger.dump("dropping query %s"%(q.value()))
                        
                        if type(q) == querymod.RawQuery:
                            self._nreceiver.unregisterraw(qid)
                            
                        if type(q) == querymod.PeriodicQuery:
                            del self._periodic[qid]

class QRecord(object):
    def __init__(self, q):
        self.query = q
        self.aset = set()

class ARecord(object):
    maxmisses = 3

    def __init__(self, addr, hb):
        self._misses = 0
        self._addr = addr
        self._qids = set()
        self._hb = datetime.timedelta(seconds=hb)
        self.stamp()

    def stamp(self):
        self._misses = 0
        self._end = datetime.datetime.now()+self._hb
        
    def add(self, qid):
        self._qids.add(qid)

    def check(self, now):
        if self._end > now: return None
        self._end = now+self._hb
        self._misses += 1
        if self._misses >= self.maxmisses:
            return self._qids
        return None
        
    @property
    def address(self):
        return self._addr
        
def setup(insock, servsock, qrysock):
    try:
        conn = connector.Connector()
        
        recvr = native.receiver.Receiver(insock, conn.loop)

        fproc = FlowProc(recvr)
        
        conn.timer(1, fproc.on_time)

        conn.listen(qrysock, fproc)
        
    except KeyboardInterrupt:
        logger.dump("closing")
    finally:
        conn.close()
