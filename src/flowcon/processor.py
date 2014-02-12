'''
Created on Nov 25, 2013

@author: schernikov
'''

import zmq, datetime, pprint

from zmq.eventloop import ioloop
ioloop.install()

import connector, flowtools.logger as logger , native.query as querymod
#import numpyfy.source as querymod
import native.receiver

class FlowProc(connector.Connection):
    def __init__(self, receiver):
        self._addresses = {}
        self._long_queries = {}
        self._periodic = {}
        self._nreceiver = receiver
        self._nbuf = native.query.QueryBuffer()
        
        receiver.sourcecallback(self.onnewsource)

    def _send(self, addr, res):
        res = zmq.utils.jsonapi.dumps(res)
        self.send_multipart([addr, res])
                    
    def _on_status(self, addr, msg):
        stats = []
        fields = {}
        for s in self._nreceiver.sources():
            fields[s.address()] = sorted([int(f) for f in s.fields()])
            stats.append(s.stats())
        res = {'fields':fields, 'stats':stats}
        self.send_multipart([addr, zmq.utils.jsonapi.dumps(res)])
             
    def on_time(self):
        # check for expired peers
        now = datetime.datetime.utcnow()
        self._check_peers(now)
        
        stamp = native.query.mkstamp(now)
        for source in self._nreceiver.sources():
            source.on_time(stamp)
            
        for per in self._periodic.values():
            res = per.on_time(self._nbuf, now, stamp)
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
                q = querymod.Query.create(query)
            except Exception, e:
                import traceback
                traceback.print_exc()
                logger.dump("bad query: %s from %s (hb:%ds): %s"%(query, [addr], hb, str(e)))
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
