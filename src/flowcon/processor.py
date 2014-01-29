'''
Created on Nov 25, 2013

@author: schernikov
'''

import zmq, datetime, pprint

from zmq.eventloop import ioloop
ioloop.install()

import connector, flowtools.logger as logger , query as querymod
#import numpyfy.source as querymod
import native.receiver

class FlowProc(connector.Connection):
    def __init__(self, receiver):
        self._addresses = {}
        self._live_queries = {}
        self._receiver = receiver

    def _send_res(self, k, q, res):
        recs = q.records()
        if not recs:
            # no-one listens to this query; remove if it's live
            if k in self._live_queries:
                del self._live_queries[k]
                logger.dump("dropping query %s"%k)
            return
        if res:
            res = zmq.utils.jsonapi.dumps(res)
            for rec in recs:
                self.send_multipart([rec.address, res])
                
    def _send(self, addr, res):
        res = zmq.utils.jsonapi.dumps(res)
        self.send_multipart([addr, res])
                    
    def _on_status(self, addr, msg):
        stats = []
        fields = {}
        for s in self._receiver.sources():
            fields[s.address()] = sorted([int(f) for f in s.fields()])
            stats.append(s.stats())
        res = {'fields':fields, 'stats':stats}
        self.send_multipart([addr, zmq.utils.jsonapi.dumps(res)])
             
    def on_time(self):
        # check for expired peers
        now = datetime.datetime.utcnow()
        self._check_peers(now)

        for source in self._receiver.sources():
            source.on_time(now)

        # check for unwanted queues and send updates to listeners
#        stamp = int(time.mktime(now.timetuple()))
#        for k, q in self._live_queries.items():
#            res = q.results(stamp)
#            self._send_res(k, q, res)

    def _check_peers(self, now):
        for addr, rec in self._addresses.items():
            if not rec.check(now):
                del self._addresses[addr]
                logger.dump("dropping peer from %s"%([addr]))

    def on_msg(self, msg):
        req = msg[1]
        addr = msg[0]
        if req == self.hbmessage:
            record = self._addresses.get(addr, None)
            if not record:
                logger.dump("unexpected heartbeat from %s"%([addr]))
                return
            record.stamp()
            return

        try:
            qry = zmq.utils.jsonapi.loads(req)
        except Exception, e:
            logger.dump("invalid query: %s"%(req))
            err = {'error':str(e), 'badmsg':req}
            self._send(addr, err)
            return
        query = qry.get('query', None)
        if query:
            #TODO pretty ugly design here; connections, queries, sources has to be rewritten
            hb = int(qry.get('heartbeat', self.heartbeat))
            logger.dump("got query from %s (hb:%ds):\n%s\n"%([addr], hb, pprint.pformat(query)))

            q = self._live_queries.get(req, None)
            if not q:
                try:
                    q = querymod.Query.create(query)
                except Exception, e:
                    logger.dump("bad query: %s from %s (hb:%ds): %s"%(query, [addr], hb, str(e)))
                    return
                if q.is_live():
                    self._live_queries[req] = q
                    self._receiver.register(q)
            if not q.is_live():
                res = q.collect_sources(self._receiver.sources())
                self._send(addr, res)
                q = None    # to record it in QRec as `no pending query`
            record = self._addresses.get(addr, None)
            if not record:
                record = QReq(addr, q, hb)
                self._addresses[addr] = record
            else:
                if record.query != q:
                    logger.dump("different query from same address")
                    record.query = q
            return
        stat = qry.get('status', None)
        if not (stat is None):
            self._on_status(addr, stat)
            return
        logger.dump("unknown request: %s"%req)
        return
        
class QReq(object):
    maxmisses = 3

    def __init__(self, addr, qry, hb):
        self._misses = 0
        self._addr = addr
        self._query = qry
        if qry: qry.addrec(self)
        self._hb = datetime.timedelta(seconds=hb)
        self.stamp()

    def stamp(self):
        self._misses = 0
        self._end = datetime.datetime.now()+self._hb
        
    def check(self, now):
        if self._end > now: return True
        self._end = now+self._hb
        self._misses += 1
        if self._misses >= self.maxmisses:
            if self._query: self._query.delrec(self)
            return False
        return True
        
    @property
    def address(self):
        return self._addr
        
    @property
    def query(self):
        return self._query

    @query.setter
    def query(self, q):
        self._query.delrec(self)
        self._query = q
        q.addrec(self)
        
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
