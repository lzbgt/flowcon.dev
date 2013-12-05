'''
Created on Nov 25, 2013

@author: schernikov
'''

import zmq, datetime, time#, zmq.utils.monitor

from zmq.eventloop import ioloop
ioloop.install()

import connector, logger, query as querymod

flowtypes = {
         '1':'Incoming flow bytes (src->dst)',
         '2':'Incoming flow packets (src->dst)',
         '4':'IP protocol byte',
         '5':'Type of service byte',
         '6':'Cumulative of all flow TCP flags',
         '7':'IPv4 source port',
         '8':'IPv4 source address',
         '9':'IPv4 source subnet mask (/<bits>)',
         '10':'Input interface SNMP idx',
         '11':'IPv4 destination port',
         '12':'IPv4 destination address',
         '13':'IPv4 dest subnet mask (/<bits>)',
         '14':'Output interface SNMP idx',
         '15':'IPv4 next hop address',
         '16':'Source BGP AS',
         '17':'Destination BGP AS',
         '21':'SysUptime (msec) of the last flow pkt',
         '22':'SysUptime (msec) of the first flow pkt',
}

class FlowRaw():
    every = datetime.timedelta(seconds=5)
    flowtags = ['8','7','4','12','11']
    hosttags = ['8', '12']

    def __init__(self, pub):
        self.pub = pub
        self.flowrepo = {}
        self.hostrepo = {}
        self.count = 0
        self.prev = None

    def on_flow(self, msg):
        #print msg
        dd = zmq.utils.jsonapi.loads(msg[1])
        flowkey = ''
        self.count += 1
        for ft in self.flowtags:
            flowkey += ':'+str(dd[ft])
        frec = self.flowrepo.get(flowkey, None)
        if frec is None:
            frec = [0]
            self.flowrepo[flowkey] = frec
            now = datetime.datetime.now()
            if self.prev is None or (self.prev + self.every) <= now:
                for ht in self.hosttags:
                    host = dd[ht]
                    hrec = self.hostrepo.get(host, None)
                    if hrec is None:
                        hrec = [0]
                        self.hostrepo[host] = hrec
                        self.pub.send_multipart(['hosts', zmq.utils.jsonapi.dumps({'host':host, 
                                                                                   'count':len(self.hostrepo)})])
                    hrec[0] += 1
                logger.dump("%4d[%d/%d] %s %s"%(len(self.flowrepo)*1000/self.count, 
                                                len(self.flowrepo), self.count, flowkey, now))
                self.prev = now
        frec[0] += 1

class FlowProc(connector.Connection):
    flowtags = ['8','7','4','12','11']

    def __init__(self):
        self._addresses = {}
        self._queries = {}
        self._sources = {}
        self.count = 0

    def _send_res(self, k, q, res):
        recs = q.records()
        if not recs:
            # no-one listens to this queue; remove
            del self._queries[k]
            logger.dump("dropping query %s"%k)
            return
        if res:
            res = zmq.utils.jsonapi.dumps(res)
            for rec in recs:
                self.send_multipart([rec.address, res])
                    
    def on_flow(self, msg):
        #print msg
        dd = zmq.utils.jsonapi.loads(msg[1])
        for k, qry in self._queries.items():
            for fk, fv in qry.checkfields:
                if dd[fk] != fv: break   # this flow is filtered out
            else:
                res = qry.collect(self._sources, dd)
                if res: self._send_res(k, qry, res)

    def on_time(self):
        # check for expired peers
        now = datetime.datetime.now()
        for addr, rec in self._addresses.items():
            if not rec.check(now):
                del self._addresses[addr]
                logger.dump("dropping peer from %s"%([addr]))
        # check for unwanted queues and send updates to listeners
        stamp = int(time.mktime(now.timetuple()))
        for k, q in self._queries.items():
            res = q.results(stamp)
            self._send_res(k, q, res)

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

        qry = zmq.utils.jsonapi.loads(req)
        query = qry.get('query', None)
        if not query:
            logger.dump("unknown request: %s"%req)
            return

        hb = int(qry.get('heartbeat', self.heartbeat))
        logger.dump("got query: %s from %s (hb:%ds)"%(query, [addr], hb))
        q = self._queries.get(req, None)
        if not q:
            q = querymod.Query(query)
            self._queries[req] = q
        record = self._addresses.get(addr, None)
        if not record:
            record = QReq(addr, q, hb)
            self._addresses[addr] = record
        else:
            if record.query != q:
                logger.dump("different query from same address")
                record.query = q
        
class QReq(object):
    maxmisses = 3

    def __init__(self, addr, qry, hb):
        self._misses = 0
        self._addr = addr
        self._query = qry
        qry.addrec(self)
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
            self._query.delrec(self)
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
        
def setup(insock, outsock, qrysock):
    try:
        conn = connector.Connector()

        #pub = conn.publish(outsock)
        #fproc = FlowRaw(pub)
        
        fproc = FlowProc()
        
        msg = "capturing tags:\n"
        for ft in fproc.flowtags:
            msg += '  %s\n'%(flowtypes[ft])
        logger.dump(msg)
        
        conn.subscribe(insock, 'flow', fproc.on_flow)

        conn.timer(1000, fproc.on_time)

        conn.listen(qrysock, fproc)
        
    except KeyboardInterrupt:
        logger.dump("closing")
    finally:
        conn.close()
        
def showflow(count, dd):
    print "flow %d"%(count)
    for k, v in dd.items():
        print "  %20s %s"%(v, flowtypes.get(k))
