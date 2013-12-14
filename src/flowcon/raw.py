'''
Created on Dec 5, 2013

@author: schernikov
'''
import datetime, zmq

import logger, connector

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
        
    def header(self):
        msg = "capturing tags:\n"
        for ft in self.flowtags:
            msg += '  %s\n'%(flowtypes[ft])
        logger.dump(msg)

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

def setup(insock, outsock, qrysock):
    try:
        conn = connector.Connector()

        pub = conn.publish(outsock)
        fproc = FlowRaw(pub)
        
        fproc.header()
        
        conn.subscribe(insock, 'flow', fproc.on_flow)

        conn.timer(1, fproc.on_time)

        conn.listen(qrysock, fproc)
        
    except KeyboardInterrupt:
        logger.dump("closing")
    finally:
        conn.close()

def showflow(count, dd):
    print "flow %d"%(count)
    for k, v in dd.items():
        print "  %20s %s"%(v, flowtypes.get(k))
