'''
Created on Nov 25, 2013

@author: schernikov
'''

import zmq.utils.jsonapi, datetime

every = datetime.timedelta(seconds=5)

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

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect('tcp://10.1.31.81:5556')
    sock.setsockopt(zmq.SUBSCRIBE, 'flow')
    flowtags = ['8','7','4','12','11']
    print "capturing tags:"
    for ft in flowtags:
        print '  %s'%(flowtypes[ft])

    flowrepo = {}
    count = 0
    now = None
    prev = None
    while True:
        msg = sock.recv_multipart()
        dd = zmq.utils.jsonapi.loads(msg[1])
        count += 1
        flowkey = ''
        for ft in flowtags:
            flowkey += ':'+str(dd[ft])
        frec = flowrepo.get(flowkey, None)
        if frec is None:
            frec = [0]
            flowrepo[flowkey] = frec
            now = datetime.datetime.now()
            if prev is None or (prev + every) <= now:
                print "%4d[%d/%d] %s %s"%(len(flowrepo)*1000/count, len(flowrepo), count, flowkey, now)
                prev = now
        frec[0] += 1
        #showflow(count, dd)

def showflow(count, dd):
    print "flow %d"%(count)
    for k, v in dd.items():
        print "  %20s %s"%(v, flowtypes.get(k))
        
if __name__ == '__main__':
    main()