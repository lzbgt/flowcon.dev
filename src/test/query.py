'''
Created on Nov 26, 2013

@author: schernikov
'''

import argparse, zmq, datetime

import names, flowcon.connector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--list', help='list all flow fields', action='store_true')
    parser.add_argument('-i', '--interface', help='interface to connect to (ex. "tcp://host:1234")')
    parser.add_argument('-f', '--field', help='IPFix field names to capture from "--list"', nargs="+")
    parser.add_argument('-p', '--period', help='reporting period in seconds')
    
    args = parser.parse_args()
    
    header = "  {0:>6s} {1:>24s} {2:<28s} {3:s}"
    if args.list:
        print "Field types (%d):"%(len(names.nameset))
        if len(names.nameset) > 0:
            print header.format('ID', 'NetFlow', 'IPFix', 'Description')
            print '-'*80
            for fid, nf, fix, desc in names.nameset:
                print header.format(fid, nf, fix, desc)
        return
    
    if args.field:
        print "Capturing fields:"
        fids = {}
        for fl in args.field:
            fset = names.namesmap.get(fl, None)
            if not fset:
                print "don't know what to do with '%s' field"%(fl)
                return
            print header.format(*fset)
            fids[fset[0]] = '*'
        print
        if not args.interface:
            print "interface is not provided"
            return
        process(args.interface, args.period, fids)
        return

    parser.print_help()

class Query(flowcon.connector.Connection):
    def __init__(self, qry):
        q = zmq.utils.jsonapi.dumps({'query':qry, 'heartbeat':self.heartbeat})
        print "sending query:", q
        self._qry = q
        self._sender = self._do_nothing
    
    def on_time(self):
        self._sender(self.hbmessage)
    
    def on_open(self, sid):
        now = datetime.datetime.now()
        print "%s: connected from %d"%(now, sid)
        self._sender = self.send
        self._sender(self._qry)
    
    def _do_nothing(self, msg):
        return

    def on_msg(self, msg):
        rep = zmq.utils.jsonapi.loads(msg[0])
        ll = rep['counts']
        print "got %d entries"%(len(ll))
        for l in ll:
            print '  %s'%l
        tots = rep['totals']
        print '  totals: %s entries: %d'%(tots['counts'], tots['entries'])
        
    def on_close(self, sid):
        now = datetime.datetime.now()
        print "%s: disconnected from %d"%(now, sid)
        self._sender = self._do_nothing

def process(addr, period, fids):
    query = {'fields':fids, 'shape':{'max':'bytes', 'count':10}}
    if period: query['period'] = period

    q = Query(query)

    try:
        conn = flowcon.connector.Connector()
        
        conn.timer(q.heartbeat, q.on_time)
        
        conn.connect(addr, q)
    except KeyboardInterrupt:
        print "closing"
    finally:
        conn.close()

if __name__ == '__main__':
    main()
