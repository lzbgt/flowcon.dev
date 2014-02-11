'''
Created on Nov 26, 2013

@author: schernikov
'''

import argparse, zmq, datetime, pprint

import names, flowcon.connector, flowcon.query

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--list', help='list all flow fields', action='store_true')
    parser.add_argument('-i', '--interface', help='interface to connect to (ex. "tcp://host:1234")')
    parser.add_argument('-f', '--field', help='IPFix field names to capture from "--list"', nargs="+")
    parser.add_argument('-p', '--period', help='reporting period in seconds', type=int)
    parser.add_argument('-m', '--method', help='sorting methods', choices=['max', 'min'])
    parser.add_argument('-s', '--sortby', help='sorting field; considered only when "--method" is provided', choices=['bytes', 'packets'])
    parser.add_argument('-c', '--count', help='max number of entries to report', type=int)
    parser.add_argument('-b', '--heartbeat', help='heartbeat interval in seconds (default: %(default)s)', 
                        default=Query.heartbeat, type=int)
    parser.add_argument('--oldest', help='ignore any records older than this in seconds from now', type=int)
    parser.add_argument('--newest', help='ignore any records newer than this in seconds from now', type=int)
    
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
    
    if args.field or args.oldest or args.newest:
        fids = {}
        if args.field:
            print "Capturing fields:"
            for fl in args.field:
                idx = fl.find('=')
                if idx > 0:
                    nm = fl[:idx]
                    val = fl[idx+1:]
                    vals = val.split(',')
                    if len(vals) > 1: val = vals
                    fset = names.namesmap.get(nm, None)
                else:
                    fset = names.namesmap.get(fl, None)
                    val = '*'
                if not fset:
                    print "don't know what to do with '%s' field"%(fl)
                    return
                print header.format(*fset)
                fids[fset[0]] = val
            print
        if not args.interface:
            print "interface is not provided"
            return
        if args.method and not args.sortby:
            print "don't know how to sort with '%s'; --sortby is not provided"%(args.method)
            return
        if args.oldest or args.newest:
            tm = (args.oldest, args.newest)
        else:
            tm = None
        if args.period:
            if args.oldest or args.newest:
                print "both period and range specified; ignoring period"
            else:
                tm = args.period

        process(args.interface, tm, args.method, args.sortby, args.count, args.heartbeat, fids)
        return

    parser.print_help()

class Query(flowcon.connector.Connection):
    def __init__(self, qry, hb=None, once=False):
        if hb is None: hb = self.heartbeat
        qobj = {'query':qry, 'heartbeat':hb}
        q = zmq.utils.jsonapi.dumps(qobj)
        print "sending query:"
        pprint.pprint(qobj)
        print
        self._once = once
        self._qry = q
        self._sender = self._do_nothing
    
    def on_time(self):
        self._sender(self.hbmessage)
    
    def on_open(self, sid):
        now = datetime.datetime.now()
        print "%s: connected with %s"%(now, sid)
        self._sender = self.send
        self._sender(self._qry)
    
    def _do_nothing(self, msg):
        return

    def on_msg(self, msg):
        rep = zmq.utils.jsonapi.loads(msg[0])
        if not isinstance(rep, dict):
            print rep
        else:
            ll = rep.get('counts', None)
            if ll is None:
                print rep
            else:
                tots = rep.get('totals', None)
                if tots:
                    totmsg = '(totals: %s entries: %d)'%(tots['counts'], tots['entries'])
                else:
                    totmsg = ''
                print "got %d entries %s"%(len(ll), totmsg)
                for l in ll:
                    print '  %s'%l
        print
        if self._once: self._once()
        
    def on_close(self, sid):
        now = datetime.datetime.now()
        print "%s: disconnected from %s"%(now, sid)
        self._sender = self._do_nothing

def fidsmod(fids):
    for v in fids.values():
        if flowcon.query.isone(v):
            if v == '*':
                return 'flows'
        else:
            for e in v:
                if e == '*':
                    return 'flows'
    return 'time'
                
def process(addr, tm, method, sortby, count, hb, fids):
    flows = {'fields':fids}
    query = {'flows':flows}
    shape = {}
    if method: shape[method] = sortby
    if count: shape['count'] = count
    if shape: flows['shape'] = shape
    once = False
    if tm:
        tmrec = {}
        try:
            iter(tm)
            # tm is iterable
            tmrec['mode'] = fidsmod(fids)
            if tm[0]: tmrec['oldest'] = tm[0]
            if tm[1]: tmrec['newest'] = tm[1]
            query['time'] = tmrec
            once = True
        except TypeError:
            tmrec['mode'] = 'periodic'
            tmrec['seconds'] = tm
        query['time'] = tmrec

    try:
        conn = flowcon.connector.Connector()
        if once: once = lambda: conn.stop()
        q = Query(query, hb, once)
        
        conn.timer(q.heartbeat, q.on_time)
        
        conn.connect(addr, q)
    except KeyboardInterrupt:
        print "closing"
    finally:
        conn.close()

if __name__ == '__main__':
    main()
