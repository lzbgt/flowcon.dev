'''
Created on Dec 9, 2013

@author: schernikov
'''

import argparse, zmq, names, pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interface', help='interface to connect to (ex. "tcp://host:1234")', required=True)
    parser.add_argument('-f', '--fields', help='show all detected IPFix fields', action='store_true')
    parser.add_argument('-s', '--status', help='show collector status', action='store_true')
    
    args = parser.parse_args()

    if not args.fields and not args.status:
        print "\nNothing to show...\n\n"
        parser.print_help()
        return
    
    process(args.interface, args.fields, args.status)
    
def process(addr, f, s):
    context = zmq.Context()
    print "Connecting to server..."
    socket = context.socket(zmq.DEALER)
    socket.connect (addr)
    
    msg = zmq.utils.jsonapi.dumps({'status':''})
    socket.send (msg)

    message = socket.recv()
    stats = zmq.utils.jsonapi.loads(message)
    if f:
        fset = set()
        for addr, flds in stats['fields'].items():
            print "%s"%(addr)
            print "  %s"%(flds)
            fset.update(flds)
        print
        for num in sorted(fset):
            nmp = names.fullmap['%d'%num]
            print " %5d %s"%(num, nmp[3])
        print
    if s:
        pprint.pprint(stats['stats'])
        apps = stats.get('apps', None)
        if apps:
            ports = apps['ports']
            tot = ports['total']
            counts = ports['counts']
            appkeys = sorted(counts, key=lambda p: counts[p], reverse=True)
            print "port counters: %d totalcount: %d"%(len(counts), tot)
            for k in appkeys[:20]:
                print "%5s: %.1f %d"%(k, 100.0*counts[k]/tot, counts[k])
                
            appset = apps['apps']
            print 'apps total: %d zeros:%d'%(appset['total'], appset['zeros'])
            for a in appset['ports'][:20]:
                p1 = a[0]
                p2 = a[1]
                try:
                    if p1 == 0:
                        print "             %5d[%5d]"%(p2, counts[str(p2)])
                    else:
                        print "%5d[%5d] %5d[%5d]"%(p1, counts[str(p1)], p2, counts[str(p2)])
                except:
                    print "%5d[%5d] %5d[%5d] error"%(p1, 0, p2, 0)

if __name__ == '__main__':
    main()
