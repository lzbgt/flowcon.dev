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
    

if __name__ == '__main__':
    main()