'''
Created on Dec 9, 2013

@author: schernikov
'''

import argparse, zmq, names, pprint
import dateutil.parser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interface', help='interface to connect to (ex. "tcp://host:1234")', required=True)
    parser.add_argument('-f', '--fields', help='show all detected IPFix fields', action='store_true')
    parser.add_argument('-s', '--status', help='show collector status', action='store_true')
    parser.add_argument('-m', '--memory', help='show collector memory usage', action='store_true')
    parser.add_argument('-d', '--debug', help='debug collector')
    
    args = parser.parse_args()

    if not args.fields and not args.status and not args.memory and not args.debug:
        print "\nNothing to show...\n\n"
        parser.print_help()
        return
    
    if args.debug is not None:
        try:
            dreq = zmq.utils.jsonapi.loads(args.debug)
        except:
            print "\ncan not convert '%s' from JSON"%(args.debug)
            return
    else:
        dreq = None
    
    process(args.interface, args.fields, args.status, args.memory, dreq)
    
def process(addr, f, s, m, d):
    context = zmq.Context()
    print "Connecting to server..."
    socket = context.socket(zmq.DEALER)
    socket.connect (addr)

    if d:
        msg = zmq.utils.jsonapi.dumps({'status':{'debug':d}})
    else:
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
        pprint.pprint(stats)

    if m:
        showsizes(stats)
        
    if d:
        pprint.pprint(stats)
        
def showstats(stats):
        apps = stats.get('apps', None)
        if apps:
            ports = apps['ports']
            print "\nports:"
            print "  count:%d  (in flows over last hour) "%(ports['total']/2)
            print "  bytes:%d  (ports buffer)"%(ports['bytes'])
            #tot = ports['total']
#            counts = ports['counts']
#            appkeys = sorted(counts, key=lambda p: counts[p], reverse=True)
#            print "port counters: %d totalcount: %d"%(len(counts), tot)
#            for k in appkeys[:20]:
#                print "%5s: %.1f %d"%(k, 100.0*counts[k]/tot, counts[k])
                
            appset = apps['apps']
            print "\napps:"
            print "  bytes: %d (apps buffer)"%(appset['bytes'])
            print "  zeros: %d (flows with zero destination port)"%(appset['zeros'])
            print "  count: %d (number of apps)"%(appset['total'])
#            print 'apps total: %d zeros:%d'%(appset['total'], appset['zeros'])
#            for a in appset['ports'][:20]:
#                p1 = a[0]
#                p2 = a[1]
#                try:
#                    if p1 == 0:
#                        print "             %5d[%5d]"%(p2, counts[str(p2)])
#                    else:
#                        print "%5d[%5d] %5d[%5d]"%(p1, counts[str(p1)], p2, counts[str(p2)])
#                except:
#                    print "%5d[%5d] %5d[%5d] error"%(p1, 0, p2, 0)

def printline(fmt, *args):
    fstr = "  %16s"%args[0]
    for pos in range(1, len(args)):
        if not isinstance(args[pos], basestring):
            fstr+= fmt%(tuple(args[pos]))
        else:
            fstr+= fmt%args[pos]
    print fstr
    
def showsizes(stats):
    w = 8
    now = stats['now']
    start = stats['oldest']
    dn = dateutil.parser.parse(now)
    ds = dateutil.parser.parse(start)
    runtime = (dn-ds).total_seconds()
    if runtime > 60*2:
        if runtime > 3600*2:
            if runtime > 3600*24*2:
                rt = "%d days"%(runtime/3600/24)
            else:
                rt = "%d hours"%(runtime/3600)
        else:
            rt = "%d minutes"%(runtime/60)
    else:
        rt = "%d seconds"%(runtime)
    print "run time: %s  server clock: %s"%(rt, now)

    acoll = stats['apps']['apps']['collector']
    print "\napps:"
    abytes = acoll['entries']['bytes']
    print "  %*s %-*s"%(w, 'entries', w, 'bytes')
    print "  %*d %-*d"%(w, acoll['entries']['count'], w, abytes)
    print "\nqbuf" 
    qbuf = stats['querybuf']
    qbytes = qbuf['entries']['bytes']
    print "  %*s %-*s"%(w, 'entries', w, 'bytes')
    print "  %*d %-*d"%(w, qbuf['entries']['size'], w, qbytes)
    print '\nsources:'
    sources = stats['stats']
    hfmt = '%s%ds'%('%', w*2+1)

    fmt = '%s%dd %s-%dd'%('%', w, '%', w)
    
    print "\n flows:"
    printline(hfmt, 'name', 'app.entries', 'app.indices', 
              'attr.entries', 'attr.indices',
              'flow.entries', 'flow.indices')

    flowcounts = []
    for src in sources:
        fls = src['flows']
        ents = fls['apps']['entries']
        inds = fls['apps']['indices']
        atents = fls['attributes']['entries']
        atinds = fls['attributes']['indices']
        flents = fls['raw']['entries']
        flinds = fls['raw']['indices']
        printpairs(flowcounts, fmt, src['address'], 
                  (ents['count'], ents['bytes']), 
                  (inds['size'], inds['bytes']),
                  (atents['count'], atents['bytes']),
                  (atinds['size'], atinds['bytes']),
                  (flents['count'], flents['bytes']),
                  (flinds['size'], flinds['bytes']))
    flowbytes = printfooter('\n flow', fmt, flowcounts)

    print " times:"
    printline(hfmt, 'name', 'seconds', 'minutes', 'hours', 'days')

    timecounts = []
    for src in sources:
        tm = src['time']
        sents = tm['seconds']['entries']
        #sticks = tm['seconds']['ticks']['size']
        ments = tm['minutes']['entries']        
        hents = tm['hours']['entries']
        dents = tm['days']['entries']

        printpairs(timecounts, fmt, src['address'], 
                  (sents['count'], sents['bytes']), 
                  (ments['count'], ments['bytes']), 
                  (hents['count'], hents['bytes']), 
                  (dents['count'], dents['bytes']))
    timebytes = printfooter('\n time', fmt, timecounts)
    
    printbytes('total bytes', timebytes+flowbytes+abytes+qbytes)
        
def printbytes(msg, num):
    K = num / 1024
    M = num / (1024*1024)
    G = num / (1024*1024*1024)
    
    if K <= 0:
        print "%s: %d\n"%(msg, num)
        return

    if M <= 0:
        print "%s: %d (%dKB)\n"%(msg, num, K)
        return
    
    if G <= 0:
        print "%s: %d (%dMB)\n"%(msg, num, M)
        return 

    print "%s: %d (%dGB)\n"%(msg, num, G)
        
def printfooter(nm, fmt, counts):
    print
    printline(fmt, 'totals:', *counts)
    cbytes = 0
    for fc in counts:
        cbytes += fc[1]
    printbytes('%s bytes'%(nm), cbytes)

    return cbytes

def printpairs(counts, fmt, nm, *pairs):
    if not counts:
        for _ in range(len(pairs)):
            counts.append([0, 0])
    for pos in range(len(pairs)):
        counts[pos][0] += pairs[pos][0]
        counts[pos][1] += pairs[pos][1]

    printline(fmt, nm, *pairs)

if __name__ == '__main__':
    main()
