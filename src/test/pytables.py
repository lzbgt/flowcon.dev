'''
Created on Jan 9, 2014

@author: schernikov
'''

import tables, re

snamere = re.compile('S(\d{1,3})_(\d{1,3})_(\d{1,3})_(\d{1,3})')

def main():
    fname = '/home/schernikov/workspace/flowcon/backup/collector.backup'
    with tables.open_file(fname) as fileh:
        grp = fileh.get_node(fileh.root, 'sources')

        for sgrp in fileh.iter_nodes(grp):
            m = snamere.match(sgrp._v_name)
            if not m: continue
            print m.groups()
            fgrp = fileh.get_node(sgrp, 'flows')
            print "flows:"
            size = fgrp._v_attrs.end
            fcount = fgrp._v_attrs.freecount
            pos = fgrp._v_attrs.freepos
            print "  size:%d count:%d pos:%d"%(size, fcount, pos)

            tbl = fileh.get_node(fgrp, 'entries')
            t = tbl.read()
            tv = t[:size]
            cnt = 0
            prev = 0
            print "  first free:%d, prev:%d"%(pos, tv[pos][7])

            while pos != 0:
                entry = tv[pos]
                if entry[8] != 0: raise Exception('refcount != 0')
                if entry[7] != prev: print '  -----prev is broken: %d != %d'%(entry[7], prev)
                prev = pos
                pos = entry[0]
                cnt += 1
            if cnt != fcount: raise Exception('free count does not match: %d != %d'%(cnt, fcount))
            print "  last free:%d, prev:%d"%(prev, tv[prev][7])
            cnt = 0
            while prev != 0:
                entry = tv[prev]
                prev = entry[7]
                cnt += 1
                if cnt == fcount:
                    print "  stopping:%d, prev:%d"%(prev, tv[prev][7]) 
                    break
            if cnt != fcount: raise Exception('reverse free count does not match: %d != %d'%(cnt, fcount))

#            nxt = tv['next']
#            nxt[pos]
#            tv[pos]
#            checkempty(colnames, t[size:])
            
            
def checkempty(ext):
    colnames = ['next', 'crc', 'protocol', 'srcport', 'srcaddr', 'dstport', 'dstaddr', 'attrindex', 'refcount']
        
    for cnm in colnames:
        for v in ext[cnm]:
            if v != 0: raise Exception('%s not empty'%(cnm))

def structs():
    [('flow', '|S13'), ('attr', '|S23'), ('flowdig', '|S20'), ('attrdig', '|S20'), ('bytes', '<u4'), ('packets', '<u4')]
    [('protocol', '|u1'), ('srcport', '<u2'), ('srcaddr', '<u4'), ('dstport', '<u2'), ('dstaddr', '<u4'), ('ingressport', '<u2'), ('dstmask', '|u1'), ('egressport', '<u2'), ('nexthop', '<u4'), ('srcas', '<u4'), ('dstas', '<u4'), ('tos', '|u1'), ('tcpflags', '<u4'), ('srcmask', '|u1'), ('tdigest', '|S20'), ('adigest', '|S20'), ('bytes', '<u4'), ('packets', '<u4')]


if __name__ == '__main__':
    main()