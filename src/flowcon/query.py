'''
Created on Dec 4, 2013

@author: schernikov
'''

import datetime

import logger

class Source(object):
    def __init__(self, addr, stamp):
        self._addr = addr
        self.stamp = stamp
    @property
    def address(self):
        return self._addr

class Query(object):
    bytesid = 1
    packetsid = 2
    srcaddress = '130'
    startstamp = '22'
    endstamp = '21'

    def __init__(self, qry):
        self._reps, self._cnts, self._chks = self._fieldlist(qry['fields'])
        self._fields = list(self._reps)
        self._fields.extend(self._cnts)
        self._period = qry.get('period', None)
        if self._period: self._period = datetime.timedelta(seconds=max(int(self._period), 1))
        self._end = None
        self._flows = {}
        self._reconrds = set()

    def addrec(self, rec):
        self._reconrds.add(rec)

    def delrec(self, rec):
        self._reconrds.discard(rec)
        
    def records(self):
        return self._reconrds
        
    @property
    def checkfields(self):
        return self._chks

    def collect(self, sources, dd):
        if not self._period: 
            return [dd[f] for f in self._fields]

        stamp = dd[self.startstamp]
        addr = dd[self.srcaddress]
        source = sources.get(addr, None)
        if source is None:
            source = Source(addr, stamp)
            sources[addr] = source
            self._end = stamp+self._period
        
        if source.stamp != stamp:
            if source.stamp < stamp:
                source.stamp = stamp
                logger.dump("%s: %d (%d)"%(source.address, stamp, self._end))
                if self._end >= stamp:
                    logger.dump("%s"%dd)
                    self._end = stamp+self._period
                    self._mkcollection(self._flows)
                    self._flows = {}
            else:
                #logger.dump("%s: skewed timing %d > %d"%(source.address, source.stamp, stamp))
                return None

        key = tuple([dd[f] for f in self._reps])
        flow = self._flows.get(key, None)
        if flow is None:
            flow = [0 for _ in self._cnts]
            self._flows[key] = flow
        pos = 0
        for f in self._cnts:
            flow[pos] += dd[f]
            pos += 1
        return None

    def results(self, now):
        if not self._period: return None
        if self._end is None or self._end <= now:
            logger.dump("time to report")
            self._end = now+self._period
        #TODO: finish
        return None

    def _mkcollection(self, flows):
        res = []
        for k, v in flows.items():
            l = list(k)
            l.extend(v)
            res.append(l)
        return res

    def _fieldlist(self, fields):
        ls = set()
        cs = []
        for fk, fv in fields.items():
            if fv == '*': 
                ls.add(int(fk))
            else:
                cs.append((fk, fv))
        counters = (self.bytesid, self.packetsid)
        ls = ['%d'%x for x in sorted(ls.difference(counters))]
        return ls, ['%d'%x for x in counters], cs
