'''
Created on Dec 4, 2013

@author: schernikov
'''

import logger

class Source(object):
    def __init__(self, addr, stamp):
        self._addr = addr
    def on_stamp(self, stamp):
        self._stamp = stamp
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
        counters = (self.bytesid, self.packetsid)
        self._reps, self._cnts, self._chks = self._fieldlist(qry['fields'], counters)
        self._fields = list(self._reps)
        self._fields.extend(self._cnts)
        self._period = qry.get('period', None)
        self._shape = self._on_shape(qry.get('shape', None), counters)
        if self._period: self._period = max(int(self._period), 1)
        self._end = None
        self._reconrds = set()
        self._reset()

    def _reset(self):
        self._flows = {}
        self._totals = [0 for _ in self._cnts]
        
    def _on_shape(self, shape, counters):
        if not shape: return None
        field = shape.get('max', None)
        if field is None:
            field = shape.get('min', None)
            if field is None:
                logger.dump("no shape function defined")
                return None
            reverse = False
        else:
            reverse = True
        if field == 'bytes':
            nm = self.bytesid
        elif field == 'packets':
            nm = self.packetsid
        elif field is None:
            pass
        else:
            logger.dump("unexpected sort field '%s'"%field)
        pos = 0
        for c in counters:
            if c == nm:
                return reverse, pos, shape.get('count', 0)
            pos += 1
        return None

    def _reshape(self, res):
        if not self._shape: return res
        reverse, pos, count = self._shape
        pos += len(self._reps)
        res = sorted(res, key=lambda l: l[pos], reverse=reverse)
        if count <= 0: return res
        return res[:count]

    def _mkreply(self, res, totals, count):
        reply = {'counts':res}
        if totals:
            tots = ['*' for _ in self._reps]
            tots.extend(totals)
            reply['totals'] = {'counts':tots, 'entries':count}
        return reply

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
            return self._mkreply([[dd[f] for f in self._fields]], None, 1)

        stamp = dd[self.startstamp]
        addr = dd[self.srcaddress]
        source = sources.get(addr, None)
        if source is None:
            source = Source(addr, stamp)
            sources[addr] = source

        source.on_stamp(stamp)
        
        key = tuple([dd[f] for f in self._reps])
        flow = self._flows.get(key, None)
        if flow is None:
            flow = [0 for _ in self._cnts]
            self._flows[key] = flow
        pos = 0
        for f in self._cnts:
            val = dd[f]
            flow[pos] += val
            self._totals[pos] += val
            pos += 1
        return None

    def results(self, now):
        if not self._end:
            if not self._period:
                return None
            self._end = now+self._period
            return None
        if self._end > now: return None
        self._end = now+self._period
        flows = self._flows
        tots = self._totals
        self._reset()
        res = []
        for k, v in flows.items():
            l = list(k)
            l.extend(v)
            res.append(l)
        return self._mkreply(self._reshape(res), tots, len(flows))

    def _mkcollection(self, flows):
        res = []
        for k, v in flows.items():
            l = list(k)
            l.extend(v)
            res.append(l)
        return res

    def _fieldlist(self, fields, counters):
        ls = set()
        cs = []
        for fk, fv in fields.items():
            if fv == '*': 
                ls.add(int(fk))
            else:
                cs.append((fk, fv))
        ls = ['%d'%x for x in sorted(ls.difference(counters))]
        return ls, ['%d'%x for x in counters], cs
