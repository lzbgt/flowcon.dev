'''
Created on Dec 4, 2013

@author: schernikov
'''
import dateutil.tz, datetime, dateutil.parser

import logger

tzutc = dateutil.tz.tzutc()

def tostamp(stamp):
    return str(stamp.replace(tzinfo=tzutc))

def _cleanupfields(fields):
    if Source.startstamp in fields:
        logger.dump("ignoring `start time`[%s] field in query"%(Source.startstamp))
        del fields[Source.startstamp]
    if Source.endstamp in fields:
        logger.dump("ignoring `end time`[%s] field in query"%(Source.endstamp))
        del fields[Source.endstamp]


class Collector(object):
    bytesid = 1
    packetsid = 2
    counters = (bytesid, packetsid)

    def __init__(self, fields):
        self._cnts = ['%d'%x for x in self.counters]
        self._reset()
        self._fieldlist(fields)

    def _reset(self):
        self._flows = {}
        self._totals = [0 for _ in self._cnts]
        
    def _collect(self, dd):
        """ Need to rethink storage options so flows could be scanned not only in time->flow order 
            but also flow->time. This way queries can accumulate flows with shape considered so there will be
            no need to accumulate huge flow collections before shaping it. This could be done in 
            streaming fashion on per flow basis.  
        """
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

    def _fieldlist(self, fields):
        ls = set()
        cs = []
        for fk, fv in fields.items():
            if fv == '*': 
                ls.add(int(fk))
            else:
                cs.append((fk, fv))
        ls = ['%d'%x for x in sorted(ls.difference(self.counters))]
        self._reps = ls  # all fields we need to consider for aggregation 
        #                # (except counters fields) and report, i.e. all `*` fields
        self._chks = cs  # fields to filter flows with 
        #                # If these fields no not match with given flow then flow is discarded

class Source(Collector):
    srcaddress = '130'
    startstamp = '22'
    endstamp = '21'
    flowtuple = ['4', '7', '8', '11', '12']
    oldestsecond = datetime.timedelta(hours=1)

    def __init__(self, addr):
        ff = {}
        for fid in self.flowtuple: ff[fid] = '*'  
        super(self.__class__, self).__init__(ff)
        self._addr = addr
        self._field = None
        self._attributes = None
        self._seconds = []

    def _update_fields(self, dd):
        self._field = set(dd.keys())
        attrs = self._field.difference(self.flowtuple)
        attrs.difference_update((self.srcaddress, self.startstamp, self.endstamp))
        self._attributes = attrs # all fields except `special` ones.

    def account(self, dd):
        if self._field is None: self._update_fields(dd)

        self._stamp = dd[self.endstamp]
        self._collect(dd)

    def on_time(self, now):
        self._seconds.append((now, self._flows))
        oldest = self._seconds[0][0]
        while (oldest+self.oldestsecond) < now:
            self._seconds.pop(0)
            oldest = self._seconds[0][0]
        self._flows = {}

    def fields(self):
        return self._field

    def stats(self):
        fids = set(self._flows.keys())
        for stamp, flows in self._seconds:
            fids.update(flows.keys()); stamp
        seconds = {'count':len(self._seconds)}
        if self._seconds:
            seconds['oldest'] = tostamp(self._seconds[0][0])
            seconds['newest'] = tostamp(self._seconds[-1][0])

        return {'address':self._addr, 
                'stamp':self._stamp,
                'flows':len(fids),
                'seconds':seconds,
                'totals':tuple(self._totals)}
        
    def history(self, collecton, newest, oldest, keycall):
        """Collect history of flows for seconds, minutes, hours or days.
           Need to make sure duplicate entries are not counted. 
           Ex. older seconds and minutes aggregated from these seconds should be either one or another, not both.    
        """
        self._history(collecton, self._seconds, newest, oldest, keycall)
        
    def _history(self, collection, group, step, newest, oldest, keycall):
        counts = range(len(self._cnts))
        if newest or oldest:
            if not newest:
                for stamp, flows in group:
                    if stamp <= oldest: continue     # not reached yet
                    self._hiscall(collection, flows, keycall, counts)
                return
            if not oldest:
                for stamp, flows in group:
                    if stamp > newest: break        # time scope is over
                    self._hiscall(collection, flows, keycall, counts)
                return
            for stamp, flows in group:
                if stamp > newest: break        # time scope is over
                if stamp <= oldest: continue     # not reached yet
                self._hiscall(collection, flows, keycall, counts)
            return
        for stamp, flows in group:
            self._hiscall(collection, stamp, flows, keycall, counts) 

    def _hiscall(self, collection, stamp, flows, keycall, counts):
        for key, flow in flows.items():
            k = keycall(key, stamp)
            if k is None: continue # ignore that flow
            entry = collection.get(k, None)
            if entry is None:
                entry = [0 for _ in self._cnts]
                collection[k] = entry
            for p in counts:
                entry[p] += flow[p]
        
    def address(self):
        return self._addr

def stamp2time(stamp, now):
    try:
        # check if it's relative time stamp in seconds back into history
        then = float(stamp)
        if then <= 0:
            raise Exception("can handle only positive number of seconds for relative period in seconds, not %s"%(stamp))
        return now-datetime.timedelta(seconds=then)
    except:
        pass
    try:
        tm = dateutil.parser.parse(stamp)
        if tm >= now:
            raise Exception("absolute time stamp %s should be older than current time (%s)"%(stamp, tostamp(now)))
    except Exception, e:
        raise Exception("don't know what to do with period stamp %s: %s"%(stamp, str(e)))
    return tm

class Query(Collector):
    @classmethod
    def create(cls, qry):
        fields = qry['fields']
        shape = qry.get('shape', None)

        per = qry.get('time', None)
        if per is None:
            return RawQuery(fields, shape)

        # remove time stamp fields
        _cleanupfields(fields)

        try:
            per = int(per)
            return PeriodicQuery(fields, shape, max(per, 1))
        except:
            pass
        try:
            iter(per)
        except TypeError:
            raise Exception("don't know what to do with `time`")
        if isinstance(per, basestring):
            raise Exception("don't know what to do with `time` string")
        stamp = per[0]
        if stamp:
            now = datetime.datetime.utcnow()
            oldest = stamp2time(stamp, now)
        else:
            oldest = None

        return FlowQuery(fields, shape, newest, oldest)

    def __init__(self, fields, shape):
        super(self.__class__, self).__init__(fields)
        self._shape = self._on_shape(shape)

    def is_live(self):
        return False

    def _on_shape(self, shape):
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
        for c in self.counters:
            if c == nm:
                return reverse, pos, shape.get('count', 0)
            pos += 1
        return None

    def _reshape(self, res, insert=None):
        if not self._shape: return res
        reverse, pos, count = self._shape
        srtd = sorted(res, key=lambda l: l[1][pos], reverse=reverse)
        if count > 0: srtd = srtd[:count]
        if insert is None: return srtd
        # need to plug value into keys
        pos, val = insert
        pos += 1
        res = []
        for k, v in srtd:
            l = list(k)
            l.insert(pos, val)
            res.append([l, v])
        return res

    def _mkreply(self, res, totals, count):
        reply = {'counts':res}
        if totals:
            tots = list(self._reps)
            tots.extend(totals)
            reply['totals'] = {'counts':tots, 'entries':count}
        return reply

    def _results(self):
        flows = self._flows
        tots = self._totals
        self._reset()
        return self._mkreply(self._reshape(flows.items()), tots, len(flows))
    
class LiveQuery(Query):
    def __init__(self, fields, shape):
        super(self.__class__, self).__init__(fields, shape)
        self._end = None
        self._records = set()

    def _filter(self, dd):
        for fk, fv in self._chks:
            if dd[fk] != fv: return False  # this flow is filtered out
        return True

    def addrec(self, rec):
        self._records.add(rec)

    def delrec(self, rec):
        self._records.discard(rec)
        
    def records(self):
        return self._records

    def is_live(self):
        return True


class RawQuery(LiveQuery):
    def __init__(self, fields, shape):
        super(self.__class__, self).__init__(fields, shape)
        
    def collect(self, source, dd):
        if not self._filter(dd): return None
        return self._mkreply([[dd[f] for f in self._reps], [dd[f] for f in self._cnts]], None, 1)
    
    def results(self, now):
        return None
    
class PeriodicQuery(LiveQuery):
    """ Aggregate last few seconds of live traffic. Ideally this should be one special case of HistoryQuery
        executed periodically on few resent seconds of history. 
        Currently it's implemented as special case of RawQuery.
        Result: counts vs. flows, no time dimension after aggregation.
    """
    def __init__(self, fields, shape, period):
        super(self.__class__, self).__init__(fields, shape)

    def collect(self, source, dd):
        if not self._filter(dd): return None
        self._collect(dd)
        return None
    
    def results(self, now):
        if not self._end:
            self._end = now+self._period
            return None
        if self._end > now: return None
        self._end = now+self._period
        return self._results()

class HistoryQuery(Query):
    def __init__(self, fields, shape, newest, oldest):
        super(self.__class__, self).__init__(fields, shape)
        self._newest = newest
        self._oldest = oldest

    def _subsource(self, sources):
        for fk, fv in self._chks:
            if fk == Source.srcaddress:  # query has specific preference for source
                # find a source this query needs
                srcs = []
                for s in sources:
                    if s.address == fv:
                        srcs.append(s)
                sources = srcs
                break
        # set or subset of sources could also be empty here
        return sources

    def _mkcaller(self):
        pos = 0
        checker = []
        keyer = []
        for ft in Source.flowtuple:
            fv = self._chks.get(ft, None)
            if fv: checker.append((pos, fv))

            fv = self._reps.get(ft, None)
            if fv: keyer.append(pos)            
            pos += 1
        onekey = tuple()
        if checker and keyer: 
            def keycall(ftuple, stamp):
                # key per flow tuple
                for p, v in checker: 
                    if ftuple[p] != v: return None  # filter out this flow
                return tuple([ftuple[p] for p in keyer])
        elif not checker and keyer:
            def keycall(ftuple, stamp):
                # key per flow tuple
                return tuple([ftuple[p] for p in keyer])
        elif checker and not keyer:
            def keycall(ftuple, stamp):
                # key per flow tuple
                for p, v in checker: 
                    if ftuple[p] != v: return None  # filter out this flow
                return onekey
        else:
            def keycall(ftuple, stamp):
                # key per flow tuple
                return onekey
        return keycall

class FlowQuery(HistoryQuery):
    """ Aggregate historic data over time.
        Result: counts vs. flows, no time dimension after aggregation. 
    """

    def __init__(self, fields, shape, newest, oldest):
        super(self.__class__, self).__init__(fields, shape, newest, oldest)
        
    def collect_sources(self, sources):
        """need to take care of all `special` fields: srcaddress, flowtuple
           then handle remaining flow attributes"""
        sources = self._subsource(sources)
        keycall = self._mkcaller()
        # check if source address needs to be reported in each flow record
        collection = []
        try:
            pos = self._reps.index(Source.srcaddress)
            # source address has to be included into keys  
            for s in sources:
                coll = {}
                s.history(coll, self._newest, self._oldest, keycall)
                collection.extend(self._reshape(coll.items(), (pos, s.address)))
        except:
            for s in sources:
                coll = {}
                s.history(coll, self._newest, self._oldest, keycall)
                collection.extend(self._reshape(coll.items()))
        return self._mkreply(self._reshape(collection), tots, len(flows))

class TemporalQuery(HistoryQuery):
    """ Aggregate historic data over flows.
        Result: counts vs. time steps, no flow dimension after aggregation. 
    """
    def __init__(self, fields, newest, oldest, step):
        """ need to ignore all 'report' (i.e. '*') fields since it will be aggregated over flows anyways """
        super(self.__class__, self).__init__(fields, None, newest, oldest)
        self._reps = [] # nullify reportable fields
        
    def collect_sources(self, sources):
        sources = self.__subsource(sources)

        keycall = self._mkcaller()
        
        collection = {}
        for s in sources:
            s.history(collection, self._newest, self._oldest, keycall)
