'''
Created on Dec 4, 2013

@author: schernikov
'''
import dateutil.tz, datetime, dateutil.parser

import logger

tzutc = dateutil.tz.tzutc()

def tostamp(stamp):
    return str(stamp.replace(tzinfo=tzutc))

def _cleanuptimefields(fields):
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
        super(Source, self).__init__(ff)
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
        
    def history(self, collecton, newest, oldest, keycall, timekey=lambda k: k):
        """Collect history of flows for seconds, minutes, hours or days.
           Need to make sure duplicate entries are not counted. 
           Ex. older seconds and minutes aggregated from these seconds should be either one or another, not both.    
        """
        self._history(collecton, self._seconds, newest, oldest, keycall, timekey)
        
    def _history(self, collection, group, newest, oldest, keycall, timekey):
        counts = range(len(self._cnts))
        if newest or oldest:
            if not newest:
                for stamp, flows in group:
                    if stamp <= oldest: continue     # not reached yet
                    self._hiscall(collection, timekey(stamp), flows, keycall, counts)
                return
            if not oldest:
                for stamp, flows in group:
                    if stamp > newest: break        # time scope is over
                    self._hiscall(collection, timekey(stamp), flows, keycall, counts)
                return
            for stamp, flows in group:
                if stamp > newest: break        # time scope is over
                if stamp <= oldest: continue     # not reached yet
                self._hiscall(collection, timekey(stamp), flows, keycall, counts)
            return
        for stamp, flows in group:
            self._hiscall(collection, timekey(stamp), flows, keycall, counts) 

    def _hiscall(self, collection, stamp, flows, keycall, counts):
        for key, flow in flows.items():
            k = keycall(key, stamp)
            if k is None: continue # ignore that flow
            entry = collection.get(k, None)
            if entry is None:
                entry = [0 for _ in self._cnts]
                collection[k] = entry
            for p in counts:
                val = flow[p]
                entry[p] += val
        
    def address(self):
        return self._addr

def stamp2time(stamp):
    if stamp is None: return None
    now = datetime.datetime.utcnow()
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
        raise Exception("don't know what to do with stamp %s: %s"%(stamp, str(e)))
    return tm

def _intfield(var):
    try:
        return int(var)
    except:
        pass
    return None

def _timefield(var):
    return        
        
class Query(Collector):
    @classmethod
    def create(cls, qry):
        flow = qry.get('flows', None)
        if flow is None: raise Exception("missing expected 'flow' attribute")
        fields = flow.get('fields', None)
        if fields is None: raise Exception("missing expected 'flow.fields' attribute")
        shape = flow.get('shape', None)

        time = qry.get('time', None)
        if time is None: return RawQuery(fields, shape)

        # remove time stamp fields
        _cleanuptimefields(fields)
        per = _intfield(time)
        if per is not None:
            return PeriodicQuery(fields, shape, max(per, 1))

        mode = time.get('mode', None)
        if mode is None: raise Exception("missing 'time.mode' attribute")
        if mode == 'periodic':
            per = _intfield(time.get('seconds', None))
            if per is None: raise Exception("missing valid 'time.seconds' for 'time.periodic' mode")
            return PeriodicQuery(fields, shape, max(per, 1))
        elif mode == 'collect':
            oldest = stamp2time(time.get('oldest', None))
            newest = stamp2time(time.get('newest', None))
            return FlowQuery(fields, shape, newest, oldest)
        elif mode == 'range':
            oldest = stamp2time(time.get('oldest', None))
            newest = stamp2time(time.get('newest', None))
            return RangeQuery(fields, newest, oldest)

        raise Exception("don't know what to do with 'time.mode==%s'"%(mode))

    def __init__(self, fields, shape):
        super(Query, self).__init__(fields)
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

    def _reshape(self, res, insert=None, totals=True):
        """Totals collection may be quite time consuming"""
        tots = [0 for _ in self._cnts]
        if totals:
            counts = range(len(self._cnts))
            for k, v in res:
                for p in counts:
                    tots[p] += v[p]
        num = len(res)

        if not self._shape: return res
        reverse, pos, count = self._shape
        srtd = sorted(res, key=lambda l: l[1][pos], reverse=reverse)
        if count > 0: srtd = srtd[:count]
        if insert is None: return srtd, tots, num
        # need to plug value into keys
        pos, val = insert
        pos += 1
        res = []
        for k, v in srtd:
            l = list(k)
            l.insert(pos, val)
            res.append([l, v])
        return res, tots, num

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
        res, _, num = self._reshape(flows.items(), totals=False)
        return self._mkreply(res, tots, num)
    
class LiveQuery(Query):
    def __init__(self, fields, shape):
        super(LiveQuery, self).__init__(fields, shape)
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
        super(RawQuery, self).__init__(fields, shape)
        
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
        super(PeriodicQuery, self).__init__(fields, shape)
        self._period = period

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
        super(HistoryQuery, self).__init__(fields, shape)
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
    def _sel(self):
        pos = 0
        checker = []
        keyer = []
        for ft in Source.flowtuple:
            for f, v in self._chks:
                if f == ft: checker.append((pos, v))
            for f in self._reps:
                if f == ft: keyer.append(pos)            
            pos += 1
        return checker, keyer

class FlowQuery(HistoryQuery):
    """ Aggregate historic data over time.
        Result: counts vs. flows, no time dimension after aggregation. 
    """

    def __init__(self, fields, shape, newest, oldest):
        super(FlowQuery, self).__init__(fields, shape, newest, oldest)

    def _mkcaller(self):
        """ Need to redesign to consider shaper
            This will allow to reduce collected dictionary size from very beginning 
            according to shaping method and required count. 
            But how do we account for total number of flows statistics then?
        """
        checker, keyer = self._sel()
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
        
    def collect_sources(self, sources):
        """need to take care of all `special` fields: srcaddress, flowtuple
           then handle remaining flow attributes"""
        sources = self._subsource(sources)
        keycall = self._mkcaller()
        # check if source address needs to be reported in each flow record
        collection = []
        totals = [0 for _ in self._cnts]
        numbers = 0
        counts = range(len(self._cnts))
        try:
            pos = self._reps.index(Source.srcaddress)
        except:
            pos = None
        if pos is not None:
            # source address has to be included into keys
            for s in sources:
                coll = {}
                s.history(coll, self._newest, self._oldest, keycall)
                
                res, tot, num = self._reshape(coll.items(), (pos, s.address))
                for p in counts: totals[p] += tot[p]
                numbers += num  # all keys are unique between sources, so combined numbers are correct
                 
                collection.extend(res)
            res, _, _ = self._reshape(collection, totals=False)
            return self._mkreply(res, totals, numbers)

        coll = {}
        for s in sources:
            s.history(coll, self._newest, self._oldest, keycall)

        res, tot, num = self._reshape(coll.items()) 
        return self._mkreply(res, tot, num)

class RangeQuery(HistoryQuery):
    """ Aggregate historic data over flows.
        Result: counts vs. time steps, no flow dimension after aggregation. 
    """
    def __init__(self, fields, newest, oldest):
        """ need to ignore all 'report' (i.e. '*') fields since it will be aggregated over flows anyways """
        super(RangeQuery, self).__init__(fields, None, newest, oldest)
        self._reps = [] # nullify reportable fields
        #TODO implement actual granule detection
        self._gradules = datetime.time.second

    def _mkcaller(self):
        checker, _ = self._sel()
        if checker:
            def keycall(ftuple, skey):
                # key per flow tuple
                for p, v in checker: 
                    if ftuple[p] != v: return None  # filter out this flow
                return skey
        else:
            def keycall(ftuple, skey):
                # key per flow tuple
                return skey

        return keycall

    def _mkkeyer(self):
        if self._gradules == datetime.time.second:
            def keyer(s):
                return datetime.datetime.combine(s.date(), datetime.time(s.hour, s.minute, s.second))
        elif self._gradules == datetime.time.minute:
            def keyer(s):
                return datetime.datetime.combine(s.date(), datetime.time(s.hour, s.minute))
        elif self._gradules == datetime.time.hour:
            def keyer(s):
                return datetime.datetime.combine(s.date(), datetime.time(s.hour))
        else:
            return None
        return keyer

    def collect_sources(self, sources):
        sources = self._subsource(sources)

        keycall = self._mkcaller()
        
        totals = [0 for _ in self._cnts]
        counts = range(len(self._cnts))

        timekey = self._mkkeyer()
        if not timekey: return self._mkreply([], totals, 0)

        collection = {}
        for s in sources:
            s.history(collection, self._newest, self._oldest, keycall, timekey)
            
        stamps = sorted(collection)
        res = []
        for s in stamps:
            v = collection[s]
            res.append([tostamp(s), v])
            for p in counts: totals[p] += v[p]

        return self._mkreply(res, totals, len(collection))
