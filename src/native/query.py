'''
Created on Jan 28, 2014

@author: schernikov
'''

import datetime, dateutil.tz

import native.types as ntypes, native.dynamo, calendar
import flowtools.logger as logger 

tzutc = dateutil.tz.tzutc()

class Query(object):
    # counter fields
    bytestp = ntypes.IntType('bytes', '1', 4)
    packetstp = ntypes.IntType('packets', '2', 4)
    
    # flow fields
    protocol = ntypes.IntType('protocol', '4', 1)
    srcaddr = ntypes.IPType('srcaddr', '8', 4)
    srcport = ntypes.IntType('srcport', '7', 2)
    dstport = ntypes.IntType('dstport', '11', 2)
    dstaddr = ntypes.IPType('dstaddr', '12', 4)

    # special fields
    endstamp = ntypes.TimeType('last', '21', 4)
    startstamp = ntypes.TimeType('first', '22', 4)
    srcaddress = ntypes.IPType('exporter', '130', 4)

    # attribute fields    
    nexthop = ntypes.IPType('nexthop', '15', 4)
    ingressport = ntypes.IntType('inpsnmp', '10', 2)
    egressport = ntypes.IntType('outsnmp', '14', 2)
    tcpflags = ntypes.IntType('tcpflags', '6', 4)
    tosbyte = ntypes.IntType('tos', '5', 1)
    srcas = ntypes.IntType('srcas', '16', 4)
    dstas = ntypes.IntType('dstas', '17', 4)
    srcmask = ntypes.IntType('srcmask', '9', 1)
    dstmask = ntypes.IntType('dstmask', '13', 1)

    flowtuple = (protocol, srcport, srcaddr, dstport, dstaddr)
    specialtuple = (bytestp, packetstp, srcaddress, startstamp, endstamp)
    counters = (bytestp, packetstp)

    @classmethod
    def create(cls, qry):
        flow = qry.get('flows', None)
        if flow is None: raise Exception("missing expected 'flow' attribute")
        fields = flow.get('fields', None)
        if fields is None: raise Exception("missing expected 'flow.fields' attribute")
        shape = flow.get('shape', None)

        time = qry.get('time', None)
        if time is None: return RawQuery(qry, fields)

        # remove time stamp fields
        _cleanuptimefields(fields)
        per = _intfield(time)
        if per is not None:
            return PeriodicQuery(qry, fields, shape, max(per, 1))

        mode = time.get('mode', None)
        if mode is None: raise Exception("missing 'time.mode' attribute")
        if mode == 'periodic':
            per = _intfield(time.get('seconds', None))
            if per is None: raise Exception("missing valid 'time.seconds' for 'time.periodic' mode")
            return PeriodicQuery(qry, fields, shape, max(per, 1))
        elif mode == 'flows':
            oldest = stamp2time(time.get('oldest', None))
            newest = stamp2time(time.get('newest', None))
            return FlowQuery(qry, fields, shape, newest, oldest)
        elif mode == 'time':
            oldest = stamp2time(time.get('oldest', None))
            newest = stamp2time(time.get('newest', None))
            return RangeQuery(qry, fields, newest, oldest)

        raise Exception("don't know what to do with 'time.mode==%s'"%(mode))

    def __init__(self, qry, fields):
        self._qry = qry
        for cnt in self.counters:
            if cnt.id in fields: del fields[cnt.id]
#        self._chks, self._reps = _fieldlist(self.counters, fields)

    def _assigntime(self, newest, oldest):
        if not newest:
            now = datetime.datetime.utcnow()
            self._newest = mkstamp(now)
        else:
            self._newest = mkstamp(newest)+1 # include newest second into consideration
        if not oldest:
            self._oldest = 0
        else:
            self._oldest = mkstamp(oldest)

    def value(self):
        return self._qry
    
    @property
    def id(self):
        return self._native.id()

class QueryBuffer(object):
    def __init__(self):
        self._native = native.dynamo.qmod.QueryBuffer()
        
class RawQuery(Query):
    def __init__(self, qry, fields):
        super(RawQuery, self).__init__(qry, fields)
        self._native = native.dynamo.genraw(fields)
        
    @property
    def native(self):
        return self._native
    
    def is_live(self):
        return True

def mkstamp(d):
    return int(calendar.timegm(d.timetuple()))


class FQuery(Query):
    def __init__(self, qry, fields, shape):
        super(FQuery, self).__init__(qry, fields)
        self._native = native.dynamo.genflow(fields)

        self._shape = self._on_shape(shape)
        
    def _on_shape(self, shape):
        if not shape: return None
        field = shape.get('max', None)
        if field is None:
            field = shape.get('min', None)
            if field is None:
                logger.dump("no shape function defined")
                return (None, None, 0)
            direction = 'min'
        else:
            direction = 'max'

        if self.bytestp.name != field and self.packetstp.name != field:
            logger.dump("unexpected sort field '%s'"%field)
            return (None, None, 0)

        count = shape.get('count', 0)
        try:
            count = int(count)
        except:
            logger.dump("unexpected count field '%s'"%(count))
            return (None, None, 0)

        if count < 0:
            logger.dump("negative count field '%d'"%(count))
            return (None, None, 0)
        
        return field, direction, count

class PeriodicQuery(FQuery):
    vicinity = 0.1 # seconds
    
    def __init__(self, qry, fields, shape, period):
        super(PeriodicQuery, self).__init__(qry, fields, shape)

        self._period = datetime.timedelta(seconds=(period-self.vicinity))
        now = datetime.datetime.utcnow()
        self._next = now+self._period
        self._prevstamp = mkstamp(now) 
        
        self._sources = set()
        self._seconds = []

    def addsource(self, src):
        if not self._native.matchsource(src.ip): return
        self._sources.add(src)
        _, _, sec = src.getcollectors()
        self._seconds.append(sec)

    def is_live(self):
        return True

    def on_time(self, qbuf, now, stamp):
        if self._next > now: return None
        self._next = now + self._period
        
        self._native.runseconds(qbuf._native, self._seconds, stamp, self._prevstamp)
        
        self._prevstamp = stamp
        
        return self._native.report(qbuf._native, *self._shape)

class FlowQuery(FQuery):
    
    def __init__(self, qry, fields, shape, newest, oldest):
        super(FlowQuery, self).__init__(qry, fields, shape)
        self._assigntime(newest, oldest)
    
    def is_live(self):
        return False
    
    def collect(self, qbuf, sources):
        seconds = []
        
        for src in sources:
            _, _, sec = src.getcollectors()
            seconds.append(sec)

        self._native.runseconds(qbuf._native, seconds, self._newest, self._oldest)
        
        return self._native.report(qbuf._native, *self._shape)

class RangeQuery(Query):

    def __init__(self, qry, fields, newest, oldest):
        super(RangeQuery, self).__init__(qry, fields)

        self._native = native.dynamo.gentime(fields)
                
        self._assigntime(newest, oldest)

    def is_live(self):
        return False    

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

def tostamp(stamp):
    return str(stamp.replace(tzinfo=tzutc))

def _cleanuptimefields(fields):
    for ftype in ntypes.TimeType.timetypes.values():
        if ftype.id in fields:
            logger.dump("ignoring `%s`[%s] field in query"%(ftype.name, ftype.id))
            del fields[ftype.id]

def _intfield(var):
    try:
        return int(var)
    except:
        pass
    return None
