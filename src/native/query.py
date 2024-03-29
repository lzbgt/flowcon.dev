'''
Created on Jan 28, 2014

@author: schernikov
'''

import datetime, dateutil.tz, dateutil.parser, calendar

import native.types as ntypes, native.dynamo
import flowtools.logger as logger, flowtools.settings

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
    def create(cls, qry, history):
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
        
        now = history.now()        

        mode = time.get('mode', None)
        if mode is None: raise Exception("missing 'time.mode' attribute")
        if mode == 'periodic':
            per = _intfield(time.get('seconds', None))
            if per is None: raise Exception("missing valid 'time.seconds' for 'time.periodic' mode")
            return PeriodicQuery(qry, fields, shape, max(per, 1))
        elif mode == 'flows':
            oldest = stamp2time(now, time.get('oldest', None))
            newest = stamp2time(now, time.get('newest', None))
            return FlowQuery(qry, fields, shape, newest, oldest, history)
        elif mode == 'time':
            oldest = stamp2time(now, time.get('oldest', None))
            newest = stamp2time(now, time.get('newest', None))
            step = _intfield(time.get('step', None))
            return RangeQuery(qry, fields, newest, oldest, step, history)

        raise Exception("don't know what to do with 'time.mode==%s'"%(mode))

    def __init__(self, qry, fields):
        self._qry = qry
        for cnt in self.counters:
            if cnt.id in fields: del fields[cnt.id]
#        self._chks, self._reps = _fieldlist(self.counters, fields)

    def _assigntime(self, nowstamp, newest, oldest, oldesthistory):
        if not newest:
            self._newest = nowstamp
        else:
            if newest <= oldesthistory:
                raise Exception("Bad time specs. Newest '%s' should be newer than oldest available '%s'"%(tostamp(newest),
                                                                                                          tostamp(oldesthistory)))

            stamp = mkstamp(newest)
            if stamp >= nowstamp:
                self._newest = nowstamp
            else:
                self._newest = stamp+1 # include requested second into consideration

        oldeststamp = mkstamp(oldesthistory)
        if not oldest or oldest < oldesthistory:
            self._oldest = oldeststamp
        else:
            self._oldest = mkstamp(oldest)
            
        if self._newest <= self._oldest:
            raise Exception('Bad time specs. Oldest (%d) should be less than newest (%d).'%(self._oldest, self._newest))

    def _mkschedule(self, step, history, nowstamp):
        schedule = {'step':step}
        # lets figure out if we need seconds at all
        minutes = history.minutes()
        newest = _mkunit(schedule, self._newest, self._oldest, step, history.seconds(), minutes)
        if newest <= self._oldest: return schedule
        # then minutes
        hours = history.hours()
        newest = _mkunit(schedule, newest, self._oldest, step, minutes, hours)
        if newest <= self._oldest: return schedule
        # then hours
        days = history.days()
        newest = _mkunit(schedule, newest, self._oldest, step, hours, days)
        if newest <= self._oldest: return schedule

        # everything else falls into days
        doldest = max(days.first, days.fromstamp(self._oldest))
        if newest > doldest:
            schedule[days.name()] = (newest, doldest)

        return schedule
    
    def showschedule(self, nm):
        msg = '%s schedule:\n'%(nm)
        for k, v in self._schedule.items():
            try:
                iter(v) 
                msg += "  %s: %s --> %s\n"%(k, tostamp(datetime.datetime.utcfromtimestamp(v[0])), 
                                               tostamp(datetime.datetime.utcfromtimestamp(v[1])))
            except:
                msg += "  %s: %s\n"%(k, str(v))

        logger.dump(msg)
    
    def collect(self, qbuf, sources):
        self._native.initbuf(qbuf._native)
        step = self._schedule.get('step', None)
        if step is None: return ''

        dayset = self._schedule.get('days', None)
        if dayset:
            days = []
            for src in sources:
                days.append(src.getdays())
            newest, oldest = dayset
            self._native.rundays(qbuf._native, days, newest, oldest, step)
        
        hourset = self._schedule.get('hours', None)
        if hourset:
            hours = []
            for src in sources:
                hours.append(src.gethours())
            newest, oldest = hourset
            self._native.runhours(qbuf._native, hours, newest, oldest, step)
        
        minset = self._schedule.get('minutes', None)
        if minset:
            minutes = []
            for src in sources:
                minutes.append(src.getminutes())
            newest, oldest = minset
            self._native.runminutes(qbuf._native, minutes, newest, oldest, step)

        secset = self._schedule.get('seconds', None)
        if secset:
            seconds = []
            for src in sources:
                seconds.append(src.getseconds())
            newest, oldest = secset
            self._native.runseconds(qbuf._native, seconds, newest, oldest, step)
        
        return self._native.report(qbuf._native, *self._shape)

    def value(self):
        return self._qry
    
    @property
    def id(self):
        return self._native.id()

class QueryBuffer(object):
    def __init__(self):
        self._native = native.dynamo.qmod.QueryBuffer()
        
    def status(self):
        return self._native.status()
        
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
        if not shape: return (None, None, 0)
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
        self._seconds.append(src.getseconds())

    def is_live(self):
        return True

    def on_time(self, qbuf, now, stamp):
        if self._next > now: return None
        self._next = now + self._period
        
        self._native.initbuf(qbuf._native)
        
        #TODO: run more than just over seconds? 
        self._native.runseconds(qbuf._native, self._seconds, stamp, self._prevstamp, 0)
        
        self._prevstamp = stamp
        
        return self._native.report(qbuf._native, *self._shape)

class FlowQuery(FQuery):
    
    def __init__(self, qry, fields, shape, newest, oldest, history):
        super(FlowQuery, self).__init__(qry, fields, shape)
        
        nowstamp = history.seconds().now
        
        self._assigntime(nowstamp, newest, oldest, history.oldest())
        self._schedule = self._mkschedule(0, history, nowstamp)

        self.showschedule('flow')
        
    def is_live(self):
        return False
    
def stepsize(interval, units):
    return max(int(round(1.0*interval/units/flowtools.settings.count)), 1)*units

class RangeQuery(Query):

    def __init__(self, qry, fields, nwst, odst, step, history):
        super(RangeQuery, self).__init__(qry, fields)

        self._native = native.dynamo.gentime(fields)
                
        nowstamp = history.seconds().now
        print "now: %s newest:%s oldest:%s"%(str(nowstamp), str(nwst), str(odst))
        self._assigntime(nowstamp, nwst, odst, history.oldest())
        
        if step is None:
            secdiff = self._newest - self._oldest
            if secdiff <= flowtools.settings.maxseconds:
                if secdiff <= 0:
                    step = history.onesecond
                else:
                    step = stepsize(secdiff, history.onesecond)
            elif secdiff <= flowtools.settings.maxminutes*history.oneminute:
                step = stepsize(secdiff, history.oneminute)
            elif secdiff <= flowtools.settings.maxhours*history.onehour:
                step = stepsize(secdiff, history.onehour)
            else:
                step = history.oneday

        self._schedule = self._mkschedule(step, history, nowstamp)

        self.showschedule('range')
        
        self._shape = (None, None, 0)

    def is_live(self):
        return False
    
def stamp2time(now, stamp):
    if stamp is None: return None

    try:
        # check if it's relative time stamp in seconds back into history
        then = float(stamp)
        if then <= 0:
            raise Exception("can handle only positive number of seconds for relative period in seconds, not %s"%(stamp))
        print "returning now:%s - %f"%(now, then)
        return now-datetime.timedelta(seconds=then)
    except:
        pass
    try:
        tm = dateutil.parser.parse(stamp)
        if tm.tzinfo:           # strip timezone info and convert into utc
            tm = datetime.datetime.combine(tm.date(), tm.time())-tm.utcoffset()
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

def _mkunit(schedule, newest, oldest, step, unit, nextunit):
    nm = unit.name()
    
    # this unit is completely out of scope    
    if newest <= unit.oldest: return newest
    
    nextminwidth = flowtools.settings.minunits*nextunit.one
    nextnewest = nextunit.fromstamp(newest)
    unewest = unit.fromstamp(newest)
    uoldest = unit.fromstamp(oldest)
    
    if step < nextminwidth:                                   # step is small enough
        nextoldest = max(nextunit.oldest, nextunit.first, oldest)
        if (step <= 0 or (step%nextunit.one) != 0 or          # nor matches exactly with next unit
            (nextoldest+step*flowtools.settings.minunits) > nextnewest):   # or not enough units are requested from next
            # seconds are needed for sure
            if uoldest >= unit.oldest:
                # only seconds are needed
                if unit.first > uoldest: uoldest = unit.first
                if unewest > uoldest:
                    schedule[nm] = (unewest, uoldest)
                    return uoldest

                return newest
            # step is small enough to go with seconds as far as possible
            nextnewest = nextunit.fromstamp(unit.oldest)
            if uoldest > nextnewest:            # if next one rewinds to older than my oldest
                nextnewest += nextunit.one      # skip one next step  

    # step is large enough; let's switch to minutes right away
    if unewest > nextnewest:
        # gap between newest time and newest minute is more than just couple of seconds
        if nextunit.first > nextnewest:     # next unit is not available yet
            nextnewest = nextunit.first
            if unewest > nextnewest:
                schedule[nm] = (unewest, nextnewest)
                return nextnewest
            uoldest = max(unit.oldest, uoldest, unit.first)
            if unewest > uoldest:
                schedule[nm] = (unewest, uoldest)
                return uoldest

            return newest

        schedule[nm] = (unewest, nextnewest)
        return nextnewest

    return newest
