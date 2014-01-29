'''
Created on Jan 28, 2014

@author: schernikov
'''

import datetime, dateutil.tz

import native.types as ntypes
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
        elif mode == 'flows':
            oldest = stamp2time(time.get('oldest', None))
            newest = stamp2time(time.get('newest', None))
            return FlowQuery(fields, shape, newest, oldest)
        elif mode == 'time':
            oldest = stamp2time(time.get('oldest', None))
            newest = stamp2time(time.get('newest', None))
            return RangeQuery(fields, newest, oldest)

        raise Exception("don't know what to do with 'time.mode==%s'"%(mode))

    def __init__(self, fields):
        for cnt in self.counters:
            if cnt.id in fields: del fields[cnt.id]
#        self._chks, self._reps = _fieldlist(self.counters, fields)

class RawQuery(Query):
    pass

class PeriodicQuery(Query):
    pass

class FlowQuery(Query):
    pass

class RangeQuery(Query):
    pass

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
    for ftype in ntypes.TimeType.timetypes:
        if ftype.id in fields:
            logger.dump("ignoring `%s`[%s] field in query"%(ftype.name, ftype.id))
            del fields[ftype.id]

def _intfield(var):
    try:
        return int(var)
    except:
        pass
    return None
