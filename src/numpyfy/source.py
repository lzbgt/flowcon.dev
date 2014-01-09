'''
Created on Dec 18, 2013

@author: schernikov
'''
#import addresses
import flows

class Source(object):
    # counter fields
    bytestp = flows.Type('bytes', '1', 4)
    packetstp = flows.Type('packets', '2', 4)
    
    # flow fields
    protocol = flows.Type('protocol', '4', 1)
    srcaddr = flows.Type('srcaddr', '8', 4)
    srcport = flows.Type('srcport', '7', 2)
    dstport = flows.Type('dstport', '11', 2)
    dstaddr = flows.Type('dstaddr', '12', 4)

    # special fields
    endstamp = flows.Type('endstamp', '21', 0)
    startstamp = flows.Type('startstamp', '22', 0)
    srcaddress = flows.Type('exportaddr', '130', 4)

    # attribute fields    
    nexthop = flows.Type('nexthop', '15', 4)
    ingressport = flows.Type('ingressport', '10', 2)
    egressport = flows.Type('egressport', '14', 2)
    tcpflags = flows.Type('tcpflags', '6', 4)
    tosbyte = flows.Type('tos', '5', 1)
    srcas = flows.Type('srcas', '16', 4)
    dstas = flows.Type('dstas', '17', 4)
    srcmask = flows.Type('srcmask', '9', 4)
    dstmask = flows.Type('dstmask', '13', 4)

    flowtuple = (protocol, srcport, srcaddr, dstport, dstaddr)
    specialtuple = (bytestp, packetstp, srcaddress, startstamp, endstamp)
    
    startsize = 256

    def __init__(self, addr):
        self._addr = addr
        self._flowids = None
        self._fields = None
        self._attrids = None
        #self._ipset = addresses.IPCollection()
        self._fset = None

    def fields(self):
        return self._fields
    
    def address(self):
        return self._addr

    def _update_fields(self, dd):
        fields = set(dd.keys())
        self._flowids = tuple([f.id for f in self.flowtuple])
        attrsids = fields.difference(self._flowids)
        self._fields = sorted(fields)
        attrsids.difference_update([s.id for s in self.specialtuple])
        attrsids.intersection_update(flows.Type.all.keys())
        self._attrids = sorted(attrsids)
        attrs = [flows.Type.all[aid] for aid in self._attrids]

        flows.FlowSet.setup(self.flowtuple, attrs)
        
        self._fset = flows.FlowSet(self.startsize if self._fset is None else self._fset.size)

    def account(self, dd):
        if self._fields is None: self._update_fields(dd)
        
        flow = self._fset.newflow()
        for fid in self._flowids:
            flow.add(dd[fid])

        for aid in self._attrids:
            flow.add(dd[aid])

        flow.done(dd[self.bytestp.id], dd[self.packettp.sid])
#        expref = self._ipset.add(dd[self.srcaddress.id])
#        srcref = self._ipset.add(dd[self.srcaddr.id])
#        dstref = self._ipset.add(dd[self.dstaddr.id])
#        
#        expref, srcref, dstref
        
    def on_time(self, now):
        fset = flows.FlowSet(self.startsize)
        self._fset = flows.FlowSet(fset.size)
        fset # send it to history
        
    def stats(self):
        iprep = self._ipset.report()
        return {'ip':iprep, 'addr':self._addr}
        
    def history(self, collecton, newest, oldest, keycall, timekey=lambda k: k):
        self
        
class Query(object):
    def create(self):
        pass
