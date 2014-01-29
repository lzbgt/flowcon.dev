'''
Created on Dec 18, 2013

@author: schernikov
'''
#import addresses
import flows, numpyfy.tools.collection as collecttool

class Source(object):
    # counter fields
    bytestp = flows.IntType('bytes', '1', 4)
    packetstp = flows.IntType('packets', '2', 4)
    
    # flow fields
    protocol = flows.IntType('protocol', '4', 1)
    srcaddr = flows.IPType('srcaddr', '8', 4)
    srcport = flows.IntType('srcport', '7', 2)
    dstport = flows.IntType('dstport', '11', 2)
    dstaddr = flows.IPType('dstaddr', '12', 4)

    # special fields
    endstamp = flows.TimeType('endstamp', '21', 4)
    startstamp = flows.TimeType('startstamp', '22', 4)
    srcaddress = flows.IPType('exportaddr', '130', 4)

    # attribute fields    
    nexthop = flows.IPType('nexthop', '15', 4)
    ingressport = flows.IntType('ingressport', '10', 2)
    egressport = flows.IntType('egressport', '14', 2)
    tcpflags = flows.IntType('tcpflags', '6', 4)
    tosbyte = flows.IntType('tos', '5', 1)
    srcas = flows.IntType('srcas', '16', 4)
    dstas = flows.IntType('dstas', '17', 4)
    srcmask = flows.IntType('srcmask', '9', 1)
    dstmask = flows.IntType('dstmask', '13', 1)

    flowtuple = (protocol, srcport, srcaddr, dstport, dstaddr)
    specialtuple = (bytestp, packetstp, srcaddress, startstamp, endstamp)
    
    startsize = 16

    def __init__(self, addr):
        self._addr = addr
        self._flowids = None
        self._fields = None
        self._attrids = None
        #self._ipset = addresses.IPCollection()
        self._fset = None
        self._seconds = []
        self._attrs = None
        self._flows = None

    def _update_fields(self, dd):
        fields = set(dd.keys())
        self._flowids = tuple([f.id for f in self.flowtuple])
        attrsids = fields.difference(self._flowids)
        self._fields = sorted(fields)
        attrsids.difference_update([s.id for s in self.specialtuple])
        attrsids.intersection_update(flows.Type.all.keys())
        self._attrids = sorted(attrsids)
        attrs = [flows.Type.all[aid] for aid in self._attrids]

        types = flows.FlowTypes(self.flowtuple, attrs)
        
        self._fset = flows.FlowSet(self.address(), self.startsize if self._fset is None else self._fset.size, types)
        self._attrs = collecttool.Collector('attributes', types.atypes)
        self._flows = collecttool.Collector('flows', types.ftypes)

    def account(self, dd):
        if self._fields is None: self._update_fields(dd)
        
        flow = self._fset.newflow()
        for fid in self._flowids:
            flow.add(dd[fid])

        for aid in self._attrids:
            flow.add(dd[aid])

        flow.done(dd[self.bytestp.id], dd[self.packetstp.id])
#        expref = self._ipset.add(dd[self.srcaddress.id])
#        srcref = self._ipset.add(dd[self.srcaddr.id])
#        dstref = self._ipset.add(dd[self.dstaddr.id])
#        
#        expref, srcref, dstref
        
    def on_time(self, now):
        fset = self._fset
        if not fset: return
        self._fset = flows.FlowSet(fset.name, fset.size, fset.ftypes)
        fset # send it to history
        #print "collected %d flows"%(len(fset))
        self._attrs.add(*fset.attrs())
        self._flows.add(*fset.flows())
        
    def stats(self):
        "return vital source stats"
#        iprep = self._ipset.report()
#        return {'ip':iprep, 'addr':self._addr}
        return {'name':self.address(), 'flows':self._flows.report(), 'attributes':self._attrs.report()}
        
    def history(self, collecton, newest, oldest, keycall, timekey=lambda k: k):
        self

    def fields(self):
        return self._fields
    
    def address(self):
        return self._addr
        
