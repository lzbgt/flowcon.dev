'''
Created on Dec 18, 2013

@author: schernikov
'''
#import addresses
import flows

class Source(object):
    bytesid = '1'
    packetsid = '2'

    protocol = '4'
    srcaddr = '8'
    srcport = '7'
    dstport = '11'
    dstaddr = '12'

    endstamp = '21'
    startstamp = '22'
    srcaddress = '130'

    flowtuple = (protocol, srcport, srcaddr, dstport, dstaddr)
    specialtuple = (bytesid, packetsid, srcaddress, startstamp, endstamp)

    def __init__(self, addr):
        self._addr = addr
        self._fields = None
        self._attributes = None
        #self._ipset = addresses.IPCollection()
        self._fset = flows.FlowSet()

    def fields(self):
        return self._fields
    
    def address(self):
        return self._addr

    def _update_fields(self, dd):
        fields = set(dd.keys())
        attrs = fields.difference(self.flowtuple)
        self._fields = sorted(fields)
        attrs.difference_update(self.specialtuple)
        self._attributes = sorted(attrs) # all fields except `special` ones.

    def account(self, dd):
        if self._fields is None: self._update_fields(dd)
        
        flow = self._fset.newflow()
        for f in self.flowtuple:
            flow.add(dd[f])

        for a in self._attributes:
            flow.add(dd[a])

        flow.done(dd[self.bytesid], dd[self.packetsid])
#        expref = self._ipset.add(dd[self.srcaddress])
#        srcref = self._ipset.add(dd[self.srcaddr])
#        dstref = self._ipset.add(dd[self.dstaddr])
#        
#        expref, srcref, dstref
        
    def on_time(self, now):
        self
        
    def stats(self):
        iprep = self._ipset.report()
        return {'ip':iprep, 'addr':self._addr}
        
    def history(self, collecton, newest, oldest, keycall, timekey=lambda k: k):
        self
        
class Query(object):
    def create(self):
        pass
