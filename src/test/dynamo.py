'''
Created on Feb 5, 2014

@author: schernikov
'''

import socket, urlparse, cPickle, datetime

import native.dynamo, native.receiver 

def main():
    fname = '/media/store/workspace/calix/captures/flows.picle'
    #capture(fname, 'udp://192.168.1.82:2059')
    #capture(fname, 'udp://10.215.1.6:2059')
    playback(fname)
    
class Srcs(object):
    def __init__(self):
        self._srcs = {}
        
    def find(self, ip):
        src = self._srcs.get(ip, None)
        if src is None:
            src = native.receiver.Sources(ip)
            self._srcs[ip] = src

        return src.getcollectors()
    
    def on_time(self, tm):
        for src in self._srcs.values():
            src.on_time(tm)
    
def playback(fname):
    #fields = {'8': ['1.2.3.4/24', '1.2.4.6', '*'], '12':'*'}
    fields = {'130': ['198.154.124.14', '*'], '12':'*'}
    #fields = {'130': ['1.2.3.4/24', '1.2.4.6']}

    with open(fname) as f:
        collect = cPickle.load(f)
        
    srcs = Srcs()
        
    rq = native.dynamo.genper(fields)
    qbuf = native.dynamo.qmod.QueryBuffer()
    
    receiver = native.receiver.recmod.Receiver(srcs)

    for d in sorted(collect.keys()):
        buf = collect[d]
        receiver.receive(buf, len(buf))
    
    srcs.on_time(1)
    
    secset = []
    for ip, s in srcs._srcs.items():
        if rq.matchsource(ip) == 0: continue
        _, _, secs = s.getcollectors()
        secset.append(secs)

    rq.runseconds(qbuf, secset, 1)

    #rq.testflow(0x01020406)

def capture(fname, addr):
    p = urlparse.urlsplit(addr)
    if not p.scheme or p.scheme.lower() != 'udp':
        raise Exception("Only udp scheme is supported for flow reception. Got '%s'"%(addr))
    if not p.port:
        raise Exception("Please provide port to receive flows on. Got '%s'"%(addr))
            
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((p.hostname, p.port))

    collect = {}
    while True:
        try:    
            data, addr = sock.recvfrom(2048); addr
            collect[datetime.datetime.now()] = data
            print len(data)
        except KeyboardInterrupt:
            break

    with open(fname, 'wb') as f:
        cPickle.dump(collect, f)
    
if __name__ == '__main__':
    main()