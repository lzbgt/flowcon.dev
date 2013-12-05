'''
Created on Dec 3, 2013

@author: schernikov
'''

import struct, zmq.utils.monitor
from zmq.eventloop import ioloop, zmqstream

ioloop.install()

class Connection(object):
    heartbeat = 10 # seconds
    hbmessage = 'hb'
    
    def send(self, msg):
        self._sock.send(msg)
        
    def send_multipart(self, msg):
        self._sock.send_multipart(msg)

    def on_open(self, sid):
        pass # do nothing
    
    def on_msg(self, msg):
        pass # do nothing
    
    def on_close(self, sid):
        pass # do nothing

class Connector(object):

    def __init__(self):
        self._ctx = zmq.Context()

    def _setup(self, sock, addr, con):
        self._con = con
        con._sock = sock
        self._sock = sock

        stream = zmqstream.ZMQStream(sock)
        stream.on_recv(self._on_msg)

        mon = self._mkmon(sock)
        mon_stream = zmqstream.ZMQStream(mon)
        mon_stream.on_recv(self._on_mon)
        
        loop = ioloop.IOLoop.instance()
        loop.start()

    def connect(self, addr, con):
        sock = self._ctx.socket(zmq.DEALER)
        sock.connect(addr)
        self._setup(sock, addr, con)

    def listen(self, addr, con):
        sock = self._ctx.socket(zmq.ROUTER)
        sock.bind (addr)
        self._setup(sock, addr, con)

    def subscribe(self, addr, subname, callback):
        sockin = self._ctx.socket(zmq.SUB)
        sockin.connect(addr)
        sockin.setsockopt(zmq.SUBSCRIBE, subname)
        
        stream_sub = zmqstream.ZMQStream(sockin)
        stream_sub.on_recv(callback)
        
    def timer(self, milliseconds, on_time):
        loop = ioloop.IOLoop.instance()
        timer = ioloop.PeriodicCallback(on_time, milliseconds, loop)
        timer.start()
        
    def publish(self, addr):
        socket_pub = self._ctx.socket(zmq.PUB)
        socket_pub.bind (addr)
        return zmqstream.ZMQStream(socket_pub)

    def _on_msg(self, msg):
        self._con.on_msg(msg)

    def _mkmon(self, sock):
        return sock.get_monitor_socket(zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED)

    def _on_mon(self, msg):
        ev = zmq.utils.monitor.parse_monitor_message(msg)
        event = ev['event']
        endpoint = ev['endpoint']
        if event == zmq.EVENT_CONNECTED:
            self._con.on_open(endpoint)
        elif event == zmq.EVENT_DISCONNECTED:
            self._con.on_close(endpoint)

    def close(self):
        self._ctx.destroy()
