'''
Created on Dec 3, 2013

@author: schernikov
'''

import zmq.utils.monitor
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
        self.loop = ioloop.IOLoop.instance()

    def _setup(self, sock, addr, con, method):
        self._con = con
        con._sock = sock
        self._sock = sock

        stream = zmqstream.ZMQStream(sock)
        stream.on_recv(self._on_msg)

        mon = self._mkmon(sock)
        mon_stream = zmqstream.ZMQStream(mon)
        mon_stream.on_recv(self._on_mon)

        getattr(sock, method)(addr)
        
        self.loop.start()

    def stop(self):
        self.loop.stop()

    def connect(self, addr, con):
        sock = self._ctx.socket(zmq.DEALER)
        self._setup(sock, addr, con, 'connect')

    def listen(self, addr, con):
        sock = self._ctx.socket(zmq.ROUTER)
        self._setup(sock, addr, con, 'bind')

    def subscribe(self, addr, subname, callback):
        sockin = self._ctx.socket(zmq.SUB)
        sockin.connect(addr)
        sockin.setsockopt(zmq.SUBSCRIBE, subname)
        
        stream_sub = zmqstream.ZMQStream(sockin)
        stream_sub.on_recv(callback)
        
    def timer(self, seconds, on_time):
        timer = ioloop.PeriodicCallback(on_time, seconds*1000, self.loop)
        timer.start()
        
    def publish(self, addr):
        socket_pub = self._ctx.socket(zmq.PUB)
        socket_pub.bind (addr)
        return zmqstream.ZMQStream(socket_pub)

    def _on_msg(self, msg):
        self._con.on_msg(msg)

    def _mkmon(self, sock):
        #return sock.get_monitor_socket(zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED)
        return sock.get_monitor_socket(zmq.EVENT_ALL)

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
