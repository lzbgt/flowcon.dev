'''
Created on Mar 17, 2014

@author: schernikov
'''

import argparse, zmq, pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interface', help='interface to connect to (ex. "tcp://host:1234")', required=True)
    
    args = parser.parse_args()
    
    process(args.interface)

def process(addr):
    context = zmq.Context()
    print "Connecting to server..."
    socket = context.socket(zmq.DEALER)
    socket.connect (addr)
    
    msg = zmq.utils.jsonapi.dumps({'backup':''})
    socket.send (msg)

    message = socket.recv()
    resp = zmq.utils.jsonapi.loads(message)

    pprint.pprint(resp)

if __name__ == '__main__':
    main()