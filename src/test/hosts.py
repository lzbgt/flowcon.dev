'''
Created on Nov 26, 2013

@author: schernikov
'''

import argparse, zmq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='publisher interface to subscribe to', required=True)
    
    args = parser.parse_args()
    
    process(args.input)

def process(insock):
    ctx = zmq.Context()

    sockin = ctx.socket(zmq.SUB)
    sockin.connect(insock)
    sockin.setsockopt(zmq.SUBSCRIBE, 'hosts')

    while True:    
        msg = sockin.recv_multipart()
        print msg[1]


if __name__ == '__main__':
    main()