'''
Created on Jan 15, 2014

@author: schernikov
'''

import socket, os, sys

libloc = os.path.join(os.path.dirname(__file__), '..', '..', 'cython')
sys.path.append(libloc)
recmod = __import__('receiver')

def main():
    myaddr = "10.215.1.6"
    myport = 2059
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((myaddr, myport))
    receiver = recmod.Receiver()
    while True:
        data, addr = sock.recvfrom(2048); addr
        receiver.receive(data, len(data))

if __name__ == '__main__':
    main()