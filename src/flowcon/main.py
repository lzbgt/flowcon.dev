'''
Created on Nov 25, 2013

@author: schernikov
'''

import sys, argparse, os

import processor, flowtools.logger as logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='publisher interface to subscribe to', required=True)
    parser.add_argument('-s', '--server', type=str, help='server interface to serve stats requests', required=True)
    parser.add_argument('-q', '--query', type=str, help='listening interface for queries', required=True)
    parser.add_argument('-d', '--dumpfile', type=str, help='output file (defaults: stdout)', default=None)
    parser.add_argument('-b', '--backup', type=str, help='backup folder', default=None)
    
    args = parser.parse_args()
    if args.dumpfile is None:
        out = sys.stdout
    else:
        out = open(args.dumpfile)
    logger.setout(out)

    if args.backup and not os.path.isdir(args.backup):
        logger.dump("backup folder does not exist: %s "%(args.backup))
        return
    try:
        processor.setup(args.input, args.server, args.query, args.backup)
    finally:
        out.close()

if __name__ == '__main__':
    main()