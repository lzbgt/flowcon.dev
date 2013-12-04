'''
Created on Nov 25, 2013

@author: schernikov
'''

import sys, argparse

import processor, logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='publisher interface to subscribe to', required=True)
    parser.add_argument('-o', '--output', type=str, help='publisher interface to provide for other subscribers', required=True)
    parser.add_argument('-q', '--query', type=str, help='listening interface for queries', required=True)
    parser.add_argument('-d', '--dumpfile', type=str, help='output file (defaults: stdout)', default=None)
    
    args = parser.parse_args()
    if args.dumpfile is None:
        out = sys.stdout
    else:
        out = open(args.dumpfile)
    logger.setout(out)

    try:
        processor.setup(args.input, args.output, args.query)
    finally:
        out.close()

if __name__ == '__main__':
    main()