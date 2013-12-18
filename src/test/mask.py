'''
Created on Dec 18, 2013

@author: schernikov
'''
import pprint
import flowcon.query

def main():
    res = flowcon.query.ipvariations('192.168.1.150/31')
    if res is None:
        print 'exact' 
        return
    if not res:
        print 'any'
        return
    pprint.pprint(sorted(res))
    print "len:",len(res)

if __name__ == '__main__':
    main()