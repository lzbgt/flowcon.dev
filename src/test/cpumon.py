'''
Created on Mar 5, 2014

@author: schernikov
'''

import argparse, psutil, time, datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--process', help='process id to monitor', required=True, type=int)

    args = parser.parse_args()

    monitor(args.process)
    
def monitor(pid):
    if not psutil.pid_exists(pid):
        print "pid %d does not exist"%(pid)
        return
    p = psutil.Process(pid)
    p.get_cpu_percent(interval=0)
    while True:
        time.sleep(1)
        cpu = p.get_cpu_percent(interval=0)
        d = datetime.datetime.now()
        print '%5.1f'%cpu,
        if d.second == 0:
            print d,
        print

if __name__ == '__main__':
    main()