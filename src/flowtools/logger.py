'''
Created on Dec 4, 2013

@author: schernikov
'''

import datetime

output = None

def setout(out):
    global output
    output = out
    
def dump(txt):
    if not output: return
    d = datetime.datetime.now()
    output.write("%s: %s\n"%(d.strftime("%Y-%m-%d %H:%M:%S"), txt))
    output.flush()