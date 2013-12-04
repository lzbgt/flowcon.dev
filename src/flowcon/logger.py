'''
Created on Dec 4, 2013

@author: schernikov
'''

output = None

def setout(out):
    global output
    output = out
    
def dump(txt):
    if not output: return
    print >>output, txt
    output.flush()