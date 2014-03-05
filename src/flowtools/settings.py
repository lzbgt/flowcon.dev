'''
Created on Feb 12, 2014

@author: schernikov
'''

count      = 1000   # approximately how many time steps to return by default
minunits   = 2      # if more than this many higher units then switch to higher units 
                    # Ex: if step > 2 minutes then switch from seconds to minutes  
maxseconds = 3600   # one hour
maxminutes = 1440   # one day
maxhours   = 720    # one month
maxdays    = 365    # one year

portrate   = 2.0    # ratio between two flow ports (src and dst) to consider one as application
minthreshold = 100  # don't consider as an application unless port count over this threshold
checkminutes = 5     # period in minutes to run app activity checker
activerate = 0.1    # share of seconds with given app seen in checkminutes to consider this app as active  