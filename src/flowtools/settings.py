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