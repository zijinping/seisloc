#!/usr/bin/evn python
# EQTransformer related functions

import obspy
from obspy import UTCDateTime

def to_mseed(trace):
    net = trace.stats.network
    sta = trace.stats.station
    chn = trace.stats.channel
    starttime = trace.stats.starttime
    endtime = trace.stats.endtime
    out_name = net+"."+sta+"."+chn+"__"+\
                starttime.strftime("%Y%m%dT%H%M%SZ")+"__"+\
                endtime.strftime("%Y%m%dT%H%M%SZ")+".mseed"\

    trace.write(out_name)

