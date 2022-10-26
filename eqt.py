#!/usr/bin/evn python
# EQTransformer related functions

import obspy
from obspy import UTCDateTime
import os

def to_mseed(trace,out_folder="./"):
    """
    Save the obspy trace to the format EQTransformer could recognized
    """
    net = trace.stats.network
    sta = trace.stats.station
    chn = trace.stats.channel
    starttime = trace.stats.starttime
    endtime = trace.stats.endtime
    out_name = net+"."+sta+"."+chn+"__"+\
                starttime.strftime("%Y%m%dT%H%M%SZ")+"__"+\
                endtime.strftime("%Y%m%dT%H%M%SZ")+".mseed"\

    trace.write(os.path.join(out_folder,out_name))

