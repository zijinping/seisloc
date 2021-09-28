#!/usr/bin/env python
# coding: utf-8
# Author: ZI,Jinping
# Revision History
#        2021-01-24 Initial coding
#----------------------------------------------------------------------
from obspy import read
import matplotlib.pyplot as plt
from cuhk_seis.utils import read_sac_ref_time,spherical_dist
import numpy as np

def wf_dist_plot(st,length=10,color=None,label_sta=True,out_format="PNG",scaling_factor=2):
    '''    
    Description
    ------------
    Plot event waveform by distance. The start time is event origin time.
    Data should be in sac format. The output is a saved file with the title
    of reference time.

    Parameters
    -----------
                st: obspy Stream object
            length: The time window is defined by length in seconds.
             color: The usage of color is the same as matplotlib.pyplot. 
                    Using default color if not defined.
         label_sta: Whether label station name on the plot.
        out_format: "png","jpg","pdf"..., The same as matplotlib.pyplot.savefig
    scaling_factor: The waveform are normalized, increase scaling_facotr to
                    make the waveform plot more obvious

    Below data information needed:
    |   P arrival: tr.stats.sac.a
    |   S arrival: tr.stats.sac.t0
    |        evla: tr.stats.sac.evla
    |        evlo: tr.stats.sac.evlo
    |        stla: tr.stats.sac.stla
    |        stlo: tr.stats.sac.stlo
    '''
    st.detrend("linear")
    st.detrend("constant")
    try:
        e_mag = st[0].stats.sac.mag
    except:
        e_mag = -9
    starttime=st[0].stats.starttime
    endtime =st[0].stats.endtime
    #Reference time shoule be the same for all traces.
    sac_ref_time = read_sac_ref_time(st[0]) #In UTCDateTime format
    o_value = st[0].stats.sac.o 
    event_time = sac_ref_time + o_value #event origin time
    st.trim(starttime = event_time, endtime = event_time + length)
    if len(st) == 0:
        print("Error: Nothing to plot!")
    #Inititae parameters
    for tr in st:
        evla = tr.stats.sac.evla
        evlo = tr.stats.sac.evlo
        stla = tr.stats.sac.stla
        stlo = tr.stats.sac.stlo
        #It is recommend to set tr.stats.distance in meters by osbpy guideline
        tr.stats.distance = spherical_dist(evlo,evla,stlo,stla)*111*1000
    #Get mininumtime, maximum time, and max distance
    min_time = st[0].stats.starttime
    max_time = st[0].stats.endtime
    max_dist = st[0].stats.distance/1000 #in km
    for tr in st[1:]:
        if tr.stats.starttime < min_time:
            min_time = tr.stats.starttime
        if tr.stats.endtime > max_time:
            max_time = tr.stats.endtime
        if tr.stats.distance/1000 > max_dist:
            max_dist = tr.stats.distance/1000
    sampling_rate = st[0].stats.sampling_rate

    #Initiate plot parameters
    plt.figure(figsize = (8,10))
    plt.xlim(0,max_time-min_time)
    plt.ylim(0,max_dist+3)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel("Time (s)",fontdict={'size':16})
    plt.ylabel("Distance (km)",fontdict={'size':16})
    #Plot trace by trace
    for tr in st:
        sta = tr.stats.station
        tr_ref_time = read_sac_ref_time(tr) 
        tr_o_value = tr.stats.sac.o
        event_time = tr_ref_time + tr_o_value
        x_start = event_time - min_time
        dist = tr.stats.distance/1000
        #Normalize the event
        disp_data = tr.data/(max(tr.data) - min(tr.data))
        disp_data = disp_data*scaling_factor
        if color == None:
            plt.plot(np.arange(0,len(tr.data))/sampling_rate+x_start,
                    disp_data+dist,
                    linewidth = 0.5)
        else:
            plt.plot(np.arange(0,len(tr.data))/sampling_rate+x_start,
                    disp_data+dist,
                    color = color,
                    linewidth = 0.5)
        if label_sta:
            plt.text(0.1,dist+0.2,sta,fontsize=12)
        #Plot P arrival if available
        try:
            a = tr.stats.sac.a
            rela_a = tr_ref_time + a - min_time
            gap = 0.5*max_dist/25
            plt.plot([rela_a,rela_a],[dist-gap,dist+gap],color='b',linewidth=2)
        except:
            pass
        #Plot S arrival if available
        try:
            t0 = tr.stats.sac.t0
            rela_t0 = tr_ref_time + t0 - min_time
            gap = 0.5*max_dist/25
            plt.plot([rela_t0,rela_t0],[dist-gap,dist+gap],color='r',linewidth=2)
        except:
            pass
        if e_mag != -9:
            plt.title(str(tr_ref_time)+f"_M{e_mag}",fontdict={'size':18})
        else:
            plt.title(str(tr_ref_time),fontdict={'size':18})
    plt.savefig(f"{sac_ref_time.strftime('%Y%m%d%H%M%S')}.png",format=out_format)
    plt.close()
