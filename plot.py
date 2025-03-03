import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt

def time_interval_ticks(b_time,e_time,interval=1,unit='month',cut_to_round=True):
    """
    Generate month list from two UTCDateTime objects
    
    Parameters:
    |   b_time: begin time in obspy UTCDateTime format
    |   e_time: end time in obspy UTCDateTime format
    |     unit: interval unit, could be 'year','month','day','hour','minute',or 'second'
  cut_to_round: round off to the unit provided (default: True).
                E.g. unit='month' will round the  base_time to be exactly UTCDateTime(year,month,1)
                else, base_time = b_time
    Return
    ------------------------------------------------
    base_time: UTCDateTime of the start of the first month
    tick_secs: array stores the tick points in seconds
    """
    assert e_time > b_time           # end time larger than begin time
    
    b_yr = b_time.year
    b_mo = b_time.month
    b_day = b_time.day
    b_hour= b_time.hour
    b_min = b_time.minute
    b_sec = b_time.second
    
    if cut_to_round == True:
        if unit == 'year':
            base_time = UTCDateTime(b_yr,1,1)
        if unit == 'month':
            base_time = UTCDateTime(b_yr,b_mo,1)
        elif unit == 'day':
            base_time = UTCDateTime(b_yr,b_mo,b_day)
        elif unit == 'hour':
            base_time = UTCDateTime(b_yr,b_mo,b_day,b_hour)
        elif unit == 'minute':
            base_time = UTCDateTime(b_yr,b_mo,b_day,b_hour,b_min)
        elif unit == 'second':
            base_time = UTCDateTime(b_yr,b_mo,b_day,b_hour,b_min,b_sec)
    else:
        base_time = b_time
    
    tick_secs = []
    loop_time = base_time
    
    if unit == 'year':
        while loop_time < e_time:
            tick_secs.append(loop_time - base_time)
            yr = loop_time.year
            mo = loop_time.month
            dy = loop_time.day
            hr = loop_time.hour
            minu = loop_time.minute
            sec = loop_time.second
            msec = loop_time.microsecond
            yr += interval
            loop_time = UTCDateTime(yr,mo,dy,hr,minu,sec,msec)
    elif unit == 'month':
        while loop_time < e_time:
            tick_secs.append(loop_time - base_time)
            yr = loop_time.year
            mo = loop_time.month
            dy = loop_time.day
            hr = loop_time.hour
            minu = loop_time.minute
            sec = loop_time.second
            msec = loop_time.microsecond
            mo += interval
            if mo > 12:
                yr += 1
                mo -= 12
            loop_time = UTCDateTime(yr,mo,dy,hr,minu,sec,msec)
    elif unit in ['day','hour','minute','second']:
        if unit == 'day':
            interval_seconds = interval*24*60*60
        elif unit == 'hour':
            interval_seconds = interval*60*60
        elif unit == 'minute':
            interval_seconds = interval*60
        elif unit == 'second':
            interval_seconds = interval

        while loop_time < e_time:
            tick_secs.append(loop_time - base_time)
            loop_time += interval_seconds
    else:
        raise Exception("'unit' not in ['year','month','day','hour','minute','second']")
        
    tick_secs.append(loop_time-base_time)

    return base_time, np.array(tick_secs)

def segmented_plot(xys,base_time,secs,tick_secs,
                   xlim=[],
                   ylim=[],
                   columns=4,
                   subplotsize=(3,3),
                   marker='o',
                   ms=1,
                   unit='month',
                   title_fontsize=12,
                   wspace=None,
                   hspace=None,
                   aspect='equal'):
    """
    Description
    Do subplots for xys by time intervals.
    
    Parameters
           xys: 2-D numpy array for location
     base_time: base UTCDateTime time, 'secs' and 'tick_secs' are relative seconds with reference to 'base_time'
     xlim,ylim: plot range, min(x),max(x),min(y),max(y) used if not set
       columns: columns for each row
  subplotsize: figure size for each subplot
     marker,ms: marker and markersize
          unit: the interval type for each subplot, it could be 'year','month','day','hour','minute', or 'second', 
                this parameter is designed to control subplot title format.
     wspace,hspace,aspect: refer to matplotlib.pyplot for details
     
     Return:
           axs: one dimensional list of axes
    """
    # format control
    assert len(xys.shape) == 2
    assert isinstance(base_time,UTCDateTime)
    assert len(secs.shape) == 1
    assert len(tick_secs.shape) == 1
   
    if xlim == []:
        xlim = [np.min(xys[:,0]),np.max(xys[:,0])]
    if ylim == []:
        ylim = [np.min(xys[:,1]),np.max(xys[:,1])]

    # set up subplots
    segment_qty = len(tick_secs) - 1
    

    if segment_qty <= columns:
        rows = 1
        columns = segment_qty
    else:
        if segment_qty % columns == 0:
            rows = int(segment_qty/columns)
        else:
            rows = int(segment_qty/columns)+1
    fig,axs = plt.subplots(rows,columns,
                           sharex=True,sharey=True,
                          figsize=(columns*subplotsize[0],rows*subplotsize[1]))
    if len(axs)>1:
        axs = axs.ravel()

    # title format
    if unit == 'year':
        title_fmt = "%Y"
    elif unit == 'month':
        title_fmt = "%Y-%m"
    elif unit == 'day':
        title_fmt = "%Y-%m-%d"
    elif unit == 'hour':
        title_fmt = "%Y-%m-%dT%H"
    elif unit == 'minute':
        title_fmt = "%Y-%m-%dT%H:%M"
    elif unit == 'second':
        title_fmt = "%Y-%m-%dT%H:%M:%S"
    else:
        raise Exception("Wrong unit type, should be 'year','month','day','hour','minute','second'")
    
    # subplots plot in here
    for i in range(segment_qty):
        plt.sca(axs[i])
        plt.xlim(xlim)
        plt.ylim(ylim)
        kk = np.where((secs>=tick_secs[i])&(secs<tick_secs[i+1]))
        loop_time = base_time + tick_secs[i]
        loop_yr = loop_time.year
        loop_mo = loop_time.month
        loop_dy = loop_time.day
        loop_hr = loop_time.hour
        loop_min = loop_time.minute
        loop_sec = loop_time.second
        plt.plot(xys[kk,0],xys[kk,1],marker=marker,c='k',mfc='none',ms=ms,zorder=10)
        
        plt.title(loop_time.strftime(title_fmt),fontsize=title_fontsize)

        plt.gca().set_aspect(aspect)
        
    for i in range(segment_qty,len(axs)):
        plt.sca(axs[i])               # set current axis
        plt.axis('off')
        
    # adjust space    
    if wspace != None or hspace != None:
        plt.subplots_adjust(wspace=wspace,hspace=hspace)
    else:
        plt.tight_layout()
    
    return axs

def intervals_plot(xys,
                rela_secs,
                reftime,
                interval=1,
                method='month',
                xlim=[],
                ylim=[],
                columns=4,
                subplotsize=(3,3),
                marker='o',
                ms=1,
                wspace=None,
                hspace=None):
    """
    Description
        Plot xy location subplots by time intervals
        
    Parameters
    """
    secs = rela_secs.copy()
    
    # format control
    assert len(xys.shape) == 2
    assert len(secs.shape) == 1
    
    if xlim == []:
        xlim = [np.min(xys[:,0]),np.max(xys[:,0])]
    if ylim == []:
        ylim = [np.min(xys[:,1]),np.max(xys[:,1])]
    min_time = reftime + np.min(secs)
    max_time = reftime + np.max(secs)
    base_time, tick_secs = time_interval_ticks(min_time,max_time,interval=interval,unit=method)
    
    diff_time = reftime - base_time
    secs += diff_time
    
    axs = segmented_plot(xys,base_time,secs,tick_secs,
                         xlim=xlim,ylim=ylim,
                         marker=marker,ms=ms,
                         columns=columns,
                         unit=method,
                         subplotsize=subplotsize,
                         wspace=wspace,
                         hspace=hspace)
    
    return axs

def depths_plot(xyz,
                deplim=[0,10],interval=1,
                xlim=[],ylim=[],
                columns=4,subplotsize=(3,3),
                marker='o',ms=1,color='k',
                zorder=0,
                wspace=None,hspace=None):
    
    assert len(xyz.shape) == 2
    
    if xlim == []:
        xlim = [np.min(xyz[:,0]),np.max(xyz[:,0])]
    if ylim == []:
        ylim = [np.min(xyz[:,1]),np.max(xyz[:,1])]

    depticks = list(range(deplim[0],deplim[1],interval))
    depticks.append(deplim[-1])
    
    subqty = len(depticks) - 1
    if subqty <= columns:
        rows = 1
        columns = subqty
    else:
        if subqty % columns == 0:
            rows = int(subqty/columns)
        else:
            rows = int(subqty/columns)+1
    fig,axs = plt.subplots(rows,columns,
                           sharex=True,
                           sharey=True,
                           figsize=(columns*subplotsize[0],rows*subplotsize[1]))
    axs = axs.ravel()
    
    for i in range(subqty):
        plt.sca(axs[i])
        plt.xlim(xlim)
        plt.ylim(ylim)
        deplow = depticks[i]
        dephigh = depticks[i+1]
        kk = np.where((xyz[:,2]>deplow)&(xyz[:,2]<dephigh))
        plt.plot(xyz[kk,0],xyz[kk,1],
                 marker=marker,c=color,ms=ms,
                 zorder=zorder)
        plt.title(f"{deplow}-{dephigh} km")
        plt.gca().set_aspect("equal")
        
    for i in range(subqty,len(axs)):
        plt.sca(axs[i])
        plt.axis("off")

    return axs
