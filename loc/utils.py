from seisloc.loc.dd import load_DD
from seisloc.loc.hypoinv import load_sum_evid
from seisloc.loc.text_io import _load_event_dat_etime
from obspy import UTCDateTime
import re
import numpy as np
import matplotlib.pyplot as plt
from seisloc.loc.cata import Catalog

def _load_cata(locFile,format="hypoDD"):
    if format == "hypoDD":
        tmpDict, _ = load_DD(locFile)
        return tmpDict
    if format == "sum": # Hypoinverse summary file
        sumDict = load_sum_evid(locFile)
        tmpDict = {}
        for evid in sumDict.keys():
            estr,evlo,evla,evdp,mag,eres = sumDict[evid]
            etime = UTCDateTime.strptime(estr,"%Y%m%d%H%M%S%f")
            tmpDict[evid] = [evlo,evla,evdp,mag,etime]
        return tmpDict
    if format == "cata":
        with open(locFile,'r') as f:
            tmpDict={}
            for line in f:
                line = line.strip()
                _evid,_evlo,_evla,_evdp,_mag,_eday,_etime=line.split()
                tmpDict[int(_evid)] = [float(_evlo),float(_evla),float(_evdp),float(_mag),UTCDateTime(_etime)]
    if format == "dat":  # event dat fromat
        tmpDict = {}  
        f = open(locFile,'r')  
        eventLines = f.readlines() 
        for eventLine in eventLines:  
            timeSeg = eventLine[:18]  
            etime = _load_event_dat_etime(timeSeg)  
            otherSeg = eventLine[18:].strip()  
            _lat,_lon,_dep,_mag,_eh,_ez,_rms,_evid = re.split(" +",otherSeg)  
            lat,lon,dep,mag,eh,ez,rms = map(float,(_lat,_lon,_dep,_mag,_eh,_ez,_rms))
            evid = int(_evid)  
            tmpDict[evid] = [lon,lat,dep,mag,etime]
        
    return tmpDict

def hypoinv2Catalog(inv):
    """
    Convert Hypoinv class to Catalog class
    """
    inv_dict={}
    for key in inv.dict_evid.keys():
        inv_dict[key] = inv.dict_evid[key][1:5]
        _time = inv.dict_evid[key][0]
        etime = UTCDateTime.strptime(_time,"%Y%m%d%H%M%S%f")
        inv_dict[key].append(etime)
    inv_cata = Catalog(locFile=None)
    inv_cata.dict = inv_dict
    inv_cata.init()
    return inv_cata

def write_txt_cata(edict,fileName,refTime,disp=False):
    f = open(fileName,'w')
    for evid in edict.keys():
        evlo,evla,evdp,mag,etime = edict[evid]
        if evdp>=6800:
            print(f"[Warning] Depth of event {evid} exceeds 6800, are you sure? ")
        relDay = (etime - refTime)/(24*60*60)
        _evid = format(evid,'8d')
        _evlo = format(evlo,'12.6f')
        _evla = format(evla,'11.6f')
        _evdp = format(evdp,'8.2f')
        _mag = format(mag,'5.1f')
        _relDay = format(relDay,'16.8f')
        line = _evid+_evlo+_evla+_evdp+_mag+_relDay+" "+str(etime)
        f.write(line+"\n")
        if disp == True:
            print("[Class Catalog] "+line)
    f.close()

def read_txt_cata(cataPth,verbose=1):
    """
    This function could be replaced by Catalog(cataPth,format="cata") [recommend]
    """
    with open(cataPth,'r') as f:
        edict={}
        for line in f:
            line = line.strip()
            _evid,_evlo,_evla,_evdp,_mag,_eday,_etime=line.split()
            edict[int(_evid)] = [float(_evlo),float(_evla),float(_evdp),float(_mag),UTCDateTime(_etime)]
    cata = Catalog(locFile=None,verbose=verbose)
    cata.dict = edict
    cata.init()
    return cata

def _plot_eqs(data,params):
    """
    Plot earthquakes for the mode of "normal" or "animation"
    If mode is "animation", the previous events will be plotted with color[0] 
    and the new events will be plotted with color[1]

    Parameters:
    |      data: data array. Each row is [x,y,mag,relDay]
    """
    #===== parameters =====
    if "edgeColors" in params.keys():
        edgeColors = params["edgeColors"]
    if "impMag" in params.keys():
        impMag = params["impMag"]
    if "eqSizeMagShift" in params.keys():
        eqSizeMagShift = params["eqSizeMagShift"]
    else:
        eqSizeMagShift = 2
    if "eqSizeRatio" in params.keys():
        eqSizeRatio = params["eqSizeRatio"]
    else:
        eqSizeRatio = 1
    if "eqSizeRatioImp" in params.keys():
        eqSizeRatioImp = params["eqSizeRatioImp"]
    else:
        eqSizeRatioImp = 1.5
    if "edgeWidth" in params.keys():
        edgeWidth = params["edgeWidth"]
    else:
        edgeWidth = 0.5
    if "cmap" in params.keys():
        cmap = params["cmap"]
    if "vmin" in params.keys():
        vmin = params["vmin"]
    if "vmax" in params.keys():
        vmax = params["vmax"]
    if "mode" in params.keys():
        mode = params["mode"]
    if "day" in params.keys():
        day = params["day"]
    else:
        day = 1E7
    if "increDay" in params.keys():
        increDay = params["increDay"]
    else:
        increDay = 1E7
    
    #===== plot =====
    ax = plt.gca()

    pcolor = edgeColors[0] # previous, used in the normal mode
    ncolor = edgeColors[1] # new

    if cmap != None:
        if vmin == None:
            vmin = np.min(data[:,-1])
        if vmax == None:
            vmax = np.max(data[:,-1])
    #===== previous =============
    #----- period selection -----
    ks = np.where(data[:,-1]<=day)[0]
    datSel = data[ks,:]
    #----- mag. selection -----
    ks = np.where(datSel[:,2]<impMag)[0]
    if len(ks)>0:
        xs = datSel[ks,0]
        ys = datSel[ks,1]
        mags = datSel[ks,2]
        plt.scatter(xs,ys,(mags+eqSizeMagShift)*eqSizeRatio,
                    edgecolors=pcolor,facecolors='none',marker="o",linewidths=edgeWidth)
        if cmap != None:
            relDays = datSel[ks,3]
            plt.scatter(xs,ys,(mags+eqSizeMagShift)*eqSizeRatio,
                        c=relDays,cmap=cmap,vmin=vmin,vmax=vmax,marker="o",linewidths=edgeWidth)
            plt.colorbar()

    ks = np.where(datSel[:,2]>=impMag)[0]
    if len(ks)>0:
        xs = datSel[ks,0]
        ys = datSel[ks,1]
        mags = datSel[ks,2]
        plt.scatter(xs,ys,(mags+eqSizeMagShift)*eqSizeRatioImp,
                    edgecolors='r',facecolors='none',zorder=5,marker="*",linewidths=edgeWidth,label=f"M$\geq${impMag}")
    #~~~~~ new ~~~~~~~~~~~
    #----- period selection -----
    if mode == "animation":
        ks = np.where((data[:,-1]>day-increDay)&(data[:,-1]<=day))[0]
        datSel = data[ks,:]
        #----- mag. selection -----
        ks = np.where(datSel[:,2]<impMag)[0]
        if len(ks)>0:
            xs = datSel[ks,0]
            ys = datSel[ks,1]
            mags = datSel[ks,2]
            plt.scatter(xs,ys,(mags+eqSizeMagShift)*eqSizeRatio,
                        edgecolors=ncolor,facecolors='none',marker="o",linewidths=edgeWidth)
            if cmap != None:
                relDays = datSel[ks,3]
                plt.scatter(xs,ys,(mags+eqSizeMagShift)*eqSizeRatio,
                            c=relDays,cmap=cmap,vmin=vmin,vmax=vmax,marker="o",linewidths=edgeWidth)
        ks = np.where(datSel[:,2]>=impMag)[0]
        if len(ks)>0:
            xs = datSel[ks,0]
            ys = datSel[ks,1]
            mags = datSel[ks,2]
            plt.scatter(xs,ys,(mags+eqSizeMagShift)*eqSizeRatioImp,
                        edgecolors=ncolor,facecolors='none',zorder=5,marker="*",linewidths=edgeWidth,label=f"M$\geq${impMag}")
