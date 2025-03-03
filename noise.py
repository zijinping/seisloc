import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np
import pickle
from tqdm import tqdm
from numba import jit

def xcorr(data1,data2,max_shift_num):
    len1 = len(data1)
    len2 = len(data2)
    min_len = min(len1,len2)
    cross_list = []
    for shift_num in np.arange(-max_shift_num,max_shift_num+1,1):
        if shift_num<0:
            correlate_value = np.correlate(data1[:min_len+shift_num],data2[-shift_num:min_len])
            cross_list.append(correlate_value.ravel())
        else:
            correlate_value = np.correlate(data2[:min_len-shift_num],data1[shift_num:min_len])
            cross_list.append(correlate_value.ravel())
    cross_list = np.array(cross_list)
    return cross_list.ravel()

@jit(nopython=True)
def xcorr_jit(data1,data2,max_shift_num):
    len1 = len(data1)
    len2 = len(data2)
    min_len = min(len1,len2)
    cross_list = np.zeros(2*max_shift_num+1)
    for i,shift_num in enumerate(np.arange(-max_shift_num,max_shift_num+1,1)):
        if shift_num<0:
            cross_list[i] = np.dot(data1[:min_len+shift_num],data2[-shift_num:min_len])
        else:
            cross_list[i] = np.dot(data2[:min_len-shift_num],data1[shift_num:min_len])

    return cross_list

def plot_CFs(ax,staPair,base="CFs_results",xlim=[-10,10],cmap="PiYG"):
    """
    Plot the ambient noise results of one station pair on ax delivered.
    
    Parameters
             ax: ax of matplot.pyplot delivered
        staPair: station pair of two stations in format "sta1_sta2"
           base: base path for the cross-correlation result
           xlim: lag time range for plot in seconds
    """
    pairBase = os.path.join(base,"pairs")
    pklLst = glob.glob(os.path.join(pairBase,staPair,"*.pkl"))
    pklLst.sort()
    pkls = []
    for file in pklLst:
        f = open(file,'rb')
        pkl = pickle.load(f)
        pkls.append(pkl)
    if len(pkls) == 0:
        return False
    data = np.zeros((len(pkls),len(pkls[0]["NCF"])))
    for i in range(len(pkls)):
        data[i,:] = pkls[i]["NCF"].ravel()
    [xi,yi]=np.meshgrid(pkls[0]['CFtime'],np.arange(0,len(pkls),1))
    ax.imshow(data,extent=[xi.min(),xi.max(),yi.min()-0.5,yi.max()+0.5],aspect='auto',cmap=cmap)
    ax.vlines(0,-1,len(pkls)+1,ls='--')
    ax.set_ylim([len(pkls)-0.5,-0.5])
    ax.set_xlim(xlim)
    ax.set_xlabel("Lag Time (s)")
    ax.set_ylabel("Day")
    ax.set_title(staPair)

    return True

def plot_pairs_CFs(base="CFs_results",
                   xlim=[-10,10],
                   figsize=(8,4),
                   cmap="PiYG"):
    """
    Plot ambient noise results of each pair under the base path

    Parameters
               base: base path for the cross-correlation result
               xlim: lag time range for plot in seconds
            figsize: figure size for plot
    """
    pairBase = os.path.join(base,"pairs")
    for pair in os.listdir(pairBase):
        pairDir = os.path.join(pairBase,pair)
        if os.path.isdir(pairDir):
            fig,ax = plt.subplots(1,1,figsize=figsize)
            status = plot_CFs(ax,pair,base,xlim=xlim,cmap=cmap)
            if status == True:
                fig.savefig(os.path.join(pairDir,f"{pair}.pdf"))


def plot_stas_CFs(staLst=[],
                   base="CFs_results",
                   xlim=[-5,5],
                   axWX=4,
                   axWY=4,
                   cmap="PiYG"):
    """
    For each sta in sta_list, extract all the pairs contain this sta and plot
    in one figure.

    Parameters
      sta_list: stations to be processed,empty means all stations under base
          base: the folder cotaining cross-correlation results
          xlim: lag time range for plot in seconds
        widthX: x size for each ax
        widthY: y size for each ax
    """
    if len(staLst)==0:         # if no specific staLst provided, then process all stations
        pairBase = os.path.join(base,"pairs")
        for pair in os.listdir(pairBase):
            pairDir = os.path.join(pairBase,pair)
            if os.path.isdir(pairDir) and \
               len(pair.split("_"))==2:
                sta1,sta2 = pair.split("_")
                if sta1 not in staLst:
                    staLst.append(sta1)
                if sta2 not in staLst:
                    staLst.append(sta2)
    for sta in staLst:
        processLst = []
        for staPair in os.listdir(pairBase):
            if sta in staPair:
                processLst.append(staPair)
        print(f"Processing {format(sta,'5s')} with {len(processLst)} pairs")
        k = len(processLst)
        if k==0:
            continue
        elif k == 1:
            fig,ax = plt.subplots(1,1,figsize=(axWX,axWY))
            plot_CFs(ax,processLst[0],base)
        elif k <=4:
            fig,axs = plt.subplots(1,k,figsize=(4*axWX,axWY))
        else:
            if (k/4-int(k/4))==0:
                rows =int(k/4)
            else:
                rows = int(k/4)+1
            columns = 4
            fig,axs = plt.subplots(rows,columns,
                                    figsize=(columns*axWX,rows*axWY)
                                    )
            axs_list = axs.ravel()
            for i in range(len(processLst)):
                plot_CFs(axs_list[i],processLst[i],base=base,xlim=xlim,cmap=cmap)
        plt.tight_layout()
        plotDir = os.path.join(base,"Plot_CFs_by_sta") 
        if not os.path.exists(plotDir):
            os.mkdir(plotDir)
        plt.savefig(os.path.join(plotDir,f"{sta}.png"))
