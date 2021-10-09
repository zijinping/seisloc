import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np
import pickle

def plot_corr(ax,sta_pair,base="CFs_Result",xlim=[-10,10]):
    """
    Plot the ambient noise results of one station pair on ax delivered.
    
    Parameters
             ax: ax of matplot.pyplot delivered
       sta_pair: station pair of two stations
           base: base path for the cross-correlation result
           xlim: lag time range for plot in seconds
    """
    file_list = glob.glob(os.path.join(base,sta_pair,"*.pkl"))
    file_list.sort()
    pkls = []
    for file in file_list:
        f = open(file,'rb')
        pkls.append(pickle.load(f))
    if len(pkls) == 0:
        return False
    data = np.zeros((len(pkls),len(pkls[0]["NCF"])))
    for i in range(len(pkls)):
        data[i,:] = pkls[i]["NCF"].ravel()
    [xi,yi]=np.meshgrid(pkls[0]['CFtime'],np.arange(0,len(pkls)+1,1))
    ax.pcolormesh(xi,yi,data,cmap="PiYG")
    ax.vlines(0,0,len(pkls))
    ax.set_ylim([len(pkls),0])
    ax.set_xlim(xlim)
    ax.set_xlabel("Lag Time (s)")
    ax.set_ylabel("Day")
    ax.set_title(sta_pair)
    return True

def plot_pairs_corr(base="CFs_Result",
                   xlim=[-10,10],
                   figsize=(8,4),
                   exclude_folders=["Logs","Plot"]):
    """
    Plot ambient noise results of each pair under the base path

    Parameters
               base: base path for the cross-correlation result
               xlim: lag time range for plot in seconds
            figsize: figure size for plot
    exclude_folders: exclude folders as they are not station pair folders
    """
    for item in os.listdir(base):
        item_path = os.path.join(base,item)
        if os.path.isdir(item_path) and item not in exclude_folders:
            sta_pair = item
            fig,ax = plt.subplots(1,1,figsize=figsize)
            status = plot_corr(ax,sta_pair,base,xlim=xlim)
            if status == True:
                fig.savefig(os.path.join(base,sta_pair,f"{sta_pair}.pdf"))


def plot_stas_corr(sta_list,base = "CFs_Result",xlim=[-5,5],ax_xsize=4,ax_ysize=4):
    """
    For each sta in sta_list, extract all the pairs contain this sta and plot
    in one figure.

    Parameters
      sta_list: stations to be processed
          base: the folder cotaining cross-correlation results
          xlim: lag time range for plot in seconds
      ax_xsize: x size for each ax
      ax_ysize: y size for each ax
    """
    for sta in sta_list:
        process_list = []
        for sta_pair in os.listdir(base):
            if sta in sta_pair:
                process_list.append(sta_pair)
        k = len(process_list)
        if k>0:
            if k == 1:
                fig,ax = plt.subplots(1,1,figsize=(ax_xsize,ax_ysize))
                plot_corr(ax,process_list[0],base)
                return
            elif k <=4:
                fig,axs = plt.subplots(1,k,figsize=(4*ax_xsize,ax_ysize))
            else:
                rows = int(k/4)+1
                columns = 4
                fig,axs = plt.subplots(rows,columns,figsize=(columns*ax_xsize,rows*ax_ysize))
            axs_list = axs.ravel()
            for i in range(len(process_list)):
                plot_corr(axs_list[i],process_list[i],base=base,xlim=xlim)
            plt.tight_layout()
            plot_folder = os.path.join(base,"Plot") 
            if not os.path.exists(plot_folder):
                os.mkdir(plot_folder)
            plt.savefig(os.path.join(base,"Plot",f"{sta}.png"))
