#----------------------------------------------------------------------------
# coding: utf-8
# Author: ZI,Jinping
# History:
#     2021-04-06 Initial coding
###################################

import os
import numpy as np
from math import radians,cos,acos,sin,asin,sqrt,ceil,pi,floor
import obspy
from obspy import Stream
import glob
import re
from obspy import UTCDateTime
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from obspy.geodetics import gps2dist_azimuth
from tqdm import tqdm
import multiprocessing as mp
import random
import shutil
import subprocess
import time
import pickle
from seisloc.geometry import in_rectangle,loc_by_width
from seisloc.io import read_y2000_event_line,read_y2000_phase_line



def invmod2vel(out_file,vp_file,vs_file="",ps_ratio=1.73,vpdamp=1,vsdamp=1):
    """
    Convert hypoinverse velocity model to the velest velocity model

    Parameters:
       out_file: the output velest velocity model file
        vp_file: input hypoinverse P velocity file
        vs_file: input hypoinverse S velocity file, if vs_file == "",
                   the output S velocity will be calculated based on P
                   velocity and ps_ratio
       ps_ratio: used when vs_file==""
         vpdamp: damping factor in the output P velocity model
         vsdamp: damping factor in the output S velocity model
    """
    vp_vels = []
    vp_lays = []
    vs_vels = []
    vs_lays = []
    with open(vp_file,'r') as f:
        cont = f.readlines()
    for i in range(1,len(cont)):                 # Start from the second line
        vp_vel,vp_lay = map(float,cont[i].split())
        vp_vels.append(vp_vel)
        vp_lays.append(vp_lay)
    if vs_file =="":
        vs_lays = vp_lays.copy()
        for vp_vel in vp_vels:
            vs_vels.append(vp_vel/ps_ratio)
    else:
        with open(vs_file,'r') as f:
            cont = f.readlines()
        for i in range(1,len(cont)):             # Start from the second line
            vs_vel,vs_lay = map(float,cont[i].split())
            vs_vels.append(vs_vel)
            vs_lays.append(vs_lay)

    f = open(out_file,'w')
    f.write("Velocity model from HYPOINVERSE\n")
    f.write(str(len(vp_vels))+"\n")
    for i in range(len(vp_vels)):
        f.write(format(vp_vels[i],'5.2f'))
        f.write("     ")
        f.write(format(vp_lays[i],'7.2f'))
        f.write("  ")
        f.write(format(vpdamp,'7.3f'))
        f.write("\n")
    f.write(str(len(vs_vels))+"\n")
    for i in range(len(vs_vels)):
        f.write(format(vs_vels[i],'5.2f'))
        f.write("     ")
        f.write(format(vs_lays[i],'7.2f'))
        f.write("  ")
        f.write(format(vpdamp,'7.3f'))
        f.write("\n")

def phs_add_mag(phs_file,mag_file):
    """
    Add magnitude information to the phs file. If no magnitude, set it to -9
    """
    out_file = phs_file+".mag"

    # load in magnitude information
    mag_dict = {}
    with open(mag_file,'r') as f:
        for line in f:
            line = line.strip()
            _evid,_mag = re.split(" +",line)
            evid = int(_evid)
            mag = float(_mag)
            mag_dict[evid] = mag
    
    output_lines = []
    evid_line_idxs = []
    evid_list = []
    i = 0
    tmp_lines = []

    with open(phs_file,'r') as f:
        for line in f:
            line = line.rstrip()
            if line[:5] != "     ":
                tmp_lines.append(line)
            if line[:5] == "     ":
                tmp_lines.append(line)
                evid = int(re.split(" +",line)[1])
                try:
                    e_mag = mag_dict[evid]
                    if e_mag < 0:
                        e_mag = 0
                except:
                    e_mag = 0
                eve_line = tmp_lines[0]
                if len(eve_line) < 126:
                    eve_line = eve_line[:36]+str(int(e_mag*100)).zfill(3)+eve_line[39:]
                output_lines.append(eve_line)  # append the event line
                for line in tmp_lines[1:]:
                    output_lines.append(line)
                tmp_lines = []                 # empty the temporary line
    f.close()

    with open(phs_file+".mag","w") as f:
        for line in output_lines:
            f.write(line+"\n")




def phs_subset(phs_file,evid_list=[],loc_filter=[]):
    """
    subset the *.phs file by event id list or by region location
    len(evid_list) == 0 means no evid filter applied
    len(loc_filter) == 0 means no location filter applied

    Parameters:
        loc_filter in format [lon_min,lon_max,lat_min,lat_max]
    """
    evid_filt = False
    loc_filt = False
    
    if len(evid_list) > 0:
        evid_filt = True

    if len(loc_filter) > 0:
        loc_filt = True
        if len(loc_filter)!=4:
            raise Exception(f"Values qty in loc_filter is {len(loc_filter)},should be 4.")
        lon_min,lon_max,lat_min,lat_max = loc_filter

    cont = [] 
    with open(phs_file,'r') as f:
        for line in f:
            line = line.rstrip()
            cont.append(line)
    output = []
    tmp = []
    for line in cont:
        tmp.append(line)
        if re.match("\d+",line[:4]):
            lat = int(line[16:18])+int(line[19:23])*0.01/60
            if line[18] == "S":
                lat = -lat
            lon = int(line[23:26])+int(line[27:31])*0.01/60
            if line[26]=="W":
                lon = -lon
        if line[:4]== "    ": # last line of one event
            record_status = True
            evid = int(line[66:72])
            if evid_filt == True:
                if evid not in evid_list:
                    record_status = False
            if loc_filt == True:
                if lat<lat_min or lat>lat_max or lon<lon_min or lon>lon_max:
                    record_status = False

            if record_status == True:
                for line in tmp:
                    output.append(line)
            tmp = []               # reset to empty
                
    with open(phs_file+".sel",'w') as f:
        for line in output:
            f.write(line+"\n")
    f.close()


def load_sum_evid(sum_file):
    """
    *.sum file is the catalog summary file after Hyperinverse.
    This function returns a dict:
        -key is event id
        -value is an array with below component:
            --Str format event time "yyyymmddhhmmss**", also the event folder.
            --event longitude
            --event latitude
            --event depth
            --event magnitude
            --event travel time residual
    """
    sum_dict = {}
    with open(sum_file,'r') as f:
        for line in f:
            eve_id=int(line[136:146])
            eve_folder = line[0:16]
            evla = int(line[16:18])+0.01*int(line[19:23])/60
            evlo = int(line[23:26])+0.01*int(line[27:32])/60
            evdp = int(line[31:36])*0.01
            e_mag = int(line[123:126])*0.01
            e_res = int(line[48:52])*0.01
            sum_dict[eve_id] = [eve_folder,evlo,evla,evdp,e_mag,e_res]
    return sum_dict

def load_sum_evstr(sum_file):
    """
    *.sum file is the catalog summary file after Hyperinverse.
    This function returns a dict:
        -key is event time in "yyyymmddhhmmss**" format, same with event folder
        -value is an array with below component:
            --event id
            --event longitude
            --event latitude
            --event depth
            --event magnitude
            --event travel time residual
    """

    sum_dict = {}
    with open(sum_file,'r') as f:
        for line in f:
            eve_id=int(line[136:146])
            eve_folder = line[0:16]
            evla = int(line[16:18])+0.01*int(line[19:23])/60
            evlo = int(line[23:26])+0.01*int(line[27:32])/60
            evdp = int(line[31:36])*0.01
            e_mag = int(line[123:126])*0.01
            e_res = int(line[48:52])*0.01
            sum_dict[eve_folder] = [eve_id,evlo,evla,evdp,e_mag,e_res]
    f.close()
    return sum_dict

def arc_filt(arc_file="Y2000.phs",min_obs=8):
    cont = []
    with open(arc_file,"r") as f:
        cont = f.readlines()
    f.close()
    filt_cont = []
    tmp_cont = []

    for line in cont:
        if line[:4] != "    ":
            tmp_cont.append(line)
        else:
            tmp_cont.append(line)
            if len(tmp_cont) >= min_obs+1+1: # 1 event line, one evid line
                for line in tmp_cont:
                    filt_cont.append(line)
            tmp_cont = []

    with open(arc_file+".filt",'w') as f:
        for line in filt_cont:
            f.write(line)
    f.close()

class Hypoinv():
    def __init__(self,sum_file="out.sum"):
        self.dict_evid = load_sum_evid(sum_file)
        self.dict_evstr = load_sum_evstr(sum_file)
        self.get_locs()
        self.gen_cata_dict()
    
    def get_locs(self):
        self.locs = []
        tmp = 0
        for key in self.dict_evid.keys():
            lon = self.dict_evid[key][1]
            lat = self.dict_evid[key][2]
            dep = self.dict_evid[key][3]
            mag = self.dict_evid[key][4]
            res = self.dict_evid[key][5]
            tmp = tmp+res
            self.locs.append([lon,lat,dep,mag])
        self.locs = np.array(self.locs)
        self.avg_res = tmp/self.locs.shape[0]

    def gen_cata_dict(self):
        self.cataDict = {}
        for key in self.dict_evid.keys():
            evstr = self.dict_evid[key][0]
            etime = UTCDateTime.strptime(evstr,"%Y%m%d%H%M%S%f")
            lon = self.dict_evid[key][1]
            lat = self.dict_evid[key][2]
            dep = self.dict_evid[key][3]
            mag = self.dict_evid[key][4]
            res = self.dict_evid[key][5]
            self.cataDict[key]=[lon,lat,dep,mag,etime]

        
    def crop(self,lonmin,lonmax,latmin,latmax):
        pop_list = []
        for key in self.dict.keys():
            lon,lat,_,_,_ = self.dict[key]
            if lon<lonmin or lon>lonmax or lat<latmin or lat>latmax:
                pop_list.append(key)
                
        for key in pop_list:
            self.dict.pop(key)
    
    def hplot(self,
              xlim=[],
              ylim=[],
              markersize=6,
              size_ratio=1,
              imp_mag=3,
              add_cross=False,
              alonlat=[104,29],
              blonlat=[105,30],
              cross_width=0.1):

        plt.scatter(self.locs[:,0],
                    self.locs[:,1],
                    (self.locs[:,3]+2)*size_ratio,
                    edgecolors = "k",
                    facecolors='none',
                    marker='o',
                    alpha=1)

        kk = np.where(self.locs[:,3]>=imp_mag)
        if len(kk)>0:
            imp = plt.scatter(self.locs[kk,0],
                        self.locs[kk,1],
                        (self.locs[kk,3]+2)*size_ratio*5,
                        edgecolors ='r',
                        facecolors='none',
                        marker='*',
                        alpha=1)
            plt.legend([imp],[f"M$\geq${format(imp_mag,'4.1f')}"])
        if add_cross == True: # draw cross-section plot
            a1lon,a1lat,b1lon,b1lat = loc_by_width(alonlat[0],
                                                   alonlat[1],
                                                   blonlat[0],
                                                   blonlat[1],
                                                   width=cross_width,
                                                   direction="right")
            a2lon,a2lat,b2lon,b2lat = loc_by_width(alonlat[0],
                                                   alonlat[1],
                                                   blonlat[0],
                                                   blonlat[1],
                                                   width=cross_width,
                                                   direction="left")
            plt.plot([a1lon,b1lon,b2lon,a2lon,a1lon],
                     [a1lat,b1lat,b2lat,a2lat,a1lat],
                     linestyle='--',
                     c='darkred')
            plt.plot([alonlat[0],blonlat[0]],[alonlat[1],blonlat[1]],c='darkred')
        if len(xlim) != 0:
            plt.xlim(xlim)
        if len(ylim) != 0: 
            plt.ylim(ylim)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()
    
    def vplot(self,alonlat=[],blonlat=[],width=0.1,depmin=0,depmax=10,size_ratio=1,imp_mag=3):
        """
        Description
        """
        length_m,_,_ = gps2dist_azimuth(alonlat[1],alonlat[0],blonlat[1],blonlat[0])
        length_km = length_m/1000
        alon = alonlat[0]; alat = alonlat[1]
        blon = blonlat[0]; blat = blonlat[1]
        results = in_rectangle(self.locs,alon,alat,blon,blat,width)
        jj = np.where(results[:,0]>0)
        plt.scatter(results[jj,1],
                    self.locs[jj,2],
                    marker='o',
                    edgecolors = "k",
                    facecolors='none',
                    s=(self.locs[jj,3]+2)*size_ratio*5)
        tmplocs = self.locs[jj]
        tmpresults = results[jj]
        kk = np.where(tmplocs[:,3]>=imp_mag)

        if len(kk)>0:                 
            imp = plt.scatter(tmpresults[kk,1],
                        tmplocs[kk,2],
                        (tmplocs[kk,3]+2)*size_ratio*10,
                        edgecolors ='red',
                        facecolors='red',
                        marker='*',
                        alpha=1)
            plt.legend([imp],[f"M$\geq${format(imp_mag,'4.1f')}"])
        
        plt.ylim([depmax,depmin])
        plt.xlim([0,length_km])
        plt.xlabel("length (km)")
        plt.ylabel("depth (km)")
        plt.show()

    def depth_hist(self,mag_threshold=-9,depthmin=0,depthmax=10,gap=0.5):
        bins=np.arange(depthmin,depthmax,gap)
        fig,ax = plt.subplots(1,1,figsize=(6,8))
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("Depth (km)",fontsize=16)
        ax.set_xlabel("Event Qty",fontsize=16)
        kk = np.where(self.locs[:,3]>=mag_threshold)
        hist,bins = np.histogram(self.locs[:,2],bins=bins)
        ax.barh(bins[:-1]+gap/2,hist,height=gap,color='gray',edgecolor='k')
        ax.set_ylim([depthmax,depthmin])
        plt.show()
        
    def __repr__(self):
        self.get_locs()
        _qty = f"Hypoinverse catlog with {len(self.locs)} events\n"
        _mag = f"Magnitue range is: {format(np.min(self.locs[:,3]),'4.1f')} to {format(np.max(self.locs[:,3]),'4.1f')}\n"
        _lon = f"Longitude range is: {format(np.min(self.locs[:,0]),'8.3f')} to {format(np.max(self.locs[:,0]),'8.3f')}\n"
        _lat = f"Latitude range is: {format(np.min(self.locs[:,1]),'7.3f')} to {format(np.max(self.locs[:,1]),'7.3f')}\n"
        _dep = f"Depth range is: {format(np.min(self.locs[:,2]),'4.1f')} to {format(np.max(self.locs[:,2]),'4.1f')}\n"
        return _qty+_mag+_lon+_lat+_dep
    
    def __getitem__(self,key):
        if isinstance(key,str):
            return self.dict_evstr[key]
        elif isinstance(key,int):
            return self.dict_evid[key]

