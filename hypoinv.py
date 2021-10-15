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
    if vs_file != "":
        with open(vs_file,'r') as f:
            cont = f.readlines()
        for i in range(1,len(cont)):             # Start from the second line
            vs_vel,vs_lay = map(float,cont[i].split())
            vs_vels.append(vp_vel)
            vs_lays.append(vp_lay)
    else:
        vs_lays = vp_lays.copy()
        for vp_vel in vp_vels:
            vs_vels.append(vp_vel/ps_ratio)

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


def load_y2000(y2000_file,print_line=False):
    """
    If print_line is true, each phase line will be printed out
    """
    phs_cont = []
    with open(y2000_file,"r") as f1:
        for line in f1:
            phs_cont.append(line.rstrip())
    f1.close()
    phs_dict = {}
    event_count = 0

    print(">>> Loading phases ... ")
    for line in tqdm(phs_cont):
        if print_line:
            print(line)
        f_para = line[0:2]     # first two characters as first parameter(f_para)
        if re.match("\d+",f_para):    # event line
            event_count += 1
            _yr=line[0:4];_mo=line[4:6];_day=line[6:8]
            _hr=line[8:10];_minute=line[10:12];
            yr = int(_yr); mo = int(_mo); day=int(_day); hr=int(_hr);minute=int(_minute);
            _seconds=line[12:14]+"."+line[14:16]
            lat_deg = int(line[16:18]) # short of latitude degree
            try:
                lat_min_int = int(line[19:21]) 
            except:
                lat_min_int = 0
            try:
                lat_min_decimal = int(line[21:23])*0.01
            except:
                lat_min_decimal = 0
            lat_min = lat_min_int + lat_min_decimal
            evla = lat_deg + lat_min/60

            lon_deg = int(line[23:26]) # short of longitude degree
            try:
                lon_min_int = int(line[27:29])
            except:
                lon_min_int = 0
            try:
                lon_min_decimal = int(line[29:31])*0.01
            except:
                lon_min_decimal = 0
            lon_min = lon_min_int + lon_min_decimal  # short of longitude minute    
            evlo = lon_deg + lon_min/60

            try:
                evdp=float(line[32:36])/100
            except:
                evdp = 0

            e_secs = float(_seconds)
            e_time = UTCDateTime(yr,mo,day,hr,minute,0)+e_secs

            str_time = e_time.strftime('%Y%m%d%H%M%S%f')
            str_time = str_time[:16]
            phs_dict[str_time] = {}
            phs_dict[str_time]["eve_loc"] = [evlo,evla,evdp]
            phs_dict[str_time]["phase"] = []
#            try:
#                evid = int(line[136:146])
#                phs_dict[str_time]["evid"] = evid
#            except:
#                print(f"Warning: no evid reads in for {event_count}")
#                pass

        elif re.match("[A-Z]+",f_para) and f_para != "  ": # phase line
            net = line[5:7]
            sta = re.split(" +",line[0:5])[0]
            year = int(line[17:21])
            month = int(line[21:23])
            day = int(line[23:25])
            hour = int(line[25:27])
            minute = int(line[27:29])
            if line[14]==" ":
                #if sec or msec is 0, it will be "  " in out.arc file
                _sec = line[41:44]; _sec_m = line[44:46]
                if _sec == "   ":
                    _sec = "000"
                if _sec_m == "  ":
                    _sec_m = "00"
                p_type="S"
                phs_time = UTCDateTime(year,month,day,hour,minute,0)+\
                                   (int(float(_sec))+int(_sec_m)*0.01)
                phs_dict[str_time]["phase"].append([net,sta,p_type,phs_time-e_time])
            
            else:
                p_type="P"
                #if sec or msec is 0, it will be "  " in out.arc file
                _sec = line[29:32]; _sec_m = line[32:34]
                if _sec == "   ":
                    _sec = "000"
                if _sec_m == "  ":
                    _sec_m = "00"
                phs_time = UTCDateTime(year,month,day,hour,minute,0)+\
                                   (int(float(_sec))+int(_sec_m)*0.01)
                phs_dict[str_time]["phase"].append([net,sta,p_type,phs_time-e_time])
        elif f_para=="  ":
            evid = int(line[66:72])
            phs_dict[str_time]["evid"]=evid
            
    out_name = y2000_file+".pkl"
    out_file = open(out_name,'wb')
    pickle.dump(phs_dict,out_file)
    out_file.close()
 
    return phs_dict


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
            print(lon,lat)
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


def load_sum(sum_file):
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

def load_sum_rev(sum_file):
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
