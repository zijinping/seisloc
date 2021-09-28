# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
#     Author: Jinping ZI
#
# Revision History
#     2021-01-24 Initiate coding
#-----------------------------------------------------------------------------

import os
import numpy as np
from math import radians,cos,acos,sin,asin,sqrt,ceil,pi
import obspy
from obspy import Stream
import glob
import re
from obspy import UTCDateTime
import pandas as pd
from obspy.geodetics import gps2dist_azimuth

def extract_set_info(pth,sta_file,depth=2):
    """
    Parameters:
        pth: path for the folder to work on
        depth: 1 or 2, 1 means pth/files, 2 means pth/folder/files
        
    Return:
        A setinfo dictionary containing keys:
        "s_times": sorted starttime list
        "e_times": sorted endtime list
        "netstas": network and station list in format <net+sta>
        "center" : mean longitude and latitude of stations,intended for tele-
                   event selection.
    """
    setinfo = {}
    sta_dict = load_sta(sta_file)
    s_times = []
    e_times = []
    netstas = []
    sta_lons = []
    sta_lats = []
    if depth == 2:                    # strcture: pth/folder/seis_file
        for item in os.listdir(pth):
            if os.path.isdir(os.path.join(pth,item)):
                for file in os.listdir(os.path.join(pth,item)):
                    try:
                        st = obspy.read(os.path.join(pth,item,file),headonly=True)
                        for tr in st:
                            net = tr.stats.network
                            sta = tr.stats.station
                            netsta = net+sta
                            if netsta not in netstas:
                                netstas.append(netsta)
                                sta_lons.append(sta_dict[net][sta][0])
                                sta_lats.append(sta_dict[net][sta][1])          
                            s_times.append(tr.stats.starttime)
                            e_times.append(tr.stats.endtime)
                    except:           # not seis file
                        pass
    if depth == 1:                    # strcture: pth/seis_file
        for item in os.listdir(pth):
            if os.path.isfile(os.path.join(pth,item)):
                try:
                    st = obspy.read(os.path.join(pth,item,file),headonly=True)
                    for tr in st:
                        net = tr.stats.network
                        sta = tr.stats.station
                        netsta = net+sta
                        if netsta not in netstas:
                            netstas.append(netsta)
                            sta_lons.append(sta_dict[net][sta][0])
                            sta_lats.append(sta_dict[net][sta][1])  
                        s_times.append(tr.stats.starttime)
                        e_times.append(tr.stats.endtime)
                except:               # not seis file
                    pass

    setinfo["s_times"] = sorted(s_times)
    setinfo["e_times"] = sorted(e_times)
    setinfo["netstas"] = netstas
    print(sta_lons)
    setinfo["center"] = [np.mean(sta_lons),np.mean(sta_lats)] 
    return setinfo

def load_sta(sta_file):
    sta_dict={}
    with open(sta_file,'r') as f:
        for line in f:
            line = line.rstrip()
            net,sta,_lon,_lat,_ele,label=re.split(" +",line)
            if net not in sta_dict:
                sta_dict[net]={}
            if sta not in sta_dict[net]:
                sta_dict[net][sta] = [float(_lon),float(_lat),float(_ele),label]
    return sta_dict

def draw_vel(ax,dep_list,vel_list,color='k',linestyle='-'):
    """
    Draw velocity line on the ax based on the depth list and velocity list
    """
    points_list = []
    points_list.append([dep_list[0],vel_list[0]])
    for i in range(1,len(dep_list)):
        points_list.append([dep_list[i],vel_list[i-1]])
        points_list.append([dep_list[i],vel_list[i]])
        
    points_list = np.array(points_list)
    line, = ax.plot(points_list[:,1],points_list[:,0],color=color,linestyle=linestyle)
    return line


def read_sac_ref_time(tr):
    """
    Read and return reference time of a sac file in obspy.UTCDateTime format.

    Parameter
    --------
    tr: Trace object of obspy
    """

    nzyear = tr.stats.sac.nzyear
    nzjday = tr.stats.sac.nzjday
    nzhour = tr.stats.sac.nzhour
    nzmin = tr.stats.sac.nzmin
    nzsec = tr.stats.sac.nzsec
    nzmsec = tr.stats.sac.nzmsec*0.001
    year,month,day = month_day(nzyear,nzjday)
    sac_ref_time = UTCDateTime(year,month,day,nzhour,nzmin,nzsec)+nzmsec
    return sac_ref_time

def get_st(net,sta,starttime,endtime,f_folder):
    """
    Read and return waveform between starttime and endtime by specified
    net and station in designated folder. It will merge waveform if include
    more than one file.
    
    The return is a obspy Stream object
    """
    inc_list=[]
    for file in os.listdir(f_folder):
        file_path = os.path.join(f_folder,file)
        try:
            st = obspy.read(file_path,headonly=True)
        except:
            continue
        t1,t2 = st[0].stats.starttime,st[0].stats.endtime
        if t2 < starttime or t1 > endtime \
            or st[0].stats.network != net\
            or st[0].stats.station != sta:
            continue
        else:
            inc_list.append(file_path)
    #Read in data
    st = Stream()
    for path in inc_list:
        st += obspy.read(path)
    if len(st) == 0:
        pass
    else:
        st.trim(starttime,endtime)
    return st

def julday(year,month,day):
    ref_time=UTCDateTime(year,1,1)
    tar_time=UTCDateTime(year,month,day)
    julday=(tar_time-ref_time)/(24*60*60)+1
    return int(julday)

def month_day(year,julday):
    """
    Transfer from julday to month and day.
    Return year,month,day
    """
    #check if year is leap year
    leap=False
    if year%100==0:
        if year%400==0:
            leap=True
    else:
        if year%4==0:
            leap=True
    normal_list=[0,31,59,90,120,151,181,212,243,273,304,334,365]
    leap_list=[0,31,60,91,121,152,182,213,244,274,305,335,366]
    if leap:
        i=0
        while leap_list[i]<julday:
            i=i+1
        month=i
        day=julday-leap_list[i-1]
        return year,month,day
    else:
        i=0
        while normal_list[i]<julday:
            i=i+1
        month=i
        day=julday-normal_list[i-1]
        return year,month,day

def find_nearest(array,value):
    """
    find the nearest value. The return is index and diff
    """
    if type(array) != np.ndarray:
        array=np.array(array)
    idx=np.abs(array-value).argmin()
    diff=array[idx]-value
    return idx,diff



def read_sta_file(sta_file):
    """
    Read information from the station file with free format: net,sta,lon,lat,ele,label.
    The label is designed with the purpose to distinguish stations into types.
    """
    cont = []
    with open(sta_file,'r') as f:
        for line in f:
            line = line.rstrip()
            net,sta,_lon,_lat,_ele,label = re.split(" +",line)
            cont.append([net,sta,float(_lon),float(_lat),int(_ele),label])
    f.close()
    if len(cont)==0:
        raise Exception(f"No content in the station file {sta_file}")
    return cont

def to_inv_sta_file(cont,out_file):
    f_inv = open(out_file,'w')
    for tmp in cont:
        lat = tmp[3]
        lon = tmp[2]
        ele = tmp[4]
        net = tmp[0]
        sta = tmp[1]
        label = tmp[5]
        net_sta = net+sta
        lon_i = int(lon)
        lon_f = lon-lon_i
        lat_i = int(lat)
        lat_f = lat-lat_i
        f_inv.write(format(sta,"<6s")+format(net,"<4s")+"SHZ  "+format(lat_i,">2d")+" "+\
                format(lat_f*60,">7.4f")+" "+format(lon_i,">3d")+" "+format(lon_f*60,">7.4f")+\
                "E"+format(ele,">4d")+"\n")
    f_inv.close()

def sta2inv(sta_file,out_file):
    """
    Convert station file into hypoinverse format
    """
    cont = read_sta_file(sta_file)      # Read in information
    to_inv_sta_file(cont,out_file)  # Write into files

def to_dd_sta_file(cont,out_file):
    f_dd = open(out_file,'w')
    for tmp in cont:
        lat = tmp[3]
        lon = tmp[2]
        ele = tmp[4]
        net = tmp[0]
        sta = tmp[1]
        label = tmp[5]
        net_sta = net+sta
        lon_i = int(lon)
        lon_f = lon-lon_i
        lat_i = int(lat)
        lat_f = lat-lat_i
        f_dd.write(format(net_sta,"<9s")+format(lat_i+lat_f,">9.6f")+format(lon_i+lon_f,">12.6f")+\
                   " "+format(ele,'>5d')+"\n")
    f_dd.close()

def sta2dd(sta_file,out_file):
    """
    Convert station file into hypoDD format
    """
    cont = read_sta_file(sta_file)      # Read in information
    to_dd_sta_file(cont,out_file)  # Write into files

def to_vel_sta_file(cont,out_file,ele_zero=True):
    f_vel = open(out_file,'w')
    f_vel.write("(a5,f7.4,a1,1x,f8.4,a1,1x,i4,1x,i1,1x,i3,1x,f5.2,2x,f5.2,3x,i1)\n")
    sta_count = 1
    for tmp in cont:
        lat = tmp[3]
        lon = tmp[2]
        ele = tmp[4]
        net = tmp[0]
        sta = tmp[1]
        label = tmp[5]
        if ele_zero:
            ele = 0
            f_vel.write(f"{format(sta,'<5s')}{format(lat,'7.4f')}N {format(lon,'8.4f')}E {format(ele,'4d')} 1 "+\
                        f"{format(sta_count,'3d')} {format(0,'5.2f')}  {format(0,'5.2f')}   1\n")
        else:
            f_vel.write(f"{format(sta,'<5s')}{format(lat,'7.4f')}N {format(lon,'8.4f')}E {format(ele,'4d')} 1 "+\
                        f"{format(sta_count,'3d')} {format(0,'5.2f')}  {format(0,'5.2f')}   1\n")
        sta_count += 1
    f_vel.write("  \n")   # signal of end of file for VELEST
    f_vel.close()

def sta2vel(sta_file,out_file,ele_zero=True):
    """
    Convert station file into VELEST format with 5 characters,
    which is applicable for the update VELEST program modified by Hardy ZI
    """
    cont = read_sta_file(sta_file)
    to_vel_sta_file(cont,out_file,ele_zero)
