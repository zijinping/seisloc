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
import logging
import pickle
from numba import cuda

cuda.jit()
def _matmul_gpu(A,B,C):
    row,col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row,k] * B[k,col]
        C[row,col] = tmp

def matmul_gpu(A,B,C,TPX=16,TPY=16):
    if len(cuda.gpus) == 0:
        raise Exception("Error, no gpu available")
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    C_global_mem = cuda.device_array(A.shape[0],B.shape[1])
    threads_per_block = (TPA,TPB)
    blocks_per_grid_x = int(math.ceil(A.shape[0]/threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(B.shape[1]/threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x,blocks_per_grid_y)

    _matmul_gpu[blocks_per_grid,blocks_per_block](A_global_mem,
                                                  B_global_mem,
                                                  C_global_mem)
    cuda.syschronize()
    C_global_gpu = C_global_mem.copy_to_host()


def init_logger(log_file,file_level=logging.DEBUG,stream_level=logging.INFO):
    '''
    Parameters:
        log_file: file to save the log information
        file_level: level for file writing
        stream_level: level for stream writing
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file,mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s-%(filename)s-%(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)


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
    logger = logging.getLogger()
    logger.info("Extract_set_info program launched...")
    parent_path = os.path.split(pth)[0]
    if parent_path=="":
        parent_path="."
    if os.path.exists(os.path.join(parent_path,'setinfo.pkl')):
        f = open(os.path.join(parent_path,'setinfo.pkl'),'rb')
        setinfo = pickle.load(f)
        logger.info("Read in the existed setinfo.pkl and return")
        return setinfo

    setinfo = {}
    sta_dict = load_sta(sta_file)
    s_times = []
    e_times = []
    netstas = []
    stalons = []
    stalats = []
    logger.debug(f"file system depth is {depth}")
    availdays = []
    if depth == 2:                    # strcture: pth/folder/seis_file
        for item in os.listdir(pth):
            if os.path.isdir(os.path.join(pth,item)):
                logging.debug(f"Process dir {item}")
                juldays = []
                for file in os.listdir(os.path.join(pth,item)):
                    if file == "availdays.txt":
                        juldays = []
                        f = open(os.path.join(pth,item,file),'r')
                        for line in f:
                            line = line.rstrip()
                            julday = int(line)
                            juldays.append(julday)
                    try:
                        st = obspy.read(os.path.join(pth,item,file),headonly=True)
                        for tr in st:
                            s_julday = tr.stats.starttime.julday
                            e_julday = tr.stats.endtime.julday
                            if s_julday not in juldays:
                                juldays.append(s_julday)
                            if e_julday not in juldays:
                                juldays.append(e_julday)
                            net = tr.stats.network
                            sta = tr.stats.station
                            netsta = net+sta
                            if netsta not in netstas:
                                netstas.append(netsta)
                                stalons.append(sta_dict[net][sta][0])
                                stalats.append(sta_dict[net][sta][1])      
                            if tr.stats.starttime not in s_times:
                                s_times.append(tr.stats.starttime)
                            if tr.stats.endtime not in e_times:
                                e_times.append(tr.stats.endtime)
                    except:           # not seis file
                        pass
            if len(juldays) > 0:
                availdays.append(sorted(juldays))
    if len(availdays)>0 and len(availdays)!= len(netstas):
        raise Exception("len(availdays) not equal len(netstas)")
    setinfo["availdays"] = availdays
    setinfo["s_times"] = sorted(s_times)
    setinfo["e_times"] = sorted(e_times)
    setinfo["stalons"] = stalons
    setinfo["stalats"] = stalats
    setinfo["netstas"] = netstas
    setinfo["center"] = [np.mean(stalons),np.mean(stalats)]
    logging.info("extract set info programme done")
    f = open(os.path.join(parent_path,"setinfo.pkl"),'wb')
    pickle.dump(setinfo,f)
    f.close()
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

def get_st(net,sta,starttime,endtime,f_folder,pad=False,fill_value=0):
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
        if len(st)>3:
            st = st.merge()
        if pad == True:
            st.detrend("constant")
            st.detrend("linear")
        st.trim(starttime,endtime,pad=pad,fill_value=fill_value)
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



