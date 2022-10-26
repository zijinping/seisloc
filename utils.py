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
from distutils.sysconfig import get_python_lib
import matplotlib.pyplot as plt

def add_path():
    """
    Add current path to the active python library
    """
    print(">>> Add path to python library ...")
    pwd = os.getcwd()
    lib_path = get_python_lib()
    path_file = os.path.join(lib_path,'added.pth')
    with open(path_file,'a') as f:
        f.write(pwd)

    print("Done!")

@cuda.jit()
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
            else:
                continue
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
    f = open(os.path.join(parent_path,"setinfo.pkl"),'wb')
    pickle.dump(setinfo,f)
    f.close()
    logging.info("extract set info programme done")
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

def draw_vel(ax,dep_list,vel_list,color='k',linestyle='-',label=""):
    """
    Draw velocity line on the ax based on the depth list and velocity list
    """
    points_list = []
    points_list.append([dep_list[0],vel_list[0]])
    for i in range(1,len(dep_list)):
        points_list.append([dep_list[i],vel_list[i-1]])
        points_list.append([dep_list[i],vel_list[i]])
        
    points_list = np.array(points_list)
    line, = ax.plot(points_list[:,1],points_list[:,0],color=color,linestyle=linestyle,label=label)
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

def get_st(net,sta,starttime,endtime,f_folder,pad=False,fill_value=None):
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
        st.trim(starttime,endtime,pad=pad,fill_value=fill_value)
    return st

def get_st_SC(net,sta,starttime,endtime,f_folder,pad=False,fill_value=None):
    """
    A modified get_st function for Agency data which stores data by UTM-8 and days.
    """
    ymd_start = starttime.strftime("%Y%m%d")
    ymd_end = endtime.strftime("%Y%m%d")
    st = Stream()
    if ymd_start == ymd_end:
        try:
            f_folder = os.path.join(f_folder,ymd_start[:4],ymd_start[4:6],ymd_start[6:])
            st += obspy.read(os.path.join(f_folder,f"*{sta}*"))
        except:
            pass
    else:
        try:
            f_folder = os.path.join(f_folder,ymd_start[:4],ymd_start[4:6],ymd_start[6:])
            st += obspy.read(os.path.join(f_folder,f"*{sta}*"))
            f_folder = os.path.join(f_folder,ymd_end[:4],ymd_end[4:6],ymd_end[6:])
            st += obspy.read(os.path.join(f_folder,f"*{sta}*"))
        except:
            pass
    if len(st) == 0:
        pass
    else:
        if len(st)>3:
            st = st.merge()
        if st[0].stats.starttime-8*60*60>=endtime or \
           st[0].stats.endtime-8*60*60<=starttime:
            st = Stream()
        else:
            st.trim(starttime-8*60*60,endtime-8*60*60,pad=pad,fill_value=fill_value)
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

def time_interval_ticks(b_time,e_time,interval=1,unit='month',cut_to_round=True):
    """
    Generate month list from two UTCDateTime objects
    
    Parameters:
    |   b_time: begin time in obspy UTCDateTime format
    |   e_time: end time in obspy UTCDateTime format
    |     unit: interval unit, could be 'year','month','day','hour','minute',or 'second'
  cut_to_round: round off to the unit provided.default True.
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

def readfile(texfile):
    cont = []
    with open(texfile,'r') as f:
        for line in f:
            cont.append(line.rstrip())
    return cont

def writefile(cont,texfile):
    with open(texfile,'w') as f:
        for line in cont:
            f.write(line+"\n")

def bdy2pts(xmin,xmax,ymin,ymax):
    """
    Convert boundary to points for plot
    """
    points = []
    points.append([xmin,ymin])
    points.append([xmin,ymax])
    points.append([xmax,ymax])
    points.append([xmax,ymin])
    points.append([xmin,ymin])
    return np.array(points)

def matrix_show(*args,**kwargs):
    """
    Show matrix values in grids shape
    Parameters:cmap="cool",gridsize=0.6,fmt='.2f',label_data=True
    """
    ws = []
    H = 0
    str_count = 0
    ndarr_count = 0
    new_args = []
    for arg in args:
        if isinstance(arg,str):
            new_args.append(arg)
            continue
        if isinstance(arg,list):
            arg = np.array(arg)
        if len(arg.shape)>2:
            raise Exception("Only accept 2D array")
        if len(arg.shape) == 1:
            n = arg.shape[0]
            tmp = np.zeros((n,1))
            tmp[:,0] = arg.ravel()
            arg = tmp
        h,w = arg.shape
        if h>H:
            H=h
        ws.append(w)
        new_args.append(arg)
        ndarr_count += 1
    W = np.sum(ws)+len(ws)    # text+matrix+text+...+matrix+text
    if W<0:
        raise Exception("No matrix provided!")
        
    fmt = '.2f'
    grid_size = 0.6
    cmap = 'cool'
    label_data = True
    for arg in kwargs:
        if arg == "fmt":
            fmt = kwargs[arg]
        if arg == 'grid_size':
            grid_size = kwargs[arg]
        if arg == 'cmap':
            cmap = kwargs[arg]
        if arg == 'label_data':
            label_data = kwargs[arg]
    fig = plt.figure(figsize=(W*grid_size,H*grid_size))
    gs = fig.add_gridspec(nrows=H,ncols=W)
    
    wloop = 0
    matrix_id = 0
    for arg in new_args:
        if isinstance(arg,str):
            ax = fig.add_subplot(gs[0:H,wloop-1:wloop])
            ax.axis("off")
            ax.set_xlim(0,1)
            ax.set_ylim(0,H)
            ax.text(0.5,H/2,arg,horizontalalignment='center',verticalalignment='center')
        if isinstance(arg,np.ndarray):
            h,w = arg.shape
            hlow = int(np.round((H-h+0.01)/2))        # Find the height grid range
            hhigh = hlow+h
            wlow = wloop
            whigh = wlow+w
#            print("H: ",H,hlow,hhigh,"; W ",W,wlow,whigh)
            ax = fig.add_subplot(gs[hlow:hhigh,wlow:whigh])
            
            plt.pcolormesh(arg,cmap=cmap)
            for i in range(1,w):
                plt.axvline(i,color='k',linewidth=0.5)
            for j in range(1,h):
                plt.axhline(j,color='k',linewidth=0.5)
            if label_data:
                for i in range(h):
                    for j in range(w):
                        plt.text(j+0.5,i+0.5,format(arg[i,j],fmt),
                                 horizontalalignment='center',
                                 verticalalignment='center')
            plt.xlim(0,w)
            plt.ylim([h,0])
            plt.xticks([])
            plt.yticks([])
            wloop+=w+1
            matrix_id+=1
    plt.show()

def layer2linear_vel(inp_depths,inp_vels,linear_nodes):
    """
    Convert from layered 1-D model to linear velocity model, designed to conver from 
    the HYPOINVERSE model to tomoDD input model. It balances the average slowness.
    Parameters:
    | inp_depths: input depths, 1-D array/list
    | inp_vels: input velocities, 1-D array/list
    | linear_nodes: the depth nodes of output velocities
    
    Return:
    | average vlocity list correponding to the input velocity nodes

    Example:
    >>> inp_depths = [ -1,0.5,1.,1.5,2.13,4.13,6.9,10.07,15.93,17.,18.,27.,31.89]
    >>> inp_vels = [3.67,4.7,5.38,5.47,5.5,5.61,5.73,6.19,6.23,6.31,6.4,6.45,6.5]
    >>> tomo_deps = [0,1.5,3.0,4.5,6.0,7.5,9.0,10.5,12,18,24]
    >>> layer2linear_vel(inp_depths,inp_vels,tomo_deps)
    """
    
    #----------- quality control --------------------------------
    if linear_nodes[-1]>inp_depths[-1]:
        raise Exception("linear_nodes[-1] should less than inp_depths[-1]")
    if inp_depths.shape[0] != inp_vels.shape[0]:
        raise Eception("Different length of input depths quantity and velocity values")
    
    #----------- load in data ------------------------------------
    pxnodes = []
    pynodes = []
    for i in range(inp_depths.shape[0]):
        if i == 0:
            pxnodes.append(inp_vels[i])
            pynodes.append(inp_depths[i])
        else:
            pxnodes.append(inp_vels[i-1])
            pxnodes.append(inp_vels[i])
            pynodes.append(inp_depths[i])
            pynodes.append(inp_depths[i])

    #----------- calculate the mean velocity ----------------------
    avg_vels = []
    for i in range(len(linear_nodes)):
        dep = linear_nodes[i]
        #---------------get left and right range--------------
        if i == 0:                                   # first node
            dep_il = linear_nodes[i]
            dep_ir = (linear_nodes[i+1]+linear_nodes[i])/2
        elif i == len(linear_nodes) - 1:             # last node
            dep_il = (linear_nodes[i-1]+linear_nodes[i])/2
            dep_ir = linear_nodes[i]+(linear_nodes[i]-linear_nodes[i-1])/2
        else:
            dep_il = (linear_nodes[i-1]+linear_nodes[i])/2
            dep_ir = (linear_nodes[i+1]+linear_nodes[i])/2

        slw_list = []                                # slow list
        embrace_first = False
        for j in range(len(pynodes)):
            pynode = pynodes[j]
            velocity = pxnodes[j]                    # velocity

            if pynode >=dep_il: # consider velocity node within the range
                if embrace_first == False:
                    slw_list.append([dep_il,1/velocity]) # embrace the left node
                    embrace_first = True

                end_velocity = velocity
                if pynode>dep_ir:
                    break
                slw_list.append([pynode,1/velocity])
        slw_list.append([dep_ir,1/end_velocity])      # embrace the right node

        # calculate average slowness
        trav = 0                                      # travel time        
        for k in range(len(slw_list)-1):
            dep_a = slw_list[k][0]
            slw_a = slw_list[k][1]
            dep_b = slw_list[k+1][0]
            slw_b = slw_list[k+1][1]
            if (dep_b - dep_a)==0 or slw_a==slw_b:
                trav += (dep_b -dep_a)*slw_a
            else:
                raise Exception("(dep_b - dep_a)==0 or vel_a==vel_b")
        avg_vel = (dep_ir-dep_il)/trav
        avg_vels.append(avg_vel)
        
    return avg_vels

def read_line_values(line,vtype="float"):
    values = []
    _values = line.strip().split()
    for _value in _values:
        if vtype == "float":
            values.append(float(_value))
        elif vtype == "int":
            values.append(int(_value))
        else:
            raise Exception("Unrecognized value type: ",vtype)
    return values
