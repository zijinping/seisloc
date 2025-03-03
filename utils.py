# -*- coding: utf-8 -*-
#-------------------------------------
#     Author: Jinping ZI
#
# Revision History
#     2021-01-24 Initiate coding
#     2025-02-07 Update content
#-------------------------------------
import re
import os
import time
import json
import obspy
import logging
import functools
import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime
from seisloc.sta import load_sta
from distutils.sysconfig import get_python_lib

def timer(func):
    functools.wraps(func)
    def wrapper_timer(*args,**kwargs):
        startTime = time.perf_counter()
        func(*args,**kwargs)
        endTime = time.perf_counter()
        runTime = endTime - startTime
        print(f"Finished {func.__name__!r} in {runTime:.4f} secs")
    return wrapper_timer

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

def init_logger(logPth,file_level=logging.DEBUG,stream_level=logging.INFO):
    '''
    Parameters:
        log_file: file to save the log information
        file_level: level for file writing
        stream_level: level for stream writing
    '''
    logDir = os.path.dirname(logPth)
    if logDir!="" and not os.path.exists(logDir):
        os.mkdir(logDir)
        
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(logPth,mode='w')
    fh.setLevel(file_level)
    formatter = logging.Formatter('%(asctime)s-%(filename)s-%(levelname)s: %(message)s')
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(stream_level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


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

def find_closest(array,value):
    """
    find the nearest value.
    Return:
    | idx
    | diff: array[id]-value
    """
    if type(array) != np.ndarray:
        array=np.array(array)
    idx=(np.abs(array-value)).argmin()
    diff=array[idx]-value
    return idx,diff

def readfile(texfile):
    cont = []
    with open(texfile,'r') as f:
        for line in f:
            cont.append(line.strip())
    return cont

def writefile(cont,texfile):
    with open(texfile,'w') as f:
        for line in cont:
            f.write(line+"\n")

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

def load_para(paraFile):
    """
    load a parameter file and return a dictionary
    """
    paraDict = {}
    with open(paraFile,'r') as f:
        for line in f:
            line = line.strip()
            if line=="":
                continue
            if line[0]=="#":
                continue
            splits = re.split("=",line,maxsplit=1)
            para = splits[0]
            val = splits[1]
            val = val.strip("\"")
            val = val.strip("\'")
            if re.match("^[+-]*[0-9]+\.*[0-9]*$",val):
                if re.match("^[+-]*\d+$",val):
                    val = int(val)
                else:
                    val = float(val)
            paraDict[para]=val

    return paraDict
