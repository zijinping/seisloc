# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
#    Author: ZI, Jinping
# Institute: The Seismology Lab, the Chinese University of Hong Kong
# Reference: Yang, H. L. Zhu, and R. Chu(2009), Fault-Plane Determination of
#            the 18 April 2008 Mt. Carmel, Illinois, Earthquake by Detection
#            and Relocating Aftershocks, BSSA, 99(6), 3413-3420
#            doi:10.1785/0120090038
#   History: 
#       2021-01-25 Initial coding
#-----------------------------------------------------------------------------

import obspy
from obspy import Stream,UTCDateTime
from obspy.io.sac import SACTrace
from obspy.geodetics import gps2dist_azimuth
from math import sqrt
import numpy as np
import sys
import os
import re
import glob
import shutil
from numba import jit
from numba.typed import List
from tqdm import tqdm
from seisloc.hypoinv import load_sum_evid,load_sum_evstr
import multiprocessing as mp

def wf_scc(st1,st2,tb,te,maxShift,marker='t0',bestO=False):
    """
    Sliding-window cross-correlation between template and target waveform
    Reference time should be the event origin time
    reference: Yang et al.,2009, BSSA.

    Parameters
    -----------
     tmplt_st: template waveform of one station
       sta_st: target waveform of the same station
        tb,te: begin and end time window for waveform center on the corresponding marker
     maxShift: maximum shift value for sliding, in seconds         
        bestO: bool. If True, output the best fit origin time

    Return
    -----------
        ccmax: maximum cross-correlation coefficient
        aamax: amplitude ratio at ccmax
     bestTime: the best arrival time
      cc_list: cross-correlation values
    """
    tmplt_st = st1.copy()
    sta_st = st2.copy()
    tmplt_st.sort()
    sta_st.sort()
    
    assert len(tmplt_st) == len(sta_st)
    ncom = len(tmplt_st)    # number of component, n = 3 means 3 components cross-correlation
    assert ncom in [1,3]
    
    assert tmplt_st[0].stats.delta == sta_st[0].stats.delta
    delta = tmplt_st[0].stats.delta
    
    markerTime1 = tmplt_st[0].stats.sac[marker]
    sac1 = SACTrace.from_obspy_trace(tmplt_st[0])
    reftime1 = sac1.reftime
    tmplt_st.trim(reftime1+markerTime1+tb,reftime1+markerTime1+te)
    tmplt_st.detrend("constant")
    
    markerTime2 = sta_st[0].stats.sac[marker]
    sac2 = SACTrace.from_obspy_trace(sta_st[0])
    reftime2 = sac2.reftime
    sta_st.trim(reftime2+markerTime2+tb-maxShift,reftime2+markerTime2+te+maxShift)
    sta_st.detrend("constant")
    assert sac1.o == 0
    assert sac2.o == 0
    
    tmplt_data = List()
    sta_data = List()
    cc_list = List()
    if ncom == 3:
        for i in range(3):
            tmplt_data.append(tmplt_st[i].data)
            sta_data.append(sta_st[i].data)
    elif ncom == 1:
        tmplt_data.append(tmplt_st[0].data)
        sta_data.append(sta_st[0].data)
    ccmax,aamax,i0,cc_list = data_scc(tmplt_data,sta_data,ncom)
    bestTime = markerTime2 + (i0-(len(cc_list)-1)/2)*delta
    
    if bestO:
        bestTime = (i0-(len(cc_list)-1)/2)*delta
    
    return ccmax,aamax,bestTime,cc_list

@jit(nopython=True)
def data_scc(tmplt_data,st_data,ncom):
    """
    Sliding-window cross-correlation between template and target waveform
    reference: Yang, H. et al.,2009, BSSA.

    Parameters
    -----------
    tmplt_data: template waveform of one station
       st_data: target waveform of the same station
          ncom: number of component, n = 3 means 3 components cross-correlation

    return
    ----------
         ccmax: maximum cross-correlation coefficient
         aamax: amplitude ratio at ccmax
            i0: the shifting index at ccmax
       cc_list: the list of cross-correlation coefficient in each step
    """

    normMaster = 0.
    ic = 0
    mm = len(tmplt_data[0])
    while ic < ncom:
        k = 0
        while k < mm:
            normMaster += tmplt_data[ic][k]*tmplt_data[ic][k]
            k += 1
        ic += 1
    normMaster = sqrt(normMaster)
    
    npts = len(st_data[0])
    norm = 0
    j = 0
    while j < mm-1:
        ic=0
        while ic < ncom:
            norm += st_data[ic][j]*st_data[ic][j]
            ic+=1
        j=j+1
    ccmax = -1
    aamax = -1
    j=0
    cc_list=[]
    while j<=npts-mm:
        cc = 0
        ic = 0
        while ic < ncom:
            norm+=st_data[ic][j+mm-1]*st_data[ic][j+mm-1]
            k=0
            while k <mm:
                cc += tmplt_data[ic][k]*st_data[ic][j+k]
                k+=1
            ic+=1
        aa = sqrt(norm)/normMaster
        cc = cc*aa/norm #cc = <f|g>/sqrt((f|f)(g|g))
        if(cc>=ccmax):
            ccmax = cc
            aamax = aa
            i0 = j
        cc_list.append(cc)
        ic = 0
        while ic < ncom:
            norm -= st_data[ic][j]*st_data[ic][j]
            ic+=1
        j=j+1
    return ccmax,aamax,i0,cc_list

def gen_dtcc(netsta_list=None,sum_file="out.sum",work_dir="./",cc_threshold=0.7,min_link=4,max_dist=4):
    '''
    This function generate dt.cc.* files from the output of SCC results
    
    Parameters:
     netsta_list: list of netstas to be processed, if set None, process stations under work_dir
        sum_file: summary(*.sum) file generated by HYPOINVERSE
        work_dir: the directory of mp_scc results
    cc_threshold: threshold value of cross_correlation
        min_link: minumum links to form an event pair
        max_dist: maximum distance accepted to form an event pair, unit km
    '''
    sum_rev= load_sum_evstr(sum_file) # dictionary {evid: [e_lon,e_lat,e_dep,e_mag]}
    sum_dict= load_sum_evid(sum_file)    # dictionary {"YYYYmmddHHMMSSff":[e_lon,e_lat,e_dep,e_mag]}
    work_dir = os.path.abspath(work_dir)
    evid_list = []                  # event list included by scc results
    to_cc_list = []                 # event list included in the ouput dt.cc.* results
    
    # Remove existing dt.cc files 
    cc_files = glob.glob(os.path.join(work_dir,"dt.cc*"))
    if len(cc_files)>0:
        raise Exception("dt.cc files exsited!")
    if netsta_list == None:
        netsta_list = []
        for folder in os.listdir(work_dir):
            if folder[-2:] in ["_P","_S"]:
                netsta = re.split("_",folder)[0]
                if netsta not in netsta_list:
                    netsta_list.append(netsta)
    print(">>> Loading in scc results ...")
    for netsta in tqdm(netsta_list):                                # Loop for station
        for pha in ["P","S"]:                                 # Loop for phases
            netsta_pha = netsta+"_"+pha
            globals()[netsta_pha+"_cc_dict"] = {}                # Initiate dictionary
            netsta_pha_path = os.path.join(work_dir,netsta_pha)
            if not os.path.exists(netsta_pha_path):
                print(netsta_pha_path+" not exist!")
                continue
            for file in os.listdir(netsta_pha_path):
                if file[-3:]!=".xc":                          # none scc results file
                    continue
                with open(os.path.join(netsta_pha_path,file),'r') as f:
                    for line in f:
                        line = line.rstrip()
                        path1,arr1,_,path2,arr2,_,_cc,_aa=re.split(" +",line.rstrip())
                        tmp = os.path.split(path1)[0]
                        eve_folder1 = os.path.split(tmp)[1]
                        evid1 = sum_rev[eve_folder1][0]
                        arr1 =float(arr1)                     # arrival time
                        tmp = os.path.split(path2)[0]
                        eve_folder2 = os.path.split(tmp)[1]
                        evid2 = sum_rev[eve_folder2][0]
                        if evid1 not in evid_list:
                            evid_list.append(evid1)
                        if evid2 not in evid_list:
                            evid_list.append(evid2)
                        arr2 = float(arr2)                    # arrival time
                        cc = float(_cc)                        # cross correlation coefficient
                        aa = float(_aa)                        # amplitude ratio
                        if cc >=cc_threshold:
                            try:
                                globals()[netsta_pha+"_cc_dict"][evid1][evid2]=[arr1,arr2,cc,aa]
                            except:
                                globals()[netsta_pha+"_cc_dict"][evid1]={} # Initiation
                                globals()[netsta_pha+"_cc_dict"][evid1][evid2]=[arr1,arr2,cc,aa]
    evid_list.sort()
    print("<<< Loading complete! <<<")
    
    print(">>> Preparing dt.cc files ...")
    ##----------------
    for i,evid1 in enumerate(evid_list):
        print(evid1,"  ",end='\r')
        evid1_evlo = sum_dict[evid1][1]
        evid1_evla = sum_dict[evid1][2]
        out_index = int(i/6000)# Every 6k events preserve in a seperate dt.cc.* file.
                               # to avoid extreme large out file size.
        for evid2 in evid_list[i+1:]:
            evid2_evlo = sum_dict[evid2][1]
            evid2_evla = sum_dict[evid2][2]
            dist,_,_ = gps2dist_azimuth(evid1_evla,evid1_evlo,evid2_evla,evid2_evlo)
            if dist/1000>max_dist:                          # discard large distance events
                continue
            link_cc=[]
            for netsta in netsta_list:                            # Loop for stations
                for pha in ["P","S"]:                       # Loop for phases
                    netsta_pha = netsta+"_"+pha               
                    try:
                        arr1,arr2,cc,aa = globals()[netsta_pha+"_cc_dict"][evid1][evid2]
                        link_cc.append([netsta,arr1-arr2,cc,pha])
                    except:
                        continue
            if len(link_cc)>=min_link:
                if evid1 not in to_cc_list:
                    to_cc_list.append(evid1)
                if evid2 not in to_cc_list:
                    to_cc_list.append(evid2)
                cc_file = os.path.join(work_dir,"dt.cc."+f"{out_index}")
                with open(cc_file,'a') as f:                # Write in results
                    f.write(f"# {format(evid1,'5d')} {format(evid2,'5d')} 0\n")
                    for record in link_cc:
                        f.write(f"{format(record[0],'<7s')} {format(record[1],'7.4f')} {format(record[2],'5.3f')} {record[3]}\n")
                f.close()
    print(">>> Number of events in dt.cc is: ",len(to_cc_list))
    
    cont = []
    cc_files = glob.glob(os.path.join(work_dir,"dt.cc*"))
    cc_files.sort()
    for cc_file in cc_files:
        with open(cc_file,'r') as f:
            for line in f:
                cont.append(line)
        f.close()
        os.remove(cc_file)
    with open(os.path.join(work_dir,"dt.cc"),'w') as f:
        for line in cont:
            f.write(line)
    f.close()    
    print("<<< dt.cc files generated! <<<")




def scc_input_load_arr(tarBase,markerP="P",markerS="S"):
    """
    Load P&S travel time from event waveforms
    """
    arrDict={}
    _days = os.listdir(os.path.join(tarBase,'eve_wf_bp'))
    _days.sort()
    for _day in _days:
        _eves = os.listdir(os.path.join(tarBase,'eve_wf_bp',_day))
        _eves.sort()
        for _eve in _eves:
            _eveDir = os.path.join(tarBase,'eve_wf_bp',_day,_eve)
            for sac in os.listdir(_eveDir):
                if sac[-1]!='z':
                    continue
                sacPth = os.path.join(_eveDir,sac)
                st = obspy.read(sacPth,headonly=True)
                sta = st[0].stats.station
                if hasattr(st[0].stats.sac,markerP):
                    travTime = getattr(st[0].stats.sac,markerP)
                    if f"{sta}_P" not in arrDict:
                        arrDict[f"{sta}_P"] = []
                    _str1 = os.path.join("eve_wf_bp",_day,_eve,sta+".z")
                    _str2 = f"  {format(travTime,'5.2f')}  1\n"
                    arrDict[f"{sta}_P"].append(_str1+_str2)
                    
                if hasattr(st[0].stats.sac,markerS):
                    travTime = getattr(st[0].stats.sac,markerS)
                    if f"{sta}_S" not in arrDict:
                        arrDict[f"{sta}_S"] = []
                    _str1 = os.path.join("eve_wf_bp",_day,_eve,sta+".z")
                    _str2 = f"  {format(travTime,'5.2f')}  1\n"
                    arrDict[f"{sta}_S"].append(_str1+_str2)
    return arrDict

def scc_input_write_arr(arrDir,arrDict):
    for key in arrDict.keys():
        arrFile = key+".arr"
        arrFilePth = os.path.join(arrDir,arrFile)
        with open(arrFilePth,'w') as f:
            for line in arrDict[key]:
                f.write(line)

def scc_input_wf_bp(srcEveDir,tarEveDir,freqmin,freqmax,zerophase=True):
    if not os.path.exists(tarEveDir):
        os.mkdir(tarEveDir)
    for sac in os.listdir(srcEveDir):
        st = obspy.read(os.path.join(srcEveDir,sac))
        chn = st[0].stats.channel
        sta = st[0].stats.station
        st.detrend("linear"); st.detrend("constant")
        st.filter("bandpass",freqmin=freqmin,freqmax=freqmax,zerophase=True)
        if chn[-1]=="N":
            st[0].write(os.path.join(tarEveDir,f"{sta}.r"),format="SAC")
        if chn[-1]=="E":
            st[0].write(os.path.join(tarEveDir,f"{sta}.t"),format="SAC")
        if chn[-1]=="Z":
            st[0].write(os.path.join(tarEveDir,f"{sta}.z"),format="SAC")    
    
def scc_input(srcWfBase,tarBase,freqmin,freqmax,markerP="a",markerS="t0",parallel=False,parallelCores=10,zerophase=True):
    """
    Prepare the sliding window cross-correlation input files
    Parameters:
      srcWfBase: The source waveform folder
      tarBase: The target multiprocessing project folder
    """
    if not os.path.exists(tarBase):
        os.mkdir(tarBase)
    arrDir = os.path.join(tarBase,'arr_files')
    if os.path.exists(arrDir):
        print(" arrDir: 'arr_files' exsited and will be removed!")
        shutil.rmtree(arrDir)
    os.mkdir(arrDir)
    bpwfDir = os.path.join(tarBase,'eve_wf_bp')
    if os.path.exists(bpwfDir):
        print("bpwfDir: 'eve_wf_bp' exsited and will be removed!")
        shutil.rmtree(os.path.join(tarBase,'eve_wf_bp'))
    os.mkdir(bpwfDir)
    _days = os.listdir(srcWfBase)
    _days.sort()
    for _day in _days:
        os.mkdir(os.path.join(tarBase,'eve_wf_bp',_day))
        _eves = os.listdir(os.path.join(srcWfBase,_day))
        _eves.sort()
        if parallel: pool = mp.Pool(processes=parallelCores)
        for _eve in _eves:
            srcEveDir = os.path.join(srcWfBase,_day,_eve)
            if not os.path.isdir(srcEveDir):
                continue
            tarEveDir = os.path.join(tarBase,'eve_wf_bp',_day,_eve)
            if parallel: pool.apply_async(scc_input_wf_bp,args=(srcEveDir,tarEveDir,freqmin,freqmax,zerophase))
            else: scc_input_wf_bp(srcEveDir,tarEveDir,freqmin=freqmin,freqmax=freqmax,zerophase=zerophase)
        if parallel: pool.close(); pool.join()
      
    print("Waveform bandpass finished!")
#----------- Generate arr files ----------------------------------------    
    print(">>> Now prepare arrival files")
    print(">>> Loading arrivals from tarWfDir 'eve_bp_wf' ")
    arrDict = scc_input_load_arr(tarBase,markerP=markerP,markerS=markerS)

    print(">>> Writing arrival files ")
    scc_input_write_arr(arrDir,arrDict)

