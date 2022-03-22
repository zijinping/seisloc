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
#
#     Usage: python scc.py [-Ccc] [-E] [-Mn] [-O] [-Tlength] [-Wt1/t2[/maxShift]]
#            -C: cross-correlation threshold (default = 0.7)
#-----------------------------------------------------------------------------

import obspy
from obspy import Stream,UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from math import sqrt
import numpy as np
import sys
import os
import re
import glob
import shutil
from numba import jit
from tqdm import tqdm
from seisloc.hypoinv import load_sum_evid,load_sum_evstr

def wf_scc(tmplt_st,sta_st,ncom):
    """
    Sliding-window cross-correlation between template and target waveform
    reference: Yang et al.,2009, BSSA.

    Parameters
    -----------
    tmplt_st: template waveform of one station
       sta_st: target waveform of the same station
         ncom: number of component, n = 3 means 3 components cross-correlation

    Return
    -----------
        ccmax: maximum cross-correlation coefficient
        aamax: amplitude ratio at ccmax
        i0: the shifting index at ccmax
    """

    tmplt_data = []
    sta_data = []
    if ncom == 3:
        tmplt_data.append([tmplt_st.select(component="*N")[0].data])
        tmplt_data.append([tmplt_st.select(component="*E")[0].data])
        tmplt_data.append([tmplt_st.select(component="*Z")[0].data])
        sta_data.append([sta_st.select(component="*N")[0].data])
        sta_data.append([sta_st.select(component="*E")[0].data])
        sta_data.append([sta_st.select(component="*Z")[0].data])
    elif ncom == 1:
        tmplt_data.append([tmplt_sta[0].data])
    dt = temp[0].stats.delta
    ccmax,aamax,i0,cc_list = data_scc(tmplt_data,st_data,ncom)

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

def eve_wf_bp(freqmin,freqmax,
              src_folder="eve_wf",
              tar_folder="eve_wf_bp",
              taper_percentage=0.05,
              zerophase=True):
    """
      bandpass event waveform
      src_folder: The source waveform folder
      tar_folder: The target waveform folder
    """
    try:
        os.mkdir(tar_folder)
    except:
        raise Exception("target folder existed!")
    _days = os.listdir(src_folder)
    _days.sort()
    for _day in _days:
        os.mkdir(os.path.join(tar_folder,_day))
        _eves = os.listdir(os.path.join(src_folder,_day))
        _eves.sort()
        for _eve in _eves:
            _eve_folder = os.path.join(tar_folder,_day,_eve)
            if not os.path.exists(_eve_folder):
                os.mkdir(_eve_folder)
            for sac in os.listdir(os.path.join(src_folder,_day,_eve)):
                st = obspy.read(os.path.join(src_folder,_day,_eve,sac))
                chn = st[0].stats.channel
                sta = st[0].stats.station
                net = st[0].stats.network
                st.detrend("linear"); st.detrend("constant")
                st.taper(max_percentage=taper_percentage)
                st.filter("bandpass",freqmin=freqmin,freqmax=freqmax,zerophase=zerophase)
                if chn[-1]=="N":
                    st[0].write(os.path.join(_eve_folder,f"{sta}.r"),format="SAC")
                if chn[-1]=="E":
                    st[0].write(os.path.join(_eve_folder,f"{sta}.t"),format="SAC")
                if chn[-1]=="Z":
                    st[0].write(os.path.join(_eve_folder,f"{sta}.z"),format="SAC")

def gen_scc_input(wf_folder,arr_folder="arr_files"):
    """
    Prepare the sliding window cross-correlation input files
    Parameters:
      wf_folder: The waveform data folder
      arr_folder: The target output folder
    """
    try:
        shutil.rmtree(arr_folder)
    except:
        pass
    os.mkdir(arr_folder)
    _days = os.listdir(wf_folder)
    _days.sort()
    for _day in _days:
        _eves = os.listdir(os.path.join(wf_folder,_day))
        _eves.sort()
        for _eve in _eves:
            _eve_folder = os.path.join(wf_folder,_day,_eve)
            if not os.path.exists(_eve_folder):
                os.mkdir(_eve_folder)
            for sac in os.listdir(os.path.join(wf_folder,_day,_eve)):
                st = obspy.read(os.path.join(wf_folder,_day,_eve,sac),headonly=True)
                chn = st[0].stats.channel
                sta = st[0].stats.station
                net = st[0].stats.network
                if chn[-1]=="Z":
                    if hasattr(st[0].stats.sac,'a'):
                        a = st[0].stats.sac.a
                        arr_file = os.path.join(arr_folder,f"{net}{sta}_P.arr")
                        with open(arr_file,'a') as f:
                            f.write(os.path.join(wf_folder,_day,_eve,sta+".z"))
                            f.write(f"  {format(a,'5.2f')}  1\n")
                    if hasattr(st[0].stats.sac,'t0'):
                        t0 = st[0].stats.sac.t0
                        arr_file = os.path.join(arr_folder,f"{net}{sta}_S.arr")
                        with open(arr_file,'a') as f:
                            f.write(os.path.join(wf_folder,_day,_eve,sta+".z"))
                            f.write(f"  {format(t0,'5.2f')}  1\n")
                        f.close()

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
    for cc_file in cc_files:
        os.remove(cc_file)
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
