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

import os
import re
import sys
import glob
import shutil
import obspy
import numpy as np
import pandas as pd
import multiprocessing as mp
from obspy import Stream,UTCDateTime
from obspy.io.sac import SACTrace
from obspy.geodetics import gps2dist_azimuth
from math import sqrt
from numba import jit
from numba.typed import List
from tqdm import tqdm
from seisloc.hypoinv import load_sum_evid,load_sum_evstr
from seisloc.sac import read_sac_ref_time

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
     bestTime: the best arrival time of st2(sta_st)
      cc_list: cross-correlation values
    """
    tmplt_st = st1.copy()
    tmplt_st.sort()
    sta_st = st2.copy()
    sta_st.sort()
    #======= quality control ==============
    assert len(tmplt_st) == len(sta_st)
    ncom = len(tmplt_st)    # number of component, n = 3 means 3 components cross-correlation
    assert ncom in [1,3]

    assert tmplt_st[0].stats.delta == sta_st[0].stats.delta
    delta = tmplt_st[0].stats.delta

    tb = int(tb/delta)*delta
    te = int(te/delta)*delta
    maxShift = int(maxShift/delta)*delta

    reftime1 = read_sac_ref_time(tmplt_st[0])
    npts1 = tmplt_st[0].stats.npts
    for tr in tmplt_st:
        assert tr.stats.sac.o==0
        assert read_sac_ref_time(tr) == reftime1
        assert tr.stats.delta == delta
        assert tr.stats.npts == npts1

    reftime2 = read_sac_ref_time(sta_st[0])
    npts2 = sta_st[0].stats.npts
    for tr in sta_st:
        assert tr.stats.sac.o==0
        assert read_sac_ref_time(tr) == reftime2
        assert tr.stats.delta == delta
        assert tr.stats.npts == npts2

    markerTime1 = tmplt_st[0].stats.sac[marker]
    deltab1 = tmplt_st[0].stats.sac.b - int(tmplt_st[0].stats.sac.b/delta)*delta
    cutb1 = markerTime1+tb+deltab1
    cute1 = markerTime1+te+deltab1
    assert cutb1>=tmplt_st[0].stats.sac.b, "try increase tb!"
    assert cute1<=tmplt_st[0].stats.sac.e, "try decrease te!"

    markerTime2 = sta_st[0].stats.sac[marker]
    deltab2 = sta_st[0].stats.sac.b - int(sta_st[0].stats.sac.b/delta)*delta
    cutb2 = markerTime2+tb+deltab2
    cute2 = markerTime2+te+deltab2
    assert cutb2>=sta_st[0].stats.sac.b, "try increase tb or decrease maxShift!"
    assert cute2<=sta_st[0].stats.sac.e, "try decrease tb or decrease maxShift!"

    tmplt_st.trim(reftime1+cutb1,reftime1+cute1)
    
    sta_st.trim(reftime2+cutb2-maxShift,reftime2+cute2+maxShift)
    
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
        if normMaster == 0:
            aa = 0
        else:
            aa = sqrt(norm)/normMaster
        if norm == 0:
            cc = 0
        else:
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

def gen_dtcc(staLst=None,sumFile="out.sum",workDir="./",cc_threshold=0.7,minLink=4):
    
    dtccDir = os.path.join(workDir,"0dtcc")
    if not os.path.exists(dtccDir):
        os.mkdir(dtccDir)
    if os.path.exists(os.path.join(dtccDir,'dt.cc')):
        raise Exception(f"dt.cc existed in {os.path.join(dtccDir,'dt.cc')}")
    for item in os.listdir(dtccDir):
        if item[-3]!="csv":
            continue
        itemPth = os.path.join(dtccDir,item)
        if not os.path.isdir(itemPth):
            os.remove(itemPth)
    
    convert_csv(staLst=staLst,sumFile=sumFile,workDir=workDir)
    write_dtcc(dtccDir=dtccDir,minLink=4)

def convert_csv(staLst=None,sumFile="out.sum",workDir="./",cc_threshold=0.7,minLink=4,max_dist=4):
    '''
    This function generate dt.cc.* files from the output of SCC results
    
    Parameters:
     staLst: list of stas to be processed, if set None, process stations under workDir
        sumFile: summary(*.sum) file generated by HYPOINVERSE
        workDir: the directory of mp_scc results
    cc_threshold: threshold value of cross_correlation
        minLink: minumum links to form an event pair
        max_dist: maximum distance accepted to form an event pair, unit km
    '''
    sum_rev= load_sum_evstr(sumFile) # dictionary {evid: [e_lon,e_lat,e_dep,e_mag]}
    sum_dict= load_sum_evid(sumFile)    # dictionary {"YYYYmmddHHMMSSff":[e_lon,e_lat,e_dep,e_mag]}
    workDir = os.path.abspath(workDir)
    to_cc_list = []                 # event list included in the ouput dt.cc.* results

    if staLst == None:
        staLst = []
        for folder in os.listdir(workDir):
            if folder[-2:] in ["_P","_S"]:
                sta = re.split("_",folder)[0]
                if sta not in staLst:
                    staLst.append(sta)
    print(">>> Convert scc results ...")
    for sta in tqdm(staLst):                                # Loop for station
        dts = []
        for pha in ["P","S"]:                                 # Loop for phases
            sta_pha = sta+"_"+pha
            globals()[sta_pha+"_cc_dict"] = {}                # Initiate dictionary
            sta_pha_path = os.path.join(workDir,sta_pha)
            if not os.path.exists(sta_pha_path):
                print(sta_pha_path+" not exist!")
                continue
            for file in os.listdir(sta_pha_path):
                if file[-3:]!=".xc":                          # none scc results file
                    continue
                with open(os.path.join(sta_pha_path,file),'r') as f:
                    for line in f:
                        line = line.rstrip()
                        path1,arr1,_,path2,arr2,_,_cc,_aa=re.split(" +",line.rstrip())
                        arr1 =float(arr1)
                        arr2 = float(arr2)                    # arrival time
                        cc = float(_cc)                        # cross correlation coefficient
                        aa = float(_aa)                        # amplitude ratio
                        tmp = os.path.split(path1)[0]
                        eve_folder1 = os.path.split(tmp)[1]
                        evid1 = sum_rev[eve_folder1][0]
                        tmp = os.path.split(path2)[0]
                        eve_folder2 = os.path.split(tmp)[1]
                        evid2 = sum_rev[eve_folder2][0]
                        if cc >=cc_threshold:
                            dts.append([evid1,evid2,sta,pha,arr1-arr2,cc])
        if len(dts)>=1:
            df = pd.DataFrame(dts,columns=["evid1","evid2","sta","pha","dt","cc"])
            df.to_csv(os.path.join(workDir,"0dtcc",f"{sta}.csv"),index=False)
    print("<<< Conversion complete! <<<")

def write_dtcc(dtccDir="0dtcc",minLink = 4):
    print(">>> Loading csv files ... ")
    dfAll = pd.DataFrame([],columns=["evid1","evid2","sta","pha","dt","cc"])
    for csv in tqdm(os.listdir(dtccDir)):
        if csv[-3:] != "csv" or csv[0] == ".":
            continue
        csvPth = os.path.join(dtccDir,csv)
        df = pd.read_csv(csvPth)
        dfAll = pd.concat([dfAll,df],ignore_index=True)
    print(dfAll)
    dfAll['evid1'] = dfAll['evid1'].astype(int)
    dfAll["evid2"] = dfAll["evid2"].astype(int)
    print(">>> Writing dt.cc file ...")
    dfAllSort = dfAll.sort_values(["evid1","evid2",'sta','pha'])
    evid1s = sorted(np.unique(dfAllSort['evid1']))
    evid1Use = -1 # set a unreasonable value
    evid2Use = -2 # set a unreasonable value
    lines = []
    f = open(os.path.join(dtccDir,'dt.cc'),'w')
    for i, df in tqdm(dfAllSort.iterrows()):
        if df.evid1 != evid1Use or df.evid2 != evid2Use:
            if len(lines) > minLink+1: # phases + headlines
                for line in lines:
                    f.write(line+"\n")
            evid1Use = df.evid1
            evid2Use = df.evid2
            # Initiate the line
            lines = [f"# {format(df.evid1,'5d')} {format(df.evid2,'5d')} 0"]
        if df.evid1 == evid1Use and df.evid2 == evid2Use:
            lines.append(f"{format(df['sta'],'<7s')} {format(df['dt'],'7.4f')} {format(df['cc'],'5.3f')} {df['pha']}")
    if len(lines) > minLink + 1:
        for line in lines:
            f.write(line+"\n")
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

def scc_input_wf_bp(srcEveDir,tarEveDir,freqmin,freqmax,zerophase=True,excludeNetstas=[]):
    if not os.path.exists(tarEveDir):
        os.mkdir(tarEveDir)
    for sac in os.listdir(srcEveDir):
        sacPth = os.path.join(srcEveDir,sac)
        try:
            st = obspy.read(sacPth)
        except:
            print("Fail to read the sac file ",sacPth)
            continue
        net = st[0].stats.network
        sta = st[0].stats.station
        chn = st[0].stats.channel
        if net+sta in excludeNetstas:
            continue
        st.detrend("linear"); st.detrend("constant")
        st.filter("bandpass",freqmin=freqmin,freqmax=freqmax,zerophase=True)
        if chn[-1]=="N":
            st[0].write(os.path.join(tarEveDir,f"{sta}.r"),format="SAC")
        if chn[-1]=="E":
            st[0].write(os.path.join(tarEveDir,f"{sta}.t"),format="SAC")
        if chn[-1]=="Z":
            st[0].write(os.path.join(tarEveDir,f"{sta}.z"),format="SAC")    
    
def scc_input(srcWfBase,tarBase,freqmin,freqmax,markerP="a",markerS="t0",zerophase=True,excludeNetstas=[]):
    """
    Prepare the sliding window cross-correlation input files
    Parameters:
      srcWfBase: The source waveform folder
      tarBase: The target multiprocessing project folder
    """
    arrDir = os.path.join(tarBase,'arr_files')
    bpwfDir = os.path.join(tarBase,'eve_wf_bp')
    if not os.path.exists(tarBase):
        os.mkdir(tarBase)
    if os.path.exists(arrDir):
        print(" arrDir: 'arr_files' exsited and will be removed!")
        shutil.rmtree(arrDir)
    os.mkdir(arrDir)
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
        for _eve in _eves:
            srcEveDir = os.path.join(srcWfBase,_day,_eve)
            if not os.path.isdir(srcEveDir):
                continue
            tarEveDir = os.path.join(tarBase,'eve_wf_bp',_day,_eve)
            scc_input_wf_bp(srcEveDir,tarEveDir,freqmin=freqmin,freqmax=freqmax,zerophase=zerophase,excludeNetstas=excludeNetstas)
      
    print("Waveform bandpass finished!")
#----------- Generate arr files ----------------------------------------    
    print(">>> Now prepare arrival files")
    print(">>> Loading arrivals from tarWfDir 'eve_bp_wf' ")
    arrDict = scc_input_load_arr(tarBase,markerP=markerP,markerS=markerS)

    print(">>> Writing arrival files ")
    scc_input_write_arr(arrDir,arrDict)


def load_dtcc(dtccPth="dt.cc"):
    """
    Load dt.cc file and return a DataFrame with columns: evid1, evid2, sta, pha, dt, cc
    """
    if dtccPth[-4:] == ".pkl":                         # .pkl file load by pickle
        f = open(dtccPth,'rb')
        dfSort = pickle.load(f)
    else:                                           # using normal processing
        data = []
        f = open(dtccPth,'r')
        for line in f:
            line = line.rstrip()
            if line[0] == "#":
                _,_evid1,_evid2,_ = re.split(" +",line)
                evid1 = int(_evid1); evid2 = int(_evid2)
            else:
                sta,_diff,_cc,pha = re.split(" +",line.strip())
                data.append([evid1,evid2,sta,pha,float(_diff),float(_cc)])
        df = pd.DataFrame(data,columns=["evid1","evid2","sta","pha","dt","cc"])
        dfSort = df.sort_values(by=["evid1","evid2"],ascending=False)

    return dfSort
