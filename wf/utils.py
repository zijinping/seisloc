# -*- coding: utf-8 -*-
#----------------------------------
#     Author: Jinping ZI
#
# Revision History
#     2025-02-26 copy and modify from the utils.py under the seisloc dir
#----------------------------------
import os
import time
import json
import obspy
import logging
import numpy as np
from obspy import Stream
import pandas as pd
from seisloc.sta import load_sta

def SC_tr_remove_spikes(tr):
    """ 
    Remove spikes of the Sichuan continous waveform data
    """
    # more than one zero values
    ks, = np.where(tr.data==0)
    if len(ks)>0:
        meanValue = np.sum(tr.data)/(tr.stats.npts - len(ks))    
        tr.data -= meanValue
        tr.data[ks] = 0
        tr.detrend("constant")
        tr.detrend("linear")
        tr.data[ks] = 0

def extract_dataset_info(wfBase,staTxt,allowExisting=True):
    """
    This function extract fundamental information of the waveform
    Parameters:
        wfBase: directory for the waveform, strcture: wfBase/staName/seis_file
        staTxt: path for the station text file
        
    Return:
        A setinfo dictionary containing keys:
        "s_times": sorted starttime list
        "e_times": sorted endtime list
        "netstas": network and station list in format <net+sta>
        "center" : mean longitude and latitude of stations,intended for tele-
                   event selection.
    """
    logger = logging.getLogger()
    logger.info("Extract_dataset_info program launched ...")
    wfBase = os.path.abspath(wfBase)
    setInfoPth = os.path.join(wfBase,"setinfo.json")
    if os.path.exists(setInfoPth) and allowExisting==True:
        with open(setInfoPth,'r') as f:
            setInfo = json.load(f)
        return setInfo

    dfStas = load_sta(staTxt)
    setInfo = {}
    setInfo["startTime"]=""
    setInfo["endTime"]=""
    setInfo["center"] = []
    setInfo["staLonLats"] = []
    setInfo["availYearDays"] = {}
    staLonLats = []
    # Loop for fundamental waveform information
    for staName in os.listdir(wfBase):
        staDir = os.path.join(wfBase,staName)
        if not os.path.isdir(staDir):
            continue
        logging.debug(f"Process dir {staDir}")
        if not "_wf_files_summary.csv" in os.listdir(staDir):
            print("gen_wf_files_summary launched!")
            gen_wf_files_summary(staDir)
        wfSumCsv = os.path.join(staDir,"_wf_files_summary.csv")
        if not "wfSumAll" in locals():
            wfSumAll = pd.read_csv(wfSumCsv)
        else:
            _wfSum = pd.read_csv(wfSumCsv)
            wfSumAll=wfSumAll.append(_wfSum)
    
    for i,row in wfSumAll.iterrows():
        net = row.net
        sta = row.sta
        netsta = net+sta
        if netsta not in setInfo['availYearDays'].keys():
            setInfo['availYearDays'][netsta]={}
            stlo = dfStas[(dfStas["net"]==net) & (dfStas["sta"]==sta)]["stlo"].values[0]
            stla = dfStas[(dfStas["net"]==net) & (dfStas["sta"]==sta)]["stla"].values[0]
            staLonLats.append([stlo,stla])
        yr = row.year
        if yr not in setInfo['availYearDays'][netsta].keys():
            setInfo['availYearDays'][netsta][yr] = []
        julDays = [int(_str) for _str in row.julDays[1:-1].split()]
        for julDay in julDays:
            if julDay not in setInfo['availYearDays'][netsta][yr]:
                setInfo['availYearDays'][netsta][yr].append(julDay)
                    
    setInfo["staLonLats"] = staLonLats
    setInfo["center"] = [list(np.median(np.array(staLonLats),axis=0))]
    wfSumAllSort = wfSumAll.sort_values(by='startTime')
    setInfo["startTime"] = wfSumAllSort.iloc[0].startTime
    wfSumAllSort = wfSumAll.sort_values(by='endTime')
    setInfo['endTime'] = wfSumAllSort.iloc[-1].endTime
    
    with open(setInfoPth,'w') as fw:
        json.dump(setInfo,fw,indent=4)    
    
    logging.info("extract set info programme done and saved in {setinfoPth}")

    return setInfo

def gen_wf_files_summary(wfDir):
    _dataFrame = []
    for item in sorted(os.listdir(wfDir)):
        itemPth = os.path.join(wfDir,item)
        try:
            st = obspy.read(itemPth,headonly=True)
        except:
            print(f"{itemPth} is not a waveform file.")
            continue
        for tr in st:
            net = tr.stats.network
            sta = tr.stats.station
            chn = tr.stats.channel
            startTime = tr.stats.starttime
            endTime = tr.stats.endtime
            julDays = []
            loopTime = startTime+0.01 # +0.01 for debug
            while loopTime < endTime:
                year = loopTime.year
                julDay = loopTime.julday
                julDays.append(julDay)
                loopTime +=24*60*60
            _dataFrame.append([item,net,sta,chn,startTime,endTime,year,julDays])

    df = pd.DataFrame(data=_dataFrame,
                      columns=["fileName","net","sta","chn","startTime","endTime",'year',"julDays"])
    df.to_csv(os.path.join(wfDir,"_wf_files_summary.csv"),index=False)

def get_st(startTime,endTime,wfDir,net=None,sta=None,pad=False,fill_value=None,DEBUG=False):
    """        
    Read and return waveform between startTime and endtime by specified
    net and station in designated folder. It will merge waveform if include
    more than one file.
               
    The return is a obspy Stream object
    """
    #----------------- Quality Control ----------------------
    sumCsvPth = os.path.join(wfDir,'_wf_files_summary.csv')
    if not os.path.exists(sumCsvPth):
        logging.info(f"gen_wf_files_summary({wfDir}) launched!")
        gen_wf_files_summary(wfDir)
    else:
        sumCsvMtime = os.path.getmtime(sumCsvPth)
        for item in os.listdir(wfDir):
            itemPth = os.path.join(wfDir,item)
            itemMtime = os.path.getmtime(itemPth) # modification time
            if itemMtime > sumCsvMtime:
                gen_wf_files_summary(wfDir)
                break
    #---------------------------------------------------------
    b = time.time()
    inc_list = []
    df = pd.read_csv(sumCsvPth)
    dfUse = df
    if net != None:
        dfUse = dfUse[dfUse.net==net]
    if sta != None:
        dfUse = dfUse[dfUse.sta==sta]
    dfUse = dfUse[~((dfUse.startTime>str(endTime)) | (dfUse.endTime<str(startTime)))]
    inc_list = pd.unique(dfUse['fileName'])

    #------------------------Read in data --------------------
    st = Stream()
    for fileName in inc_list:
        filePth = os.path.join(wfDir,str(fileName))
        if DEBUG:
            print(f"get_st debug: st+=obspy.read({filePth})")
        try:
            st += obspy.read(filePth)
        except:
            raise Exception(f"Error in st += obspy.read({filePth})")
    if len(st) == 0:
        pass   
    else:
        st.trim(startTime,endTime,pad=pad,fill_value=fill_value)
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

def check_wf_status_net_sta(workDir,dfStas):
    """
    Quality control:
    (1) Print names of unrelated waveforms
    (1) Station waveforms have correct net and sta.
    (2) The station
    Note: The day_split.mp_day_split function will automatically correct net and sta 
          accordingly to station file. So not necessary to modify raw data and take 
          information by this function as sidenotes if you are planning
          to use splitted waveforms.
    """        
    falseRecs = []
    for sta in os.listdir(workDir): # inDir should only include dirs with name of station
        staDir = os.path.join(workDir,sta)
        if not os.path.isdir(staDir) or sta[0] =="\.":   
            continue    # skip non-directory files and folder starts with "."
        logging.info("func check_trace_...>> station: "+sta)
        staDir = os.path.join(workDir,sta)
        net = dfStas[dfStas["sta"]==sta]['net'].values[0]
        for wfnm in os.listdir(staDir):
            wfPth = os.path.join(workDir,sta,wfnm)
            try:
                st = obspy.read(wfPth,headonly=True)
            except:
                logging.info("func check_trace_...>> inconsistent file: "+wfPth)
                continue
            if st[0].stats.network != net:
                logging.warning(f"func check_trace_...>> {wfPth}: inconsistent st[0].stats.network [{net}]")
                falseRecs.append([wfPth,st[0].stats.network,st[0].stats.station])
            if st[0].stats.station != sta:
                logging.warning(f"func check_trace_...>> {wfPth}: inconsistent st[0].stats.station [{net}]")
                falseRecs.append([wfPth,net,sta])

    if len(falseRecs)>0:
        if not os.path.exists("Reports"):
            os.mkdir("Reports")
        f = open("Reports/check_wf_status_net_sta.err",'w')
        for rec in falseRecs:
            f.write(f"Error net or sta in folder: {rec[0]} net:{rec[1]} sta:{rec[2]}")
        f.close()
        statement="func check_trace_...>> Error net or sta in trace status, check Reports/check_wf_status_net_sta.err for details"
        logging.info(statement)
        raise Exception(statement)

def check_sta_names(rawDir,dfStas):
    """
    Check whether station names are in the station dataframe and each station name is uniqe
    """
    for sta in os.listdir(rawDir): # inDir should only include dirs with name of station
        staDir = os.path.join(rawDir,sta)
        if os.path.isdir(staDir) and sta[0] !="\.":   # folder starts with "." are hidden dirctories
            dfSta = dfStas[dfStas["sta"]==sta]
            if len(dfSta) == 0:
                raise Exception(f"Station {sta} not in the station file! \
                                Please correct the station directory name!")
            if len(dfSta)>1:
                raise Exception(f"Station {sta} is repeated in the station file")

def raw_status_control(rawDir:str,staTxt:str)->None:
    """
    Quality control of the raw dataset:
    (1) Ensure that station under rawDir is included in the staFile.
    (2) Check the raw data waveforms have correct net and sta

    RawDir: directory of the raw data, the structure should rawDir/sta/wfFiles
    """
    dfStas = load_sta(staTxt)
    check_sta_names(rawDir,dfStas)             # dir name should be sta 
    check_wf_status_net_sta(rawDir, dfStas)   # trace net and sta

def to_EQT_mseed(trace,outDir="./"):
    """
    Save the obspy trace to the format EQTransformer could recognized
    """
    net = trace.stats.network
    sta = trace.stats.station
    chn = trace.stats.channel
    starttime = trace.stats.starttime
    endtime = trace.stats.endtime
    name = net+"."+sta+"."+chn+"__"+\
                starttime.strftime("%Y%m%dT%H%M%SZ")+"__"+\
                endtime.strftime("%Y%m%dT%H%M%SZ")+".mseed"\

    trace.write(os.path.join(outDir,name),format="MSEED")
