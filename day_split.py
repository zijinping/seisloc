import numpy as np
import os
import re
import logging
from seisloc.sta import load_sta
from tqdm import tqdm
import obspy
from obspy import UTCDateTime
import multiprocessing as mp

def time_file_list(staDir,prjBtime=None,prjEtime=None):
    '''
    Get the starttime and file path in sequence, append the last
    end time to the time list.
    '''
    dataPths = []
    timePoints = []
    max_endtime = UTCDateTime(1900,1,1)
    for file in os.listdir(staDir):
        file_path = os.path.join(staDir,file)
        assert 'obspy' in globals()
        try:
            st=obspy.read(file_path,headonly=True)
        except:
            continue
        starttimes = []
        endtimes = []
        #----------- Avoid the influence of error times segments ------------    
        if prjBtime!=None:
            for tr in st:
                if tr.stats.starttime < prjBtime:
                    st.remove(tr)
        if prjEtime!=None:
            for tr in st:
                if tr.stats.endtime > prjEtime:
                    st.remove(tr)
        if len(st) == 0:
            continue
        for tr in st:
            starttimes.append(tr.stats.starttime)
            endtimes.append(tr.stats.endtime)
        timePoints.append(min(starttimes))
        if max(endtimes)>max_endtime:
            max_endtime = max(endtimes)
        dataPths.append(file_path)
    tmp = np.array(timePoints)
    dataPths = np.array(dataPths)
    k = tmp.argsort()
    dataPths = dataPths[k]
    timePoints.sort()
    timePoints.append(max_endtime)

    return timePoints,dataPths

def gen_day_split_time_nodes(timePoints,shiftHour):
    """
    Generate time nodes for time split
    """
    tts=timePoints[0] #total start time is the start time of the first file 
    tte=timePoints[-1]#total end time is the end time of the last file
    #start to get the trim point to form day_sac files
    dayTimeNodes=[]
    dayTimeNodes.append(tts)#the first point is the start time of the first file
    #information of the first day
    f_year=timePoints[0].year
    f_month=timePoints[0].month
    f_day=timePoints[0].day
    #the second point should be the start time of the second day
    trim_node=UTCDateTime(f_year,f_month,f_day)+24*60*60+shiftHour
    #if the second point is less than the total end time, add it into list and move to next day
    while trim_node < tte:
        dayTimeNodes.append(trim_node)
        trim_node+=24*60*60
    #append the last time
    dayTimeNodes.append(tte)

    return dayTimeNodes

def write_avail_days(outFdr,dayTimeNodes):
    """
    Generate a available days text file for cross-correlation usage
    """
    f = open(os.path.join(outFdr,'availdays.txt'),'w')
    for time in dayTimeNodes:                
        f.write(f"{time.year} {time.julday}\n")   
    f.close()                       

def write_st_to_file(st,outFdr,trimBtime,trimEtime,fileFmt="mseed"):
    """
    Save traces in the Stream to the desginiated folder in format
    """
    for tr in st:
        net=tr.stats.network
        sta=tr.stats.station
        chn=tr.stats.channel
        if fileFmt.upper()=="SAC":
            fname=net+"."+sta+"."+chn+"__"+\
                   trimBtime.strftime("%Y%m%dT%H%M%SZ")+"__"+\
                   trimEtime.strftime("%Y%m%dT%H%M%SZ")+".SAC"
        if fileFmt.upper()=="MSEED":
            fname=net+"."+sta+"."+chn+"__"+\
                trimBtime.strftime("%Y%m%dT%H%M%SZ")+"__"+\
                trimEtime.strftime("%Y%m%dT%H%M%SZ")+".mseed"
        tr.write(outFdr+"/"+fname,format=fileFmt)
        logging.info(fname+" writed.")

def load_stream_channels(st):
    chns = []
    for tr in st:
        if tr.stats.channel not in chns:
            chns.append(tr.stats.channel)

    return chns

def get_trim_idxs(trimBtime,trimEtime,timePoints):
    for j in range(len(timePoints)-1):  
        if trimBtime>=timePoints[j] and trimBtime<timePoints[j+1]:
            idxa=j                   
    for k in range(len(timePoints)-1):  
        if trimEtime>=timePoints[k] and trimBtime<timePoints[k+1]:
            idxb=k

    return idxa,idxb

def cut_and_save_day_wf(dataPths,idxa,idxb,trimBtime,trimEtime,fileFmt,outFdr):
    st=obspy.read(dataPths[idxa])
    while idxa < idxb:
        idxa+=1
        st+=obspy.read(dataPths[idxa])
    #merge data, using try except to avoid mistake len(st)==1
    chns = load_stream_channels(st)
    for chn in chns:
        stUse = st.select(channel=chn)
        if len(stUse)==0:
            continue
        if len(stUse)>1:
            stUse.merge(method=1,fill_value="interpolate")
        stUse.trim(starttime=trimBtime,endtime=trimEtime)
        write_st_to_file(stUse,outFdr,trimBtime,trimEtime,fileFmt=fileFmt)
    
def sta_in_dict(sta,staDict):
    flag = 0
    staList = []
    for netName in staDict.keys():
        for staName in staDict[netName].keys():
            if staName == sta:
                staList.append([netName,staName])
    if len(staList)==0:
        flag = 0
    if len(staList)==1:
        flag = 1
    if len(staList)>1:
        flag = 2
        _tmp = ""
        for [netName,staName] in staList:
            _tmp += netName+" "+staName+"; "
        logging.error("func sta_in_dict: more than one station pair: "+_tmp)
    return flag

def check_sta(sta,staDict):
    flag = sta_in_dict(sta,staDict)
    if flag == 0:
        statement=f"{sta} not in station file"
        logging.error(statement)
        raise Exception(statement)
    if flag == 2:
        statement=f"{sta} in more than one network or provided more than once in station file"
        logging.error(statement)
        raise Exception(statement)

def check_station_directories(inDir,staDict):
    stasUse=[]
    for station in os.listdir(inDir): # inDir should only include dirs with name of station
        stationPth = os.path.join(inDir,station)
        if os.path.isdir(stationPth) and station[0] !="\.":
            check_sta(station,staDict)    # check whether sta is in the station list
            stasUse.append(station)
    logging.info("func check_station_...>> Sta qty to be processed: "+str(len(stasUse)))

    return stasUse

def check_trace_status_net_sta(stasUse,staDict,baseDir):
    """
    Check whether the net and sta is correct
    """
    status = True          
    falseRecs = []
    for station in stasUse:
        logging.info("func check_trace_...>> station: "+station)
        stationPth = os.path.join(baseDir,station)
        for item in os.listdir(stationPth):
            itemPth = os.path.join(baseDir,station,item)
            if os.path.isdir(itemPth):
                logging.info("func check_trace_...>> inconsistent folder: "+itemPth)
                continue
            try:
                st = obspy.read(itemPth,headonly=True)
            except:
                logging.info("func check_trace_...>> inconsistent file: "+itemPth)
                continue
            net = st[0].stats.network
            sta = st[0].stats.station
            if net not in staDict:
                logging.info(f"func check_trace_...>> {itemPth}: net [{net}] not in station file")
                falseRecs.append([itemPth,net,sta])
            else:      
                if sta not in staDict[net]:
                    logging.info(f"func check_trace_...>> {itemPth}: sta [{sta}] in station file")
                    falseRecs.append([itemPth,net,sta])
                if sta != station:
                    logging.info(f"func check_trace_...>> {itemPth}: sta [{sta}] != 'dir name' ")
                    falseRecs.append([itemPth,net,sta])

    if len(falseRecs)>0:
        status = False
        if not os.path.exists("Reports"):
            os.mkdir("Reports")
        f = open("Reports/check_trace_status_net_sta.err",'w')
        for rec in falseRecs:
            f.write(f"Error net or sta in folder: {record[0]} net:{record[1]} sta:{record[2]}")
        f.close()
        statement="func check_trace_...>> Error net or sta in trace status, check Reports/check_trace_status_net_sta.err for details"
        logging.info(statement)
        raise Exception(statement)
        
def raw_status_control(rawDir,staFile):
    staDict = load_sta(staFile)
    stasUse = check_station_directories(rawDir,staDict)  # dir name should be sta 
    check_trace_status_net_sta(stasUse,staDict,rawDir)   # trace net and sta

def mp_day_split(inDir,outDir,staFile,fileFmt="mseed",shiftHour=0,parallel=True,cutData=True,prjBtime=None,prjEtime=None):
    '''
    This function reads in waveform data and split them by days.
    The inDir should be in the strcutre: inDir/staName/dataFiles
    Parameters
        inDir: base folder containing the raw data
       outDir: output directory.
      staFile: station files containing station location information
      fileFmt: "SAC" or "MSEED"
    shiftHour: Shift trimming times to adjust time zones
     parallel: apply multiprocessing
      cutData: if only wish to generate availdays.txt, set cutData to false
     prjBtime: Project begin time. Data should be within the project time span
     prjEtime: Project end time.
    '''
    #----------------- Initiation --------------------
    staDict = load_sta(staFile)
    if not os.path.exists(outDir):
        os.mkdir(outDir)
        logging.info(f" Output dirctory {outDir} created")
    #----------------- Clean up information ----------
    for sta in os.listdir(inDir):
        staDir = os.path.join(inDir,sta)
        timePoints,dataPths = time_file_list(staDir,
                                             prjBtime=prjBtime,
                                             prjEtime=prjEtime)
        if len(dataPths)==0: 
            logging.info(f"{sta} has no available data files")
            continue           
        logging.info(f"{sta} Time span: {timePoints[0]} {timePoints[-1]}")
        #------------- Split the data ----------------
        outFdr = os.path.join(outDir,sta)
        dayTimeNodes = gen_day_split_time_nodes(timePoints,shiftHour)
        write_avail_days(outFdr,dayTimeNodes)
        if cutData == False: 
            logging.info(f"{sta} cutData is set False, no day split will be conducted!")
            continue                  #only generate availdays.txt
        if parallel:
            pool = mp.Pool(processes=mp.cpu_count())
        for i in range(len(dayTimeNodes)-1):
            trimBtime=dayTimeNodes[i] #trim start time
            trimEtime=dayTimeNodes[i+1]#trim end time
            #get the index of trim_s and trim_e file
            idxa,idxb = get_trim_idxs(trimBtime,trimEtime,timePoints)
            #read in coresponding files
            if parallel:
                pool.apply_async(cut_and_save_day_wf,
                                args=(dataPths,idxa,idxb,trimBtime,trimEtime,fileFmt,outFdr))
            else:
                cut_and_save_day_wf(dataPths,idxa,idxb,trimBtime,trimEtime,fileFmt,outFdr)
        if parallel:
            pool.close()
            pool.join()
