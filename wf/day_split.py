import os
import logging
import numpy as np
import obspy
from obspy import Stream
from obspy import UTCDateTime
import multiprocessing as mp
from seisloc.sta import load_sta

def time_file_list(staDir,prjBtime=None,prjEtime=None):
    '''
    Get the starttime and file path in sequence, append the last
    end time to the time list.

      staDir: directory containing waveform files of a station
    prjBtime: Project begin time. Data should be within the project time span, default None
    prjEtime: Project end time, default None
    '''
    dataPths = []
    timeNodes = []
    endtimeMax = UTCDateTime(1900,1,1) # A unreasonably early time that must be replaced
    for wfFile in os.listdir(staDir):
        wfPth = os.path.join(staDir,wfFile)
        assert 'obspy' in globals()
        try:   # only read waveform files and skip other files
            st=obspy.read(wfPth,headonly=True)
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
        timeNodes.append(min(starttimes))
        if max(endtimes)>endtimeMax:
            endtimeMax = max(endtimes)
        dataPths.append(wfPth)
    tmp = np.array(timeNodes)
    dataPths = np.array(dataPths)
    k = tmp.argsort()
    dataPths = dataPths[k]
    timeNodes.sort()
    timeNodes.append(endtimeMax)

    return timeNodes,dataPths

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

def cut_and_save_day_wf(dataPths,idxa,idxb,trimBtime,trimEtime,fileFmt,outFdr,net,sta):
    print(dataPths,idxa,idxb,trimBtime,trimEtime,fileFmt,outFdr,net,sta)
    #---------------- get channel list -------------------------
    fileChnList = []
    chns = []
    st = Stream()
    while idxa <= idxb:
        st+=obspy.read(dataPths[idxa],headonly=True)
        fileChnList.append([dataPths[idxa],st[-1].stats.channel])
        if st[-1].stats.channel not in chns:
            chns.append(st[-1].stats.channel)
        idxa+=1
    del st
    #---------------- process each channel ---------------------
    for chn in chns:
        stUse = Stream()
        for dataPth,channel in fileChnList:
            if channel != chn:
                continue
            stUse += obspy.read(dataPth)
            assert stUse[-1].stats.channel == chn
        if len(stUse)==0:
            continue
        if len(stUse)>1:
            stUse.merge(method=1,fill_value=0)
        for tr in stUse:     # reset network and station from the station file
            tr.stats.network = net
            tr.stats.station = sta
        stUse.trim(starttime=trimBtime,endtime=trimEtime)
        write_st_to_file(stUse,outFdr,trimBtime,trimEtime,fileFmt=fileFmt)
        del stUse

def mp_day_split(inDir,outDir,staPth,fileFmt="mseed",shiftHour=0,parallel=True,
                 processes=10,cutData=True,prjBtime=None,prjEtime=None):
    '''
    The function reads in dataset waveforms data and split them by days. 
    The inDir should be in the strcutre: inDir/staName/waveformFiles. 
    Note: The saved waveforms will use "net" and "sta" from the station file(staPth).
          The reason to do so is that some raw waveforms are using wrong net or sta 
          information that should be updated during the data cut process. 
    Parameters
        inDir: base folder containing the raw data
       outDir: output directory.
       staPth: station files containing station location information
      fileFmt: "SAC" or "MSEED"
    shiftHour: Shift trimming times to adjust time zones
     parallel: apply multiprocessing
      cutData: if only wish to generate availdays.txt, the information file for 
               ambient noise cross-correlation, set cutData=False to save time
     prjBtime: Project begin time. Data should be within the project time span, 
               default None(no constraint)
     prjEtime: Project end time, default None
staCorrection: If True, output waveforms use net and sta from the station file 
               rather than the internal values of input waveforms
    '''
    #----------------- Initiation --------------------
    df = load_sta(staPth)
    if not os.path.exists(outDir):
        os.mkdir(outDir)
        logging.info(f" Output dirctory {outDir} created")
    #----------------- Clean up information ----------
    for sta in os.listdir(inDir):
        staDir = os.path.join(inDir,sta)
        if not os.path.isdir(staDir):
            continue
        net = df[df["sta"]==sta]["net"].values[0]
        timePoints,dataPths = time_file_list(staDir,
                                             prjBtime=prjBtime,
                                             prjEtime=prjEtime)
        if len(dataPths)==0: 
            logging.info(f"{sta} has no available data files")
            continue           
        logging.info(f"{sta} Time span: {timePoints[0]} {timePoints[-1]}")
        #------------- Split the data ----------------
        outFdr = os.path.join(outDir,sta)
        if not os.path.exists(outFdr):
            os.makedirs(outFdr)
        dayTimeNodes = gen_day_split_time_nodes(timePoints,shiftHour)
        if cutData == False: 
            logging.info(f"{sta} cutData is set False, no day split will be conducted!")
            continue                  #only generate availdays.txt
        if parallel:
            pool = mp.Pool(processes=processes)
        for i in range(len(dayTimeNodes)-1):
            trimBtime=dayTimeNodes[i]     #trim start time
            trimEtime=dayTimeNodes[i+1]   #trim end time
            #get the index of waveform files for trimming waveforms
            idxa,idxb = get_trim_idxs(trimBtime,trimEtime,timePoints)
            #read in coresponding files
            if parallel:
                pool.apply_async(cut_and_save_day_wf,
                                args=(dataPths,idxa,idxb,trimBtime,trimEtime,fileFmt,outFdr,net,sta))
            else:
                print("Here")
                cut_and_save_day_wf(dataPths,idxa,idxb,trimBtime,trimEtime,fileFmt,outFdr,net,sta)
        if parallel:
            pool.close()
            pool.join()
        #gen_wf_files_summary(outFdr)



'''
def cut_single_sta_day_wf(prjBtime,prjEtime,net,sta,rawBase,saveBase,saveFmt,staCorrection=False,fill_value=0):
    wfDir = os.path.join(rawBase,sta)
    saveDir = os.path.join(saveBase,sta)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    loopTime = prjBtime
    while loopTime < prjEtime:
        if staCorrection:
            st = get_st(loopTime,loopTime+24*60*60,wfDir,net=None,sta=None,fill_value=0)
            for tr in st:
                tr.stats.network = net
                tr.stats.station = sta
        else:
            st = get_st(loopTime,loopTime+24*60*60,wfDir,net=net,sta=net,fill_value=0)
        #if len(st)==0:
        #    pass
        st.merge(method=1,fill_value=fill_value)
        for tr in st:
            chn = tr.stats.channel
            if saveFmt.upper()=="SAC":
                fname=net+"."+sta+"."+chn+"_"+\
                       loopTime.strftime("%Y%m%d")+".SAC"
            if saveFmt.upper()=="MSEED":
                fname=net+"."+sta+"."+chn+"_"+\
                       loopTime.strftime("%Y%m%d")+".mseed"
            tr.write(saveDir+"/"+fname,format=saveFmt)
            logging.info(fname+" writed.")
        loopTime += 24*60*60
'''
