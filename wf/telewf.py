import os
import obspy
from obspy import UTCDateTime,Stream
from obspy.io.sac.sactrace import SACTrace
import re
import matplotlib.pyplot as plt
from seisloc.utils import get_st,extract_set_info
from seisloc.geometry import spherical_dist
from seisloc.sta import load_sta
import numpy as np
from obspy.taup import TauPyModel
from obspy.clients.fdsn import Client
import pandas as pd
import logging
from tqdm import tqdm
from seisloc.noise import xcorr
from numpy import polyfit
import glob


def gen_tele_files( datasetPth,
                    staFile,
                    minMag,
                    dist_range,
                    client_name = "IRIS",
                    taup_model="iasp91",
                    tele_dir = "tele_event"):
    """
    Function first searches for suitable tele-event based on condtions provided:
        > starttime
        > endtime
        > minmagnitude
        > distance range in degree[d1, d2]
    Then calculates arrival times for stations and writes into files.
    
    Parameters:
      datasetPth: the path of the dataset. The program will extract information
                    of this dataset, including starttime and endtime
          staFile: station file for station longitude and latitude information
           minMag: minimum magnitude of tele-event for plot
        dist_range: [d1,d2] in degree
       client_name: default "IRIS", check obspy.clients.fdsn.client.Client.__init__()
                     for detail
        taup_model: default "iasp91", check folder obspy/taup/data/ for more model
          tele_dir: target dir

    """    
    logger = logging.getLogger()
    #---------- main program ------------------------------------------------
    if not os.path.exists(tele_dir):
        os.mkdir(tele_dir)
        logger.info("tele dir created.")
    set_info = extract_set_info(datasetPth,staFile)
    c_lon,c_lat = set_info["center"][0]
    starttime = UTCDateTime(set_info['startTime'][:-1])  
    endtime = UTCDateTime(set_info['endTime'][:-1])
    netstas = list(set_info["availYearDays"].keys())
    
    sta_dict = load_sta(staFile)
    client=Client(client_name)
    event_list=client.get_events(starttime=starttime,
                                 endtime=endtime,
                                 minmagnitude=minMag)
    
    columns=["e_id","e_time","e_lon","e_lat","e_dep","e_dist","e_mag","e_mag_type"]
    df = pd.DataFrame(columns=columns)
    for event in event_list:
        e_id = str(event["origins"][0]["resource_id"])[-8:]
        e_time = event["origins"][0]["time"]
        e_lon = event["origins"][0]["longitude"]
        e_lat = event["origins"][0]["latitude"]
        e_dep = event["origins"][0]["depth"]
        e_mag = event['magnitudes'][0]["mag"]
        e_mag_type = event['magnitudes'][0]["magnitude_type"]
        ec_dist = spherical_dist(e_lon,e_lat,c_lon,c_lat) #distance from dataset center
        if ec_dist>=dist_range[0] and ec_dist <= dist_range[1]:
            df.loc[df.shape[0]+1]=[e_id,e_time,e_lon,e_lat,e_dep,ec_dist,e_mag,e_mag_type]

    model = TauPyModel(taup_model)
    for index,row in df.iterrows():
        e_time=row["e_time"]
        e_lon=row["e_lon"]
        e_lat=row["e_lat"]
        e_dep=row["e_dep"]/1000
        e_mag=row["e_mag"]
        e_mag_type=row["e_mag_type"]
        logger.info(f"Now process event: time:{e_time} mag:{e_mag}")
        with open(os.path.join(tele_dir,e_time.strftime("%Y%m%d%H%M%S")+".tele"),"w") as f:
            f.write(f"{e_time} {e_lon} {e_lat} {e_dep} {e_mag} {e_mag_type}\n")
            cont_list=[]
            dist_list=[]
            for netsta in netstas:
                net = netsta[:2]
                sta = netsta[2:]
                sta_lon = sta_dict[net][sta][0]
                sta_lat = sta_dict[net][sta][1]
                es_dist = spherical_dist(e_lon,e_lat,sta_lon,sta_lat)
                arrivals=model.get_travel_times(source_depth_in_km=e_dep,
                                                distance_in_degree=es_dist,
                                                phase_list=["P","S"])
                try:#In case no content error
                    p_arrival=arrivals[0].time
                    s_arrival=arrivals[1].time
                    cont_list.append([netsta,p_arrival,s_arrival,es_dist*111])
                    dist_list.append([es_dist*111])
                    dist_list_sort.append([es_dist*111])
                except:
                    continue
            if len(dist_list) == 0:
                continue
            for dist in sorted(dist_list):
                idx=dist_list.index(dist)
                f.write(f"{cont_list[idx][0]} ")       # netsta       
                f.write(f"{cont_list[idx][1]} ")       # P arrival time
                f.write(f"{cont_list[idx][2]} ")       # S arrival time
                f.write(f"{cont_list[idx][3]}\n")      # event distance
        f.close()


def readTelePhase(cont):
    """
    Read stations tele event arrival time from the tele file
    Parameters:
    |    cont: content list of tele file
    """
    staPhaList=[]              # Array to store the P&S phase time information
    for line in cont[1:]:
        netsta,_P_time,_S_time,_dist=line.split()
        net = netsta[:2]
        sta = netsta[2:]
        P_time = float(_P_time)
        S_time = float(_S_time)
        dist=float(_dist)
        staPhaList.append([netsta,P_time,S_time,dist])
    return staPhaList

def _trimStaTeleWf(net,sta,startTime,endTime,wfFolder,teleWfDir,mode,pad,fill_value):
        if mode == "normal":
            st=get_st(net,sta,startTime,endTime,wfFolder,pad=True,fill_value=0)
        if mode == "SC":
            st=get_st_SC(net,sta,startTime,endTime,wfFolder,pad=True,fll_value=0)
        st = st.select(component="*Z")                # Use Z component
        if len(st) != 0:                                  # len(st)==0 means no waveform
            st[0].write(os.path.join(teleWfDir,net+"_"+sta+".mseed"))  

def trimTeleWf(teleFile,wfRoot,pBefore=50,sAfter=50,mode="normal"):
    """
    test
    Description:
        To plot tele-event waveform, the steps are:
            1. Generate tele-event files
            2. Cut tele-event waveforms of all stations
            3. Make plot of tele-event waveforms
        this functions corresponds to step 2, the waveforms of all stations will
        be saved in one miniseed file under the same tele-file folder with the
        same title but different suffix(".mseed").

        Retrieve from online resources to be developed

    Parameters:
        teleFile: the path of tele-file
          wfRoot: root path for waveform library
         pBefore: start trim point is pBefore seconds before the earliest P
          sAfter: end trim point the pAster seconds after the latest S
            mode: default "normal". "SC" indicates retrieving Sichuan
                    Agency Dataset,uncommon used
    """

    logger = logging.getLogger()
    logger.info(f"Trim tele waveform of tele-event: {teleFile}")
    
    #-------------- load tele file---------------------------------------------
    cont=[]                                 
    with open(teleFile,"r") as f:          
        for line in f:
            cont.append(line.rstrip())
    if len(cont)==1:               # No record, first line is event line
        logger.warn("No station record in tele_file")
        return                              
    
    _etime,_elon,_elat,_edep,_emag,etype = cont[0].split()
    etime=UTCDateTime(_etime[:-1])
    staPhaList = readTelePhase(cont)

    min_P = staPhaList[0][1]
    max_S = staPhaList[-1][2]
                     
    startTime = etime+min_P-pBefore
    endTime = etime+max_S+sAfter

    stsum = Stream()
    for staPhase in tqdm(staPhaList):
        netsta = staPhase[0]
        net = netsta[:2]
        sta = netsta[2:]
     
        wfFolder = os.path.join(wfRoot,sta) # Waveform folder
        if mode == "normal":
            st=get_st(startTime,endTime,wfFolder,net,sta,pad=True,fill_value=0)                                                                                                                            
        if mode == "SC":
            st=get_st_SC(net,sta,startTime,endTime,wfFolder,pad=True,fll_value=0)
        st = st.select(component="*Z")                # Use Z component
        if len(st) != 0:                                  # len(st)==0 means no waveform
            stsum.append(st[0])
    stsum.write(os.path.join(teleFile[:-5]+".mseed"))


def trimTeleWfs(teleRoot="tele_event",wfRoot="day_data",processes=1):
    teleFiles = glob.glob(os.path.join(teleRoot,"*tele"))
    pool = mp.Pool(processes=processes)
    for teleFile in teleFiles:           # Loop for each event
        #trimTeleWf(teleFile,wfRoot)
        pool.apply_async(trimTeleWf,args=(teleFile,wfRoot))
    print(f"Multiprocessing with cores = {processes}")
    pool.close()
    pool.join()

def axisRange(staPhaList,plotPhase,xOffsets,yOffsetRatios):
    """
    Get the axis range for tele waveform plot
    """
    minP = staPhaList[0][1]
    maxP = staPhaList[-1][1]
    minS = staPhaList[0][2]
    maxS = staPhaList[-1][2]
    minDist = staPhaList[0][3]
    maxDist = staPhaList[-1][3]

    ystart = minDist-yOffsetRatios[0]*(maxDist-minDist+0.1)                        
    yend = maxDist+yOffsetRatios[1]*(maxDist-minDist+0.1)

    if plotPhase == "P":
        xstart = minP+xOffsets[0]
        xend = maxP+xOffsets[1]

    elif plotPhase == "S":
        xstart = minS+xOffsets[0]
        xend = maxS+xOffsets[1]

    elif plotPhase == "PS":
        xstart = minP+xOffsets[0]
        xend = maxS+xOffsets[1]
    else:
        raise Exception(f"'plotPhase' parameter {plotPhase} not in ['P','S','PS']")
        
    return xstart,xend,ystart,yend

def plotTeleWf(teleFile,
              wfRoot = "day_data",
              staSel="all",
              staExclude = [],
              staHighlight=[],
              plotPhase = "P",
              bpRange= [0.5,2],
              xOffsets=[-10,20],
              yOffsetRatios=[0.5,0.5],
              wfNormalize=True,
              wfScaleFactor=1,
              labelStas = "all",
              figsize=(6,8),
              linewidth=1,
              tickHeight=0.05,
              oFormat="pdf"):
    """
    Parameters:
          wfRoot: The folder containing wf data in strcture wf_dir/sta_name/'wf files'
         staSel: list or , station list selected for plot, "all": plot all
     staExclude: stations excluded in the plot,used for problem staion
   staHighlight: station waveform to be highlighted will be drawn in green
      plotPhase: "P","S" or "PS". "P" only plot P arrival, "S"  only plot S arrival.
                    "PS" means both P arrival and S arrival will be presented
      labelStas: False means no label, empty list means all, else label station in list
    """
    logger = logging.getLogger()
    logger.info("plot tele event: "+teleFile)

    mseed_file = teleFile.rsplit(".",1)[0]+".mseed"
    if not os.path.exists(mseed_file):
        logging.error("mseed file not exits, did you run the trimTeleWf?")
        return                              # end the programme
    else:
        saved_st = obspy.read(mseed_file)
    #-------------- load tele file---------------------------------------------
    cont=[]                                 # Store content from file
    with open(teleFile,"r") as f:          # Load starts
        for line in f:
            cont.append(line.rstrip())
    f.close()                               # Load finishes
    if len(cont)==1:
        logger.warn("No station record in tele_file")
        return                              # No record, first line is event line
    _etime,_elon,_elat,_edep,_emag,etype = cont[0].split()
    etime=UTCDateTime(_etime[:-1])
    elon=float(_elon)
    elat=float(_elat)
    emag=float(_emag)

    staPhaList = readTelePhase(cont)

    xstart,xend,ystart,yend = axisRange(staPhaList,plotPhase,xOffsets,yOffsetRatios)

    plt.close()                               # close previous figure as this is inside loop 
    fig,ax = plt.subplots(1,1,figsize=figsize) # Initiate a figure with size 8x10 inch        

    plt.axis([xstart,xend,ystart,yend]) # Set axis
    tele = re.split("/",teleFile)[-1]
    title = etime.strftime("%Y-%m-%d %H:%M:%S")
    plt.title(f'Tele Event {title} M{emag}')           # Set the title
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (km)")
    # Draw event waveform, P and S arrival markers for each station
    for staPhase in tqdm(staPhaList):
        netsta = staPhase[0]
        net = netsta[:2]
        sta = netsta[2:]

        if staSel != "all": # Then staSel should be a list
            if sta not in staSel:
                continue
        if sta in staExclude:
            continue

        P_time = staPhase[1]
        S_time = staPhase[2]
        dist = staPhase[3]
        st = saved_st.select(network=net,station=sta)
        st.trim(starttime = etime+xstart,
                    endtime = etime+xend,
                    pad = True)

        if len(st) != 0:                                  # len(st)==0 means no waveform
            st = st.select(component="*Z")                # Use Z component
            sampling_rate = st[0].stats.sampling_rate
            chn = st[0].stats.channel
            st[0].detrend("linear")                       # Remove linear trend
            st[0].detrend("constant")                     # Remove mean
            st.filter("bandpass",freqmin=bpRange[0],freqmax=bpRange[1],corners=2)
            if wfNormalize:
                data = st[0].data.copy()
                if max(data) != min(data):
                    st[0].data=data/(max(data) - min(data)) # Normalize data
                st[0].data = st[0].data * wfScaleFactor
            # Draw waveform
            if sta in staHighlight:                               # color='k' means black
                plt.plot(np.arange(0,len(st[0].data))*1/sampling_rate+xStart,
                         st[0].data+dist,
                         color='darkred',
                         linewidth=3*linewidth)
            else:                                         # color='g' means green
                plt.plot(np.arange(0,len(st[0].data))*1/sampling_rate+xstart,
                         st[0].data+dist,
                         color='k',
                         linewidth=linewidth)
            # Plot P arrival marker in red
            P_marker, = plt.plot([P_time,P_time],[dist-tickHeight,dist+tickHeight],color='r',linewidth=2)
            # Plot S arrival marker in blue
            S_marker, = plt.plot([S_time,S_time],[dist-tickHeight,dist+tickHeight],color='b',linewidth=2)
            if labelStas=="all":
                plt.text(xstart,dist,f'{sta}',color='darkred',fontsize=12)
            elif sta in labelStas:
                plt.text(xstart,dist,f'{sta}',color='darkred',fontsize=12)
    plt.legend([P_marker,S_marker],['tele P','tele S'],loc='upper right')
    plt.tight_layout()
    if oFormat.lower()=="pdf":
        plt.savefig(os.path.join(teleFile[:-5]+".pdf"))
    if oFormat.lower()=="jpg" or oFormat.lower=="jpeg":
        plt.savefig(os.path.join(teleFile[:-5]+".jpg"))
    if oFormat.lower()=="png":
        plt.savefig(os.path.join(teleFile[:-5]+".png"))

def plotTeleWfs(teleRoot="tele_event",wfRoot="day_data",
                          staSel='all',
                          staExclude = [],
                          plotPhase = "P",
                          bpRange= [0.5,2],
                          xOffsets=[-10,20],
                          yOffsetRatios=[0.01,0.05],
                          wfNormalize=True,
                          wfScaleFactor=0.06,
                          labelStas = "all",
                          figsize=(8,12),
                          linewidth=0.5,
                          tickHeight=0.01,
                          oFormat="png"):
    teleFiles = glob.glob(os.path.join(teleRoot,"*tele"))
    for teleFile in teleFiles:           # Loop for each event
            plotTeleWf(teleFile,
                          wfRoot = wfRoot,
                          staSel=staSel,
                          staExclude = staExclude,
                          plotPhase = plotPhase,
                          bpRange= bpRange,
                          xOffsets=xOffsets,
                          yOffsetRatios=yOffsetRatios,
                          wfNormalize=wfNormalize,
                          wfScaleFactor=wfScaleFactor,
                          labelStas = labelStas,
                          figsize=figsize,
                          linewidth=linewidth,
                          tickHeight=tickHeight,
                          oFormat=oFormat)


def plotTeleDiffs(plotPhase="P",
                 tb=-5,
                 te=20,
                 root="tele_event",
                 maxlag=100,
                 freqRange=[0.5,2],
                 decimateFactor=5,
                 threshold = 0.5,
                 figsize=(10,6)):
    """
    This function reads in the corresponding tele event minseed files,
    do cross-correlation to find large shift stations and make plot.
    
    Parameter:
      maxlag: maxlag data points. The total calculation times is 2*maxlag+1
    """
    logger = logging.getLogger()
#---------------------------------------------------------------------------    
    tele_files = glob.glob(os.path.join(root,"*tele"))
    for tele_file in tqdm(tele_files):
        with open(tele_file,'r') as f:
            lines = f.readlines()
        f.close()
        if not os.path.exists(tele_file[:-4]+"mseed"):
            print(tele_file[:-4]+"mseed "+"not existed!")
            continue
        st = obspy.read(tele_file[:-4]+"mseed")
        st.detrend("linear")
        st.detrend("constant")
        st_new = st.decimate(factor=decimateFactor)
        st_new.filter("bandpass",freqmin=freqRange[0],freqmax=freqRange[1],zerophase=True)

        ref_sta_set =False

        maxtimes = []
        sta_sequence = []
        delta = st_new[0].stats.delta
        
        str_time,_,_,_dep,_mag,type = lines[0].split()
        etime = UTCDateTime(str_time)
        
        for line in lines[1:]:
            line = line.rstrip()
            netsta,_p,_s,_dist = line.split()
            net = netsta[:2]
            sta = netsta[2:]
            st_sel = st_new.select(network=net,station=sta,component='*Z')
            if len(st_sel)==0:
                continue
            if plotPhase == "P":
                ttb = etime + float(_p) + tb
                tte = etime + float(_p)+ te
            if plotPhase == "S":
                ttb = etime + float(_s) + tb
                tte = etime + float(_s)+ te
            if plotPhase == "PS":
                ttb = etime + float(_p) + tb
                tte = etime + float(_s)+ te
            st_sel = st_sel.trim(starttime=ttb,endtime=tte)
            sta_tr = st_sel[0]
            if not ref_sta_set:
                ref_sta = sta
                ref_sta_set = True
                continue
            ref_tr = st_new.select(station=ref_sta,component="*Z")[0]
            try:
                corr_result = xcorr(sta_tr.data,ref_tr.data,maxlag)
            except:
                continue
            corr_result = list(corr_result/(np.linalg.norm(sta_tr.data)*np.linalg.norm(ref_tr.data)))
            max_corr = max(corr_result)
            if max_corr < 0.7:
                continue
            max_index = corr_result.index(max_corr)
            maxtimes.append((max_index-maxlag)*delta)
            sta_sequence.append(sta)

        fig,ax = plt.subplots(1,1,figsize=figsize)
        try:
            p = polyfit(np.arange(len(maxtimes)),maxtimes,deg=1)
        except:
            continue
        x = np.linspace(0,len(maxtimes)-1,len(maxtimes))
        y = x*p[0]+p[1]
        diffs = maxtimes-y
        plt.scatter(x,diffs,c=np.abs(diffs),s=40,edgecolor='k',cmap="rainbow",vmin=0.2,vmax=4)
        #plt.ylim([-1,5])
        plt.xlim([0,len(maxtimes)])
        plt.ylabel("Time (s)")

        for i in range(len(sta_sequence)):
            if np.abs(diffs[i])>threshold:
                plt.text(i,diffs[i],sta_sequence[i])
        plt.title(f"Tele Event {etime} M{_mag}")
        plt.savefig(tele_file[:-4]+"jpg")
