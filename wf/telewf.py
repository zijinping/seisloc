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

def gen_tele_files( dataset_path,
                    sta_file,
                    min_mag,
                    dist_range,
                    client_name = "IRIS",
                    taup_model="iasp91",
                    tele_dir = "tele_event"):
    """
    This function first search for suitable tele-event based on the contion provided:
        > starttime
        > endtime
        > minmagnitude
        > distance range in degree[d1, d2]
    The second step is calculate the arrival time for stations and write into document for further process.
    
    Parameters:
      dataset_path: the path of the dataset. The program will extract information
                    of this dataset, including starttime and endtime
          sta_file: station file for station longitude and latitude information
           min_mag: minimum magnitude of tele-event for plot
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
    set_info = extract_set_info(dataset_path,sta_file)
    c_lon,c_lat = set_info["center"] 
    starttime = set_info['s_times'][0]   
    endtime = set_info['e_times'][-1]
    netstas = set_info["netstas"]
    
    sta_dict = load_sta(sta_file)
    client=Client(client_name)
    event_list=client.get_events(starttime=starttime,endtime=endtime,minmagnitude=min_mag)
    
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


def read_tele_phase(cont,sta_sel=[]):
    sta_pha_list=[]              # Array to store the phase arrival time information
    for line in cont[1:]:
        netsta,_P_time,_S_time,_dist=line.split()
        net = netsta[:2]
        sta = netsta[2:]
        P_time = float(_P_time)
        S_time = float(_S_time)
        dist=float(_dist)

        if len(sta_sel) == 0:
            sta_pha_list.append([netsta,P_time,S_time,dist])
        elif sta in sta_sel:                # Only process station in station list
            sta_pha_list.append([netsta,P_time,S_time,dist])
    return sta_pha_list

def tele_file_plot(tele_file,
              wf_dir = "day_data",
              sta_sel=[],
              sta_exclude = [],
              sta_highlight=[],
              plot_phase = "P",
              p_method="dist",
              bp_range= [0.5,2],
              x_offsets=[10,20],
              y_offset_ratio=[0.5,0.5],
              wf_normalize=True,
              wf_scale_factor=1,
              label_stas = [],
              figsize=(6,8),
              linewidth=1,
              o_format="pdf",
              from_saved_wf=False,
              save_result_wf=True):
    """
    Parameters:
          wf_dir: The folder containing wf data in strcture wf_dir/sta_name/'wf files'
         sta_sel: stations selected for plot, empty for all stations
     sta_exclude: stations excluded in the plot,used for problem staion
   sta_highlight: station waveform to be highlighted will be drawn in green
      plot_phase: "P","S" or "PS". "P" only plot P arrival, "S"  only plot S arrival.
                    "PS" means both P arrival and S arrival will be presented
        p_method: 'dist' means plot by distance, "average" means the vertical 
                  gap between stations are the same
    from_save_wf: load saved miniseed waveform from previous run
  save_result_wf: miniseed file will be saved the same name will telefile
      label_stas: False means no label, empty list means all, else label station in list
    """
    logger = logging.getLogger()
    logger.info(tele_file)

    if from_saved_wf == True:
        mseed_file = tele_file.rsplit(".",1)[0]+".mseed"
        if not os.path.exists(mseed_file):
            logging.error("mseed file not exits, set from_saved_wf False")
            from_saved_wf = False
        else:
            saved_st = obspy.read(mseed_file)
    #-------------- load tele file---------------------------------------------
    cont=[]                                 # Store content from file
    with open(tele_file,"r") as f:          # Load starts
        for line in f:
            cont.append(line.rstrip())
    f.close()                               # Load finishes
    if len(cont)==1:
        logger.warn("No station record in tele_file")
        return                              # No record, first line is event line
    _e_time,_e_lon,_e_lat,_e_dep,_e_mag,e_type = cont[0].split()
    e_time=UTCDateTime(_e_time[:-1])
    e_lon=float(_e_lon)
    e_lat=float(_e_lat)
    e_mag=float(_e_mag)

    sta_pha_list = read_tele_phase(cont,sta_sel)

    try:
        del min_P,max_P,max_S,min_dist,max_dist # Remove parameters
    except:
        pass
    min_P = sta_pha_list[0][1]
    max_P = sta_pha_list[-1][1]
    min_S = sta_pha_list[0][2]
    max_S = sta_pha_list[-1][2]
    min_dist = sta_pha_list[0][3]
    max_dist = sta_pha_list[-1][3]

    y_start = min_dist-y_offset_ratio[0]*(max_dist-min_dist-0.1)                        
    y_end = max_dist+y_offset_ratio[1]*(max_dist-min_dist+0.1)                      

    if plot_phase == "P":
        x_start = min_P-x_offsets[0]
        x_end = max_P+x_offsets[1]

    elif plot_phase == "S":
        x_start = min_S-x_offsets[0]
        x_end = max_S+x_offsets[1]

    elif plot_phase == "PS":
        x_start = min_P-x_offsets[0]
        x_end = max_S+x_offsets[1]
    else:
        raise Exception(f"'plot_phase' parameter {plot_phase} not in 'P','S','PS'")

    plt.close()                               # close previous figure as this is inside loop 
    fig,ax = plt.subplots(1,1,figsize=figsize) # Initiate a figure with size 8x10 inch        

    plt.axis([x_start,x_end,y_start,y_end]) # Set axis
    tele = re.split("/",tele_file)[-1]
    title = tele[:4]+"-"+tele[4:6]+"-"+tele[6:8]+" "+tele[8:10]+":"+tele[10:12]+":"+tele[12:14]
    plt.title(f'Tele Event {title} M{e_mag}')           # Set the title
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (km)")
    # Draw event waveform, P and S arrival markers for each station
    st_sum = Stream()
    for sta_phase in tqdm(sta_pha_list):
        netsta = sta_phase[0]
        net = netsta[:2]
        sta = netsta[2:]
        if sta in sta_exclude:
            continue
        P_time = sta_phase[1]
        S_time = sta_phase[2]
        dist = sta_phase[3]
        if from_saved_wf==True:
            st = saved_st.select(network=net,station=sta)
            st.trim(starttime = e_time+x_start,
                    endtime = e_time+x_end,
                    pad = True)
        else:
            wf_folder = os.path.join(wf_dir,sta) # Waveform folder
            st=get_st(net,sta,e_time+x_start,e_time+x_end,wf_folder,pad=True) 
        if len(st) != 0:                                  # len(st)==0 means no waveform
            st = st.select(component="*Z")                # Use Z component
            st_sum.append(st[0])
            sampling_rate = st[0].stats.sampling_rate
            chn = st[0].stats.channel
            st[0].detrend("linear")                       # Remove linear trend
            st[0].detrend("constant")                     # Remove mean
            st.filter("bandpass",freqmin=bp_range[0],freqmax=bp_range[1],corners=2)
            if wf_normalize:
                data = st[0].data.copy()
                if max(data) != min(data):
                    st[0].data=data/(max(data) - min(data)) # Normalize data
                st[0].data = st[0].data * wf_scale_factor
            # Draw waveform
            if sta in sta_highlight:                               # color='k' means black
                plt.plot(np.arange(0,len(st[0].data))*1/sampling_rate+x_start,
                         st[0].data+dist,
                         color='darkred',
                         linewidth=3*linewidth)
            else:                                         # color='g' means green
                plt.plot(np.arange(0,len(st[0].data))*1/sampling_rate+x_start,
                         st[0].data+dist,
                         color='k',
                         linewidth=linewidth)
            # Plot P arrival marker in red
            P_marker, = plt.plot([P_time,P_time],[dist-0.5,dist+0.5],color='r',linewidth=2)
            # Plot S arrival marker in blue
            S_marker, = plt.plot([S_time,S_time],[dist-0.5,dist+0.5],color='b',linewidth=2)
            if label_stas!=False and len(label_stas)==0:
                plt.text(x_start,dist+0.1,f'{sta}',color='darkred',fontsize=12)
            elif label_stas!=False and sta in label_stas:
                plt.text(x_start,dist+0.1,f'{sta}',color='darkred',fontsize=12)
    plt.legend([P_marker,S_marker],['tele P','tele S'],loc='upper right')
    plt.tight_layout()
    if o_format.lower()=="pdf":
        plt.savefig(os.path.join(tele_file[:-5]+".pdf"))
    if o_format.lower()=="jpg" or o_format.lower=="jpeg":
        plt.savefig(os.path.join(tele_file[:-5]+".jpg"))
    if o_format.lower()=="png":
        plt.savefig(os.path.join(tele_file[:-5]+".png"))

    if save_result_wf == True:
        st_sum.write(os.path.join(tele_file[:-5]+".mseed"))
