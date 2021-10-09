#!/usr/bin/env python
# coding: utf-8

import obspy
import glob
import os
import re
from obspy import UTCDateTime
import warnings
import numpy as np
import multiprocessing as mp
import time
from tqdm import tqdm
from seisloc.sta import load_sta
warnings.filterwarnings("ignore")

def pos_write(o_folder,sta_dict):
    '''
    position write of station information
    '''
    sac_list=glob.glob(o_folder+"/"+"*.SAC")
    inp_status = False
    st = obspy.read(sac_list[0])
    net = st[0].stats.network
    sta = st[0].stats.station
    for sac in sac_list:
        st=obspy.read(sac)
        st[0].stats.sac.stla=sta_dict[net][sta][0]
        st[0].stats.sac.stlo=sta_dict[net][sta][1]
        st[0].stats.sac.stel=sta_dict[net][sta][2]
        st[0].write(sac)
        
def time_file_list(i_path,folder):
    '''
    Get the starttime and file path in sequence, append the last
    end time to the time list.
    '''
    file_list=os.listdir(os.path.join(i_path,folder))
    data_path_list = []
    t_point_list = []
    max_endtime = UTCDateTime(1900,1,1)
    for file in file_list:
        file_path = os.path.join(i_path,folder,file)
        try:
            st=obspy.read(file_path,headonly=True)
            starttime = st[0].stats.starttime
            t_point_list.append(starttime)
            endtime = st[0].stats.endtime
            if endtime>max_endtime:
                max_endtime = endtime
            data_path_list.append(file_path)
        except:    # not seismic file
            continue     
    tmp = np.array(t_point_list)
    data_path_list = np.array(data_path_list)
    k = tmp.argsort()
    data_path_list = data_path_list[k]
    t_point_list.sort()
    t_point_list.append(max_endtime)
    return t_point_list,data_path_list


def day_split(i_path,o_path,folder,sta_dict,format,shift_hour=0,cut_data=True):
        # get the ordered time list and corresponding file path 
        t_point_list,data_path_list = time_file_list(i_path,folder)
        
        if len(data_path_list)==0:
            return
 
        st = obspy.read(data_path_list[0],headonly=True)
        net = st[0].stats.network
        sta = st[0].stats.station

        o_folder = os.path.join(o_path,sta)
        if not os.path.exists(o_folder):
            os.mkdir(o_folder)

        for chn in ["BHE","BHN","BHZ"]:

            tts=t_point_list[0] #total start time is the start time of the first file 
            tte=t_point_list[-1]#total end time is the end time of the last file
            #start to get the trim point to form day_sac files
            trim_point_list=[]
            trim_point_list.append(tts)#the first point is the start time of the first file
            #information of the first day
            f_year=t_point_list[0].year
            f_month=t_point_list[0].month
            f_day=t_point_list[0].day
            #the second point should be the start time of the second day
            trim_node=UTCDateTime(f_year,f_month,f_day)+24*60*60+shift_hour
            #if the second point is less than the total end time, add it into list and move to next day
            while trim_node < tte:
                trim_point_list.append(trim_node)
                trim_node+=24*60*60
            #add the total time end to the last
            trim_point_list.append(tte)
            # write into txt file available days
            f = open(os.path.join(o_folder,'availdays.txt'),'w')
            tmp_list = []
            for time in trim_point_list:
                if time.julday not in tmp_list:
                    f.write(f"{time.julday}\n")
                    tmp_list.append(time.julday)
            f.close()
            if cut_data == False:
                return                  #only generate availdays.txt
            #for each time segment, need to check the corresponding files then load, merge and trim.
            for i in range(len(trim_point_list)-1):
                trim_s=trim_point_list[i] #trim start time
                trim_e=trim_point_list[i+1]#trim end time
                #get the index of trim_s and trim_e file
                for j in range(len(t_point_list)-1):
                    if trim_s>=t_point_list[j] and trim_s<t_point_list[j+1]:
                        count_s=j
                for k in range(len(t_point_list)-1):
                    if trim_e>=t_point_list[k] and trim_s<t_point_list[k+1]:
                        count_e=k
                #read in the coresponding file
                st=obspy.read(data_path_list[count_s])
                #read more file to include the trim_s and trim_e time
                while count_s < count_e:
                    count_s+=1
                    st+=obspy.read(data_path_list[count_s])
                #merge data, using try except to avoid mistake len(st)==1
                try:
                    st.merge(method=1,fill_value="interpolate")
                except:
                    pass
                #trim the data
                st.trim(starttime=trim_s,endtime=trim_e)
                #save the data
                for tr in st:
                    chn=tr.stats.channel
                    if format.upper()=="SAC":
                        f_name=net+"."+sta+"."+chn+"__"+\
                               trim_s.strftime("%Y%m%dT%H%M%SZ")+"__"+\
                               trim_e.strftime("%Y%m%dT%H%M%SZ")+".SAC"
                    if format.upper()=="MSEED":
                        f_name=net+"."+sta+"."+chn+"__"+\
                               trim_s.strftime("%Y%m%dT%H%M%SZ")+"__"+\
                               trim_e.strftime("%Y%m%dT%H%M%SZ")+".mseed"
                    tr.write(o_folder+"/"+f_name,format=format)
                    
        if format=="SAC":
            pos_write(o_folder,sta_dict)
            
            
def mp_day_split(i_path,o_path,sta_file,format="mseed",shift_hour=0,parallel=True,cut_data=True):
    '''
    This function reads in the data from QS5A devices and split them by days
    Parameters:
    i_path: base folder containing all the source files. base_folder/sta_folder/seismic_files
    o_path: output path. output_path/sta_name/seismic_files.
    sta_file: station files containing the longitude and latitude information
    format: "SAC" or "MSEED"
    shift_hour: Shift the trimming time with reference to hour 0. The main purpose is the adjust time zone issue
    '''
    # First load in the station file
    print("# Loading station files ...")
    sta_dict = load_sta(sta_file)

    dir_list = os.listdir(i_path) # Get dir list before create below folder 
    if not os.path.exists(o_path):
        os.mkdir(o_path)
    folder_list=[]
    for item in dir_list:
        if os.path.isdir(os.path.join(i_path,item)) and item[0] !="\.":
            folder_list.append(item)
    print("# Folders to be processed: ",len(folder_list))
    
    # Now check whether all the stations 
    status = True
    false_folder = []
    print("# Check the network and station name...")
    for folder in tqdm(folder_list):
        file_list=os.listdir(os.path.join(i_path,folder))
        for file in file_list:
            try:
                st = obspy.read(os.path.join(i_path,folder,file),headonly=True)
                net = st[0].stats.network
                sta = st[0].stats.station
                if net not in sta_dict:
                    false_folder.append([folder,net,sta])
                else:
                    try:
                        if sta not in sta_dict[net]:
                            false_folder.append([folder,net,sta])
                        else:
                            pass
                    except:
                        false_folder.append([folder,net,sta])
                break
            except:
                continue
    if len(false_folder)>0:
        status = False
        for record in false_folder:
            print(f"Error net or sta in folder: {record[0]} net:{record[1]} sta:{record[2]}")
        
    if status == True and parallel==True:
        print("# Now multiprocessing...")
        cores = int(mp.cpu_count()/2)
        tasks = []
        for folder in folder_list:
            tasks.append([i_path,o_path,folder,sta_dict,format,shift_hour,cut_data])
        pool = mp.Pool(processes=cores)
        rs = pool.starmap_async(day_split,tasks,chunksize=1)
        pool.close()
        while True:
            remaining = rs._number_left
            print(f"Finished {len(tasks)-remaining}/{len(tasks)}",end='\r')
            if(rs.ready()):
                break
            time.sleep(0.5)
    if status == True and parallel==False:
        for folder in folder_list:
            day_split(i_path,o_path,folder,sta_dict,format,shift_hour,cut_data=cut_data)

if __name__=="__main__":
    """
    This function reads in the data from QS5A devices and split them by days
    Parameters:
    i_path: base folder containing all the source files. base_folder/sta_folder/seismic_files
    o_path: output path. output_path/sta_name/seismic_files.
    sta_file: station files containing the longitude and latitude information
    """
    i_path = "./raw_data"
    o_path = './day_data'
    sta_file = './sta.txt'
    
    mp_day_split(i_path,o_path,sta_file)
