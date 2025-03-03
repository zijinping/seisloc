import re
import os
import shutil
import random
from obspy import UTCDateTime
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import subprocess
from seisloc.geometry import spherical_rotate
from seisloc.utils import readfile,writefile
import pandas as pd

def load_PC(catalog="/home/zijinping/Desktop/projects/wy_eq/2018_2019_PC/2018_2019_hypoDD.reloc",
            start_evid=300000):
    """
    Read in Pengcheng's earthquake catalog
    """
    evid = start_evid+1
    catalog_dict = {}
    cont = []
    with open(catalog,'r') as f:
        cont = f.readlines()
    for line in cont:
        _time,_,_lat,_lon,_dep,_mag, = re.split(" +",line.rstrip())[:6]
        etime = UTCDateTime.strptime(_time,'%Y%m%d%H%M%S')
        catalog_dict[evid] = [float(_lon),float(_lat),float(_dep),float(_mag),etime]
        evid += 1
    return catalog_dict

def load_CEDC(catalog="/home/zijinping/Dropbox/resources/catalog/CEDC/20090101_20201231.txt",start_evid=100000):
    """
    Read in China Earthquake Data Center catalog
    """
    evid = start_evid+1
    catalog_dict = {}
    cont = []
    with open(catalog,'r') as f:
        cont = f.readlines()
    for line in cont:
        _time,_lon,_lat,_dep,M,_mag,_,_ = re.split(",",line.rstrip())
        #print(_time,_lon,_lat,_dep,_mag)
        _date,_hr_min = re.split(" ",_time)
        _yr,_mo,_dy = re.split("\/",_date)
        _hr,_min = re.split(":",_hr_min)
        yr = int(_yr); mo = int(_mo); dy = int(_dy)
        hr = int(_hr); minute = int(_min)
        etime = UTCDateTime(yr,mo,dy,hr,minute,0)
        catalog_dict[evid] = [float(_lon),float(_lat),float(_dep),float(_mag),etime]
        evid += 1
    return catalog_dict


def dd_event_sel(evid_list=[],event_dat="event.dat",event_sel="event.sel"):
    '''
    select events in the "event.dat" file and output them into
    the "event.sel" file by the event ID list provided
    '''
    content = []
    # Read in data
    with open(event_dat,'r') as f:
        for line in f:
            line = line.rstrip()
            evid = int(line[-8:])
            if evid in evid_list:
                content.append(line)
    f.close()

    # Output into target file
    with open(event_sel,'w') as f:
        for line in content:
            f.write(line+"\n")
    f.close()

def dtct_sel(evid_list,input_file):
    """
    Output clean dtct file with event id list provided
    """
    out_file = input_file+".sel"
    out_cont = []
    record_status = False # Set initiate value
    with open(input_file,'r') as f:
        for line in f:
            if line[0]=="#": # pair line
                print(re.split(" +",line.rstrip()))
                _,ID1,ID2 = re.split(" +",line.rstrip())
                ID1 = int(ID1)
                ID2 = int(ID2)
                if (ID1 in evid_list) and (ID2 in evid_list):
                    record_status = True
                    out_cont.append(line.rstrip())
                else:
                    record_status = False
            elif record_status==True:
                out_cont.append(line.rstrip())
    f.close()

    with open(out_file,'w') as f:
        for line in out_cont:
            f.write(line+"\n")


def hypoDD_rmdup(in_file="total_hypoDD.reloc"):
    """
    remove duplicated events and take mean values
    """
    count=0
    evid_list = []
    evid_mapper = {}
    with open(in_file,'r') as f:
        for line in f:
            evid = int(line[0:11])
            try:
                evid_mapper[evid].append(line)
                count += 1
            except:
                evid_mapper[evid]=[line]
    f.close()

    evid_list = list(evid_mapper)
    evid_list.sort()

    log_record = []
    f=open(in_file+".rm",'w')
    f.close()
    for evid in evid_list:
        if len(evid_mapper[evid])>1:
            lon_list = []
            lat_list = []
            dep_list = []
            for i in range(len(evid_mapper[evid])):
                lon = float(evid_mapper[evid][i][22:32])
                lon_list.append(lon)
                lat = float(evid_mapper[evid][i][11:20])
                lat_list.append(lat)
                dep = float(evid_mapper[evid][i][36:42])
                dep_list.append(dep)
            lon_mean = np.mean(lon_list)
            lat_mean = np.mean(lat_list)
            dep_mean = np.mean(dep_list)
            lon_str = format(lon_mean,'10.6f')
            lat_str = format(lat_mean,'9.6f')
            dep_str = format(dep_mean,'6.3f')
            log_record.append([evid,lon_str,lat_str,dep_str])
            firstr = evid_mapper[evid][0]  #Use the first record as template
            outstr = firstr.replace(line[22:32],lon_str,1) #Replace lon
            outstr = outstr.replace(line[11:20],lat_str,1) #Replace lat
            outstr = outstr.replace(line[22:32],dep_str,1) #Replace dep
            with open("total_hypoDD.reloc.rm",'a') as f:
                f.write(outstr)
            f.close()
        else:
            with open(in_file+".rm",'a') as f:
                f.write(evid_mapper[evid][0])
            f.close()
    print(log_record)

def hypoDD_ref_days(reloc_file,ref_time,shift_hours=0):
    """
    Add one column to the last of hypoDD files, calculate the length of time 
    between the referece time and the event time in days.
    The output is a file with the same title with reloc_file and add ".add" as
    suffix.

    Parameters
    ----------
     reloc_file: The hypoDD relocation file.
       ref_time: Reference time in UTCDateTime format
    shift_hours: Used when event time is not in UTC time zone
    """

    new_add=[]
    with open(reloc_file,"r") as f:
        for line in f:
            year = int(re.split(" +",line)[11])
            month = int(re.split(" +",line)[12])
            day = int(re.split(" +",line)[13])
            hour = int(re.split(" +",line)[14])
            minute = int(re.split(" +",line)[15])
            seconds = float(re.split(" +",line)[16])
            eve_time = UTCDateTime(year,month,day,hour,minute)+seconds
            days = (eve_time - ref_time)*1.0/(24*60*60)
            new_line=line[:-1]+" "+format(days,'4.2f')
            new_add.append(new_line)
    f.close()
    with open(reloc_file+".add","w") as f:
        for line in new_add:
            f.write(line+"\n")
    f.close()

def compare_DD(dd1_path,dd2_path):
    dd1,_ = load_DD(dd1_path)
    dd2,_ = load_DD(dd2_path)
    f = open("dd_diff.dat",'w')
    for key in dd1:
        try: 
            lon1 = dd1[key][0] 
            lon2 = dd2[key][0] 
            lat1 = dd1[key][1] 
            lat2 = dd2[key][1] 
            dep1 = dd1[key][2] 
            dep2 = dd2[key][2]
            print("dx,dy,dz:",abs(lon1-lon2)*111*1000,abs(lat1-lat2)*111*1000,abs(dep1-dep2)*1000)
            f.write(f"{abs(lon1-lon2)*111*1000} {abs(lat1-lat2)*111*1000} {abs(dep1-dep2)*1000}\n")
        except: 
            pass 
    f.close()

def hypoDD_mag_mapper(reloc_file,out_sum,magcolumn_index=128):
    """    
    The output of hypoDD doesn't contain magnitude information.
    This function reads magnitude information from *.sum file, which is the
    output of hyperinverse and provide to hypoDD file.
           
    The results will cover the input reloc_fiie
    """    
           
    #get the magnitude dictionary
    event_mag_list = {}
    with open(out_sum,"r") as f_obj:
        for line in f_obj:
            event_id = int(line[136:146])
            event_mag = int(line[123:126])*0.01
            event_mag_list[event_id]=event_mag
    f_obj.close()
    print(len(event_mag_list.keys()))
    #add in the magnitude

    new_dd = []
    with open(reloc_file,"r") as f_obj:
        for line in f_obj:
            dd_event_id = int(line[0:9])
            dd_event_mag = event_mag_list[dd_event_id]
            new_line=line[:magcolumn_index]+format(dd_event_mag,'5.2f')+line[magcolumn_index+4:]
            new_dd.append(new_line)
    f_obj.close()
    with open(reloc_file+".mag","w") as f_obj:
        for line in new_dd:
            f_obj.write(line)
    f_obj.close()

def load_DD(reloc_file="hypoDD.reloc",shift_hour=0):
    """
    load results of hypoDD
    return eve_dict, df

    Parameters
    ----------
    If the time of results is not in UTC time zone, a time shift might needed.
    For example, Beijing time zone is 8 hours early than UTC time, 8 hours 
    should be deducted so as to be consistent with UTC time.
    """

    eve_dict={}
    columns = ["ID","LAT","LON","DEPTH","X","Y","Z","EX","EY","EZ",\
           "YR","MO","DY","HR","MI","SC","MAG",\
           "NCCP","NCCS","NCTP","NCTS","RCC","RCT","CID"]
    number = 0
    dataset = []
    with open(reloc_file,'r') as f:
        for line in f:
            line = line.rstrip()
            if "*" not in line:
                data = list(map(float,line.split()))
                dataset.append(data)
    dataset = np.array(dataset)
    
    if dataset.shape[1] == 24: # format of hypoDD.reloc file
        columns = ["ID","LAT","LON","DEPTH","X","Y","Z","EX","EY","EZ",\
                   "YR","MO","DY","HR","MI","SC","MAG",\
                   "NCCP","NCCS","NCTP","NCTS","RCC","RCT","CID"]
    if dataset.shape[1] == 25: # format of modified hypoDD.reloc, note the last column
        columns = ["ID","LAT","LON","DEPTH","X","Y","Z","EX","EY","EZ",\
                   "YR","MO","DY","HR","MI","SC","MAG",\
                   "NCCP","NCCS","NCTP","NCTS","RCC","RCT","CID","DAY"]
    if dataset.shape[1] == 18: # format of hypoDD.loc file
        columns = ["ID","LAT","LON","DEPTH","X","Y","Z","EX","EY","EZ",\
                   "YR","MO","DY","HR","MI","SC","MAG","CID"]

    
    for i,data in enumerate(dataset):
        eve_id = data[0]
        eve_lat = data[1]
        eve_lon = data[2]
        eve_dep = data[3]
        eve_mag = data[16]
        eve_time = UTCDateTime(int(data[10]),int(data[11]),int(data[12]),int(data[13]),int(data[14]),0)+data[15] - shift_hour*50*60
        eve_dict[int(eve_id)]=[float(eve_lon),float(eve_lat),float(eve_dep),float(eve_mag),eve_time]

    df = pd.DataFrame(data=dataset,columns=columns)
    return eve_dict,df


def run_dd(base_dir="./",work_dir='hypoDD',inp_file="hypoDD.inp"):
    os.chdir(base_dir)
    os.chdir(work_dir)
    subprocess.run(["hypoDD",inp_file])
    os.chdir(base_dir)

def dd_bootstrap(base_folder="hypoDD",times=10,method="event",samp_ratio=0.75,cores=2):
    """
    Randomly run hypoDD with randomly selected events to show the results variation
    Parameters:
        base_folder: the basement folder which should include the material for hypoDD, 
            including dt.ct, dt.cc,hypoDD.inp, event.dat, station.dd
        times: number of hypoDD runs
        method: "event" means sample events; "phase" means sample phases
        samp_ratio: the ratio of events to be relocated in the run
    """
    # Load in event.dat file
    base_dir = os.getcwd()
    e_dat = []
    if method == "event":
        with open(os.path.join(base_folder,"event.dat"),'r') as f:
            for line in f:
                evid = int(line[84:91])
                e_dat.append(line)
        e_qty = len(e_dat)
        s_qty = int(e_qty*samp_ratio) # sample qty

        tar_folders = []
        tasks = []
        # Prepare the subroutine files
        for i in range(1,times+1):
            tar_folder = base_folder + str(i).zfill(3)
            shutil.copytree(base_folder,tar_folder)
            sel_idxs = random.sample(range(e_qty),s_qty)
            with open(os.path.join(tar_folder,'event.sel'),'w') as f:
                for idx in sel_idxs:
                    f.write(e_dat[idx])
            f.close()
            tar_folders.append(tar_folder)
            tasks.append([base_dir,tar_folder])
            
    if method == "phase":
        dtct = []
        out_dtct = []
        with open(os.path.join(base_folder,"dt.ct"),'r') as f:
            for line in f:
                dtct.append(line)        
        len_dtct = len(dtct)
        
        tar_folders = []
        tasks = []
        # Prepare the subroutine files
        for i in range(1,times+1):
            tar_folder = base_folder + str(i).zfill(3)
            shutil.copytree(base_folder,tar_folder)
            with open(os.path.join(tar_folder,'dt.ct'),'w') as f:
                
                for i,line in enumerate(dtct):
                    if line[0] == "#":    # event line
                        f.write(dtct[i])
                        tmp = []
                        j = i+1
                        while j<len_dtct and dtct[j][0]!="#":
                            tmp.append(dtct[j])
                            j=j+1
                        pha_qty = len(tmp)
                        sample_qty = int(pha_qty*samp_ratio+0.5)
                        sel_idxs = random.sample(range(pha_qty),sample_qty)
                        for idx in sel_idxs:
                            f.write(tmp[idx])


            tar_folders.append(tar_folder)
            tasks.append([base_dir,tar_folder])
            
    pool = mp.Pool(processes=cores)
    rs = pool.starmap_async(run_dd,tasks,chunksize=1)
    while True:
        remaining = rs._number_left
        print(f"Finished {len(tasks)-remaining}/{len(tasks)}",end='\r')
        if(rs.ready()):
            break
        time.sleep(0.5)
    print("\nDone!!!")
    
def bootstrap_summary(times,base_folder="hypoDD"):
    """
    """
    rand_dict = {}
    
    with open(os.path.join(base_folder,"event.dat"),'r') as f:
        for line in f:
            evid = int(line[84:91])
            rand_dict[evid] = []

    
    tar_folders = []
    for i in range(1,times+1):
        tar_folder = base_folder + str(i).zfill(3)
        tar_folders.append(tar_folder)

    # Load hypoDD results
    print("Loading results ...")
    for tar_folder in tar_folders:
        reloc_file = os.path.join(tar_folder,'hypoDD.reloc')
        dd_dict,_ = load_DD(reloc_file)
        for key in dd_dict.keys():
            rand_dict[key].append(dd_dict[key][:4])
    
    print("Write out results ... ")
    f = open("hypoDD.rand",'w')
    for key in rand_dict.keys():
        cont = rand_dict[key]
        if len(cont) == 0:
            continue
        else:
            lon_list = []
            lat_list = []
            dep_list = []
            for tmp in cont:
                lon_list.append(tmp[0])
                lat_list.append(tmp[1])
                dep_list.append(tmp[2])
            record_qty = len(cont)
            mean_lon = np.mean(lon_list)
            mean_lat = np.mean(lat_list)
            mean_dep = np.mean(dep_list)
            std_lon = np.std(lon_list)
            std_lat = np.std(lat_list)
            std_herr = np.sqrt(std_lon**2+std_lat**2)
            std_dep = np.std(dep_list)
            f.write(format(key,'7d')+" "+
                format(mean_lon,'8.4f')+" "+
                format(mean_lat,'7.4f')+" "+
                format(mean_dep*1000,'9.3f')+" "+
                format(std_herr*111.1*1000*2,"9.3f")+" "+
                format(std_dep*1000*2,"8.3f")+" "+
                format(record_qty,'3d')+"\n")
    f.close()

def pha_subset(pha_file,loc_filter,obs_filter=8,out_file=None):
    """
    *.pha file is the input file for hypoDD ph2dt, this function subset the
    pha file by the boundary condition and the minimum observation condition.
    The output file is a file with ".subset" suffix

    Parameters
    ----------
    pha_file: Str. The input file.
    loc_filter: array in format [lon_min, lon_max, lat_min, lat_max]
    obs_filter: The minimum observation
    out_path: file path for the target file
    """

    lon_min, lon_max, lat_min, lat_max = loc_filter
    if out_file == None:
        out_file = pha_file+".subset"
    f = open(out_file,"w")
    f.close()
    pha_content = []
    with open(pha_file,"r") as f:
        for line in f:
            pha_content.append(line.rstrip())
    f.close()
    i = 0
    j = 0
    record_list=[]
    for line in pha_content:
        if line[0]=="#":
            if i>0 and len(record_list) > (obs_filter+1):
                j=j+1
                with open(out_file,"a") as f:
                    for record in record_list:
                        f.write(record+"\n")
                f.close()
                record_list = []
                record_list.append(line)
            else:
                record_list = []
                record_list.append(line)
            i=i+1
            lat = float(re.split(" +",line)[7])
            lon = float(re.split(" +",line)[8])
            if lat>lat_min and lat<lat_max and lon>lon_min and lon<lon_max:
                region_pass = True
            else:
                region_pass = False
        else:
            if region_pass:
                record_list.append(line)
    if i>0 and len(record_list) > (obs_filter+1):
        j=j+1
        with open(out_file,"a") as f:
            for record in record_list:
                f.write(record+"\n")
        f.close()
    print("Event before filtering",i)
    print("Events qty after filtering",j)

def pha_sel(pha_file,e_list=[],remove_net=False):
    '''
    Select phases of events in e_list
    if need to remove net name.
    '''
    out = []   # output
    with open(pha_file,'r') as f:
        for line in f:
            line = line.rstrip()
            if line[0]=="#":
                _evid = re.split(" +",line)[-1]
                evid = int(_evid)
                if evid in e_list or e_list==[]:
                    status = True
                    out.append(line)
                else:
                    status = False
            else:
                if status==True:
                    if remove_net:
                        out.append(line[2:])
                    else:
                        out.append(line)
    f.close()

    with open(pha_file+".sel",'w') as f:
        for line in out:
            f.write(line+'\n')
    f.close()

def inv_dd_compare(inv,dd,keys=[],xlim=[],ylim=[],aspect='auto'):
    inv_locs = []
    dd_locs = []
    for key in keys:
        key = int(key)
        inv_lon = inv[key][1]
        inv_lat = inv[key][2]
        dd_lon = dd[key][0]
        dd_lat = dd[key][1]
        inv_locs.append([inv_lon,inv_lat])
        dd_locs.append([dd_lon,dd_lat])
    inv_locs = np.array(inv_locs)
    dd_locs = np.array(dd_locs)
    plt.plot(inv_locs[:,0],inv_locs[:,1],'kx')
    plt.plot(dd_locs[:,0],dd_locs[:,1],'rv')
    if len(xlim)>0:
        plt.xlim(xlim)
    if len(ylim)>0:
        plt.ylim(ylim)
    plt.gca().set_aspect(aspect)

def rotDD(ddfile,center,rotate):
    """
    Parameters:
    |  center: the rotation center [lon,lat], positive for E and N
    |  rotate: rotate angle in degree, postive for anticlockwise
    """
    cont = readfile(ddfile)
    
    new_cont = []
    for line in cont:
        _evid,_lat,_lon = line.split()[:3]
        lat = float(_lat)
        lon = float(_lon)
        rotated = spherical_rotate([lon,lat],center=center,degree=rotate)
        new_lon = rotated[0,0]
        new_lat = rotated[0,1]
        new_line = line[:9] + format(new_lat,'>11.6f')+format(new_lon,'>12.6f')+line[32:]
        new_cont.append(new_line)
    
    writefile(new_cont,ddfile+".rot")

def subset_dtcc(evids,ccFilePth,saveFilePth="",useStas=[]):
    linesUse = []      
    ccFileName = os.path.basename(ccFilePth)
    with open(ccFilePth,'r') as f:
        for line in tqdm(f):  
            if line[0] == "#":
                record=False
                _,_evid1,_evid2,_ = line.split()
                if int(_evid1) in evids or int(_evid2) in evids:
                    linesUse.append(line)
                    record = True
            else:          
                if record==True:
                    if useStas==[]:
                        linesUse.append(line)
                    else:
                        sta = re.split(" +",line)[0]
                        if sta in useStas:
                            linesUse.append(line)
    if saveFilePth=="":
        saveFilePth=ccFileName+'.subset'
    with open(saveFilePth,'w') as f:
        for line in linesUse:
            f.write(line)

def subset_dd_file(evids,ddFilePth,saveFilePth=""):
    linesUse = []      
    ddFileName = os.path.basename(ddFilePth)
    with open(ddFilePth,'r') as f:
        for line in tqdm(f):  
                _evid = line[:9]
                if int(_evid) in evids:
                    linesUse.append(line)
    if saveFilePth=="":
        saveFilePth=ddFileName+'.subset'
    with open(saveFilePth,'w') as f:
        for line in linesUse:
            f.write(line)
