from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
import re
from seisloc.hypoinv import load_sum_evstr,load_sum_evid
import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DD():
    def __init__(self,reloc_file="hypoDD.reloc"):
        self.dict,_ = load_DD(reloc_file)
        self.locs = []
        for key in self.dict.keys():
            lon = self.dict[key][0]
            lat = self.dict[key][1]
            dep = self.dict[key][2]
            mag = self.dict[key][3]
            self.locs.append([lon,lat,dep,mag])
        self.locs = np.array(self.locs)
    def plot(self,xlim=[],ylim=[],markersize=6,size_ratio=1,imp_mag=3):

        plt.scatter(self.locs[:,0],
                    self.locs[:,1],
                    (self.locs[:,3]+2)*size_ratio,
                    edgecolors = "k",
                    facecolors='none',
                    marker='o',
                    alpha=1)

        kk = np.where(self.locs[:,3]>=imp_mag)
        if len(kk)>0:
            imp = plt.scatter(self.locs[kk,0],
                        self.locs[kk,1],
                        (self.locs[kk,3]+2)*size_ratio*5,
                        edgecolors ='r',
                        facecolors='none',
                        marker='*',
                        alpha=1)
            plt.legend([imp],[f"M$\geq${format(imp_mag,'4.1f')}"])
        if len(xlim) != 0:
            plt.xlim(xlim)
        if len(ylim) != 0: 
            plt.ylim(ylim)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()
    
    def __repr__(self):
        _qty = f"HypoDD relocation catlog with {len(self.dict.keys())} events\n"
        _mag = f"Magnitue range is: {format(np.min(self.locs[:,3]),'4.1f')} to {format(np.max(self.locs[:,3]),'4.1f')}\n"
        _lon = f"Longitude range is: {format(np.min(self.locs[:,0]),'8.3f')} to {format(np.max(self.locs[:,0]),'8.3f')}\n"
        _lat = f"Latitude range is: {format(np.min(self.locs[:,1]),'7.3f')} to {format(np.max(self.locs[:,1]),'7.3f')}\n"
        _dep = f"Depth range is: {format(np.min(self.locs[:,2]),'4.1f')} to {format(np.max(self.locs[:,2]),'4.1f')}\n"
        return _qty+_mag+_lon+_lat+_dep
    
    def __getitem__(self,key):
        return self.dict[key]


def event_sel(evid_list=[],event_dat="event.dat",event_sel="event.sel"):
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

def gen_dtcc(sta_list,sum_file="out.sum",work_dir="./",cc_threshold=0.7,min_link=4,max_dist=4):
    '''
    This function generate dt.cc.* files from the output of SCC results
    
    Parameters:
        sta_list: list of stations to be processed
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
        
    print(">>> Loading in scc results ...")
    for sta in tqdm(sta_list):                                # Loop for station
        for pha in ["P","S"]:                                 # Loop for phases
            sta_pha = sta+"_"+pha
            globals()[sta_pha+"_cc_dict"] = {}                # Initiate dictionary
            sta_pha_path = os.path.join(work_dir,sta_pha)
            for file in os.listdir(sta_pha_path):
                if file[-3:]!=".xc":                          # none scc results file
                    continue
                with open(os.path.join(sta_pha_path,file),'r') as f:
                    for line in f:
                        line = line.rstrip()
                        path1,arr1,_,path2,arr2,_,cc,aa=re.split(" +",line.rstrip())
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
                        cc = float(cc)                        # cross correlation coefficient
                        aa = float(aa)                        # amplitude ratio
                        if cc >=cc_threshold:
                            try:
                                globals()[sta_pha+"_cc_dict"][evid1][evid2]=[arr1,arr2,cc,aa]
                            except:
                                globals()[sta_pha+"_cc_dict"][evid1]={} # Initiation
                                globals()[sta_pha+"_cc_dict"][evid1][evid2]=[arr1,arr2,cc,aa]
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
            for sta in sta_list:                            # Loop for stations
                for pha in ["P","S"]:                       # Loop for phases
                    sta_pha = sta+"_"+pha               
                    try:
                        arr1,arr2,cc,aa = globals()[sta_pha+"_cc_dict"][evid1][evid2]
                        link_cc.append([sta,arr1-arr2,cc,pha])
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
                        f.write(f"{record[0][:2]}{record[0]} {format(record[1],'7.4f')} {record[2]} {record[3]}\n")
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

def hypoDD_hist(dd_file="hypoDD.reloc",ref_time=UTCDateTime(2019,3,1,0,0,0)):
    """
    Plot events by day-quantity in a histogram plot.
    Parameters:
        -dd_file: Path of hypoDD file
        -ref_time: Reference time for plot
    """
    ref_list = []
    time_list = []
    number=0
    if isinstance(dd_file,str):
        with open(dd_file,"r") as f:
            for line in f:
                number = number+1
                if number%10==0:
                    print("Current in process %d    "%number,end='\r')
                data=re.split(" +",line.rstrip())[1:]
                try:
                    data_arr = np.vstack((data_arr,data))
                except:
                    data_arr = np.array(data)
                eve_id = data[0]
                eve_lat = data[1]
                eve_lon = data[2]
                eve_dep = data[3]
                year = int(data[10])
                month = int(data[11])
                day = int(data[12])
                hour = int(data[13])
                minute = int(data[14])
                seconds = float(data[15])
                eve_time = UTCDateTime(year,month,day,hour,minute)+seconds
                time_list.append(eve_time)
                gap_time = eve_time - ref_time
                ref_list.append((eve_time-ref_time)/(60*60*24))
    elif isinstance(dd_file,list)and len(dd_file)!=0:
        for dd_f in dd_file:
            with open(dd_f,"r") as f:
                for line in f:
                    number = number+1
                    if number%10==0:
                        print("Current in process %d    "%number,end='\r')
                    data=re.split(" +",line.rstrip())[1:]
                    try:
                        data_arr = np.vstack((data_arr,data))
                    except:
                        data_arr = np.array(data)
                    eve_id = data[0]
                    eve_lat = data[1]
                    eve_lon = data[2]
                    eve_dep = data[3]
                    year = int(data[10])
                    month = int(data[11])
                    day = int(data[12])
                    hour = int(data[13])
                    minute = int(data[14])
                    seconds = float(data[15])
                    eve_time = UTCDateTime(year,month,day,hour,minute)+seconds
                    time_list.append(eve_time)
                    gap_time = eve_time - ref_time
                    ref_list.append((eve_time-ref_time)/(60*60*24))
    min_day=floor(min(ref_list))
    max_day=ceil(max(ref_list))
    bins = np.linspace(min_day,max_day,max_day-min_day+1)
    fig1 = plt.figure(1,figsize=(8,4))
    ax1 = plt.subplot(1,1,1)
    ax1.hist(ref_list,bins)
    # The bottom x-axis is in days
    ax1.set_xlim([0,max_day])
    # The top x-axis marks year and month in YYYYMM
    tick_list_1 = [] # Store the position number
    tick_list_2 = [] # Store the tick text
    ref_year = ref_time.year
    ref_month = ref_time.month
    ref_day = ref_time.day
    if ref_day == 1:
        tick_list_1.append(0)
        tick_list_2.append(str(ref_year)+str(ref_month).zfill(2))
    status = True # Start to loop month by month
    loop_time = UTCDateTime(ref_year,ref_month,1) # Initiate loop time
    step = 32 #32 > 31. Make sure each step pass to next month
    while status==True:
        loop_time = loop_time + step*24*60*60
        tmp_year = loop_time.year
        tmp_month = loop_time.month
        loop_time = UTCDateTime(tmp_year,tmp_month,1)
        diff_days = (loop_time - ref_time)/(24*60*60)
        if diff_days > (max_day):
            status=False
        else:
            tick_list_1.append(diff_days)
            tick_list_2.append((str(tmp_month).zfill(2)))
    ax2 = ax1.twiny()
    ax2.set_xlim([0,max_day])
    ax2.plot(0,0,'k.')
    plt.xticks(tick_list_1,tick_list_2)
    ax1.set_xlabel("Time, days")
    ax1.set_ylabel("event quantity")
    ax2.set_xlabel("date")
    plt.show()

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

def hypoDD_mag_mapper(reloc_file,out_sum):
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
            new_line=line[:128]+format(dd_event_mag,'5.2f')+line[132:]
            new_dd.append(new_line)
    f_obj.close()
    with open(reloc_file,"w") as f_obj:
        for line in new_dd:
            f_obj.write(line)
    f_obj.close()

def load_DD(reloc_file="hypoDD.reloc",shift_hour=0):
    """
    load results of hypoDD
    return eve_dict, df

    Parameters
    ----------
    If the time of results is not in UTC time zone, a time shift is needed.
    For example, Beijing time zone is 8 hours early than UTC time, 8 hours 
    should be deducted so as to be consistent with UTC time.
    """

    eve_dict={}
    columns = ["ID","LAT","LON","DEPTH","X","Y","Z","EX","EY","EZ",\
           "YR","MO","DY","HR","MI","SC","MAG",\
           "NCCP","NCCS","NCTP","NCTS","RCC","RCT","CID"]
    number = 0

    dataset = np.loadtxt(reloc_file)
    
    if dataset.shape[1] == 24:
        columns = ["ID","LAT","LON","DEPTH","X","Y","Z","EX","EY","EZ",\
                   "YR","MO","DY","HR","MI","SC","MAG",\
                   "NCCP","NCCS","NCTP","NCTS","RCC","RCT","CID"]
    if dataset.shape[1] == 25:
        columns = ["ID","LAT","LON","DEPTH","X","Y","Z","EX","EY","EZ",\
                   "YR","MO","DY","HR","MI","SC","MAG",\
                   "NCCP","NCCS","NCTP","NCTS","RCC","RCT","CID","DAY"]
    
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

def dd_bootstrap(base_folder="hypoDD",times=10,samp_ratio=0.75,cores=2):
    """
    Randomly run hypoDD with randomly selected events to show the results variation
    Parameters:
        base_folder: the basement folder which should include the material for hypoDD, 
            including dt.ct, dt.cc,hypoDD.inp, event.dat, station.dd
        times: number of hypoDD runs
        samp_ratio: the ratio of events to be relocated in the run
    """
    # Load in event.dat file
    base_dir = os.getcwd()
    e_dat = []
    rand_dict = {}
    with open(os.path.join(base_folder,"event.dat"),'r') as f:
        for line in f:
            evid = int(line[84:91])
            rand_dict[evid] = []
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
    
    pool = mp.Pool(processes=cores)
    rs = pool.starmap_async(run_dd,tasks,chunksize=1)
    while True:
        remaining = rs._number_left
        print(f"Finished {len(tasks)-remaining}/{len(tasks)}",end='\r')
        if(rs.ready()):
            break
        time.sleep(0.5)
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
            std_dep = np.std(dep_list)
            f.write(format(key,'7d')+" "+
                format(mean_lon,'8.4f')+" "+
                format(mean_lat,'7.4f')+" "+
                format(mean_dep*1000,'9.3f')+" "+
                format(std_lon*111.1*1000,"9.3f")+" "+
                format(std_lat*111.1*1000,"9.3f")+" "+
                format(std_dep*1000,"8.3f")+" "+
                format(record_qty,'3d')+"\n")

    f.close()
    print("Done")

def pha_subset(pha_file,loc_filter,obs_filter=8,out_file=None):
    """
    *.pha file is the input file for hypoDD ph2dt, this function subset the
    pha file by the boundary condition and the minimum observation condition.
    The output file is a file with ".st" suffix

    Parameters
    ----------
    pha_file: Str. The input file.
    loc_filter: array in format [lon_min, lon_max, lat_min, lat_max]
    obs_filter: The minimum observation
    out_path: file path for the target file
    """

    lon_min, lon_max, lat_min, lat_max = loc_filter
    if out_file == None:
        out_file = pha_file+".st"
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
