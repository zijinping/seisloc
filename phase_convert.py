#-----------------------------------------------------------------------------
# coding: utf-8
# author: Hardy ZI
# history:
#		...
#       2021-03-15 Add CN2fsdn function
#

import re
from obspy import UTCDateTime
import os
import sys
import time
from seisloc.dd import load_DD
import time
from seisloc.geopara import WY_para
from tqdm import tqdm


def arc2cnv(arc_file,mag_threshold=-9,qty_limit=None):
    """
    convert y2000 archieve file into velest format file
    """
    count = 0
    cont = []    # Save the content of the phase file
    cont_sel = [] # content selected
    out_dict = {}     # output content
    with open(arc_file,'r') as f:
        for line in f:
            cont.append(line.rstrip())
    f.close()

    for line in tqdm(cont):
        if re.match("\d+",line[:6]):    # line start with digit is summary line
            e_label = line[:16]
            e_time = UTCDateTime.strptime(line[:12],'%Y%m%d%H%M')+int(line[12:16])*0.01
            e_lat = int(line[16:18])+int(line[19:23])*0.01/60
            e_lon = int(line[23:26])+int(line[27:31])*0.01/60
            e_dep = int(line[31:36])*0.01
            record_status = True
            try:
                if len(line) < 126:
                    e_mag = int(line[36:39])*0.01
                else:
                    e_mag = int(line[123:126])*0.01
            except:
                e_mag = 0
            if e_mag<mag_threshold:
                record_status=False
            if e_dep==0:                # no zero depth for VELEST
                record_status=False
            if qty_limit!=None and count >= qty_limit:
                record_status = False

            if record_status == False:
                continue
            
            cont_sel.append(line)
            count += 1
            out_dict[e_label] = {}    # dict with event line str as key
            out_dict[e_label]["e_time"]=e_time
            out_dict[e_label]["e_lat"]=e_lat
            out_dict[e_label]["e_lon"]=e_lon
            out_dict[e_label]["e_dep"]=e_dep
            out_dict[e_label]["e_mag"]=e_mag
            out_dict[e_label]["phase"]=[]   # array to store phases
        elif line[:6]=='      ':        # The last line indicate the evid
            if record_status==False:
                continue
            cont_sel.append(line)
            out_dict[e_label]["evid"]=int(line[63:72])
        else:                           # Station phase line
            if record_status==False:
                continue
            cont_sel.append(line)
            sta = line[:5].split()
            sta = sta[0]
            net = line[5:7]
            if line[14] == "P":
                pha = "P"
                phs_time = UTCDateTime.strptime(line[17:29],"%Y%m%d%H%M")+\
                            int(line[30:34])*0.01
                diff_time = phs_time - e_time
            elif line[47] == "S":
                pha = "S"
                phs_time = UTCDateTime.strptime(line[17:29],"%Y%m%d%H%M")+\
                            int(line[42:46])*0.01
                diff_time = phs_time - e_time
            pha_record = sta.ljust(5," ")+pha+"1"+format(diff_time,'6.2f')
            out_dict[e_label]["phase"].append(pha_record)
    print(f"# Total {count} events!")
    cnv_file = arc_file+".cnv"
    f = open(cnv_file,'w')
    for key in out_dict.keys():      # Loop for each event
        e_lat = out_dict[key]["e_lat"]
        e_lon = out_dict[key]["e_lon"]
        e_dep = out_dict[key]["e_dep"]
        e_mag = out_dict[key]["e_mag"]
        phases = out_dict[key]["phase"]
        evid = out_dict[key]["evid"]
        part1 = key[2:8]+" "+key[8:12]+" "+key[12:14]+"."+key[14:16]+" "
        part2 = format(e_lat,"7.4f")+"N"+" "+format(e_lon,"8.4f")+"E"+" "
        part3 = format(e_dep,"7.2f")+"  "+format(e_mag,"5.2f")
        
        f.write(part1+part2+part3)
        i = 0
        for phase in phases:
            if i%6==0:
                f.write("\n")
            f.write(phase)
            i=i+1
        tmp = i%6
        if tmp>0:
            f.write((6-tmp)*13*" "+"\n") # add space to fill line
        elif tmp==0:
            f.write("\n")
        f.write(f"   {str(evid).zfill(6)}\n")
    f.write("9999")              # indicates end of file for VELEST
    f.close()

    f = open(arc_file+".sel",'w')
    for line in cont_sel:
        f.write(line+"\n")
    f.close()

def dd2fdsn(in_file,subset=None):
    '''
    Convert the hypoDD reloc file into fdsn
    format that could be read by zmap	
    '''
    T0 = time.time()
    print("The start time is 0.")
    if subset == None:
        filt = False
    else:
        filt = True
        [lon_min,lon_max,lat_min,lat_max] = subset
    events = []
    out_file = "dd.fdsn"
    f=open(out_file,'w')
    f.write("#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName\n")
    f.close()
    eve_dict,df = load_DD(reloc_file=in_file)
    T1 = time.time()
    print("%f seconds passed to load hypoDD file" %(T1-T0))
    eve_list = list(eve_dict)
    f = open(out_file,'a')
    for eve in eve_list:
        evid = eve_dict[eve][4]
        e_time = UTCDateTime.strptime(eve,"%Y%m%d%H%M%s%f")
        e_lon = eve_dict[eve][0]
        e_lat = eve_dict[eve][1]
        e_dep = eve_dict[eve][2]
        e_mag = eve_dict[eve][3] 
        if filt:
            if e_lat>lat_max or e_lat<lat_min or e_lon>lon_max or e_lon<lon_min:
                continue
        mag_type = 'ML'
        f.write('{:0>6d}'.format(evid)+"|")
        f.write(str(e_time)+"|")
        f.write(format(e_lat,'6.3f')+"|")
        f.write(format(e_lon,'7.3f')+"|")
        f.write(format(e_dep,'6.2f')+"|")
        f.write("Hardy|")
        f.write("SC|")
        f.write("SC|")
        f.write("01|")
        f.write(mag_type+'|')
        f.write(format(e_mag+0.01,'5.2f')+"|")
        f.write("SC Agency|")
        f.write("SC\n")
    f.close()

def sum2fdsn(in_file,subset=None):
    '''
    Convert the out.sum file into fdsn
    format that could be read by zmap	
    '''
    if subset == None:
        filt = False
    else:
        filt = True
        [lon_min,lon_max,lat_min,lat_max] = subset
    events = []
    if isinstance(in_file,list):
        for file in in_file: #input file, *.csv, without two header line
            with open(file,'r',encoding='UTF-8') as f:
                for line in f:
                    events.append(line.rstrip())
            f.close()
    else:
        with open(in_file,'r',encoding='UTF-8') as f:
            for line in f:
                events.append(line.rstrip())
        f.close()
    out_file = "sum.fdsn"
    f=open(out_file,'w')
    f.write("#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName\n")
    f.close()
    f = open(out_file,'a')
    for event in events:
        evid = int(event[141:146])
        e_time = UTCDateTime(event[:16],"%Y%m%d%H%M%S%f")
        e_lat = int(event[16:18])+int(event[19:23])/100.0/60.0
        e_lon = int(event[23:26]) + int(event[27:31])/100.0/60.0
        if filt:
            if e_lat>lat_max or e_lat<lat_min or e_lon>lon_max or e_lon<lon_min:
                continue
        e_dep = int(event[31:36])/100.0
        e_mag = int(event[123:126])/100.0
        mag_type = 'ML'
        f.write('{:0>6d}'.format(evid)+"|")
        f.write(str(e_time)+"|")
        f.write(format(e_lat,'6.3f')+"|")
        f.write(format(e_lon,'7.3f')+"|")
        f.write(format(e_dep,'6.2f')+"|")
        f.write("Hardy|")
        f.write("SC|")
        f.write("SC|")
        f.write("01|")
        f.write(mag_type+'|')
        f.write(format(e_mag+0.01,'5.2f')+"|")
        f.write("SC Agency|")
        f.write("SC\n")
    f.close()


def SC2fdsn(in_file,subset=None):
    '''
    Convert the event file from the SC catalog into fdsn
    format that could be read by zmap	
    '''
    if subset == None:
        filt = False
    else:
        filt = True
        [lon_min,lon_max,lat_min,lat_max] = subset
    events = []
    if isinstance(in_file,list):
        for file in in_file: #input file, *.csv, without two header line
            with open(file,'r',encoding='UTF-8') as f:
                for line in f:
                    events.append(line.rstrip())
            f.close()
    else:
        with open(in_file,'r',encoding='UTF-8') as f:
            for line in f:
                events.append(line.rstrip())
        f.close()
    out_file = "out.fdsn"
    f=open(out_file,'w')
    f.write("#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName\n")
    f.close()
    f = open(out_file,'a')
    for event in events:
        evid = int(event[:5])
        e_time = UTCDateTime(event[6:27])
        e_lat = event[28:34]
        e_lon = event[35:42]
        if filt:
            if float(e_lat)>lat_max or float(e_lat)<lat_min or float(e_lon)>lon_max or float(e_lon)<lon_min:
                continue
        e_dep = str(int(int(event[47:])/1000))
        e_mag = float(event[43:46])+0.01
        mag_type = 'ML'
        f.write('{:0>6d}'.format(evid)+"|")
        f.write(str(e_time)+"|")
        f.write(e_lat+"|")
        f.write(e_lon+"|")
        f.write(e_dep+"|")
        f.write("Hardy|")
        f.write("SC|")
        f.write("SC|")
        f.write("01|")
        f.write(mag_type+'|')
        f.write(format(e_mag,'5.2f')+"|")
        f.write("SC Agency|")
        f.write("SC\n")
    f.close()

def CN2fdsn(in_file):
    '''
    Convert the event file from the China National Data Center into fdsn
    format that could be read by zmap	
    '''
    events = []
    if isinstance(in_file,list):
        for file in in_file: #input file, *.csv, without two header line
            with open(file,'r',encoding='UTF-8-sig') as f:
                for line in f:
                    events.append(line.rstrip())
            f.close()
    else:
        with open(in_file,'r',encoding='UTF-8-sig') as f:
            for line in f:
                events.append(line.rstrip())
        f.close()
    out_file = "out.fdsn"
    f=open(out_file,'w')
    f.write("#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName\n")
    f.close()
    evid = 0
    f = open(out_file,'a')
    for event in events:
        evid = evid+1
        e_time,e_lon,e_lat,e_dep,mag_type,e_mag,_,_ = re.split(",",event)
        e_date,e_hm = re.split(" ",e_time)
        e_year,e_month,e_day = re.split("/",e_date)
        e_year=int(e_year)
        e_month = int(e_month)
        e_day = int(e_day)
        e_hr,e_min = re.split(":",e_hm)
        e_hr = int(e_hr)
        e_min = int(e_min)
        e_time = UTCDateTime(e_year,e_month,e_day,e_hr,e_min,0,0)
        f.write('{:0>6d}'.format(evid)+"|")
        f.write(str(e_time)+"|")
        f.write(e_lat+"|")
        f.write(e_lon+"|")
        f.write(e_dep+"|")
        f.write("Hardy|")
        f.write("SC|")
        f.write("SC|")
        f.write("01|")
        f.write(mag_type+'|')
        f.write(format(float(e_mag)+0.01,'5.2f')+"|")
        f.write("SC Agency|")
        f.write("SC\n")
    f.close()

def IRIS2fdsn(events):
    '''
    Convert the event file from the python client into fdsn
    format that could be read by zmap	
    '''
    out_file = "out.fdsn"
    f=open(out_file,'w')
    f.write("#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName\n")
    f.close()
    evid = 0
    f = open(out_file,'a')
    for event in events:
        evid = evid+1
        e_time = event["origins"][0]["time"]
        e_lon = event["origins"][0]["longitude"]
        e_lat = event["origins"][0]["latitude"]
        e_mag = event["magnitudes"][0]["mag"]
        try:
            e_dep = event["origins"][0]["depth"]/1000
        except:
            e_dep = 0
        try:
            mag_type = event["magnitudes"][0]["magnitude_type"]
        except:
            mag_type = None
        f.write('{:0>6d}'.format(evid)+"|")
        f.write(str(e_time)+"|")
        f.write(str(e_lat)+"|")
        f.write(str(e_lon)+"|")
        f.write(str(e_dep)+"|")
        f.write("Hardy|")
        f.write("IRIS|")
        f.write("IRIS|")
        f.write("01|")
        f.write(str(mag_type)+'|')
        f.write(format(float(e_mag)+0.01,'5.2f')+"|")
        f.write("SC Agency|")
        f.write("SC\n")
    f.close()

def ncsn2pha(source_file,target_file):
    input_content=[]
    output_content=[]
    with open(source_file,"r") as f:
        for line in f:
            input_content.append(line.rstrip())
    f.close()
    for line in input_content:
        if line[0:7]=="       ": # Pass event id line
            continue
        if re.match("\d+",line[0:7]): # A event line
            date = line[0:8]
            mo = line[5:6]
            dy = line[6:8]
            hr = line[8:10]
            min = line[10:12]
            sec = line[12:16]
            deglat = line[16:18]
            minlat = line[19:23]
            deglon = line[23:26]
            minlon = line[27:31]
            depth = int(line[31:36])
            mag = line[147:150]
            res = line[48:52]
            herr = line[85:89]
            verr = line[89:93]
            cuspid = line[136:146]

            year = line[0:4]
            lat = int(deglat)+int(minlat)*0.01/60
            lon = int(deglon)+int(minlon)*0.01/60

            part1 = "# "+year+str(int(mo)).rjust(3," ")+str(int(dy)).rjust(3," ")
            part2 = str(int(hr)).rjust(3," ")+str(int(min)).rjust(3," ")+format(int(sec)*0.01,'6.2f')+'  '
            part3 = format(lat,"7.4f")+"  "+format(lon,"8.4f")+"   "+format(depth*0.01,'5.2f')+"  "+format(int(mag)*0.01,'4.2f')
            part4 = " "+format(int(herr)*0.01,'5.2f')+" "+format(int(verr)*0.01,'5.2f')+" "+format(int(res)*0.01,'5.2f')+" "+cuspid
            output_content.append(part1+part2+part3+part4)
        else:
            c1 = line[3:4]
            c2 = line[9:10]
            c4 = line[11:12]
            premk = line[13:15]
            sremk = line[46:48]
            sta = line[5:7]+line[0:5]
            pqual = int(line[16:17])
            squal = int(line[49:50])
            p_min = line[27:29]
            p_sec = line[29:34]
            s_sec = line[41:46]
            p_res = int(line[34:39])*0.01
            s_res = int(line[50:54])*0.01
            p_imp = line[100:104]
            try:
                p_imp = int(p_imp)*0.001
            except:
                continue
            s_imp = line[104:108]
            try:
                s_imp = int(s_imp)*0.001
            except:
                continue
            if pqual < 4 and premk != "  " and c4 == "Z":
                if p_res > 0.2:
                    continue
                if pqual == 0:
                    p_weight = 1.0
                elif pqual == 1:
                    p_weight = 0.5
                elif pqual == 2:
                    p_weight = 0.2
                elif pqual == 3:
                    p_weight = 0.1
                else:
                    p_weight = 0.0

                if p_imp > 0.5:
                    p_weight = -1*p_weight
                if int(p_min) < int(min): #60 -> 01 of another hour
                    dmin = int(p_min) + 60 - int(min)
                else:
                    dmin = int(p_min) - int(min)
                p_time = dmin*60 + (int(p_sec)-int(sec))*0.01
                output_content.append(sta+"    "+format(p_time,'6.3f')+format(p_weight,'8.3f')+"   P")

            if squal < 4 and sremk != "  " and c4 == "Z":
                if s_res>0.2:
                    continue
                if squal == 0:
                    s_weight = 1.0
                elif squal == 1:
                    s_weight = 0.5
                elif squal == 2:
                    s_weight = 0.2
                elif squal == 3:
                    s_weight = 0.1
                else:
                    s_weight = 0.0

                if s_imp > 0.5:
                    s_weight = -1*s_weight
                if int(p_min) < int(min): #60 -> 01 of another hour
                    dmin = int(p_min) + 60 - int(min)
                else:
                    dmin = int(p_min) - int(min)
                s_time = dmin*60 + (int(s_sec)-int(sec))*0.01
                output_content.append(sta+"    "+format(s_time,'6.3f')+format(s_weight,'8.3f')+"   S")
    with open(target_file,"w") as f:
        for line in output_content:
            f.write(line)
            f.write('\n')
                
def sc2phs(file_list=[],region_condition="-9/-9/-9/-9",mag_condition=-9):
    #initiate
    lon_filt=True
    lat_filt=True
    mag_filt=True #filt magnitude
    time_filt=True
    lon_min,lon_max,lat_min,lat_max= re.split("/",region_condition)
    if lon_min=="-9":
        lon_filt=False
    else:
        lon_min = float(lon_min)
        lon_max = float(lon_max)
    if lat_min=="-9":
        lat_filt=False
    else:
        lat_min = float(lat_min)
        lat_max = float(lat_max)
    if mag_condition==-9:
        mag_filt = False
    if file_list == []:
        for file in os.listdir("./"):
            if file[-4:]==".adj":
                file_list.append(file)
    file_list.sort()
    #initiate part
    output_content=[]
    input_content = []
    for file in file_list:
        with open(file,"r") as f:
            for line in f:
                input_content.append(line.rstrip())
        f.close()
    event_id=0
    for line in input_content:
        if re.match("\d+",line[3:7]) and line[7]=="/":#then it is a event line
            record_status=True
            e_year = line[3:7]
            e_month = line[8:10]
            e_day = line[11:13]
            e_hour = line[14:16]
            e_minute = line[17:19]
            e_second_int = line[20:22]
            e_second_left = line[23:24]
            e_lat=line[26:32]
            if e_lat=='      ':
                record_status=False
                continue
            else:
                e_lat=float(e_lat)
                if lat_filt:
                    if e_lat<lat_min or e_lat>lat_max:
                        record_status=False
                        continue
            e_lon=line[34:41]
            if e_lon=='       ':
                record_status=False
                continue
            else:
                e_lon=float(e_lon)
                if lon_filt:
                    if e_lon<lon_min or e_lon>lon_max:
                        record_status=False
                        continue
            e_dep=line[43:45]
            if e_lat=='  ':
                record_status=False
                continue
            else:
                e_lat=float(e_lat)
            e_mag=line[47:50]
            if e_mag=='   ':
                record_status=False
                continue
            else:
                e_mag=float(e_mag)
                if mag_filt:
                    if e_mag < mag_condition:
                        record_status=False
                        continue
            if record_status==True:
                if event_id!=0:
                    output_content.append(format(str(event_id).rjust(72," ")))
                event_id+=1
                print("Process event     {0}    ".format(event_id),end='\r')
                part1 = e_year+e_month+e_day+e_hour+e_minute+e_second_int+e_second_left+'0'
                part2 = str(int(e_lat))+" "+str(int((e_lat-int(e_lat))*60*100)).zfill(4)
                part3 = str(int(e_lon))+"E"+str(int((e_lon-int(e_lon))*60*100)).zfill(4)
                part4 = str(int(e_dep)*100).rjust(5," ")+"000"+"L".rjust(84," ")+str(int(e_mag*100)).zfill(3)
                output_content.append(part1+part2+part3+part4)
        
        elif record_status==True:
            if line[0:2]!="  ":
                net = line[0:2]
                sta = re.split(" +",line[3:8])[0]
            p_type = line[17:19]
            p_hour = line[32:34]
            p_minute = line[35:37]
            p_seconds = float(line[38:43])
            p_residual=line[45:50]
            if p_residual=="     ":
                continue
            else:
                try:
                    p_residual = float(p_residual)
                except:
                    continue
            if p_type =="Pg":
                part1 = sta.ljust(5," ")+net+"  SHZ IPU2"+e_year+e_month+e_day+p_hour+p_minute+" "+str(int(p_seconds*100)).zfill(4)
                part2 = str(int(p_residual*100)).rjust(4," ")+"  0    0   0   0"
                output_content.append(part1+part2)
            elif p_type == "Sg":
                part1 = sta.ljust(5," ")+net+"  SHZ    2"+e_year+e_month+e_day+p_hour+p_minute+"    0   0  0 "
                part2 = str(int(p_seconds*100)).zfill(4)+"ES 0"+str(int(p_residual*100)).rjust(4," ")
                output_content.append(part1+part2)
    output_content.append(format(str(event_id).rjust(72," ")))
    with open("out.phs","w") as f:
        for line in output_content:
            f.write(line+"\n")
    print("  ") #for window output

def real2arc(input_file,phase_filt=8,region_filt=[0,0,0,0]):
    """
    change from REAL association result to the file that could be read by hypo-inverse
    """
    evlo_min,evlo_max,evla_min,evla_max = region_filt
    if evlo_min == evlo_max:
        region_filt_status=False
    else:
        region_filt_status=True

    input_title = re.split("\.",input_file)[:-1]
    input_title = "".join(input_title)
    output_file = input_title+".phs"

    f_content=[]
    with open(input_file,"r") as f:
        for line in f:
            f_content.append(line)
    f.close()
    event_id = 0
    filt_status = False #initiate the status
    with open(output_file,"w") as f:
        for line in f_content:
            print(line)
            f_para = re.split(" +",line)[1] # first parameter
            # if it is digital number, then it is an event line
            if re.match("\d+",f_para): 
                filt_status = False   #first set to False and then if nt (number of total phase) > threshold
                _,no,year,month,day,o_time,ab_sec,res,lat,lon,dep,mag,mag_res,np,ns,nt,sta_gap=re.split(" +",line)
                if region_filt_status:
                    if float(lon)<evlo_min or float(lon)>evlo_max or float(lat)<evla_min or float(lat)>evla_max:
                        continue
                if int(nt) >= phase_filt:
                    filt_status = True
                    # REAL provide seconds below zero and above 60, need to handle it
                    e_hr,e_min,e_sec=re.split(":",o_time)
                    e_sec = float(e_sec)
                    e_time = UTCDateTime(year+"-"+month+"-"+day+"T"+e_hr+":"+e_min+":"+"00")+e_sec
                    # re-get the value
                    e_year = e_time.year
                    e_month = e_time.month
                    e_day = e_time.day
                    e_hr = e_time.hour
                    e_min = e_time.minute
                    e_sec = e_time.second+e_time.microsecond/1000000
                    # actions on lon and lat
                    lat = float(lat)
                    lat_i = int(lat) # int part, unit degree
                    lat_f = lat - lat_i # float part, unit degree
                    lon = float(lon)
                    lon_i = int(lon) # int part, unit degree
                    lon_f = lon - lon_i # float part, unit degree
                    #actions on depth
                    dep = float(dep) #unit km
                    #actions on magnitude, if it is Null, then set to zero
                    if mag == "-inf":
                        mag = 0.0
                        mag_res = 0.0
                    else:
                        mag = float(mag)
                        mag_res = float(mag_res)
                    if event_id > 0:
                        f.write(format(event_id,">72d")+"\n")
                    f.write(format(e_year,"4d")+format(e_month,"0>2d")+format(e_day,"0>2d")+\
                        format(e_hr,"0>2d")+format(e_min,"0>2d")+format(e_sec*100,"0>4.0f")+\
                        format(lat_i,"0>2d")+" "+format(lat_f*60*100,"0>4.0f")+\
                        format(lon_i,"0>3d")+"E"+format(lon_f*60*100,"0>4.0f")+\
                        format(dep*100,">5.0f")+format(mag*100,"0>3.0f")+"\n")                
                    event_id = event_id + 1
            else:
                if filt_status == True:
                    _,net,sta,p_type,ab_sec,ref_sec,amp,res,weight,azimuth=re.split(" +",line)
                    res=float(res)
                    weight=float(weight)
                    #special action
                    weight=1.0
                    p_time = e_time + float(ref_sec)
                    p_year = p_time.year
                    p_month = p_time.month
                    p_day = p_time.day
                    p_hr=p_time.hour
                    p_min=p_time.minute
                    p_sec=p_time.second+p_time.microsecond/1000000
                    p_sec_i = p_time.second #int part
                    p_sec_f = p_time.microsecond/1000000 #float part
                    #write in content
                    if p_type=="P":
                        f.write(format(sta,"<5s")+format(net,"2s")+"  SHZ IPU2"+\
                            format(p_year,"4d")+format(p_month,"0>2d")+format(p_day,"0>2d")+\
                            format(p_hr,"0>2d")+format(p_min,"0>2d")+\
                            format(p_sec*100,">5.0f")+\
                            format(res*100,">4.0f")+"  0    0   0   0"+"\n")
                    if p_type=="S":
                        f.write(format(sta,"<5s")+format(net,"2s")+"  SHZ    2"+\
                            format(p_year,"4d")+format(p_month,"0>2d")+format(p_day,"0>2d")+\
                            format(p_hr,"0>2d")+format(p_min,"0>2d")+"    0   0  0"+\
                            format(p_sec*100,">5.0f")+"ES 1"+format(res*100,">4.0f")+"\n")
        f.write(format(event_id,">72d")+"\n")
        f.close()
