#-----------------------------------------------------------------------------
# coding: utf-8
# author: Hardy ZI
# history:
#		...
#       2021-03-15 Add CN2fsdn function
#       2025-03-05 From seisloc/phase_convert --> seisloc/loc/phase_convert
#-------------------------------------------------------------------------

import re
from obspy import UTCDateTime
import os
from seisloc.loc.dd import load_DD
import time
from seisloc.geopara import WYpara
from tqdm import tqdm
from seisloc.loc.text_io import load_y2000,load_cnv,_write_cnv_file,write_arc


def cata2fdsn(cata,saveFile="dd.fdsn"):
    f = open(saveFile,'w')
    f.write("#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName\n")

    for evid in cata.dict.keys():
            etime = cata[evid][4]
            evlo = cata[evid][0]
            evla = cata[evid][1]
            evdp = cata[evid][2]
            mag = cata[evid][3]
            magType = 'ML'
            f.write('{:0>6d}'.format(evid)+"|")
            f.write(str(etime)+"|")
            f.write(format(evla,'6.3f')+"|")
            f.write(format(evlo,'7.3f')+"|")
            f.write(format(evdp,'6.2f')+"|")
            f.write("Hardy|")
            f.write("SC|")
            f.write("SC|")
            f.write("01|")
            f.write(magType+'|')
            f.write(format(mag+0.01,'5.2f')+"|")
            f.write("SC Agency|")
            f.write("SC\n")
    f.close()



def dd2fdsn(ddFile):
    '''
    Convert the hypoDD reloc file into fdsn
    format that could be read by zmap
    '''
    cata = Catalog(ddFile)
    cata2fdsn(cata)

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
        f.write(format(e_mag+0.01,'5.2f')+"|")
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

def cata2fdsn(cata,author="Hardy",catalog="SC",
              cont="SC",contID="01",magtype="ML",
              magauthor="SC Agency",elocname="SC",out_file='cata.fdsn'):
    """
    Outout fdsn format to be read by ZMAP
    """
    keys = cata.keys
    f = open(out_file,'w')
    f.write("#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog"
            +"|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName\n")
    for evid in keys:
        e_time = cata[evid][4]
        e_lon = cata[evid][0]
        e_lat = cata[evid][1]
        e_dep = cata[evid][2]
        e_mag = cata[evid][3]
        magtype = 'ML'
        f.write('{:0>6d}'.format(evid)+"|")
        f.write(str(e_time)+"|")
        f.write(format(e_lat,'6.3f')+"|")
        f.write(format(e_lon,'7.3f')+"|")
        f.write(format(e_dep,'6.2f')+"|")
        f.write(author+"|")
        f.write(catalog+"|")
        f.write(cont+"|")
        f.write(contID+"|")
        f.write(magtype+'|')                                                                                                                                                                               
        f.write(format(e_mag+0.01,'5.2f')+"|")
        f.write(magauthor+"|")
        f.write(elocname+"\n")
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
                
def sc2phs(file_list=[],trims=[None,None,None,None],magThreshold=None,baseid=1,outfile='out.phs'):
    """
    convert Sichuan earthquake administration report into file could be
    recognized by Hypoinverse.
    """
    if file_list == []:
        for file in os.listdir("./"):
            if file[-6:]==".phase":
                file_list.append(file)
    file_list.sort()
 
    lon_filt=True
    lat_filt=True
    mag_filt=True #filt magnitude
    time_filt=True
    lon_min,lon_max,lat_min,lat_max= trims
    if lon_min==None:
        lon_filt=False
    if lat_min==None:
        lat_filt=False
    if magThreshold==None:
        mag_filt = False
    #initiate part
    output_content=[]
    input_content = []
    for file in file_list:
        try:
            with open(file,"r",encoding="utf-8") as f:
                for line in f:
                    input_content.append(line.rstrip())
            f.close()
        except:
            with open(file,"r",encoding="gbk") as f:
                for line in f:
                    input_content.append(line.rstrip())
            f.close()
    evid= baseid-1
    for line in tqdm(input_content):
        print(line)
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
            if e_dep == "  ":
                record_status = False
                continue
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
                    if e_mag < magThreshold:
                        record_status=False
                        continue  
            if record_status==True:      
                if evid!=baseid-1:   
                    output_content.append(format(str(evid).rjust(72," ")))
                evid+=1     
                part1 = e_year+e_month+e_day+e_hour+e_minute+e_second_int+e_second_left+'0'
                part2 = str(int(e_lat))+" "+str(int((e_lat-int(e_lat))*60*100)).zfill(4)
                part3 = str(int(e_lon))+"E"+str(int((e_lon-int(e_lon))*60*100)).zfill(4)
                part4 = str(int(e_dep)*100).rjust(5," ")+"000"+"L".rjust(84," ")+str(int(e_mag*100)).zfill(3)
                part5 = " ".rjust(21," ")+str(int(e_mag*100)).zfill(3)
                output_content.append(part1+part2+part3+part4+part5)
                                  
        elif record_status==True:   # process phase line
            # >>>>> a bug >>>>>
            if len(line)>=26:
                if line[25] == "-":
                    line = line[:25]+line[26:]
            # <<<<<<<<<<<<<<<<<
            if line[0:2]!="  ":   
                net = line[0:2]   
                sta = re.split(" +",line[3:8])[0]
            p_type = line[17:19]  
            p_weight = float(line[25:28])
            if p_weight >= 0.75:  
                weightCode = 1    
            elif p_weight >= 0.5: 
                weightCode = 2    
            elif p_weight >= 0.0: 
                weightCode = 3    
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
                part1 = sta.ljust(5," ")+net+"  SHZ IPU"+str(weightCode)+e_year+e_month+e_day+p_hour+p_minute+" "+str(int(p_seconds*100)).zfill(4)
                part2 = str(int(p_residual*100)).rjust(4," ")+"  0    0   0   0"
                output_content.append(part1+part2)
            elif p_type == "Sg": 
                part1 = sta.ljust(5," ")+net+"  SHZ     "+e_year+e_month+e_day+p_hour+p_minute+"    0   0  0 "
                part2 = str(int(p_seconds*100)).zfill(4)+"ES "+str(weightCode)+str(int(p_residual*100)).rjust(4," ")
                output_content.append(part1+part2) 
    output_content.append(format(str(evid).rjust(72," ")))
    with open(outfile,"w") as f:
        for line in output_content:
            f.write(line+"\n")

def getWeightCode(weight,stds):
    """
    Get the corresponding weight code as HYPOINVERSE
    """
    if weight < stds[3]:
        weightCode = 4
    elif weight < stds[2]:
        weightCode = 3
    elif weight < stds[1]:
        weightCode = 2
    elif weight < stds[0]:
        weightCode = 1
    else:
        weightCode = 0
        
    return weightCode

def real2arc(inFile,
             minObs=8,
             cmp="SHZ",
             boundCond=[],
             weightCodeStds=[0.95,0.75,0.5,0.25],
             startId = 0):
    """
    change from REAL association result to the file that could be read by HYPOINVERSE
    outfile have the same name of input file with suffix '.phs'
    
    Parameters:
        inFile: input file generated by REAL
        minObs: minimum observation by stations, including P and S
        boundCond: boundary condition, [lonMin,lonMax,latMin,latMax]
        weightCodeStds: standards for weight code assignment,corresponding weight codes
                        are [0,1,2,3,4]
        startId: Initial event id. Default is zero, and events will receive IDs start from startID+1.
                 Under cases user have demand to merge different catalogs, it is necessary to set up 
                 different event ids to avoid event id confliction
        
    """
    inFile = os.path.abspath(inFile)
    if len(boundCond) == 0:
        boundFilt=False
    elif len(boundCond) == 4:
        boundFilt=True
        lonMin,lonMax,latMin,latMax = boundCond
    else:
        raise Exception("boundCond should be with length 0 or 4!")
        
    inTitle = re.split("\.",inFile)[:-1]
    inTitle = "".join(inTitle)
    outFile = inTitle+".phs"

    cont=[]
    with open(inFile,"r") as f:
        for line in f:
            cont.append(line.rstrip())
    f.close()
    loopId = startId                          # initiate event ID
    f = open(outFile,"w")
    for line in cont:
        f_para = re.split(" +",line)[1]       # first parameter            
        if re.match("\d+",f_para):            # event line start with digital number
            status = False                    # initiate the status to be False
            _,_no,_yr,_mo,_dy,_otime,_absec,_timeRes = re.split(" +",line)[:8]
            _lat,_lon,_dep,_mag,_magRes = re.split(" +",line)[8:13]
            _numP,_numS,_numT,_staGap = re.split(" +",line)[13:]
            lon = float(_lon); lat = float(_lat);dep = float(_dep)
            numT = int(_numT)
            if boundFilt:
                if lon<lonMin or lon>lonMax or lat<latMin or lat>latMax:
                    continue
            if numT < minObs:
                continue
            status = True
            # REAL provide seconds below zero and above 60, need to handle it
            _ehr,_emin,_esec=re.split(":",_otime)
            esec = float(_esec)
            etime = UTCDateTime(_yr+"-"+_mo+"-"+_dy+"T"+_ehr+":"+_emin+":"+"00")+esec
            # re-get the value
            eyear = etime.year
            emonth = etime.month
            eday = etime.day
            ehr = etime.hour
            emin = etime.minute
            esec = etime.second+etime.microsecond/1000000

            latInt = int(lat)             # int part, unit degree
            latDec = lat - latInt         # float part, unit degree
            lonInt = int(lon)             # int part, unit degree
            lonDec = lon - lonInt         # float part, unit degree

            if _mag == "-inf":
                mag = 0.0
                mag_res = 0.0
            else:
                mag = float(_mag)
                mag_res = float(_magRes)
            if loopId > startId:
                f.write(format(loopId,">72d")+"\n")
            f.write(format(eyear,"4d")+format(emonth,"0>2d")+format(eday,"0>2d")+\
                format(ehr,"0>2d")+format(emin,"0>2d")+format(esec*100,"0>4.0f")+\
                format(latInt,"0>2d")+" "+format(latDec*60*100,"0>4.0f")+\
                format(lonInt,"0>3d")+"E"+format(lonDec*60*100,"0>4.0f")+\
                format(dep*100,">5.0f")+format(mag*100,"0>3.0f")+"\n")                
            loopId = loopId + 1
        else:
            if status == False:
                continue
            _,net,sta,phaType,_absSec,_relSec,_amp,_res,_weight,_az = re.split(" +",line)
            res=float(_res);weight=float(_weight)
            weightCode = getWeightCode(weight,weightCodeStds)
            phaTime = etime + float(_relSec)
            phaYr = phaTime.year
            phaMo = phaTime.month
            phaDy = phaTime.day
            phaHr=phaTime.hour
            phaMin=phaTime.minute
            phaSec=phaTime.second+phaTime.microsecond/1000000
            phaSecInt = phaTime.second #int part
            phaSecDec = phaTime.microsecond/1000000 #float part
            #write in content
            if phaType=="P":
                f.write(format(sta,"<5s")+format(net,"2s")+"  "+format(cmp,'3s')+\
                    " IPU"+str(weightCode)+\
                    format(phaYr,"4d")+format(phaMo,"0>2d")+format(phaDy,"0>2d")+\
                    format(phaHr,"0>2d")+format(phaMin,"0>2d")+\
                    format(phaSec*100,">5.0f")+\
                    format(res*100,">4.0f")+"  0    0   0   0"+"\n")
            if phaType=="S":
                f.write(format(sta,"<5s")+format(net,"2s")+"  "+format(cmp,'3s')+"     "+\
                    format(phaYr,"4d")+format(phaMo,"0>2d")+format(phaDy,"0>2d")+\
                    format(phaHr,"0>2d")+format(phaMin,"0>2d")+"    0   0  0"+\
                    format(phaSec*100,">5.0f")+"ES "+str(weightCode)+format(res*100,">4.0f")+"\n")
    if loopId != startId:
        f.write(format(loopId,">72d")+"\n")
    print(f"{loopId - startId} events after conversion!")
    f.close()

def arc2cnv(arcFile,magThred=-9,minAz=360,qty_limit=None,staChrLen=5):
    """
    convert y2000 archieve file into velest format file
    minAz: minium azimuth angle
    """
    count = 0
    arc = load_y2000(arcFile)
    arcSel = arc.copy()
    for evstr in arcSel.keys():
        evla = arc[evstr]["evla"]
        evlo = arc[evstr]["evlo"]
        evdp = arc[evstr]["evdp"]
        emag = arc[evstr]["emag"]
        maxStaAzGap = arc[evstr]['maxStaAzGap']
        #--------- quality control ------------------------------
        # no zero depth for VELEST
        if emag<magThred or evdp==0 or maxStaAzGap > minAz or\
        (qty_limit!=None and count >= qty_limit):
            del arcSel[evstr]
            continue
        count += 1
    
    cnvFile = arcFile+".cnv"
    _write_cnv_file(cnvFile,arcSel,staChrLen=staChrLen)
    print(f"# {cnvFile} saved! Total {count} events! This is the value for the parameter 'neqs' in Velest!")
    arcSelFile = arcFile+".sel"
    write_arc(arcSelFile,arcSel)
    print(f"# {arcSelFile} saved!")
    
def cnv2pha(cnvFile):
    cnv = load_cnv(cnvFile)
    f = open(cnvFile+".pha",'w')
    for evstr in cnv.keys():
        etime = cnv[evstr]['etime']
        evlo = cnv[evstr]['evlo']
        evla = cnv[evstr]['evla']
        evdp = cnv[evstr]['evdp']
        emag = cnv[evstr]['emag']
        rms = cnv[evstr]['rms']
        evid = cnv[evstr]['evid']
        errh = 0
        errz = 0
        secs = etime.second+etime.microsecond/1000000
        _str1 = f"# "+etime.strftime("%Y %m %d %H %M ")+format(secs,'5.2f')
        _str2 = f" {format(evla,'8.4f')} {format(evlo,'9.4f')} {format(evdp,'7.2f')} {format(emag,'5.2f')}"
        _str3 = f" {format(errh,'4.2f')} {format(errz,'4.2f')} {format(rms,'4.2f')} {format(evid,'9d')}"
        print(_str1+_str2+_str3)
        f.write(_str1+_str2+_str3+"\n")
        for sta,phsType,travTime,weightCode in cnv[evstr]['phases']:
            weightValue = 1-weightCode/4
            print(f"{format(sta,'<5s')} {format(travTime,'10.3f')} {format(weightValue,'7.3f')}   {phsType}")
            f.write(f"{format(sta,'<5s')} {format(travTime,'10.3f')} {format(weightValue,'7.3f')}   {phsType}"+"\n")


def arc2phs(arc,outFile="out.phs"):
    keys = list(arc.keys())
    f = open(outFile,'w')
    for key in sorted(keys):
        evlo = arc[key]['evlo']
        evla = arc[key]['evla']
        evdp = arc[key]['evdp']
        emag = arc[key]['emag']
        evid = arc[key]['evid']
        _date=key[:8]
        _time=key[8:16]
        _evla=str(int(evla))+"N"+str(int(np.round((evla-int(evla))*60*100,0))).zfill(4)
        _evlo=str(int(evlo))+"E"+str(int(np.round((evlo-int(evlo))*60*100,0))).zfill(4)
        _evdp=format(int(evdp*100),'5d')
        _magAmp=format(int(emag*100),'3d')
        _tmp = " "*83+"L"
        f.write(_date+_time+_evla+_evlo+_evdp+_magAmp+_tmp+_magAmp+"\n")

        for net,sta,pt,ptimeBJ,res,wt in arc[key]['phase']:
            _secs=format(int((ptimeBJ.second + ptimeBJ.microsecond/1000000)*100),">5d")
            if pt == "P":
                f.write(format(sta,'<5s')+format(net,'2s')+" "*2+"SHZ"+" "+"IPU"+str(wt)+ptimeBJ.strftime("%Y%m%d%H%M")+_secs+"   0  0    0   0   0\n")
            else: # pt == "S":
                f.write(format(sta,'<5s')+format(net,'2s')+" "*2+"SHZ"+" "+"   "+" "    +ptimeBJ.strftime("%Y%m%d%H%M")+"    0   0  0"+_secs+"ES "+str(wt)+"   0\n")
        f.write(format(evid,">72d")+"\n")
