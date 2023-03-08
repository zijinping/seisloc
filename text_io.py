import os
import re
import numpy as np
from tqdm import tqdm
from obspy import UTCDateTime

def load_taup_tvel(tvelFile):
    tvelDict = {}
    f = open(tvelFile,'r')
    lines = f.readlines()
    tvelDict["commentLines"] = lines[:2]
    tvelDict["deps"] = []
    tvelDict["vps"]  = []
    tvelDict["vss"]  = []
    tvelDict["rhos"] = []
    for line in lines[2:]:
        dep,vp,vs,rho = re.split(" +",line.strip())
        tvelDict["deps"].append(float(dep))
        tvelDict["vps"].append(float(vp))
        tvelDict["vss"].append(float(vs))
        tvelDict["rhos"].append(float(rho))
    return tvelDict

def _load_event_dat_etime(timeSeg):
    etime = UTCDateTime.strptime(timeSeg[:8],'%Y%m%d')
    _hr = timeSeg[10:12]
    if _hr == "  ":_hr=" 0"
    _mi = timeSeg[12:14]
    if _mi == "  ":_mi=" 0"
    _sec = timeSeg[14:16]
    if _sec == "  ":_sec=" 0"
    _secf = timeSeg[16:18]
    etime = etime + int(_hr)*60*60 + int(_mi)*60 + int(_sec) + int(_secf)*0.01
    return etime

def load_event_dat(eventDatFile="./event.dat",showExampleLine=False):
    eventDatDict = {}
    f = open(eventDatFile,'r')
    eventLines = f.readlines()
    if showExampleLine:
        print("Head line:"+eventLines[0].rstrip())
    for eventLine in eventLines:
        timeSeg = eventLine[:18]
        etime = _load_event_dat_etime(timeSeg)
        otherSeg = eventLine[18:].strip()
        _lat,_lon,_dep,_mag,_eh,_ez,_rms,_evid = re.split(" +",otherSeg)
        lat,lon,dep,mag,eh,ez,rms = map(float,(_lat,_lon,_dep,_mag,_eh,_ez,_rms))
        evid = int(_evid)
        eventDatDict[evid] = {}
        eventDatDict[evid]['etime'] = etime
        eventDatDict[evid]['evlo'] = lon
        eventDatDict[evid]['evla'] = lat
        eventDatDict[evid]['evdp'] = dep
        eventDatDict[evid]['emag'] = mag
        eventDatDict[evid]['errH'] = eh
        eventDatDict[evid]['errZ'] = ez
        eventDatDict[evid]['rms']  = rms
    return eventDatDict

def write_event_dat(eventDatDict,eventDatFile="event.dat"):
    if os.path.exists(eventDatFile):
        raise Exception("Target eventDatFile existed!")
    f = open(eventDatFile,'w')
    #_estr=
    for evid in eventDatDict.keys():   
        f.write(eventDatDict[evid]['etime'].strftime("%Y%m%d  %H%M%S%f")[:18]+
                format(eventDatDict[evid]['evla'],'>10.4f')+
                format(eventDatDict[evid]['evlo'],'>11.4f')+
                format(eventDatDict[evid]['evdp'],'>11.2f')+
                format(eventDatDict[evid]['emag'],'>7.1f')+
                format(eventDatDict[evid]['errH'],'>8.2f')+
                format(eventDatDict[evid]['errZ'],'>8.2f')+
                format(eventDatDict[evid]['rms'],'>7.2f')+
                format(evid,'>11d')+"\n")

def _write_cnv_file(cnvFile,arc,staChrLen=6):
    f = open(cnvFile,'w')
    for evstr in arc.keys():      # Loop for each event
        evla = arc[evstr]["evla"]
        evlo = arc[evstr]["evlo"]
        evdp = arc[evstr]["evdp"]
        emag = arc[evstr]["emag"]
        etime = arc[evstr]["etime"]
        maxStaAzGap = arc[evstr]['maxStaAzGap']
        rms = arc[evstr]['rms']
        
        _phases = []
        for net,sta,phsType,phsTime,res,weightCode in arc[evstr]["phase"]:
            travTime = phsTime - etime
            phaRecord = sta.ljust(staChrLen," ")+phsType+str(weightCode)+format(travTime,'6.2f')
            _phases.append(phaRecord)
        evid = arc[evstr]["evid"]
        part1 = evstr[2:8]+" "+evstr[8:12]+" "+evstr[12:14]+"."+evstr[14:16]+" "
        part2 = format(evla,"7.4f")+"N"+" "+format(evlo,"8.4f")+"E"+" "
        part3 = format(evdp,"7.2f")+"  "+format(emag,"5.2f")+" "
        part4 = format(maxStaAzGap,'>6d')+" "+format(rms,'9.2f')
        
        f.write(part1+part2+part3+part4)
        i = 0
        for _phase in _phases:
            if i%6==0:
                f.write("\n")
            f.write(_phase)
            i=i+1
        tmp = i%6
        if tmp>0:
            f.write((6-tmp)*(8+staChrLen)*" "+"\n") # add space to fill line
        elif tmp==0:
            f.write("\n")
        if staChrLen==6:
            f.write("    \n")
        else:
            f.write(f"   {str(evid).zfill(6)}\n")
    f.write("9999")              # indicates end of file for VELEST
    f.close()

def write_arc(fileName,arc):
    f = open(fileName,'w')
    for evstr in arc.keys():
        for line in arc[evstr]["lines"]:
            f.write(line+"\n")
    f.close()    

def write_cnv(cnv,cnvFilePth="vel.cnv",staChrLen=6):
    f = open(cnvFilePth,'w')
    for evstr in cnv.keys():
        evla = cnv[evstr]["evla"]
        evlo = cnv[evstr]["evlo"]
        evdp = cnv[evstr]["evdp"]
        emag = cnv[evstr]["emag"]
        rms = cnv[evstr]["rms"]
        maxStaAzGap = cnv[evstr]["maxStaAzGap"]
        part1 = evstr[2:8]+" "+evstr[8:12]+" "+evstr[12:14]+"."+evstr[14:16]+" "
        part2 = format(evla,"7.4f")+"N"+" "+format(evlo,"8.4f")+"E"+" "
        part3 = format(evdp,"7.2f")+"  "+format(emag,"5.2f")+" "
        part4 = format(maxStaAzGap,'>6d')+" "+format(rms,'9.2f')
        f.write(part1+part2+part3+part4)
        i=0
        for sta,phsType,travTime,weightCode in cnv[evstr]['phases']:
            _phase=sta.ljust(staChrLen," ")+phsType+str(weightCode)+format(np.round(travTime,2),'6.2f')
            if i%6==0:
                f.write("\n")
            f.write(_phase)
            i=i+1
        tmp = i%6
        if tmp>0:
            f.write((6-tmp)*(8+staChrLen)*" "+"\n") # add space to fill line
        elif tmp==0:
            f.write("\n")
        if staChrLen==6:
            f.write("    \n")
        else:
            f.write(f"   {str(evid).zfill(6)}\n")
    f.write("9999")              # indicates end of file for VELEST
    f.close()
    
def read_y2000_event_line_mag(line):
    try:
        if len(line) < 126:
            emag = int(line[36:39])*0.01
        else:
            emag = int(line[123:126])*0.01
    except:
        emag = -9
    return emag
    
def read_y2000_event_line(line):
    evstr = line[:16]
    etime = UTCDateTime.strptime(evstr,'%Y%m%d%H%M%S%f')
    evla = int(line[16:18])+int(line[19:23])*0.01/60
    evlo = int(line[23:26])+int(line[27:31])*0.01/60
    evdp = int(line[31:36])*0.01
    emag = read_y2000_event_line_mag(line)
    maxStaAzGap = int(line[42:45])
    rms = int(line[48:52])*0.01
    
    return evstr,etime,evlo,evla,evdp,emag,maxStaAzGap,rms

def read_y2000_phase_line(line):
    sta = re.split(" +",line[:5])[0]
    net = line[5:7]
    phsTimeMinute= UTCDateTime.strptime(line[17:29],"%Y%m%d%H%M")
    if line[14]=="P" and line[47]==" ":
        phsType = "P"
        _secInt = line[29:32]; _secDecimal = line[32:34]
        res = int(line[34:38])*0.01
        weightCode=int(line[16])
    elif line[14]==" " and line[47]=="S":
        phsType = "S"
        _secInt = line[41:44]; _secDecimal = line[44:46]
        res =int(line[50:54])*0.01
        weightCode=int(line[49])
    else:
        raise Exception("Error phase type: line[14] '{line[14]}' and line[47] '{line[47]}'")
        
    if _secInt == "   ": _secInt = "0"
    if _secDecimal == "  ": _secDecimal = "0"
    phsTime = phsTimeMinute+int(_secInt)+int(_secDecimal)*0.01
    
    return sta,net,phsType,phsTime,res,weightCode
    
def load_y2000(y2000File,printLine=False,savePkl=False):
    """
    If printLine is true, each phase line will be printed out
    """
    with open(y2000File,"r") as f1:
        phsLines = f1.readlines()
    arc = {}
    ecount = 0

    print(">>> Loading phases ... ")
    for line in tqdm(phsLines):
        line = line.rstrip()
        if printLine:
            print(line)
        if re.match("\d+",line[:6]):    # line start with digit is summary/event line
            funcReturn = read_y2000_event_line(line)
            evstr,etime,evlo,evla,evdp,emag,maxStaAzGap,rms = funcReturn[:8]
            ecount += 1
            netstas = []
            arc[evstr] = {} 
            arc[evstr]["etime"] = etime
            arc[evstr]["evlo"] = evlo
            arc[evstr]["evla"] = evla
            arc[evstr]["evdp"] = evdp
            arc[evstr]["emag"] = emag
            arc[evstr]["phase"] = []
            arc[evstr]['nsta'] = 0
            arc[evstr]['rms'] = rms
            arc[evstr]['maxStaAzGap']=maxStaAzGap
            arc[evstr]["lines"] = []

        elif line[:6]=='      ':         # The last line
            evid = int(line[66:72])
            arc[evstr]["evid"]=evid            
        else:                            # phase line
            funcReturn = read_y2000_phase_line(line)
            sta,net,phsType,phsTime,res,weightCode = funcReturn[:6]
            travTime = phsTime - etime
            if net+sta not in netstas:
                netstas.append(net+sta)
                arc[evstr]['nsta'] += 1
            arc[evstr]["phase"].append([net,sta,phsType,phsTime,res,weightCode])
        arc[evstr]["lines"].append(line)

    if savePkl:  
        outFile = y2000File+".pkl"
        f = open(out_name,'wb')
        pkl.dump(arc,f)
        f.close()
 
    return arc
    
    
def read_cnv_event_line(line):
    yr = int(line[:2])        
    mo = int(line[2:4])       
    dy = int(line[4:6])   
    hr = int(line[7:9])       
    mi = int(line[9:11])
    secs = float(line[12:17])
    #etime = UTCDateTime(yr+2000,mo,dy,hr,mi,secs)
    etime = UTCDateTime(yr+2000,mo,dy)+\
        hr*60*60+mi*60+secs #modified due to a bug
    evstr = etime.strftime("%Y%m%d%H%M%S%f")[:16]
    evla = float(line[18:25])
    if line[25]=="S":         
        lat = -lat         
    evlo = float(line[27:35])
    if line[35]=="W":         
        lon = -lon         
    evdp = float(line[36:43])
    emag = float(line[45:50])
    maxStaAzGap=int(line[54:57])                                                                                                                                                                      
    rms = float(line[63:67])
    
    return evstr,etime,evlo,evla,evdp,emag,maxStaAzGap,rms

def read_cnv_phase_line(line):
    phsRecs = []
    if len(line)%14 == 0:
        segLen = 14
        staChrLen = 6
        nu = int(len(line)/14)
    elif len(line)%13 == 0:
        segLen = 13
        staChrLen = 5
        nu = int(len(line)/13)
    for i in range(nu):
        sta = line[i*segLen:i*segLen+staChrLen].strip()
        phsType = line[i*segLen+staChrLen]
        weightCode = int(line[i*segLen+staChrLen+1])
        travTime = float(line[i*segLen+staChrLen+1+1:i*segLen+staChrLen+1+1+6])
        phsRecs.append([sta,phsType,travTime,weightCode])
    return phsRecs
        

def load_cnv(cnv_file="velout.cnv"):
    """                            
    Load VELEST output file and return a dict with event id as key
    the content is [lon,lat,dep,rms]
    """       
    cnv = {}
    lines = []                  
    with open(cnv_file,'r') as f:  
        for line in f: lines.append(line.rstrip())                
    ecount = 0                  
    for line in lines:
        if re.match('\d+',line[:2]):  # event line
            if line[:4] == "9999":    # end marker
                continue
            funcReturn = read_cnv_event_line(line)
            evstr,etime,evlo,evla,evdp,emag,maxStaAzGap,rms = funcReturn[:8]
            ecount+=1
            availStas = []
            cnv[evstr] = {}  
            cnv[evstr]["etime"] = etime
            cnv[evstr]["evlo"] = evlo
            cnv[evstr]["evla"] = evla
            cnv[evstr]["evdp"] = evdp
            cnv[evstr]["emag"] = emag
            cnv[evstr]["rms"] = rms
            cnv[evstr]["phases"] = []
            cnv[evstr]['nsta'] = 0 
            cnv[evstr]['maxStaAzGap']=maxStaAzGap
            cnv[evstr]["lines"] = []
        elif len(line)==0:
            cnv[evstr]['evid'] = ecount
        elif line[0]==" ":
            cnv[evstr]['evid'] = int(line)
        elif line[:4] == "9999":
            continue
        else:
            phsRecs = read_cnv_phase_line(line)
            for sta,phsType,travTime,weightCode in phsRecs:
                if sta not in availStas:
                    cnv[evstr]['nsta'] += 1
                    availStas.append(sta)
                cnv[evstr]['phases'].append([sta,phsType,travTime,weightCode])
    return cnv    
    
