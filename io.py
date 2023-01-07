import os
import re
import numpy as np
from tqdm import tqdm
from obspy import UTCDateTime

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
    
    return evstr,etime,evlo,evla,evdp,emag,maxStaAzGap

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
            evstr,etime,evlo,evla,evdp,emag,maxStaAzGap = funcReturn[:7]
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
            arc[evstr]["phase"].append([net,sta,phsType,travTime,res,weightCode])
            
        arc[evstr]["lines"].append(line)

    if savePkl:  
        outFile = y2000File+".pkl"
        f = open(out_name,'wb')
        pkl.dump(arc,f)
        f.close()
 
    return arc
