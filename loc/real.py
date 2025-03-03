import os
import re
import subprocess
from obspy import UTCDateTime
import pandas as pd
from seisloc.sta import load_sta,sta2REAL,getNet

def loadEQTphases(df):
    pdict = {}
    sdict = {}
    for index,rows in df.iterrows():
        ptime = rows["p_arrival_time"]
        pprob = rows["p_probability"]
        stime = rows['s_arrival_time']
        sprob = rows["s_probability"]
        if not pd.isna(ptime):
            ptime = UTCDateTime(ptime)
            pyr = ptime.year
            pmo = ptime.month
            pdy = ptime.day
            psecs = ptime - UTCDateTime(pyr,pmo,pdy)
            pdayStr = ptime.strftime("%Y%m%d")
            if pdayStr not in pdict.keys():
                pdict[pdayStr] = []
            else:
                pdict[pdayStr].append([psecs,pprob])
        if not pd.isna(stime):
            stime = UTCDateTime(stime)
            syr = stime.year
            smo = stime.month
            sdy = stime.day
            ssecs = stime - UTCDateTime(syr,smo,sdy)
            sdayStr = stime.strftime("%Y%m%d")
            if sdayStr not in sdict.keys():
                sdict[sdayStr] = []
            else:
                sdict[sdayStr].append([ssecs,sprob])
    return pdict,sdict

def writeREALpicks(net,sta,pdict,sdict,pickPath):
    """
    pdict,sdict in format of "YYYYMMDD"
    """
    for _day in pdict.keys():
        dayPath = os.path.join(pickPath,_day)
        if not os.path.exists(dayPath):
            os.mkdir(dayPath)
        pFilePath = os.path.join(dayPath,f"{net}.{sta}.P.txt")
        fp = open(pFilePath,'w')
        for record in pdict[_day]:
            psecs = record[0]
            pprob = record[1]
            fp.write(f"{format(psecs,'.3f')} {format(pprob,'4.2f')} 0\n")
        fp.close()
    for _day in sdict.keys():
        dayPath = os.path.join(pickPath,_day)
        if not os.path.exists(dayPath):
            os.mkdir(dayPath)        
        sFilePath = os.path.join(dayPath,f"{net}.{sta}.S.txt")        
        fs = open(sFilePath,'w')        
        for record in sdict[_day]:
            ssecs = record[0]
            sprob = record[1]
            fs.write(f"{format(ssecs,'.3f')} {format(sprob,'4.2f')} 0\n")
        fs.close()

def runREAL(workdir):
    os.chdir(workdir)
    subprocess.run(["perl","runREAL.pl"])

def writeREALpl(assoPath,pickPath,
                R='0.3/20/0.02/1/3',
                G="1.4/20/0.01/1",
                V="6.0/3.3",
                S="5/0/12/1/0.5/0/1.3/1.8",
                workdir="../../Picks/$year$mon$day",
                station="../../sta.real",
                ttime="../../ttdb.txt"):
    """
    Prepare perl scripts for REAL run
    """
    for date in os.listdir(pickPath):
        dayResultPath = os.path.join(assoPath,date)
        if not os.path.exists(dayResultPath):
            os.mkdir(dayResultPath)
        REALplPath = os.path.join(dayResultPath,'runREAL.pl')
        year=date[:4]
        month=date[4:6]
        day=date[6:8]
        with open(os.path.join(REALplPath),"w") as f:
            f.write("#!/usr/bin/perl -w\n")
            f.write(f"$year = \"{year}\";\n")
            f.write(f"$mon = \"{month}\";\n")
            f.write(f"$day = \"{day}\";\n")
            f.write("\n")
            f.write("$D = \"$year/$mon/$day\";\n")
            f.write(f"$R = \"{R}\";\n")
            f.write(f"$G = \"{G}\";\n")
            f.write(f"$V = \"{V}\";\n")
            f.write(f"$S = \"{S}\";\n")
            f.write("\n")
            f.write(f"$dir = \"{workdir}\";\n")
            f.write(f"$station = \"{station}\";\n")
            f.write(f"$ttime = \"{ttime}\";\n")
            f.write("\n")
            f.write("system(\"REAL -D$D -R$R -G$G -S$S -V$V $station $dir $ttime\");\n")
            f.write("print\"REAL -D$D -R$R -G$G -S$S -V$V $station $dir $ttime\";\n")
