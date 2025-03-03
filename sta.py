import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from obspy.geodetics import gps2dist_azimuth

def load_sta(staPth:str,skipRows=0)->pd.DataFrame:
    """
    Read the staFile information and return a pandas dataframe.

    Parameter:
        staPth: Path for station file in format "net sta chn stlo stla stel marker1 marker2\n",
                the markers are reserved for further classification. Leave them 
                any character, e.g. "_", if not necessary.
      skipRows: skip head lines
    Return:
        pd.DataFrame with columns ["net","sta","chn","stlo","stla","stel","mkr1","mkr2"]
    """
    df = pd.read_csv(staPth,
                     names=["net","sta","chn","stlo","stla","stel","mkr1","mkr2"],
                     skiprows=skipRows,
                     sep=r'\s*[,;]\s*|\s+',
                     engine='python')
    df["net"]  = df["net"].astype(str)
    df["sta"]  = df["sta"].astype(str)
    df["chn"]  = df["chn"].astype(str)
    df["stlo"] = df["stlo"].astype(float)
    df["stla"] = df["stla"].astype(float)
    df["stel"] = df["stel"].astype(float)
    df["mkr1"] = df["mkr1"].astype(str)
    df["mkr2"] = df["mkr2"].astype(str)

    return df

def to_INV_sta_file(df,outPth="sta.inv",eleZero=True):
    """
    Convert station file into HYPOINVERSE format
    """
    f = open(outPth,'w')
    for net in sorted(df["net"].unique()):
        dfNet = df[df["net"]==net].sort_values(by="sta")
        for idx,row in dfNet.iterrows():
            stlo = row["stlo"]; stla = row["stla"]; stel = row["stel"]
            stloMkr="E"
            if stlo < 0:
                stloMkr = "W"
                stlo = -stlo

            stloInt = int(stlo)       # integer part
            stloDec = stlo-stloInt    # decimal part
            stlaInt = int(stla)       # integer part
            stlaDec = stla-stlaInt    # decimal part
            if eleZero: stel = 0
            f.write(format(row["sta"],"<6s")+format(net,"<4s")+"SHZ  "+format(stlaInt,">2d")+" "+\
                format(stlaDec*60,">7.4f")+" "+format(stloInt,">3d")+" "+format(stloDec*60,">7.4f")+\
                stloMkr+format(stel,">4d")+"\n")
    f.close()

def sta2INV(staPth,outPth="sta.inv",eleZero=True)->None:
    """
    Convert station file into HYPOINVERSE format
    """
    df = load_sta(staPth)
    to_INV_sta_file(df,outPth,eleZero=True)  # Write into files

def to_DD_sta_file(df,outPth="sta.dd",eleZero=True)->None:
    """
    Convert station file into hypoDD format
    """
    f = open(outPth,'w')
    for net in sorted(df["net"].unique()):
        dfNet = df[df["net"]==net].sort_values(by="sta")
        for idx,row in dfNet.iterrows():
            stlo = row["stlo"]; stla = row["stla"]; stel = row["stel"]
            sta = row["sta"]
            netSta = net+sta
            if eleZero:
                stel = 0
            f.write(format(netSta,"<9s")+format(stlo,">9.6f")+format(stla,">12.6f")+\
                   " "+format(stel,'>5d')+"\n")
    f.close()

def sta2DD(staPth,outPth="sta.dd",eleZero=True):
    """
    Convert station file into hypoDD format
    """
    df = load_sta(staPth)
    to_DD_sta_file(df,outPth,eleZero=eleZero)  # Write into files

def to_VELEST_sta_file(df,outPth="sta.vel",eleZero=True):
    f = open(outPth,'w')
    f.write("(a5,f7.4,a1,1x,f8.4,a1,1x,i4,1x,i1,1x,i3,1x,f5.2,2x,f5.2,3x,i1)\n")
    staQty = 1
    for net in sorted(df["net"].unique()):
        dfNet = df[df["net"]==net].sort_values(by="sta")
        for idx,row in dfNet.iterrows():
            stlo = row["stlo"]; stla = row["stla"]; stel = row["stel"]
            sta = row["sta"]
            if eleZero: stel = 0
            f.write(f"{format(sta,'<5s')}{format(stla,'7.4f')}N {format(stlo,'8.4f')}E {format(stel,'4d')} 1 "+\
                f"{format(staQty,'3d')} {format(0,'5.2f')}  {format(0,'5.2f')}   1\n")
            staQty += 1
    f.write("  \n")   # signal of end of file for VELEST
    f.close()

def sta2VEL(staPth,outPth="sta.vel",eleZero=True):
    """
    Convert station file into VELEST format with 5 characters,
    which is applicable for the update VELEST program modified by Hardy ZI
    """
    dd = load_sta(staPth)
    to_VELEST_sta_file(dd,outPth,eleZero)

def to_REAL_sta_file(df,outPth='sta.real',eleZero=True):
    """
    Convert station file into REAL format
    """
    f = open(outPth,'w')
    for net in sorted(df["net"].unique()):
        dfNet = df[df["net"]==net].sort_values(by="sta")
        for idx,row in dfNet.iterrows():
            stlo = row["stlo"];stla = row["stla"];stel = row["stel"]
            sta = row["sta"];chn = row["chn"]
            if eleZero: stel = 0
            f.write(f"{format(stlo,'10.6f')} {format(stla,'10.6f')} {net} {format(sta,'5s')} {chn} {format(stel/1000,'5f')}\n")
    f.close()

def sta2REAL(staPth,outPth='sta.real',eleZero=True):
    """
    Convert station file into REAL format.
    """
    df = load_sta(staPth)
    to_REAL_sta_file(df,outPth,eleZero)


def to_EQT_sta_file(df,outPth="staEQT.json",eleZero=False):
    """
    Convert station file into EQTransformer format
    """
    staDict = {}
    for net in sorted(df["net"].unique()):
        dfNet = df[df["net"]==net].sort_values(by="sta")
        for idx,row in dfNet.iterrows():
            stlo = row["stlo"]; stla = row["stla"]; stel = row["stel"]
            sta = row["sta"]; chn = row["chn"]; ch = chn[:-1]
            if eleZero: stel = 0
            staDict[sta]={}
            staDict[sta]["network"]=net
            staDict[sta]["channels"]=[ch+"N",ch+"E",ch+"Z"]
            staDict[sta]["coords"] = [stla,stlo,stel]
    with open(outPth,'w') as f:
        json.dump(staDict,f)
    f.close()

def sta2EQT(staPth,outPth="staEQT.json",eleZero=False):
    """
    Convert station file into EQTransformer format
    """
    df = load_sta(staPth)
    to_EQT_sta_file(df,outPth,eleZero)
    
class Station():
    """
    Text file in the format "net sta chn stlo stla stel stdp mkr1 mkr2"
    """
    def __init__(self,staPth:str,skipRows=0):
        """
        staPth: path of the station file in the format "net sta chn stlo stla stel stdp mkr1 mkr2"
        skipRows: skip head lines
        """
        self.staFile = staPth
        self.df = load_sta(staPth,skipRows=skipRows)
        
    def plot(self,xlim=[],ylim=[],size=20,txtOffsets=[0.001,0.001],fontsize=12):
        dfPlt = self.df.copy()
        if len(xlim) != 0:
            plt.xlim(xlim)
            dfPlt = dfPlt[(dfPlt["stlo"]>=xlim[0]) & (dfPlt["stlo"]<=xlim[1])]
        if len(ylim) != 0:
            plt.ylim(ylim)
            dfPlt = dfPlt[(dfPlt["stla"]>=ylim[0]) & (dfPlt["stla"]<=ylim[1])]
        
        # symbols
        plt.scatter(dfPlt["stlo"],dfPlt["stla"],s=size,
            edgecolors = "k",facecolors='none',marker='^',
            alpha=1)
        # station names
        for i, txt in enumerate(dfPlt["sta"]):
            plt.text(dfPlt["stlo"].iloc[i] + txtOffsets[0],
                     dfPlt["stla"].iloc[i] + txtOffsets[1],
                     txt,
                     fontsize=fontsize)
            
        medStla = np.median(dfPlt["stla"])
        plt.gca().set_aspect(np.cos(np.deg2rad(medStla)))
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
    def copy(self):
        return copy.deepcopy(self)
    
    def select(self,net=None,sta=None,chn=None):
        """
        Select stations by network or station name. "net", "sta" and "chn" could 
        be either str or list. Return a new Sta object.
        """
        dfSel = self.df.copy()
        if net != None:
            if isinstance(net, str):
                dfSel = dfSel[dfSel["net"] == net]
            elif isinstance(net, list):
                dfSel = dfSel[dfSel["net"].isin(net)]
        if sta != None:
            if isinstance(sta, str):
                dfSel = dfSel[dfSel["sta"] == sta]
            elif isinstance(sta, list):
                dfSel = dfSel[dfSel["sta"].isin(sta)]
        if chn != None:
            if isinstance(chn, str):
                dfSel = dfSel[dfSel["chn"] == chn]
            elif isinstance(chn, list):
                dfSel = dfSel[dfSel["chn"].isin(chn)]

        self.df = dfSel
    
    def trim(self,lomin,lomax,lamin,lamax):
        """
        Select stations by lon and lat range
        """
        self.df = self.df[(self.df["stlo"]>=lomin) & (self.df["stlo"]<=lomax)& \
                          (self.df["stla"]>=lamin) & (self.df["stla"]<=lamax)]
        return self
    
    def trim_by_distance(self,clo,cla,radius):
        """
        Select stations by distance(km) from a center point
        """
        dists = self.df.apply(lambda x: gps2dist_azimuth(cla,clo,x["stla"],x["stlo"])[0],axis=1)
        self.df = self.df[dists/1000<=radius]
        return self
    
    def form_pairs_by_distance(self,qty=6):
        """
        Form unique station pairs by distance for ambient noise cross-correlation.

        Parameters:
            netstas: station list in format network(2 char)+station(maximum 5 
                    character
        qty: quantity of cloest stations to form pairs.
        """
        pairLst = []
        for i,dfSta1 in self.df.iterrows():
            netsta1 = dfSta1["net"]+dfSta1["sta"]
            stlo1 = dfSta1["stlo"]
            stla1 = dfSta1["stla"]
            distLst = []
            for j,dfSta2 in self.df.iterrows():
                if i == j: distLst.append(0);continue   # skip self
                netsta2 = dfSta2["net"]+dfSta2["sta"]
                stlo2 = dfSta2["stlo"]
                stla2 = dfSta2["stla"]
                dist,_,_ = gps2dist_azimuth(stla1,stlo1,stla2,stlo2)
                distLst.append(dist)
            distLstSort = sorted(distLst)
            for k in range(1,min([qty+1,len(distLst)])):
                dist = distLstSort[k]
                idx = distLst.index(dist)
                netsta2 = self.df.iloc[idx]["net"]+self.df.iloc[idx]["sta"]
                if netsta1 == netsta2: continue
                if [netsta1,netsta2] not in pairLst and [netsta2,netsta1] not in pairLst:
                    pairLst.append([netsta1,netsta2])
        return pairLst

    def to_sta_file(self,format,outPth,eleZero=True):
        """
        Convert station file into different format
        """
        if format == "INV":
            to_INV_sta_file(self.df,outPth,eleZero=eleZero)
        elif format == "DD":
            to_DD_sta_file(self.df,outPth,eleZero=eleZero)
        elif format == "VELEST":
            to_VELEST_sta_file(self.df,outPth,eleZero=eleZero)
        elif format == "REAL":
            to_REAL_sta_file(self.df,outPth,eleZero=eleZero)
        elif format == "EQT":
            to_EQT_sta_file(self.df,outPth,eleZero=eleZero)
        elif format == "original":
            self.df.to_csv(outPth,header=False,index=False,sep=" ")
        else:
            print("Invalid format")
    
    def __repr__(self):
        _qty = f"Total {len(self.df)} stations\n"
        _lon = f"Longitude range is: {format(np.min(self.df['stlo']),'8.3f')} to {format(np.max(self.df['stlo']),'8.3f')}\n"
        _lat = f"Latitude range is:  {format(np.min(self.df['stla']),'7.3f')} to {format(np.max(self.df['stla']),'7.3f')}\n"
        _dep = f"Elevation range is: {format(np.min(self.df['stel']),'4.1f')} to {format(np.max(self.df['stel']),'4.1f')}\n"
        return _qty+_lon+_lat+_dep
    
    def copy(self):
        return deepcopy(self)

