import re
from obspy.geodetics import gps2dist_azimuth
import json
import copy
import numpy as np
import matplotlib.pyplot as plt

def sta_dist_pairs(netstas,sta_dict,group_qty=6):
    """
    For station list provided, form unique pairs of stations by cloest
    distance.

    Parameters:
        netstas: station list in format network(2 char)+station(maximum 5 
                 character
        stadict: loaded station file containing station lon and lat information
      group_qty: control the quantity of cloest stations to form pairs.
    """
    pair_list = []
    for netsta1 in netstas:     # loop for each sta
        net1 = netsta1[:2]
        sta1 = netsta1[2:]
        lon1 = sta_dict[net1][sta1][0]
        lat1 = sta_dict[net1][sta1][1]
        dist_list = []
        for netsta2 in netstas: # calculate distance one by one
            net2 = netsta2[:2]
            sta2 = netsta2[2:]
            lon2 = sta_dict[net2][sta2][0]
            lat2 = sta_dict[net2][sta2][1]  
            dist,_,_ = gps2dist_azimuth(lat1,lon1,lat2,lon2)
            dist_list.append(dist)
        dist_list_cp = dist_list.copy()
        dist_list_cp.sort()
        for i in range(1,min([group_qty+1,len(netstas)])): # 0 idx is self
            dist = dist_list_cp[i]
            idx = dist_list.index(dist)
            tmp_netsta = netstas[idx]
            if [netsta1,tmp_netsta] not in pair_list and\
               [tmp_netsta,netsta1] not in pair_list:
                pair_list.append([netsta1,tmp_netsta])      
    return pair_list

def load_sta(sta_file):
    """
    Load in station information.

    Parameter:
        sta_file: text file in free format "net sta lon lat ele label\n"
        The label is intended for mark special stations purpose 
    Return:
        dictionary in structure sta_dict[net][sta]=[lon,lat,ele,label]
    """
    sta_dict={}
    with open(sta_file,'r') as f:
        for line in f:
            line = line.rstrip()
            splits = re.split("[ ,;]+",line)
            if len(splits) == 7:
                net,sta,_lon,_lat,_ele,mkr1,mkr2 = splits
            elif len(splits) == 6:
                net,sta,_lon,_lat,_ele,mkr1 = splits
            else:
                raise Exception("Wrong station file format!!!")

            if net not in sta_dict:
                sta_dict[net]={}
            if sta not in sta_dict[net]:
                if len(splits)==7:
                    sta_dict[net][sta] = [float(_lon),float(_lat),int(float(_ele)),mkr1,mkr2]
                elif len(splits)==6:
                    sta_dict[net][sta] = [float(_lon),float(_lat),int(float(_ele)),mkr1]
    return sta_dict

def getNet(sta,staFile):
    staDict = load_sta(staFile)
    findStatus = False
    for net in staDict.keys():
        for key in staDict[net].keys():
            if key == sta:
                findStatus = True
                return net
    if findStatus == False:
        raise Exception("Sta not in station file.")
    return net

def to_inv_sta_file(sta_dict,out_file="sta.inv",ele_zero=True):
    f_inv = open(out_file,'w')
    for net in sta_dict.keys():
        for sta in sta_dict[net].keys():
            lon = sta_dict[net][sta][0]
            lon_marker="E"
            if lon < 0:
                lon_marker = "W"
                lon = -lon
            lat = sta_dict[net][sta][1]
            ele = sta_dict[net][sta][2]
            label = sta_dict[net][sta][3]
            net_sta = net+sta
            lon_i = int(lon)
            lon_f = lon-lon_i
            lat_i = int(lat)
            lat_f = lat-lat_i
            if ele_zero:
                ele = 0
            f_inv.write(format(sta,"<6s")+format(net,"<4s")+"SHZ  "+format(lat_i,">2d")+" "+\
                format(lat_f*60,">7.4f")+" "+format(lon_i,">3d")+" "+format(lon_f*60,">7.4f")+\
                lon_marker+format(ele,">4d")+"\n")
    f_inv.close()

def sta2inv(sta_file,out_file="sta.inv",ele_zero=True):
    """
    Convert station file into HYPOINVERSE format
    """
    sta_dict =load_sta(sta_file)
    to_inv_sta_file(sta_dict,out_file,ele_zero=True)  # Write into files

def to_dd_sta_file(sta_dict,out_file="netsta.dd",ele_zero=True):
    f_dd = open(out_file,'w')
    for net in sta_dict.keys():
        for sta in sta_dict[net].keys():
            lon = sta_dict[net][sta][0]
            lat = sta_dict[net][sta][1]
            ele = sta_dict[net][sta][2]
            label = sta_dict[net][sta][3]
            net_sta = net+sta
            lon_i = int(lon)
            lon_f = lon-lon_i
            lat_i = int(lat)
            lat_f = lat-lat_i
            if ele_zero:
                ele = 0
            f_dd.write(format(net_sta,"<9s")+format(lat_i+lat_f,">9.6f")+format(lon_i+lon_f,">12.6f")+\
                   " "+format(ele,'>5d')+"\n")
    f_dd.close()

def sta2dd(sta_file,out_file="sta.dd",ele_zero=True):
    """
    Convert station file into hypoDD format
    """
    sta_dict = load_sta(sta_file)
    to_dd_sta_file(sta_dict,out_file,ele_zero=ele_zero)  # Write into files

def to_vel_sta_file(sta_dict,out_file="sta.vel",ele_zero=True):
    f_vel = open(out_file,'w')
    f_vel.write("(a5,f7.4,a1,1x,f8.4,a1,1x,i4,1x,i1,1x,i3,1x,f5.2,2x,f5.2,3x,i1)\n")
    sta_count = 1
    for net in sta_dict.keys():
        for sta in sta_dict[net].keys():
            lon = sta_dict[net][sta][0]
            lat = sta_dict[net][sta][1]
            ele = sta_dict[net][sta][2]
            label = sta_dict[net][sta][3]
            if ele_zero:
                ele = 0
            f_vel.write(f"{format(sta,'<5s')}{format(lat,'7.4f')}N {format(lon,'8.4f')}E {format(ele,'4d')} 1 "+\
                f"{format(sta_count,'3d')} {format(0,'5.2f')}  {format(0,'5.2f')}   1\n")
            sta_count += 1
    f_vel.write("  \n")   # signal of end of file for VELEST
    f_vel.close()

def sta2vel(sta_file,out_file="sta.vel",ele_zero=True):
    """
    Convert station file into VELEST format with 5 characters,
    which is applicable for the update VELEST program modified by Hardy ZI
    """
    sta_dict = load_sta(sta_file)
    to_vel_sta_file(sta_dict,out_file,ele_zero)

def toREAL(sta_dict,outFile):
    fsta = open(outFile,'w')
    for net in sta_dict.keys():
        for sta in sta_dict[net].keys():
            lon = sta_dict[net][sta][0]
            lat = sta_dict[net][sta][1]
            ele = sta_dict[net][sta][2]
            label = sta_dict[net][sta][3]
            fsta.write(f"{format(lon,'10.6f')} {format(lat,'10.6f')} {net} {format(sta,'5s')} BHZ {format(ele/1000,'5.3f')}\n")
    fsta.close()

def sta2REAL(sta_file,out_file):
    """
    Convert station file into VELEST format with 5 characters,
    which is applicable for the update VELEST program modified by Hardy ZI
    """
    sta_dict = load_sta(sta_file)
    toREAL(sta_dict,out_file)

def sta_sel(sta_file,c_lon,c_lat,nets=[],stas=[],radius=100):
    """
    select stations inside radius of a give lon and lat and output.
    output is a {sta_file}.sel file.
    Parameters:
        c_lon: longitude of the center. If c_lon<-180 or >180,radius filter will be passed
        c_lat: latitude of the center. If c_lat<-90 or >90, radisu filter will be passed
        nets: select nets if nets not empty
        stas: select stas if stas not empty 
    """
    select_net = False
    select_sta = False
    if len(nets)!=0:
        select_net = True
    if len(nets)!=0:
        select_sta = True

    out_file = sta_file+".sel"
    f1 = open(out_file,'w')
    with open(sta_file,'r') as f2:
        for line in f2:
            line = line.rstrip()
            net,sta,_lon,_lat,_ele,label = line.split()
            if len(nets)>0:   # select network
                if net not in nets:
                    continue
            if len(stas)>0:   # select station
                if sta not in stas:
                    continue
            lon = float(_lon)
            lat = float(_lat)
            if c_lon<=180 and c_lon>=-180 and c_lat>=-90 and c_lat<=90:
                dist,_,_ = gps2dist_azimuth(c_lat,c_lon,lat,lon)
                dist_km = dist/1000
                if dist_km > radius:
                    continue
            f1.write(line+"\n")
    f2.close()
    f1.close()

def sta2eqt(sta_file,out_file):
    """
    Convert station file into EQTransformer format
    """
    eqt_sta_dict = {}
    with open(sta_file,'r') as f:
        for line in f:
            line = line.rstrip()
            net,sta,_lon,_lat,_ele,_ = re.split(" +",line[:42])
            lon = float(_lon)
            lat = float(_lat)
            ele = float(_ele)
            eqt_sta_dict[sta]={}
            eqt_sta_dict[sta]["network"]=net
            eqt_sta_dict[sta]["channels"]=["BHN","BHE","BHZ"]
            eqt_sta_dict[sta]["coords"] = [lat,lon,ele]
    f.close()
    with open(out_file,'w') as dump_f:
        json.dump(eqt_sta_dict,dump_f)
    dump_f.close()


class Sta():
    def __init__(self,sta_file="/home/jinping/Dropbox/Weiyuan_resources/geo_elements/stations/sta_sum_202207.txt"):
        self.sta_file = sta_file
        self.dict = load_sta(sta_file)
        self.get_locs()
        
    def get_locs(self):
        self.locs = []
        for net in self.dict.keys():
            for sta in self.dict[net].keys():
                lon = self.dict[net][sta][0]
                lat = self.dict[net][sta][1]
                ele = self.dict[net][sta][2]
                marker = self.dict[net][sta][3]
                self.locs.append([lon,lat,ele])
        self.locs = np.array(self.locs)
        
    def plot(self,xlim=[],ylim=[],markersize=6,size=20,offsets=[0.1,0.1],fontsize=12):

        for net in self.dict.keys():
            for sta in self.dict[net].keys():
                lon = self.dict[net][sta][0]
                lat = self.dict[net][sta][1]
                if len(xlim) > 0:
                    if lon<xlim[0] or lon>xlim[1]:
                        continue
                if len(ylim) > 0:
                    if lat<ylim[0] or lat>ylim[1]:
                        continue
                plt.scatter(lon,
                    lat,
                    s=size,
                    edgecolors = "k",
                    facecolors='none',
                    marker='^',
                    alpha=1)
                plt.text(lon+offsets[0],
                    lat+offsets[1],
                    sta,
                    fontsize=fontsize)
                
        if len(xlim) != 0:
            plt.xlim(xlim)
        if len(ylim) != 0: 
            plt.ylim(ylim)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()
        
    def copy(self):
        return copy.deepcopy(self)
    
    def select(self,net=None):
        if net != None:
            dict_copy = self.dict.copy()
            self.dict = {}
            if isinstance(net,str):
                self.dict[net] = dict_copy[net]
            elif isinstance(net,list):
                for kk in net:
                    self.dict[kk] = dict_copy[kk]
        self.get_locs()
        return self
    
    def subset(self,xlim=[],ylim=[]):
        dict_subset = {}
        for net in self.dict.keys():
            dict_subset[net] = {}
            for sta in self.dict[net].keys():
                lon = self.dict[net][sta][0]
                lat = self.dict[net][sta][1]
                if lon>=xlim[0] and lon<=xlim[1] and\
                   lat>=ylim[0] and lat<=ylim[1]:
                    dict_subset[net][sta] = self.dict[net][sta]
        self.dict = dict_subset
        
        return self
    
    def sta2inv(self,ele_zero=True):
        to_inv_sta_file(self.dict,ele_zero=ele_zero)
        
    def sta2dd(self,ele_zero=True):
        to_dd_sta_file(self.dict,ele_zero=ele_zero)
        
    def sta2vel(self,ele_zero=True):
        to_vel_sta_file(self.dict,ele_zero=ele_zero)
    
    def __repr__(self):
        self.get_locs()
        _qty = f"Including {len(self.locs)} stations\n"
        _lon = f"Longitude range is: {format(np.min(self.locs[:,0]),'8.3f')} to {format(np.max(self.locs[:,0]),'8.3f')}\n"
        _lat = f"Latitude range is: {format(np.min(self.locs[:,1]),'7.3f')} to {format(np.max(self.locs[:,1]),'7.3f')}\n"
        _dep = f"Elevation range is: {format(np.min(self.locs[:,2]),'4.1f')} to {format(np.max(self.locs[:,2]),'4.1f')}\n"
        return _qty+_lon+_lat+_dep
    
    def __getitem__(self,net):
            return self.dict[net]
