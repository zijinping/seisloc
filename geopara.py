import re
import os
import numpy as np
from seisloc.geometry import lonlat_by_dist
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import collections
import matplotlib.patches as patches

class WYpara():
    '''
    This class reads parameters in "wy.para", which contains parameters for GMT plot.
    '''
    def __init__(self,paraFile="wy.para",workDir="/home/jinping/Dropbox/Weiyuan_share",mode='normal'):
        self.dict={}
        self.dict['workDir']=workDir
        paraPth = os.path.join(workDir,paraFile)
        with open(paraPth) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if mode=='debug':
                    print(line)
                if line[:2]=="#=":                      # after this are functions
                    break
                if len(line)==0 or line[0]=='#' or \
                   re.split(" ",line)[0] in ["gmt","if","","elif","then","fi",""]: # ignore comment lines, gmt set lines, and shell related lines
                    continue
                content = re.split(" +",line.rstrip())[0]
                para,info = re.split("=",content)
                para = para.strip()                     # remove fore and end spaces
                info = info.strip()
                if len(re.split("\$",info))>1:          # $ indicates citation of other parameters
                    for seg in re.split("\$",info)[1:]: # seperate each citation
                        sub = re.split("[/]",seg)[0]    # get the cited parameter name
                        info=info.replace("$"+sub,self.dict[sub])
                self.dict[para]=info
        f.close()
        #===============================================================================================
        # load in information
        # load_list=["city_loc",'city_label','ml_fault','zg_fault','Neo_fault','sta_loc','well']
        tmp_arr = []
        with open(self.dict['ml_fault'],'r') as f:
            for line in f:
                line = line.rstrip()
                _lon,_lat = re.split(" +",line)
                tmp_arr.append([float(_lon),float(_lat)])
        f.close()
        self.dict['ml_fault']=np.array(tmp_arr)
        #---------------------------------------------------------
        tmp_dict={}
        count = 0
        with open(self.dict['zg_faults'],'r') as f:
            for line in f:
                line = line.rstrip()
                if line[0] == "#":
                    continue     # pass comment line
                elif line[0]==">":
                    count+=1
                    tmp_dict[count]=[]
                else:
                    _lon,_lat = re.split(" +",line)
                    tmp_dict[count].append([float(_lon),float(_lat)])
        f.close()
        self.dict['zg_faults'] = tmp_dict
        #---------------------------------------------------------
        tmp_dict={}
        count = 0
        with open(self.dict['Neo_faults'],'r') as f:
            for line in f:
                line = line.rstrip()
                if line[0] == "#":
                    continue     # pass comment line
                elif line[0]==">":
                    count+=1
                    tmp_dict[count]=[]
                else:
                    _lon,_lat = re.split(" +",line)
                    tmp_dict[count].append([float(_lon),float(_lat)])
        f.close()
        for key in tmp_dict:
            tmp_dict[key] = np.array(tmp_dict[key])
        self.dict['Neo_faults'] = tmp_dict
        #---------------------------------------------------------
        tmp_dict={}
        count = 0
        with open(self.dict['WY_Neo_faults'],'r') as f:
            for line in f:
                line = line.rstrip()
                if line[0] == "#":
                    continue     # pass comment line
                elif line[0]==">":
                    count+=1
                    tmp_dict[count]=[]
                else:
                    _lon,_lat = re.split(" +",line)
                    tmp_dict[count].append([float(_lon),float(_lat)])
        f.close()
        for key in tmp_dict:
            tmp_dict[key] = np.array(tmp_dict[key])
        self.dict['WY_Neo_faults'] = tmp_dict
        #---------------------------------------------------------
        tmp_arr = []
        with open(self.dict['city_locs'],'r') as f:
            for line in f:
                line = line.rstrip()
                _lon,_lat,_lvl,name = re.split(" +",line)[:4]
                tmp_arr.append([float(_lon),float(_lat),int(_lvl),name])
        f.close()
        self.dict['city_locs']=tmp_arr
        f.close()
        #----------------------------------------------------------------
        tmp_arr = []
        with open(self.dict['sta_locs'],'r') as f:
            for line in f:
                line = line.rstrip()
                net,sta,_lon,_lat,_ele,marker = re.split(" +",line)
                tmp_arr.append([float(_lon),float(_lat),float(_ele),net,sta,marker])
        f.close()
        self.dict["sta_locs"]=tmp_arr
        f.close()
        #---------------------------------------------------------------
        tmp_arr = []
        with open(self.dict['wells'],'r') as f:
            for line in f:
                line = line.rstrip()
                if len(line)==0:
                    continue
                if line[0]=="#":
                    continue
                _lon,_lat,name,marker = re.split(" +",line)[:4]
                tmp_arr.append([float(_lon),float(_lat),name,marker])
        f.close()
        self.dict["wells"]=tmp_arr
        f.close()
        #---------------------------------------------------------------
        self.vel_depths = [0.00,0.38,1.64,2.97,4.41,6.04,6.95,8.50,10.0,12.0,33.9,36.0,37.9,39.9,43.9,45.9]
        self.vel_vp =     [3.38,4.92,4.92,5.19,5.46,6.10,6.57,6.60,6.61,6.63,6.70,6.85,7.09,7.27,7.44,7.61]
        self.vel_vs =     [1.61,2.68,2.73,2.83,3.14,3.42,3.42,3.82,3.83,3.84,3.87,3.96,4.10,4.20,4.30,4.40]

    def wellpad(self,pad_name,platform_edgecolor='k',platform_facecolor='white',well_edgecolor='k',lw=2):
        """
        Read in designated pad_name pad, the file name should be pad_name+'.pad'
        """
        pad_dir = self.dict["pad_dir"]
        pad_file = os.path.join(pad_dir,pad_name+'.pad')
        col = wellpad(pad_file,
                platform_edgecolor = platform_edgecolor,
                platform_facecolor = platform_facecolor,
                well_edgecolor = well_edgecolor,
                lw = lw)
        return col
    
    def wellpads(self,platform_edgecolor='k',platform_facecolor='white',well_edgecolor='k',lw=2):
        """
        Read in all wellpads with file name end with ".pad"
        """
        cols = []
        for file in os.listdir(self.dict["pad_dir"]):
            if file[-3:] == "pad":
                pad_name = file[:-4]
                col = self.wellpad(pad_name,
                              platform_edgecolor = platform_edgecolor,
                              platform_facecolor = platform_facecolor,
                              well_edgecolor = well_edgecolor,
                              lw = lw)
                cols.append(col)
        return cols
    
    def __str__(self):
        return "%s" %str(self.dict.keys())
    def __repr__(self):
        return "%s" %str(self.dict.keys())
    def __getitem__(self,item):
        return self.dict[item]
    
def read_pad_file(pad_file):
    """
    Line start with '#' is comment line.

    The first line is basic information line with format:
    Well_pad_name well_pad_lon well_pad_lat
    The reason for such format is because that the information is extracted from image, 
    It is better to describe the relative position between horizontal well controlling points and platform
    e.g. W204H37 104.8075537 29.58421817

    For each later line, it presents one horizontal well, it is constrained in the format:
    dx1 dy1 dx2 dy2, ..., dxs,dys # unit in km, with reference to the platform
    The longitude and latitude of controlling points is transferred by:
    lon1,lat1 = lonlat_by_dist(platform_lon,plaform_lat, dx_km,dy_km)

    The estimated uncertainty is ~3.3%
    """
    cont = []
    with open(pad_file,'r') as f:
        i = 0 # line counter
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == "#": # empty line and comment line
                continue
            line = re.split("#+",line)[0]     # remove comment after "#"
            if i == 0:
                tmps = re.split(" +",line)
                if len(tmps) == 3:
                    pad_name, _lon,_lat = re.split(" +",line)
                    cont.append([pad_name,float(_lon),float(_lat)])
                elif len(tmps) == 4:
                    pad_name, _lon,_lat,_sf = re.split(" +",line)
                    cont.append([pad_name,float(_lon),float(_lat),float(_sf)])
                else:
                    raise Exception("Wrong header line, should contain 3 or 4 elements")
            else:
                tmp_list = []
                for _tmp in re.split(" +",line):
                    if len(_tmp)==0:
                        continue
                    tmp_list.append(float(_tmp))
                cont.append(tmp_list)
            i = i+1
    return cont

def wellpad(pad_file,platform_edgecolor='k',platform_facecolor='white',well_edgecolor='k',lw=2):
    """
    Read in designated pad_name pad, the file name should be pad_name+'.pad'
    """
    if not os.path.exists(pad_file):
        print("Pad not in the pads library")
    new_mode = True
    cont = read_pad_file(pad_file)
    collect = []
    if len(cont[0]) == 3:
        padname = cont[0]; lon = cont[0][1]; lat = cont[0][2];
    elif len(cont[0]) == 4:
        padname = cont[0]; lon = cont[0][1]; lat = cont[0][2]; sf =cont[0][3]
        new_mode = False

    for _,args in enumerate(cont[1:]):    # Horizontal wells
        verts = [(lon,lat)]
        codes = [Path.MOVETO]
        if len(args) == 0:
            raise Exception("Error: No point information in the horizontal well line")
        if len(args)%2 == 1:
            raise Exception("Error: dx,dy list not in pairs")
        for i in range(int(len(args)/2)):
            dx1 = args[2*i]; dy1 = args[2*i+1]
            if new_mode:
                lon1,lat1 = lonlat_by_dist(lon,lat,dx1,dy1)
            else:
                dlon1 = dx1 * sf; lon1 = lon + dlon1;
                dlat1 = dy1 * sf; lat1 = lat + dlat1;
            verts.append((lon1,lat1))
            codes.append(Path.LINETO)
        path = Path(verts,codes)
        patch = patches.PathPatch(path,facecolor='none',edgecolor=well_edgecolor,lw=lw)
        collect.append(patch)
    # For platform square
    verts = [(lon+0.0009*2,lat+0.0009*2),
             (lon+0.0009*2,lat-0.0009*2),
             (lon-0.0009*2,lat-0.0009*2),
             (lon-0.0009*2,lat+0.0009*2),
             (lon+0.0009*2,lat+0.0009*2)]
    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY]
    path = Path(verts,codes)
    patch = patches.PathPatch(path,facecolor=platform_facecolor,edgecolor=platform_edgecolor,lw=lw)
    collect.append(patch)
    col = collections.PatchCollection(collect,match_original=True)
    return col
