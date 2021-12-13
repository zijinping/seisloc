import re

class WY_para():
    '''
    This class reads parameters in "wy.para", which contains parameters for GMT plot.
    '''
    def __init__(self,para_path="/home/zijinping/Desktop/zijinping/resources/wy.para"):
        self.dict={}
        with open(para_path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if len(line)==0 or line[0]=='#' or line[:4]=="gmt ": # ignore comment line and gmt set line
                    continue
                if line[:9]=="root_path":
                     self.dict["root_path"] = re.split("=",line.rstrip())[1]
                     continue
                content = re.split(" +",line.rstrip())[0]
                para,info = re.split("=",content)
                if len(re.split("\$",info))>1: # $ indicates citation of other parameters
                    for seg in re.split("\$",info)[1:]: # seperate each citation
                        sub = re.split("[/]",seg)[0]   # get the cited parameter name
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
        self.dict['ml_fault']=tmp_arr
        #------------------------------------------------------------------------------------------------
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
        self.dict['Neo_faults'] = tmp_dict
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
                _lon,_lat,name,marker = re.split(" +",line)
                tmp_arr.append([float(_lon),float(_lat),name,marker])
        f.close()
        self.dict["wells"]=tmp_arr
        f.close()
        #---------------------------------------------------------------
        self.vel_depths = [0.00,0.38,1.64,2.97,4.41,6.04,6.95,8.50,10.0,12.0,33.9,36.0,37.9,39.9,43.9,45.9]
        self.vel_vp =     [3.38,4.92,4.92,5.19,5.46,6.10,6.57,6.60,6.61,6.63,6.70,6.85,7.09,7.27,7.44,7.61]
        self.vel_vs =     [1.61,2.68,2.73,2.83,3.14,3.42,3.42,3.82,3.83,3.84,3.87,3.96,4.10,4.20,4.30,4.40]

    def __str__(self):
        return "%s" %str(self.dict.keys())
    def __repr__(self):
        return "%s" %str(self.dict.keys())
    def __getitem__(self,item):
        return self.dict[item]
