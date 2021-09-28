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
