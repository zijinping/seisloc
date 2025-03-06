import re
import pickle
import pandas as pd
from tqdm import tqdm
from seisloc.loc.hypoinv import Hypoinv
from obspy import UTCDateTime


def subset_dtcc(evids,ccFilePth,saveFilePth="",useStas=[]):
    linesUse = []      
    ccFileName = os.path.basename(ccFilePth)
    with open(ccFilePth,'r') as f:
        for line in tqdm(f):  
            if line[0] == "#":
                record=False
                _,_evid1,_evid2,_ = line.split()
                if int(_evid1) in evids or int(_evid2) in evids:
                    linesUse.append(line)
                    record = True
            else:          
                if record==True:
                    if useStas==[]:
                        linesUse.append(line)
                    else:
                        sta = re.split(" +",line)[0]
                        if sta in useStas:
                            linesUse.append(line)
    if saveFilePth=="":
        saveFilePth=ccFileName+'.subset'
    with open(saveFilePth,'w') as f:
        for line in linesUse:
            f.write(line)

def load_dtcc(dtccPth="dt.cc"):
    """
    Load dt.cc file and return a DataFrame with columns: evid1, evid2, sta, pha, dt, cc
    """
    if dtccPth[-4:] == ".pkl":                         # .pkl file load by pickle
        f = open(dtccPth,'rb')
        dfSort = pickle.load(f)
    else:                                           # using normal processing
        data = []
        f = open(dtccPth,'r')
        for line in f:
            line = line.rstrip()
            if line[0] == "#":
                _,_evid1,_evid2,_ = re.split(" +",line)
                evid1 = int(_evid1); evid2 = int(_evid2)
            else:
                sta,_diff,_cc,pha = re.split(" +",line.strip())
                data.append([evid1,evid2,sta,pha,float(_diff),float(_cc)])
        df = pd.DataFrame(data,columns=["evid1","evid2","sta","pha","dt","cc"])
        dfSort = df.sort_values(by=["evid1","evid2"],ascending=False)

    return dfSort

def dtcc_otc(dtccPth,invOldPth,invNewPth):
    """
    Conduct dtcc origin time correction in updated out.sum
    The output file is a new file with suffix .otc after the input file
    """
    #------------ load data ---------------------
    with open(dtccPth,'r') as f:
        lines = f.readlines()
    invOld = Hypoinv(invOldPth)
    invNew = Hypoinv(invNewPth)
    #------------ processing --------------------
    f = open("dtcc.otc",'w')
    for line in tqdm(lines):
        line = line.rstrip()
        if line[0] == "#":     # event pair line
            status = True
            _,_id1,_id2,_otc = line.split()
            id1 = int(_id1)
            id2 = int(_id2)
            otc = float(_otc)
            # ------------- read old event time ---------------
            _et1Old = invOld[id1][0]
            et1Old = UTCDateTime.strptime(_et1Old,'%Y%m%d%H%M%S%f')
            _et2Old = invOld[id2][0]
            et2Old = UTCDateTime.strptime(_et2Old,'%Y%m%d%H%M%S%f')
            # ------------- read new event time ---------------
            try:
                _et1New = invNew[id1][0]
            except:
                status = False       # events not included in the new set
                continue
            et1New = UTCDateTime.strptime(_et1New,'%Y%m%d%H%M%S%f')
            try:
                _et2New = invNew[id2][0]
            except:
                status = False       # events not included in the new set
                continue
            et2New = UTCDateTime.strptime(_et2New,'%Y%m%d%H%M%S%f')
            # ------------- calculate otc --------------------
            det1 = et1New - et1Old    # Event origin time difference
            det2 = et2New - et2Old   
            otc = otc+(det1-det2)
            # ------------- prepare writting -----------------
            line = line[:14]+format(otc,'.2f')
        if status == True:
            f.write(line+"\n")
    f.close()
