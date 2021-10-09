import os
import shutil
import numpy as np
import re
import time
import obspy
import scipy.signal as signal
from obspy.geodetics import gps2dist_azimuth
import pdb
import math
import matplotlib.pyplot as plt
import scipy.io as scio
import glob
import argparse
import pickle
from obspy import Stream
import logging
import multiprocessing as mp

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations",
                        type=str,
                        help="station1/station2 to do the waveform cross-correlation")
    parser.add_argument("--data_dir",
                        default="day_data",
                        help="day waveform data directory")
    parser.add_argument('--period_range',
                        default="2/5",
                        type=str,
                        help="Period range for bandpass")
    parser.add_argument("--fs_new",
                        default=50,
                        type=int,
                        help="New sampling rate")
    parser.add_argument("--white_spec",
                        default=True,
                        type=bool,
                        help="Whether do spectrum whitening")
    parser.add_argument("--corr_method",
                        default=2,
                        type=int,
                        help="1: one-bit cross-correlation; 2: temporal normalization")
    parser.add_argument("--year_range",
                        default='2020/2020',
                        type=str,
                        help="year range year1/year2")
    parser.add_argument('--day_range',
                        default='131/131',
                        type=str,
                        help='day range day1/day2')
    parser.add_argument("--max_lagtime",
                         default=40,
                         type=int,
                         help="Maximum time length (in seconds) for each side of CFs")
    parser.add_argument("--seg_hour",
                        default=2,
                        type=int,
                        help="Data segment length for data preprocessing")
    parser.add_argument('--cmp',
                        default='BHZ/BHZ',
                        help="component for cross correlation")
    parser.add_argument("--out_dir",
                        default="CFs_Result",
                        help="output directory results")
    parser.add_argument("--mp",
                        default=1,
                        type=int,
                        help="default(1) use multiprocessing")
    parser.add_argument("--cpu_cores",
                        default=0,
                        type=int,
                        help="0 indicates using all cores")
    parser.add_argument("--sta_file",
                        default="./sta.txt",
                        type=str,
                        help="station file")
    args = parser.parse_args()

    tmp = args.period_range
    _tmp1,_tmp2 = re.split("\/",tmp)
    tmp1 = float(_tmp1); tmp2 = float(_tmp2)
    args.period_band=np.array([[tmp1,tmp2]])
    
    tmp = args.year_range
    _tmp1,_tmp2 = re.split("\/",tmp)
    tmp1 = int(_tmp1);  tmp2 = int(_tmp2)
    args.year_range=[tmp1,tmp2]

    tmp = args.day_range
    _tmp1,_tmp2 = re.split("\/",tmp)
    tmp1 = int(_tmp1); tmp2 = int(_tmp2)
    args.day_range=[tmp1,tmp2]
    
    tmp = args.cmp
    args.cmp1,args.cmp2 = re.split("\/",tmp)
    
    tmp = args.stations
    args.sta1,args.sta2 = re.split("\/",tmp)

    args.sta_file = "sta.txt"
    args.sta_dict = {}
    with open(args.sta_file,'r') as f:
        for line in f:
	        line = line.rstrip()
	        net,sta,_lon,_lat,_ele,_ = re.split(" +",line[:42])
	        lon = float(_lon)
	        lat = float(_lat)
	        ele = float(_ele)
	        args.sta_dict[sta]={}
	        args.sta_dict[sta]["network"]=net
	        args.sta_dict[sta]["channels"]=["BHN","BHE","BHZ"]
	        args.sta_dict[sta]["coords"] = [lat,lon,ele]
    f.close()
    args.sta_list = args.sta_dict.keys()

    return args

def xcorr(data1,data2,max_shift_num):
    len1 = len(data1)
    len2 = len(data2)
    min_len = min(len1,len2)
    cross_list = []
    for shift_num in np.arange(-max_shift_num,max_shift_num+1,1):
        if shift_num<0:
            correlate_value = np.correlate(data1[:min_len+shift_num],data2[-shift_num:min_len])
            cross_list.append(correlate_value.ravel())
        else:
            correlate_value = np.correlate(data2[:min_len-shift_num],data1[shift_num:min_len])
            cross_list.append(correlate_value.ravel())
    cross_list = np.array(cross_list)
    return cross_list.ravel()

def amp_avg(b,x):
    y = []
    for i in range(len(x)):
        tmp = 0
        if i < len(b):
            for j in range(i+1):
                #print(j,x[j],b[len(b)-j])
                tmp += x[j]*b[len(b)-j-1]
        else:
            for j in range(len(b)):
                tmp += x[i-len(b)+j+1]*b[j]
        y.append(tmp)
    return y

def st_check(tr):
    if tr.stats.npts > 3600 * tr.stats.sampling_rate:
        return True
    else:
        return False


def cross_correlation(tr1,tr2,period_band,max_lagtime,corr_method):
    sampleF = tr1.stats.sampling_rate
    sampleT = tr1.stats.delta
    if sampleF == 0:
        print("Error ! Sampling rate of data should not be zero! ")
    starttime1 = tr1.stats.starttime
    endtime1 = tr1.stats.endtime
    starttime2 = tr2.stats.starttime
    endtime2 = tr2.stats.endtime
    delta_initial = round((starttime2-starttime1)*sampleF)
    
    if delta_initial>0:
        if (tr1.stats.npts-delta_initial) > 0:
            tr1.data[0:tr1.stats.npts-delta_initial] = tr1.data[delta_initial:]
            tr1.data[delta_initial:] = 0
        else:
            tr1.data = 0
        delta_initial = 0
    if delta_initial<0:
        delta_initial = abs(delta_initial)
        if (tr2.stats.npts-delta_initial) > 0:
            tr2.data[0:tr2.stats.npts-delta_initial] = tr2.data[delta_initial:]
            tr2.data[delta_initial:] = 0
        else:
            tr2.data = 0
        delta_initial = 0
        
    point_num = min(tr1.stats.npts,tr2.stats.npts)
    max_shift_num = round(max_lagtime/sampleT)
    min_shift_num = -max_shift_num
    shift_num = max_shift_num - min_shift_num + 1
    num_PB = len(period_band[:,0])
    CFcnMB = np.zeros((shift_num,num_PB))
    
    for n in range(num_PB):
        # Obtain the start and end period for the period band
        startT = period_band[n,0]
        endT = period_band[n,1]
        lowF = 1/endT/(sampleF/2)
        highF = 1/startT/(sampleF/2)
        [B,A] = signal.butter(2,[lowF,highF],btype="bandpass")
        tr1.data = signal.filtfilt(B,A,tr1.data)
        tr2.data = signal.filtfilt(B,A,tr2.data)
        if corr_method == 1:
            tr1.data = np.sign(tr1.data)
            tr2.data = np.sign(tr2.data)
        elif corr_method == 2:  # Temporal normalization by dividing the smooth amplitude (from unning average)
            winsize = int(round(endT*2*sampleF))
            if winsize%2==0:
                winsize = winsize + 1
            shiftpt = round((winsize+1)/2)
            tmp1 = np.ones((1,winsize))*np.abs(tr1.data[0])
            tmp2 = np.ones((1,winsize))*np.abs(tr1.data[-1])
            tmpamp = np.concatenate((tmp1.ravel(),np.abs(tr1.data),tmp2.ravel()))
            tmpamp2 = np.correlate(tmpamp,np.ones((1,winsize)).ravel()/winsize,'full')
            tmpamp2 = tmpamp2[shiftpt+winsize-1:shiftpt+winsize+tr1.stats.npts-1]
            KK = np.where(tmpamp2>0)
            JJ = np.where(np.isnan(tmpamp2)==True)
            tr1.data[JJ] = 0
            tr1.data[KK] = tr1.data[KK]/tmpamp2[KK]

            tmp1 = np.ones((1,winsize))*np.abs(tr2.data[0])
            tmp2 = np.ones((1,winsize))*np.abs(tr2.data[-1])
            tmpamp = np.concatenate((tmp1.ravel(),np.abs(tr2.data),tmp2.ravel()))
            tmpamp2 = np.correlate(tmpamp,np.ones((1,winsize)).ravel()/winsize,'full')
            tmpamp2 = tmpamp2[shiftpt+winsize-1:shiftpt+winsize+tr1.stats.npts-1]
            KK = np.where(tmpamp2>0)
            JJ = np.where(np.isnan(tmpamp2)==True)
            tr2.data[JJ] = 0
            tr2.data[KK] = tr2.data[KK]/tmpamp2[KK]
        
        logger.info(f"sta1 and sta2 for correlation: {tr2.stats.station} {tr1.stats.station}")
        one_bit_cross = xcorr(tr2.data,tr1.data,max_shift_num)
        one_bit_cross = signal.filtfilt(B,A,one_bit_cross)
        CFcnMB[:,n] = one_bit_cross

    return CFcnMB

class CFdata():
    def __init__(self,nptCF,num_PB):
        self.year = 0
        self.day = 0
        self.NCF = np.zeros((nptCF,num_PB))

def get_day_data(data_dir,sta,yr,day,cmp):
    st = Stream()
    for seisfile in os.listdir(os.path.join(data_dir,sta)):
        try:
            st = obspy.read(os.path.join(data_dir,sta,seisfile),headonly=True)
            starttime = st[0].stats.starttime+0.1
            chn = st[0].stats.channel
            npts = st[0].stats.npts
            delta = st[0].stats.delta
            if starttime.year != yr or starttime.julday != day:
                continue
            if chn != cmp.upper():
                continue
            if int(npts*delta/60/60)==24:
                st = obspy.read(os.path.join(data_dir,sta,seisfile))
                break
        except:
            pass
    return st

def resample_data(tr,fs_new):
    decimateR_org = tr.stats.sampling_rate/fs_new
    decimateR = round(decimateR_org)
    if abs(decimateR_org - decimateR) > 0.001:
        logger.error("Error: resampling frequency error!")
        raise Exception("Error: resampling frequency!")
    nn = math.floor(tr.stats.npts/decimateR)
    if(nn*decimateR+1)<=tr.stats.npts:
        resample_data = signal.decimate(tr.data[:nn*decimateR+1],decimateR)
    else:
        resample_data = signal.decimate(
                                    np.concatenate((tr.data[:nn*decimateR],
                                                    tr.data[nn*decimateR])),
                                    decimateR)
    logger.info(f"resample before and after: {tr.stats.npts} {len(resample_data)}")
    tr.stats.delta = 1.0/fs_new
    tr.stats.npts = len(resample_data)
    tr.data = resample_data

    return tr

def noisecorr(sta1,sta2,cmp1,cmp2,yr,day,max_lagtime,fs_new,period_band,data_dir):

    # Frequency band info
    Tmax = period_band[0,1]
    Tmin = period_band[0,0]
    Trange = np.array([0.75*Tmin,Tmin,Tmax,1.5*Tmax]) # Period range of initial band pass filtering
    freq_range = 1.0/Trange[::-1]                     # Freq range for initial band apss filtering
                                                      # [f_low_cut,f_low_pass,f_high_pass,f_high_cut]
    
#    logger.info(f"station pair = {sta1} {sta2}")
    max_shift_num = round(max_lagtime*fs_new)
    CFtime = np.linspace(-max_shift_num,max_shift_num,2*max_shift_num+1)/fs_new
    nptCF = len(CFtime)
    CFdict = {}

    st1 = get_day_data(data_dir,sta1,yr,day,cmp1)
    st2 = get_day_data(data_dir,sta2,yr,day,cmp2)
    tr1 = st1[0]
    tr2 = st2[0]
    if len(tr1.data)==0 or len(tr2.data)==0:
        logger.info(f"load data failed: {yr} {day} {sta1} {sta2}")
        return

    logger.info(f"load data success: {yr} {day} {sta1} {sta2}")
    if st_check(tr1) and st_check(tr2):
        logging.info(f"Now processing: {yr} {day} ......")
        if tr1.stats.delta != tr2.stats.delta:
            logger.error("Error: data sampling rate is not the same!")
            raise Exception("Error: data sampling rate is not the same!")
        # Now resample data
        if(tr1.stats.sampling_rate)>fs_new:
            logger.info(f"resample data of {sta1}")
            tr1 = resample_data(tr1,fs_new)
            logger.info(f"resampled frequency is {tr1.stats.sampling_rate}")
        if(tr2.stats.sampling_rate)>fs_new:
            logger.info(f"resample data of {sta2}")
            tr2 = resample_data(tr2,fs_new)
        # Bandpass the data
        tr1.detrend("linear")
        tr1.detrend("constant")
        tr1.filter("bandpass",freqmin=freq_range[0],freqmax=freq_range[-1],zerophase=True)
        tr2.detrend("linear")
        tr2.detrend("constant")
        tr2.filter("bandpass",freqmin=freq_range[0],freqmax=freq_range[-1],zerophase=True)
        dist,_,_ = gps2dist_azimuth(args.sta_dict[args.sta1]["coords"][0],
                                    args.sta_dict[args.sta1]["coords"][1],
                                    args.sta_dict[args.sta2]["coords"][0],
                                    args.sta_dict[args.sta2]["coords"][1])
        sta_dist = dist/1000                               # kilometer
                # noise cross-correlation with time-domain normalization
        CFcnMB = cross_correlation(tr2,tr1,args.period_band,max_lagtime,args.corr_method)
        CFcnMB = CFcnMB/np.max(CFcnMB) # normalization
        CFtime = np.linspace(-max_lagtime,max_lagtime,2*max_lagtime*fs_new+1)
        if sum(np.isnan(CFcnMB))==0:  # No NaN data in CFcn
            CFdict={'year':yr,'day':day,"NCF":CFcnMB,"CFtime":CFtime}
            CFdict['period_band'] = args.period_band
            if not os.path.exists(os.path.join(args.out_dir,f"{sta1}_{sta2}")):
                os.mkdir(os.path.join(args.out_dir,f"{sta1}_{sta2}"))
            CF_file = os.path.join(args.out_dir,f"{sta1}_{sta2}",f"{sta1}_{sta2}_{yr}_{day}.pkl")
            out_file = open(CF_file,'wb')
            pickle.dump(CFdict,out_file)
            out_file.close()
            logger.info(f"complete {yr} {day} {sta1} {sta2}")

def noisecorr_mp(args):
    """
    Do waveform cross-correlation
    """
    sta1 = args.sta1
    sta2 = args.sta2
    cmp1 = args.cmp1
    cmp2 = args.cmp2
    max_lagtime=args.max_lagtime
    fs_new = args.fs_new
    data_dir = args.data_dir
    period_band = args.period_band
    year_range = args.year_range
    day_range = args.day_range
    # Frequency band info
    
    logger.info(f"station pair = {sta1} {sta2}")
    
    tasks = []
    # loop for years
    for yr in range(year_range[0],year_range[1]+1):
        yr_str = str(yr)
        # loop for days
        for day in range(day_range[0],day_range[1]+1):
            tasks.append([sta1,sta2,cmp1,cmp2,yr,day,max_lagtime,fs_new,period_band,data_dir])
    if args.mp==0:
        for sta1,sta2,cmp1,cmp2,yr,day,max_lagtime,fs_new,period_band,data_dir in tasks:
            noisecorr(sta1,sta2,cmp1,cmp2,yr,day,max_lagtime,fs_new,period_band,data_dir)
    elif args.mp==1:
        print("# now multiprocessing...")
        cores = args.cpu_cores
        if cores == 0:
            cores = int(mp.cpu_count())
        pool = mp.Pool(processes=cores)
        rs = pool.starmap_async(noisecorr,tasks,chunksize=1)
        pool.close()
        while True:
            remaining = rs._number_left
            print(f"Finished {len(tasks)-remaining}/{len(tasks)}",end='\r')
            if(rs.ready()):
                break
            time.sleep(0.5)
    else:
        raise Exception("Wrong args.mp value, should be 1(True) or 0(False)")

if __name__ == "__main__":
    '''
    Description:
        Script for day waveform cross-correlation to check time shift problem.
        Modified from matlab script by Professor Huajian Yao, USTC.
    Usage:
        "python noisecorr_mp.py --stations station1/station2"
    '''
    
    args = read_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    try:
        os.mkdir((os.path.join(args.out_dir,"Logs")))
    except:
        pass

    #------- logger setup ---------------------------
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_file = os.path.join(args.out_dir,"Logs",f"{args.sta1}_{args.sta2}.log")
    fh = logging.FileHandler(log_file,mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s-%(filename)s-%(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    #------- main program ---------------------------
    noisecorr_mp(args)
