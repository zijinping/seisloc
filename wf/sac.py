import obspy
from obspy import UTCDateTime
from scipy import interpolate
import numpy as np
import os
from seisloc.utils import month_day

def calc_snr(signal, noise):
    """
    Calculate the signal-to-noise ratio (SNR) of a given signal and noise.
    
    Parameters:
        signal (numpy.ndarray): Array representing the signal.
        noise (numpy.ndarray): Array representing the noise.
    
    Returns:
        float: The signal-to-noise ratio (SNR) in decibels (dB).
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def tr_snr(tr,noiseLen=0.5,sigLen=0.5,shift=0,mkr='a'):
    """
    Calculate the signal-to-noise ratio (SNR) of a given trace.
    
    Parameters:
    |    tr (obspy.Trace): The trace to calculate the SNR for.
    |    noiseLen, sigLen: noise window length and signal window length wrt. mkr
    |    shift: separation between signal and noise will be t(mkr)+shift
    |    mkr: the marker in the sac file
    
    Returns:
    |    float: The signal-to-noise ratio (SNR) in decibels (dB).
    """
    starttime = tr.stats.starttime

    delta = tr.stats.delta
    b = tr.stats.sac.b
    mkrVal = getattr(tr.stats.sac,mkr)
    signalBeginTob = mkrVal-b+shift # separation points w.r.t the b value
    idxSignal = int(signalBeginTob/delta)
    idxNoise = int((mkrVal-b-noiseLen+shift)/delta)
    signal = tr.data[idxSignal:idxSignal+int(sigLen/delta)]
    noise = tr.data[idxNoise:idxSignal]
    
    # Calculate the SNR
    snr = calc_snr(signal, noise)
    return snr

def get_trav(st,sta,marker="a"):
    """
    Get the marker time from SAC header
    if marker is a string, then direct get the marker time. The absence 
    of the corresponding marker time will lead to the return of "0, 'none'".

    If marker is a list, then will sequential process marker one by one and 
    will return upon the successful load of the travel time. 

    For example, for a dataset of events, the P arrivals are first picked with 
    marker 'a'. After that, some P arrivals are refined with marker "t3" and 
    further refined with marker 't5'. Then to get the best travel time, one can
    use:
    >>> get_trav(st,sta,marker=['t5','t3','a'])

    Return:
    marker_time, marker
    """
    status = False   # whether load travel time successfully
    st=st.select(station=sta)
    if len(st)==0:
        raise Exception("No data for station "+sta)
    if isinstance(marker,str):
        if hasattr(st[0].stats.sac,marker):
            trav=getattr(st[0].stats.sac,marker)
            status = True

    elif isinstance(marker,list):
        markers = marker
        for marker in markers:
            if hasattr(st[0].stats.sac,marker):
                trav=getattr(st[0].stats.sac,marker)
                status = True
                break
    else:
        raise Exception("marker should be either a 'str' or a 'list'")

    if status == False:
        trav=0
        marker="none"

    return trav,marker

def aligned_sac_datas(tr1,tr2,scc_dt,tb,te,marker='t0'):
    """
    Parameters:
       tr1,tr2: obspy trace
        scc_dt: time difference measured by cross-correlation, t1-t2
            tb: Waveform cut time before the S arrival
            te: Waveform cut time after the S arrival
        marker: 'a' for P arrival time, 't0' for S arrival time
    """
    tr1.detrend("linear")
    tr1.detrend("constant")
    tr2.detrend("linear")
    tr2.detrend("constant")
    
    t01 = getattr(tr1.stats.sac,marker)
    t02 = getattr(tr2.stats.sac,marker)
    b1 = tr1.stats.sac.b
    b2 = tr2.stats.sac.b
    trim_b1 = tr1.stats.starttime+(t01-b1)+scc_dt+tb
    trim_e1 = trim_b1+(te-tb)

    trim_b2 = tr2.stats.starttime+(t01-b2)+tb
    trim_e2 = trim_b2+(te-tb)

    tr1.trim(starttime = trim_b1, endtime = trim_e1)
    tr2.trim(starttime = trim_b2, endtime = trim_e2)
    
    tr1.detrend("constant")
    tr1.detrend("linear")
    tr2.detrend("constant")
    tr2.detrend("linear")

    return tr1.data,tr2.data

def sac_interp(inS,factor=10):
    """
    Interpolate waveform data using the quadratic fitting method provided by scipy.interpolate.interp1d module
    """
    data = inS[0].data
    delta = inS[0].stats.delta
    xs = np.arange(len(data))
    f = interpolate.interp1d(xs,data,kind='quadratic')
    
    deltaNew = delta/factor
    ixs = np.arange((len(data)-1)*factor+1)*1/factor
    dataInterp = f(ixs)
    outS = inS.copy()
    outS[0].data = dataInterp
    outS[0].stats.delta = deltaNew
    return outS

def sac_file_interpolation(insacPth,outsacPth,factor=10):
    """
    Interpolate waveform data using the quadratic fitting method provided by scipy.interpolate.interp1d module
    """
    insacPth = os.path.abspath(insacPth)
    st = obspy.read(insacPth)
    stInterp = sac_interp(st,factor=factor)
    stInterp[0].write(outsacPth,format="SAC")
    
def read_sac_ref_time(tr):
    """
    Read and return reference time of a sac file in obspy.UTCDateTime format.

    Parameter
    --------
    tr: Trace object of obspy
    """

    nzyear = tr.stats.sac.nzyear
    nzjday = tr.stats.sac.nzjday
    nzhour = tr.stats.sac.nzhour
    nzmin = tr.stats.sac.nzmin
    nzsec = tr.stats.sac.nzsec
    nzmsec = tr.stats.sac.nzmsec*0.001
    year,month,day = month_day(nzyear,nzjday)
    sac_ref_time = UTCDateTime(year,month,day,nzhour,nzmin,nzsec)+nzmsec
    return sac_ref_time


def get_tr_marker_idx(tr,marker='a'):
    refTime = read_sac_ref_time(tr)
    sp = tr.stats.sampling_rate
    starttime = tr.stats.starttime
    markerTime = refTime+tr.stats.sac[marker]
    idx = np.round(int((markerTime - starttime)*sp),0)
    return idx

def get_tr_data(tr,idx,tb,te):
    """
    tb,te: negative/positive for waveform before/after the marker
    
    """
    delta = tr.stats.delta
    shift1 = int(tb/delta)
    shift2 = int(te/delta)
    data = tr.data[idx+shift1:idx+shift2]
    return data
