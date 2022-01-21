import obspy
import numpy as np

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
