import os
import obspy
import numpy as np
import time

def SC_tr_remove_spikes(tr):
    """ 
    Remove spikes of the Sichuan continous waveform data
    """
    # more than one zero values
    ks, = np.where(tr.data==0)
    '''
    ascOrder = np.zeros(len(ks))
    desOrder = np.zeros(len(ks))
    ascDiff = ks[1:] - ks[:-1]
    aks, = np.where(ascDiff==1)
    ascOrder[aks] = 1 
    desDiff = ks[:-1] - ks[1:]
    bks, = np.where(desDiff==-1)
    desOrder[bks+1] = 1 
    allOrder = ascOrder.astype('int')|desOrder.astype('int')
    cks = np.where(allOrder==1)
    multiSpikes = ks[cks]
    # only one zero value points
    dks = np.where(allOrder==0)
    singleZeroIdxs = ks[dks]
    ascOrder = np.zeros(len(singleZeroIdxs))
    desOrder = np.zeros(len(singleZeroIdxs))
    ascDelta = tr.data[singleZeroIdxs] - tr.data[singleZeroIdxs-1]
    desDelta = tr.data[singleZeroIdxs+1] - tr.data[singleZeroIdxs]
    multiplyDelta = ascDelta * desDelta
    fks = np.where(np.abs(multiplyDelta)>100000000)
    singleSpikes = singleZeroIdxs[fks]
    print(len(cks[0])+len(fks[0]))
    # update trace
    meanValue = np.sum(tr.data)/(tr.stats.npts - len(multiSpikes)-len(singleSpikes))
    '''
    if len(ks)>0:
        meanValue = np.sum(tr.data)/(tr.stats.npts - len(ks))    
        tr.data -= meanValue
        tr.data[ks] = 0
        tr.detrend("constant")
        tr.detrend("linear")
        tr.data[ks] = 0

