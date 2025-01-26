import numpy as np
import obspy
import matplotlib.pyplot as plt
import os
from scipy.sparse.linalg import lsqr
from seisloc.sac import sac_interp

class StfEgfLsqr():
    def __init__(self, sta, chn,dirTar, dirEgf):
        """
        Empirical Green Function (egf) inversion for source time function (stf) 
        using the LSQR algorithm. Input and output are sac files with picks. 
        Output are saved in the directory "Results/sta".

        Parameters
        |    sta: station name
        |    chn: channel name, e.g., '*Z'
        | dirTar: directory of the target event, sac files are named as "*sta*"
        | dirEgf: directory of the egf event, sac files are name as "*sta*"
        """
        self.sta = sta              
        rawStar,rawSegf = self.read_data(dirTar,dirEgf)    # read data
        self.rawStar = rawStar.select(channel="*"+chn[-1]) # select channel
        self.rawSegf = rawSegf.select(channel="*"+chn[-1]) # select channel
        self.delta = self.rawSegf[0].stats.delta
        self.chn = self.rawStar[0].stats.channel           # exact channel name

        self.ppStar = self.rawStar.copy()                  # copy for processing
        self.ppSegf = self.rawSegf.copy()                  # copy for processing

        if not os.path.exists(f"Results/{self.sta}"):      # output directory
            os.makedirs(f"Results/{self.sta}")

    def read_data(self,dirTar,dirEgf):
        # raw target stream and raw egf stream
        rawStar = obspy.read(os.path.join(dirTar, f"*{self.sta}.*"))
        rawSegf = obspy.read(os.path.join(dirEgf, f"*{self.sta}.*"))
        return rawStar,rawSegf

    def pp_detrend(self,mkrTar=True,mkrEgf=True):
        """
        mkrTar: (boolean), marker for the target stream
        mkrEgf: (boolean), marker for the egf stream
        mkrZeroShift: (boolean), marker for shifting the the first point to zero
        """
        if mkrTar == True:
            self.rawSegf.detrend("demean")
            self.rawSegf.detrend("linear")
            self.ppStar.detrend("demean")
            self.ppStar.detrend("linear")
        if mkrEgf == True:
            self.rawSegf.detrend("demean")
            self.rawSegf.detrend("linear")
            self.ppSegf.detrend("demean")
            self.ppSegf.detrend("linear")

    def pp_interp(self,factor):
        """
        Interpolate the data with a factor, new delta is delta/factor
        """
        self.ppStar = sac_interp(self.ppStar,factor=factor)
        self.ppSegf = sac_interp(self.ppSegf,factor=factor)
        self.delta = self.ppStar[0].stats.delta

    def pp_zero_shift(self):
        """
        Shift the data so that the first data point is 0
        """
        self.ppStar[0].data -= self.ppStar[0].data[0]
        self.ppSegf[0].data -= self.ppSegf[0].data[0]

    def pp_filter(self,bpMkr="LP", bpL=None, bpH=None,taper_percentage=0.05,
                  type='hann'):
        """
        | bpMkr: bandpass marker, 'LP': lowpass, 'HP': highpass, 'BP': bandpass
        |   bpL: low frequency of filter,  used for BP and HP
        |   bpH: high frequency of filter, used for BP and LP
        """
        self.bpMkr = bpMkr
        self.bpL = bpL
        self.bpH = bpH

        self.ppStar.taper(max_percentage=taper_percentage, type=type)
        self.ppSegf.taper(max_percentage=taper_percentage, type=type)

        if self.bpMkr == "BP":
            self.ppStar.filter("bandpass", freqmin=self.bpL, freqmax=self.bpH)
            self.ppSegf.filter("bandpass", freqmin=self.bpL, freqmax=self.bpH)
        elif self.bpMkr == "LP":
            self.ppStar.filter("lowpass", freq=self.bpH)
            self.ppSegf.filter("lowpass", freq=self.bpH)
        elif self.bpMkr == "HP":
            self.ppStar.filter("highpass", freq=self.bpL)
            self.ppSegf.filter("highpass", freq=self.bpL)

    def pp_trim(self,mkr='a',cbTtar=0, ceTtar=0.9, cd=0):
        """
        Trim target and egf traces for inversion.

        | mkr: marker for the beginning of the waveform, e.g., 'a' for P arrival
        | cbTtar: cut begin time of target waveform relative to the marker
        | ceTtar: cut end time of target waveform relative to the marker
        | cd: time difference of cut points between target and egf, positive for earlier egf.
        """
        self.mkr = mkr
        self.cbTtar = cbTtar
        self.ceTtar = ceTtar
        self.cbTegf = cbTtar + cd
        self.ceTegf = ceTtar + cd

        invTtar = self.ppStar[0].copy()
        self.btar = invTtar.stats.sac.b
        refTimeTar = invTtar.stats.starttime - self.btar
        self.mkrTtar = getattr(invTtar.stats.sac, self.mkr)  # marker time of target waveform
        tarCutb = self.mkrTtar + self.cbTtar
        tarCute = self.mkrTtar + self.ceTtar
        invTtar.trim(refTimeTar + tarCutb, refTimeTar + tarCute)

        invTegf = self.ppSegf[0].copy()
        self.begf = invTegf.stats.sac.b
        refTimeEgf = invTegf.stats.starttime - self.begf
        self.mkrTegf = getattr(invTegf.stats.sac, self.mkr)  # marker time of egf waveform
        egfCutb = self.mkrTegf + self.cbTegf
        egfCute = self.mkrTegf + self.ceTegf
        invTegf.trim(refTimeEgf + egfCutb, refTimeEgf + egfCute)

        self.invTtar = invTtar
        self.invTegf = invTegf

    def egf_G(self, Tegf, R, smooth):
        """
        Construct the matrix G for the LSQR algorithm.
        | Tegf: empirical green function obspy.trace
        | R: length of target waveform
        | smooth: smoothing factor for the second derivative of the source time function
        """
        Gls = np.zeros((R, R))
        for i in range(R):
            Gls[i:, i] = Tegf.data[:R-i]
        Gsm = np.zeros((R, R))
        for i in range(R):
            if i-1 >= 0:
                Gsm[i, i-1] = 1
            Gsm[i, i] = -2
            if i+1 < R:
                Gsm[i, i+1] = 1
        G = np.vstack((Gls, smooth*Gsm))
        return G, Gls, Gsm

    def egf_lsqr(self, Dtar, Degf, damp, smooth, iterSetN, iterN):
        R = len(Dtar)
        D = np.zeros((2*R, 1))
        D[:R, 0] = Dtar.data[:R]
        G, Gls, Gsm = self.egf_G(Degf, R, smooth)
        M = np.zeros((R, 1))
        m = np.zeros(R)
        d = D.copy()
        for i in range(iterSetN):
            M[:R, 0] += m
            M = np.where(M > 0, M, 0)
            d[:R] = D[:R] - Gls.dot(M)
            m = lsqr(G, d, damp=damp, show=False, iter_lim=iterN)[0]
        M[:R, 0] += m
        return M[:, 0]

    def bp_label(self):
        if not hasattr(self,'bpMkr'):
            return "no filter"
        if self.bpMkr == "BP":
            return f"bandpass:{self.bpL}-{self.bpH}Hz"
        elif self.bpMkr == "LP":
            return f"lower-pass:{self.bpH}Hz"
        elif self.bpMkr == "HP":
            return f"higher-pass:{self.bpL}Hz"

    def plot(self):
        """
        Make result plots.
        axs[0,0]: Raw waveform and pre-processed waveform of target
        axs[1,0]: Raw waveform and pre-processed waveform of egf
        axs[0,1]: Source Time Function
        axs[1,1]: Synthetic waveforms and original preprocessed waveform
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()
        #------------------------------------------------
        plt.sca(axs[0])
        plt.title("Target waveform")
        plt.axvspan(-10, self.cbTtar, color='lightgray', alpha=0.5)
        plt.axvspan(self.cbTtar, self.ceTtar, color='lightpink', alpha=0.5)
        plt.axvline(0, color='black')
        rawStarTimes = self.rawStar[0].times() - self.mkrTtar + self.btar
        plt.plot(rawStarTimes, self.rawStar[0].data)
        ppStarTimes = self.ppStar[0].times() - self.mkrTtar + self.btar
        plt.plot(ppStarTimes, self.ppStar[0].data, 'r--', label=self.bp_label())
        plt.legend()
        #------------------------------------------------
        plt.sca(axs[2])
        plt.title("EGF waveform")
        plt.axvspan(-10, self.cbTegf, color='lightgray', alpha=0.5)
        plt.axvspan(self.cbTegf, self.ceTegf, color='lightpink', alpha=0.5)
        plt.axvline(0, color='black')
        rawSegfTimes = self.rawSegf[0].times() - self.mkrTegf + self.begf
        plt.plot(rawSegfTimes, self.rawSegf[0].data)
        ppSegfTimes = self.ppSegf[0].times() - self.mkrTegf + self.begf
        plt.plot(ppSegfTimes, self.ppSegf[0].data, 'r--', label=self.bp_label())
        plt.legend()
        #------------------------------------------------
        plt.sca(axs[1])
        plt.title("Source Time Function")
        plt.axvspan(self.cbTtar, self.ceTtar, color='lightpink', alpha=0.5)
        plt.axvline(0, color='black')
        self.M0times = np.arange(len(self.M0)) * self.delta + self.cbTtar
        plt.plot(self.M0times, self.M0ave)
        #------------------------------------------------
        TtarTimes = np.arange(len(self.invTtar.data))*self.delta + self.cbTtar
        synTar = np.convolve(self.invTegf.data,self.Mave)
        synTimes = np.arange(len(self.DsynTar))*self.delta + self.cbTtar
        plt.sca(axs[3])
        plt.title("Convolution")
        plt.axvspan(self.cbTtar,self.ceTtar,color='lightpink',alpha=0.5)
        plt.axvline(0,color='black')
        plt.plot(TtarTimes,self.invTtar.data,'b',label='target waveform')
        plt.plot(synTimes,synTar,'r--',label='synthetic waveform')
        plt.legend()

        axs[0].set_xlim(self.cbTtar-0.5, self.ceTtar+1)
        axs[0].set_xticks([])
        axs[1].set_xticks([])
        axs[2].set_xlim(self.cbTtar-0.5, self.ceTtar+1)
        axs[1].set_xlim(self.cbTtar, self.ceTtar)
        axs[3].set_xlim(self.cbTtar, self.ceTtar)
        ymin3 = np.min(self.invTtar.data)-np.abs(np.min(self.invTtar.data))*0.1
        ymax3 = np.max(self.invTtar.data)+np.abs(np.max(self.invTtar.data))*0.1
        axs[3].set_ylim(ymin3, ymax3)

        plt.tight_layout()
        plt.savefig(f"Results/{self.sta}/{self.sta}_{self.chn}.png")

    def save_results(self,zeropadLen=1):
        """
        ouput results in sac files and padding zeros before if necessary:
        | outTtar: processed waveform that was used for inversion
        |     TM0: source time function from inversion output
        |  TM0ave: smoothed source time function from inversion output
        | TsynTar: synthetic waveform from convolution of source time function 
        and empirical green function
        """
        zerosN = np.zeros(int(zeropadLen/self.delta))

        self.outTtar = self.invTtar.copy()  # target waveform used for inversion
        self.outTtar.stats.sac.b -= zeropadLen
        self.outTtar.stats.starttime -= zeropadLen
        self.outTtar.data = np.concatenate((zerosN, self.invTtar.data))
        self.outTtar.write(f"Results/{self.sta}/Ttar.{self.chn}", format='sac')

        self.TM0 = self.outTtar.copy()      # source time function 
        self.TM0.data = np.concatenate((zerosN, self.M0))
        self.TM0.write(f"Results/{self.sta}/M0.{self.chn}", format='sac')

        self.TM0ave = self.outTtar.copy()   # smoothed source time function
        self.TM0ave.data = np.concatenate((zerosN, self.M0ave))
        self.TM0ave.write(f"Results/{self.sta}/M0ave.{self.chn}", format='sac')

        self.TsynTar = self.outTtar.copy()  # synthetic waveform
        self.DsynTar = np.convolve(self.invTegf.data,self.Mave)
        self.TsynTar.data = np.concatenate((zerosN, self.DsynTar))
        self.TsynTar.write(f"Results/{self.sta}/synTtar.{self.chn}", format='sac')

    def stf_egf_lsqr(self,MaveLen=1,damp=1,smooth=1,iterSetN=1,iterN=1):
        """
        Parameters
        |     damp: damping factor for the LSQR algorithm
        |   smooth: smoothing factor, not usre whether it is necessary
        | iterSetN: number of iteration sets for the whole inversion process
        |    iterN: number of iterations in each set
        """
        M = self.egf_lsqr(self.invTtar.data, self.invTegf.data, damp=damp, 
                          smooth=smooth, iterSetN=iterSetN, iterN=iterN)
        self.Mave = np.convolve(M, np.ones(MaveLen)/MaveLen, mode='same')
        # padding zeros to be the same length with the target waveform
        tmp = np.zeros(int(-self.cbTtar/self.delta))
        self.M0 = np.concatenate((tmp, M))
        self.M0ave = np.concatenate((tmp, self.Mave))
