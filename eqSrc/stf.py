import numpy as np
import obspy
import matplotlib.pyplot as plt
import os
from scipy.sparse.linalg import lsqr

class StfEgfLsqr():
    def __init__(self, sta, dirTar, dirEgf):
        """
        Empirical Green Function (egf) inversion for source time function (stf) using the LSQR algorithm.
        Input and output are sac files with picks and reference time at event origin time.
        Output are saved in the directory "Results/sta".

        Parameters
        | sta: station name
        | dirTar: directory of target waveforms, sac files are named as *sta.*Z
        | dirEgf: directory of empirical green functions, sac files are name as *sta.*Z

        | MaveLen: length of moving average window for result source time function.
        """
        self.sta = sta                                    
        self.load_stream(dirTar,dirEgf)
        if not os.path.exists(f"Results/{self.sta}"):
            os.makedirs(f"Results/{self.sta}")

    def load_stream(self,dirTar,dirEgf):
        # raw target stream and raw egf stream
        self.rawStar = obspy.read(os.path.join(dirTar, f"{self.sta}.*HZ"))
        self.rawSegf = obspy.read(os.path.join(dirEgf, f"SC.{self.sta}.*HZ"))
        self.rawStar.detrend("demean")
        self.rawStar.detrend("linear")
        self.rawSegf.detrend("demean")
        self.rawSegf.detrend("linear")
        self.delta = self.rawSegf[0].stats.delta

    def pp_filter(self,bpMkr="LP", bpL=None, bpH=None,taper_max_percentage=0.05, type='hann'):
        """
        | bpMkr: bandpass filter marker, 'LP' for lowpass, 'HP' for highpass, 'BP' for bandpass
        | bpL: low frequency for filter, used for bandpass(BP) and highpass(HP)
        | bpH: high frequency of filter, used for bandpass(BP) and lowpass(LP)
        """
        self.bpMkr = bpMkr
        self.bpL = bpL
        self.bpH = bpH

        self.ppStar = self.rawStar.copy()    # preprocessed target stream
        self.ppStar.detrend("demean")
        self.ppStar.detrend("linear")
        self.ppStar.taper(max_percentage=0.05, type=type)
        self.ppSegf = self.rawSegf.copy()    # preprocessed egf stream
        self.ppSegf.detrend("demean")
        self.ppSegf.detrend("linear")
        self.ppSegf.taper(max_percentage=0.05, type=type)

        self.ppSegf.taper(max_percentage=0.05, type=type)
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

        Ttar = self.ppStar[0].copy()
        self.btar = Ttar.stats.sac.b
        refTimeTar = Ttar.stats.starttime - self.btar
        self.mkrTtar = getattr(Ttar.stats.sac, self.mkr)  # marker time of target waveform
        tarCutb = self.mkrTtar + self.cbTtar
        tarCute = self.mkrTtar + self.ceTtar
        Ttar.trim(refTimeTar + tarCutb, refTimeTar + tarCute)

        Tegf = self.ppSegf[0].copy()
        self.begf = Tegf.stats.sac.b
        refTimeEgf = Tegf.stats.starttime - self.begf
        self.mkrTegf = getattr(Tegf.stats.sac, self.mkr)  # marker time of egf waveform
        egfCutb = self.mkrTegf + self.cbTegf
        egfCute = self.mkrTegf + self.ceTegf
        Tegf.trim(refTimeEgf + egfCutb, refTimeEgf + egfCute)

        self.Ttar = Ttar
        self.Tegf = Tegf

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
            m, _, _, _, _, _, _, _, _, _ = lsqr(G, d, damp=damp, show=False, iter_lim=iterN)
        return M[:, 0]

    def _bp_label(self):
        if self.bpMkr == "BP":
            return f"bandpass:{self.bpL}-{self.bpH}Hz"
        elif self.bpMkr == "LP":
            return f"lower-pass:{self.bpH}Hz"
        elif self.bpMkr == "HP":
            return f"higher-pass:{self.bpL}Hz"

    def plot(self):
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
        plt.plot(rawStarTimes, self.ppStar[0].data, 'r--', label=self._bp_label())
        plt.legend()
        #------------------------------------------------
        plt.sca(axs[2])
        plt.title("EGF waveform")
        plt.axvspan(-10, self.cbTegf, color='lightgray', alpha=0.5)
        plt.axvspan(self.cbTegf, self.ceTegf, color='lightpink', alpha=0.5)
        plt.axvline(0, color='black')
        rawSegfTimes = self.rawSegf[0].times() - self.mkrTegf + self.begf
        plt.plot(rawSegfTimes, self.rawSegf[0].data)
        plt.plot(rawSegfTimes, self.ppSegf[0].data, 'r--', label=self._bp_label())
        plt.legend()
        #------------------------------------------------
        plt.sca(axs[1])
        plt.title("Source Time Function")
        plt.axvspan(self.cbTtar, self.ceTtar, color='lightpink', alpha=0.5)
        plt.axvline(0, color='black')
        self.M0times = np.arange(len(self.M0)) * self.delta + self.cbTtar
        plt.plot(self.M0times, self.M0ave)
        #------------------------------------------------
        TtarTimes = np.arange(len(self.Ttar.data))*self.delta + self.cbTtar
        synTar = np.convolve(self.Tegf.data,self.Mave)
        synTimes = np.arange(len(self.synTar))*self.delta + self.cbTtar
        plt.sca(axs[3])
        plt.title("Convolution")
        plt.axvspan(self.cbTtar,self.ceTtar,color='lightpink',alpha=0.5)
        plt.axvline(0,color='black')
        plt.plot(TtarTimes,self.Ttar.data,'b')
        plt.plot(synTimes,synTar,'r--')

        axs[0].set_xlim(self.cbTtar-0.5, self.ceTtar+1)
        axs[0].set_xticks([])
        axs[1].set_xticks([])
        axs[2].set_xlim(self.cbTtar-0.5, self.ceTtar+1)
        axs[1].set_xlim(self.cbTtar, self.ceTtar)
        axs[3].set_xlim(self.cbTtar, self.ceTtar)
        axs[3].set_ylim(np.min(self.Ttar.data)*1.1, np.max(self.Ttar.data)*1.1)

        plt.tight_layout()
        plt.savefig(f"Results/{self.sta}/{self.sta}.png")

    def save_results(self,zeropadLen=1):
        """
        ouput result sac files:
        | Ttar: processed waveform that was used for inversion
        | TM0: source time function from inversion output
        | TM0ave: smoothed source time function from inversion output
        | TsynTar: synthetic waveform from convolution of source time function and empirical green function
        Provide additional padding zeros to before the output results.
        """
        zerosN = np.zeros(int(zeropadLen/self.delta))

        self.outTtar = self.Ttar.copy()
        self.outTtar.stats.sac.b -= zeropadLen
        self.outTtar.stats.starttime -= zeropadLen
        self.outTtar.data = np.concatenate((zerosN, self.Ttar.data))
        self.outTtar.write(f"Results/{self.sta}/Ttar.sac", format='sac')

        self.TM0 = self.outTtar.copy()
        self.TM0.data = np.concatenate((zerosN, self.M0))
        self.TM0.write(f"Results/{self.sta}/M0.sac", format='sac')

        self.TM0ave = self.outTtar.copy()
        self.TM0ave.data = np.concatenate((zerosN, self.M0ave))
        self.TM0ave.write(f"Results/{self.sta}/M0ave.sac", format='sac')

        self.TsynTar = self.outTtar.copy()
        self.synTar = np.convolve(self.Tegf.data,self.Mave)
        self.TsynTar.data = np.concatenate((zerosN, self.synTar))
        self.TsynTar.write(f"Results/{self.sta}/synTtar.sac", format='sac')

    def stf_egf_lsqr(self,MaveLen=1,damp=1,smooth=1,iterSetN=1,iterN=1):
        """
        damp: damping factor for the LSQR algorithm
        smooth: smoothing factor, not usre whether it is necessary
        iterSetN: number of iteration sets for the whole inversion process
        iterN: number of iterations in each set
        """
        M = self.egf_lsqr(self.Ttar.data, self.Tegf.data, damp=damp, smooth=smooth, iterSetN=iterSetN, iterN=iterN)
        self.Mave = np.convolve(M, np.ones(MaveLen)/MaveLen, mode='same')
        # padding zeros to be the same length with the target waveform
        M0 = np.zeros(int(-self.cbTtar/self.delta))
        self.M0 = np.concatenate((M0, M))
        self.M0ave = np.concatenate((M0, self.Mave))
