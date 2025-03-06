import os
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from math import floor,ceil
from obspy import UTCDateTime
from seisloc.geometry import loc_by_width
from seisloc.statistics import sum_count_Mo,write_sum_count_Mo
from seisloc.loc.phase_convert import cata2fdsn
from seisloc.loc.utils import _load_cata
from seisloc.loc.utils import _plot_eqs
from matplotlib.collections import LineCollection
from seisloc.geometry import signed_projections_sphere,dist_az_sphere

class Catalog():
    def __init__(self,locFile="hypoDD.reloc",format="hypoDD",verbose=1):
        """
        The programme will read in catalog in format ("hypoDD", "sum", and "cata"). If no hypoDD
        file provided (locFile=None), it will generate an empty catalog. 
        A user can set up a new catalog by providing a dict in the form:
            dict[evid] = [lon,lat,dep,mag,UTCDateTime]
        example:
        >>> cata = Catalog(locFile=None)
        >>> cata.dict = cataDict  # cata_dict is a dictionary follows above format
        >>> cata.init()            # initiation of the class
        >>> print(cata)            # basic information will be printed
        """
        self.verbose = verbose
        if locFile != None:
            if not os.path.exists(locFile):
                raise Exception(f"{locFile} not existed!")
            self.dict = _load_cata(locFile,format=format)
            self._verbose_print("[Class Catalog] successfully loaded the catalog file: "+locFile)
            self.init()
        else:
            self._verbose_print("[Class Catalog] ~~~~~~~~~~~~~~~ Warning ~~~~~~~~~~~~~~~~~")
            self._verbose_print("[Class Catalog] No hypoDD .reloc file provided, an empty Catalog created.")
            self._verbose_print("[Class Catalog] You can define self.dict[evid] = [lon,lat,dep,mag,UTCDateTime]}")
            self._verbose_print("[Class Catalog] Then run: .init() to initiate the catalog.")
            self.dict = {}

    def _verbose_print(self,message:str):
        if self.verbose: print(message)

    def init(self):
        """
        Initiate the catalog
        """
        self.baseTime = UTCDateTime(2000,1,1)
        data = []
        for evid in self.dict.keys():
            evlo = self.dict[evid][0]
            evla = self.dict[evid][1]
            evdp = self.dict[evid][2]
            mag = self.dict[evid][3]
            etime = self.dict[evid][4]
            esec = etime - self.baseTime   # relative seconds
            data.append([evid,evlo,evla,evdp,mag,esec])
        self.data = np.array(data)
        self.evids = self.data[:,0].astype(int)
        self.data = self.data[self.data[:,-1].argsort()]
        self.yxratio = 1/np.cos(np.median(self.data[:,2])*np.pi/180)

        self._verbose_print("[Class Catalog] Initiation completed! ")

    def update(self, idxs: np.ndarray):
        self.data = self.data[idxs]
        self.evids = self.data[:, 0]  # first column is evid
        self.dict = {evid: self.dict[evid] for evid in self.evids}
        
    def crop(self, lomin: float, lomax: float, lamin: float, lamax: float):
        idxs = np.where((self.data[:, 1] >= lomin) & (self.data[:, 1] <= lomax) &
                        (self.data[:, 2] >= lamin) & (self.data[:, 2] <= lamax))
        self.update(idxs)

    def magsel(self, magMin: float, magMax: float = 10):
        idxs = np.where((self.data[:, 4] >= magMin) & (self.data[:, 4] <= magMax))
        self.update(idxs)

    def trim(self, starttime: UTCDateTime, endtime: UTCDateTime):
        minSecs = starttime - self.baseTime
        maxSecs = endtime - self.baseTime
        idxs = np.where((self.data[:, 5] >= minSecs) & (self.data[:, 5] <= maxSecs))
        self.update(idxs)

    def general_plot(self, secAlonlat: list = [], secBlonlat: list = [], secNormalRange: list = [],
                     edgeColor: str = 'grey', edgeWidth: float = 0.5,
                     eqSizeMagShift: float = 2, eqSizeRatio: float = 1,
                     refTime: UTCDateTime = None, impMag: float = 3,
                     depLim: list = [10, 0], cmap=None, vmin=None, vmax=None):
        """
        Make a general 2x2 plot of the catalog 
        axs[0,0]: mapview
        axs[1,0]: section view along longitude or along the section trace (addSection==True)
        axs[0,1]: section view along latitude or tranverse the section trace (addSection==True)
        axs[1,1]: MT plot of the catalog or the cross-section events (addSection==True)

        Map view plot of earthquakes,earthquake denoted default by black circle
        Parameters:
        |    secAlonlat: the [lon,lat] of the section start point 'a'
        |    secBlonlat: the [lon,lat] of the section end point 'b'
        |secNormalRange: list, range of the section normal distance, right-hand is positive
        |     edgeColor: earthquake marker(circle) edgeColor
        |     edgeWidth: edgeWidth of the earthquake marker
        |  eqSizeMagShift: magnitude shift for plotting negative magnitude events 
        |   eqSizeRatio: size ratio of the earthquake marker
        |        impMag: important magnitude. Magnitude larger than this level will be 
        |                highlighted
        |       refTime: reference time in UTCDateTime used to constrain colormap, if
        |                no colormap provided, seismicity will be plotted wrt. the first event
        |          cmap: colormap, check 'matplotlib' for more detail.
        |     vmin,vmax: the minimum and maximum value for colormap
        |       alonlat: the [lon,lat] of the section start point 'a'
        |       blonlat: the [lon,lat] of the section end point 'b'
        """
        eqsParams= {"edgeColors":[edgeColor,edgeColor],
                    "edgeWidth":edgeWidth,
                    "eqSizeMagShift":eqSizeMagShift,
                    "eqSizeRatio":eqSizeRatio,
                    "refTime":refTime,
                    "impMag":impMag,
                    "cmap":cmap,
                    "vmin":vmin,
                    "vmax":vmax,
                    "mode":"normal"}

        fig, axs = plt.subplots(2,2,figsize=(10,8))

        if len(secAlonlat) == 2 and len(secBlonlat) == 2 and \
                len(secNormalRange) == 2: # projection processing
            
            distKm,az = dist_az_sphere(secAlonlat[1],secAlonlat[0],secBlonlat[1],secBlonlat[0])
            pxs,pys = signed_projections_sphere(secAlonlat[1],secAlonlat[0],secBlonlat[1],secBlonlat[0],
                                      self.data[:,2],self.data[:,1])

            ks = np.where((pxs>=0)&(pxs<=distKm)&(pys>=secNormalRange[0])&(pys<=secNormalRange[1]))
            dataSel = self.data[ks]

            # shift the section trace to the left and right for the boundary
            a1la,a1lo,b1la,b1lo = loc_by_width(secAlonlat[1],secAlonlat[0],
                                               secBlonlat[1],secBlonlat[0],
                                               width=np.abs(secNormalRange[0]),
                                               direction="right")
            a2la,a2lo,b2la,b2lo = loc_by_width(secAlonlat[1],secAlonlat[0],
                                               secBlonlat[1],secBlonlat[0],
                                               width=np.abs(secNormalRange[1]),
                                               direction="left")
            addSection = True
        elif len(secAlonlat) == 0 and len(secBlonlat) == 0 and len(secNormalRange) == 0:
            addSection = False
        else:
            raise Exception("The length of secAlonlat, secBlonlat and secWidths should be 2 or 0")

        if refTime == None:  # reference time for calculation of relative time
            refTime = self.baseTime + np.min(self.data[:,5])
            deltaSec = refTime - self.baseTime

        #========= axs [0,0] =============
        plt.sca(axs[0,0])
        if addSection:
            # plot the section trace
            plt.plot([secAlonlat[0],secBlonlat[0]],[secAlonlat[1],secBlonlat[1]],c='darkred')
            plt.plot([a1lo,b1lo,b2lo,a2lo,a1lo],
                     [a1la,b1la,b2la,a2la,a1la],
                    linestyle='--',
                    c='darkred') # This is the trace of the cross-section
        #----- plot earthquakes ----------
        xyMagReldays = self._prep_eqs_plot(self.data[:,[1,2,4,5]],deltaSec)
        _plot_eqs(xyMagReldays,eqsParams)
        #========= axs [0,1] along latitude or perpendicular to section =============
        plt.sca(axs[0,1])
        if addSection:
            xyMagReldays = self._prep_eqs_plot(dataSel[:,2:6],deltaSec)
            xyMagReldays[:,0] = -pys[ks]         # update with pys, negative: left--right
            plt.xlabel("Transverse-section distance(km)")
        else: # then plot along longitude sections
            xyMagReldays = self._prep_eqs_plot(self.data[:,2:6],deltaSec)
            plt.xlabel("Latitude")
        _plot_eqs(xyMagReldays,eqsParams)
        plt.ylim(depLim)
        plt.ylabel("Depth (km)")
        plt.legend()

        #========= axs [1,0] ===========================
        # along longitude or along section =============
        plt.sca(axs[1,0])
        if addSection:
            xyMagReldays = self._prep_eqs_plot(dataSel[:,[1,3,4,5]],deltaSec) # deltaSec: correction between refTime and baseTime
            xyMagReldays[:,0] = pxs[ks]          # replace with pxs
            plt.xlabel("Parallel-section distance (km)")
        else: # then plot along longitude sections
            xyMagReldays = self._prep_eqs_plot(self.data[:,[1,3,4,5]],deltaSec)
            plt.xlabel("Longitude")
        _plot_eqs(xyMagReldays,eqsParams)
        plt.ylim(depLim)
        plt.ylabel("Depth (km)")
        plt.legend()

        #========= axs [1,1] MT plot =============
        plt.sca(axs[1,1])
        tbtime = self.baseTime + np.min(self.data[:,5]) # datetime begin
        tetime = self.baseTime + np.max(self.data[:,5]) # datetime end
        #----------------- processing ------------
        if addSection:
            xyMagReldays = self._prep_eqs_plot(dataSel[:,[5,4,4,5]],deltaSec)
        else:
            xyMagReldays = self._prep_eqs_plot(self.data[:,[5,4,4,5]],deltaSec)
        xyMagReldays[:,0] = xyMagReldays[:,3] # x is relative days
        #------ vertical lines for MT --------
        relDays = xyMagReldays[:,-1]
        mags = xyMagReldays[:,2]
        
        xs = np.array([relDays,relDays]).T
        minMag = np.min(xyMagReldays[:,2])
        maxMag = np.max(xyMagReldays[:,2])
        ylim = [minMag-0.1,maxMag+0.5]
        ys = np.array([np.ones_like(relDays)*ylim[0],mags]).T
        lines = np.stack((xs,ys),axis=2).reshape(-1,2,2)
        lc = LineCollection(lines,colors='grey',linewidths=0.5,zorder=-10)
        plt.gca().add_collection(lc)
        #------ plot earthquakes -------
        _plot_eqs(xyMagReldays,eqsParams)
        plt.xlabel("Time (days)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.text(0.02,0.95,f"refTime: "+str(refTime)[:19],transform=axs[1,1].transAxes)
        plt.ylim(ylim)

        plt.tight_layout()

    def TD_plot(self,refEvid=None,
                  refLoc = [],refTime=None,
                  xlim=[],xunit="day",
                  depLim=[100,-4],
                  distLim=[],
                  impMag=5,
                  figsize=(8,6),
                  diffCFs=[]):
        """
        Distance vs. time plot using a given event id or a given location(evlo,evla,evdp)

        refEvid: reference event id
         refLoc: [lon,lat,dep], reference site longitude, latitude and depth 
        refTime: reference time in UTCDateTime with which the relative time of events will be calculated
           xlim: range for time
          xunit: unit for time
         depLim: depth range
        distLim: distance range
         impMag: important magnitude
        figsize: figure size
        diffCFs: diffusion coefficients
        """
        if refEvid == None and refLoc == []:
            self._verbose_print("[Catalog.TD_plot] At least one of refEvid or refLoc should be provided.")
            return
        if refEvid != None and refLoc != []:
            self._verbose_print("[Catalog.TD_plot] Both refEvid and refLoc are provided. refLoc will be used.")
        #------------- initiation --------------------
        fig,axs = plt.subplots(2,1,figsize=figsize)
        if xunit == "day":
            denominator = (24*60*60)
            plt.xlabel("Time (day)")
        elif xunit == "hour":
            denominator = (60*60)
            plt.xlabel("Time (hour)")
        elif xunit == "second":
            denominator = 1
            plt.xlabel("Time (second)")
        if len(xlim)>0:
            axs[0].set_xlim(xlim)
            axs[1].set_xlim(xlim)
        axs[0].set_ylabel("Depth (km)")
        axs[1].set_ylabel("Distance (km)")
        axs[0].set_ylim(depLim)
        if len(distLim)>0:
            axs[1].set_ylim(distLim)
        axs[0].grid(axis="y")
        axs[1].grid(axis="y")
        if refEvid != None:
            refLo = self.dict[refEvid][0]
            refLa = self.dict[refEvid][1]
            refDp = self.dict[refEvid][2]
        if len(refLoc)>0:
            refLo,refLa,refDp = refLoc
        if refTime == None:
            refTime = self.baseTime + np.min(self.data[:,5])
            deltaSec = refTime - self.baseTime
        relVals = (self.data[:,5] - deltaSec)/denominator
        dlos = self.data[:,1]-refLo
        dlas = self.data[:,2]-refLa
        ddps = self.data[:,3]-refDp
        dxs = dlos*111.19/self.yxratio
        dys = dlas*111.19
        dists = np.sqrt(dxs**2+dys**2+ddps**2)
        mags = self.data[:,4]

        #----- diffusion coefficient lines ---------------
        if len(diffCFs)>0:
            for diffCF in diffCFs:
                if xunit == "day":
                    x = np.linspace(0,np.max(relVals),int(np.max(relVals)*20)+1)
                    y = np.sqrt(4*np.pi*diffCF*x*24*60*60)/1000
                elif xunit == "hour":
                    x = np.linspace(0,np.max(relVals),int(np.max(relVals)*20)+1)
                    y = np.sqrt(4*np.pi*diffCF*x*60*60)/1000
                axs[1].plot(x,y,label=f"Diffusion Coefficient: {diffCF} m^2/s")
                axs[1].legend()
        #----- different symbols by magnitudes ---------------
        idxs = np.where(mags<impMag)
        axs[0].scatter(relVals[idxs],self.data[idxs,3],marker='o',c='k') # depth
        axs[1].scatter(relVals[idxs],dists[idxs],marker='o',c='k')
        idxs = np.where(mags>=impMag)
        axs[0].scatter(relVals[idxs],self.data[idxs,3],marker='o',c='r') # depth
        lg = axs[1].scatter(relVals[idxs],dists[idxs],marker='o',c='r',label=f"M$\geq${impMag}")
        plt.legend([lg],[f"M$\geq${impMag}"])
        plt.tight_layout()
        
    def depth_hist(self, magThred=-9, dpMin=0, dpMax=10, binWidth=0.5, ax=None, color='grey', edgecolor='k'):
        bins = np.arange(dpMin, dpMax, binWidth)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("Depth (km)", fontsize=14)
        ax.set_xlabel("Event Qty", fontsize=14)
        kk = np.where(self.data[:, 4] >= magThred)
        hist, bins = np.histogram(self.data[:, 4], bins=bins)
        ax.barh(bins[:-1] + binWidth / 2, hist, height=binWidth, color=color, edgecolor=edgecolor)
        ax.set_ylim([dpMax, dpMin])

        return ax
        
    def day_hist(self,refTime=UTCDateTime(2019,1,1,0,0,0),
                 xlim=[],ylim=[],
                 color='b',
                 edgeColor='k',
                 plotMonths=True,
                 figSize=(8,4)):
        """
        Plot events by day-quantity in a histogram plot.
        Parameters:
            refTime: Reference time for plot
        """
        ref_list = []
        deltaSec = refTime - self.baseTime
        relSecs = self.data[:,5] - deltaSec
        relDays = relSecs/(24*60*60)
        minDay = floor(min(relDays))
        maxDay = ceil(max(relDays))
        bins = np.linspace(minDay,maxDay,maxDay-minDay+1)
        fig1 = plt.figure(1,figsize=figSize)
        ax1 = plt.subplot(1,1,1)
        ax1.hist(relDays,bins,color=color,edgecolor=edgeColor)
        # The bottom x-axis is in days
        ax1.set_xlim([0,maxDay])
        # The top x-axis marks year and month in YYYYMM
        tickLst1 = [] # Store the position number
        tickLst2 = [] # Store the tick text
        refYr = refTime.year
        refMo = refTime.month
        refDy = refTime.day
        if refDy == 1:
            tickLst1.append(0)
            tickLst2.append(str(refYr)+str(refMo).zfill(2))
        status = True # Start to loop month by month
        loopTime = UTCDateTime(refYr,refMo,1) # Initiate loop time
        step = 32 #32 > 31. Make sure each step pass to next month
        while status==True:
            loopTime = loopTime + step*24*60*60
            tmpYr = loopTime.year
            tmpMo = loopTime.month
            loopTime = UTCDateTime(tmpYr,tmpMo,1)
            deltaDays = (loopTime - refTime)/(24*60*60)
            if deltaDays > (maxDay):
                status=False
            else:
                tickLst1.append(deltaDays)
                tickLst2.append((str(tmpMo).zfill(2)))
        if plotMonths:
            ax2 = ax1.twiny()
            ax2.set_xlim([0,maxDay])
            ax2.plot(0,0,'k.')
            plt.xticks(tickLst1,tickLst2)
            ax2.set_xlabel("date")
        if xlim!=[]:
            ax1.set_xlim(xlim)
            if plotMonths:
                ax2.set_xlim(xlim)
        if ylim!=[]:
            plt.ylim(ylim)
        ax1.set_xlabel("Time (days)")
        ax1.set_ylabel("event quantity")

    def animation(self,increDay=1,
                  timeB=None,timeE=None,
                  xlim=[],ylim=[],
                  mkrSizeMagShift=2,mkrSizeRatio=1):
        """
        Generate gif animation file
        increment: Time increased for each plot. Unit: hour
        """
        os.makedirs("animation")
        if xlim == []:
            xlim = [np.min(self.data[:,1]),np.max(self.data[:,1])]
        if ylim == []:
            ylim = [np.min(self.data[:,2]),np.max(self.data[:,2])]
        
        if timeB == None:
            timeB = self.baseTime+np.min(self.data[:,5])
        if timeE == None:
            timeE = self.baseTime+np.max(self.data[:,5])
        print("[Class Catalog] Movie start time is: ",timeB)
        print("[Class Catalog]   Movie end time is: ",timeE)

        dayB = (timeB - self.baseTime)/(24*60*60)
        dayE = (timeE - self.baseTime)/(24*60*60)
        loopDay = dayB
        figId = 1
        while loopDay <= dayE:
            fig = plt.figure(1,figsize=(8,8))
            ax1 = fig.add_subplot(111)
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax1.set_aspect(self.yxratio)
            ax1.set_xlabel("Lon(degree)",fontsize=18)
            ax1.set_ylabel("Lat(degree)",fontsize=18)
            ax1.set_title(f"{str(self.baseTime+loopDay)[:19]}",fontsize=16)
            #---------- plot Earthquakes ------------------
            #---------- (1) plot previous earthquakes -----
            relSecs = self.data[:,5]
            ks = np.where(relSecs<loopDay*24*60*60)
            ax1.scatter(self.data[ks,1],
                                self.data[ks,2],
                                s=(self.data[ks,4]+mkrSizeMagShift)*mkrSizeRatio,
                                c='grey',
                                label="Previous Events")
            #---------- (2) plot current earthquakes -----
            ks = np.where((relSecs>=loopDay*24*60*60)&(relSecs<(loopDay+increDay)*24*60*60))
            ax1.scatter(self.data[ks,1],
                                self.data[ks,2],
                                s=(self.data[ks,4]+mkrSizeMagShift)*mkrSizeRatio,
                                c='red',
                                label="Current Events")
            ##------------- save results --------------------------------------------
            plt.savefig(f"animation/{str(figId).zfill(3)}.png")
            loopDay += increDay
            figId+=1
            plt.close()
        #-------------------- gif -----------------------------
        imgs = []
        for i in range(1,figId):
            figPth = f'animation/{str(i).zfill(3)}.png'
            tmp = Image.open(figPth)
            imgs.append(tmp)
        imgs[0].save("animation.gif",save_all=True,append_images=imgs,duration=10)
    
    def intervals_plot(self,interval=1,method='month',
                       xlim=[],ylim=[],
                       columns=4,subplotsize=(3,3),
                       marker='o',ms=1,
                       wspace=None,hspace=None):

        from seisloc.plot import intervals_plot
        plt.close()
        axs = intervals_plot(xys=self.data[:,1:3],
                rela_secs=self.data[:,5],
                reftime=self.baseTime,
                interval=interval,method=method,
                xlim=xlim,ylim=ylim,
                columns=columns,subplotsize=subplotsize,
                marker=marker,ms=ms,
                wspace=wspace,hspace=hspace)
        plt.tight_layout()
        return axs

    def depths_plot(self,
                deplim=[0,10],interval=1,
                xlim=[],ylim=[],
                columns=4,subplotsize=(3,3),
                marker='o',ms=1,color='k',
                zorder=0,
                wspace=None,hspace=None):
        from seisloc.plot import depths_plot
        axs = depths_plot(xyz=self.data[:,1:4],
                          deplim=deplim,interval=interval,
                          xlim=xlim,ylim=ylim,
                          columns=columns,subplotsize=subplotsize,
                          marker=marker,ms=ms,color=color,
                          zorder=zorder,
                          wspace=None,hspace=None)
        plt.tight_layout()
        return axs

    def sum_count_Mo(self,starttime,endtime,outFile='sum_count_Mo.txt',mode='day'):
        self.dict_count_Mo = sum_count_Mo(self,starttime,endtime)
        write_sum_count_Mo(self.dict_count_Mo,outFile,mode)
    
    def write_txt_cata(self,fileName=None,refTime=None,disp=False):
        from seisloc.loc.utils import write_txt_cata

        if fileName==None:
            nowTime = UTCDateTime.now()
            fileName = "Catalog_"+nowTime.strftime("%Y%m%d%H%M%S")+".txt"
        if refTime == None:
            refTime = self.baseTime
        self._verbose_print("[Class Catalog] The reference time is: {refTime}")
        self._verbose_print("[Class Catalog] Catalog information write into: {fileName}")
        write_txt_cata(self.dict,fileName,refTime,disp)
        
    def cata2fdsn(self,author="Hardy",catalog="SC",
                  cont="SC",contID="01",magtype="ML",
                  magauthor="SC Agency",elocname="SC",out_file='cata.fdsn'):
        cata2fdsn(self,author=author,catalog=catalog,
                  cont=cont,contID=contID,magtype=magtype,
                  magauthor=magauthor,elocname=elocname,out_file=out_file)

    def copy(self):
        return copy.deepcopy(self)
    
    def merge(self,dd2):
        for evid in dd2.keys:
            if evid in self.keys:
                raise Exception(f"Key error {evid}. Please avoid using the same key value")
            self.dict[evid]=dd2.dict[evid]
        self.init()

    def __repr__(self):
        _qty = f"HypoDD relocation catalog with {len(self.dict.keys())} events\n"
        _time= f"     Time range is: {self.baseTime+np.min(self.data[:,5])} to {self.baseTime+np.max(self.data[:,5])}\n"
        _mag = f" Magnitue range is: {format(np.min(self.data[:,4]),'4.1f')} to {format(np.max(self.data[:,4]),'4.1f')}\n"
        _lon = f"Longitude range is: {format(np.min(self.data[:,1]),'8.3f')} to {format(np.max(self.data[:,1]),'8.3f')}\n"
        _lat = f" Latitude range is: {format(np.min(self.data[:,2]),'7.3f')} to {format(np.max(self.data[:,2]),'7.3f')}\n"
        _dep = f"    Depth range is: {format(np.min(self.data[:,3]),'4.1f')} to {format(np.max(self.data[:,3]),'4.1f')}\n"
        return _qty+_time+_mag+_lon+_lat+_dep

    def __getitem__(self,evid):
        return self.dict[evid]

    def _prep_eqs_plot(self,data,deltaSec):
        """
        Prepare data for plotting
        data columns: x, y , mag, relSecs
        """
        xyMagReldays = np.zeros((data.shape[0],4))
        xyMagReldays[:,0] = data[:,0]  # evlo
        xyMagReldays[:,1] = data[:,1]  # evla
        xyMagReldays[:,2] = data[:,2]  # mag
        xyMagReldays[:,3] = (data[:,3]-deltaSec)/(24*60*60)
        return xyMagReldays