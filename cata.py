import os
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from seisloc.dd import loadDD
from seisloc.geometry import in_rectangle,loc_by_width
from math import floor,ceil


class Catalog():
    def __init__(self,locfile="hypoDD.reloc"):
        """
        The programme will read in hypoDD relocation file by default. If no hypoDD
        file provided, it will generate an empty catalog. A user can set up a new 
        catalog by providing a dict in the form:
            dict[evid] = [lon,lat,dep,mag,UTCDateTime]
        example:
        >>> cata = Catalog(locfile=None)
        >>> cata.dict = cata_dict  #cata_dict is a dictionary follows above format
        >>> cata.init()
        >>> print(cata)
        """
        if locfile != None:
            if not os.path.exists(locfile):
                raise Exception(f"{locfile} not existed!")
            self.dict,_ = loadDD(locfile)
            print("successfully load catalog file: "+locfile)
            self.init()
        else:
            print("No hypoDD data provided, a empty Catalog created.")
            print("You can define self.dict[evid] = [lon,lat,dep,mag,UTCDateTime]}")
            print("Then run: .init() function")
            self.dict = {}

    def init(self):
        self.init_keys()
        self.init_locs()
        self.init_relative_seconds()

    def init_keys(self):
        """
        Turn dict keys into numpy array
        """
        self.keys = list(self.dict.keys())
        self.keys = np.array(self.keys)
        self.keys = self.keys.astype(int)

    def init_locs(self):
        """
        Generate numpy array in format lon, lat, dep, mag
        """
        self.locs = []
        for key in self.keys:
            lon = self.dict[key][0]
            lat = self.dict[key][1]
            dep = self.dict[key][2]
            mag = self.dict[key][3]
            self.locs.append([lon,lat,dep,mag])
        self.locs = np.array(self.locs)

    def init_relative_seconds(self):
        """
        Numpy array to save relative seconds to the first event
        """
        self.first_key = self.keys[0]
        self.first_time = self.dict[self.first_key][4]
        self.relative_seconds = []
        for key in self.keys:
            etime = self.dict[key][4]
            self.relative_seconds.append(etime-self.first_time)
        self.relative_seconds = np.array(self.relative_seconds)

    def update_keys(self,idxs):
        """
        Update keys array with indexs
        """
        self.keys = self.keys[idxs]

    def update_dict(self):
        """
        Update dictionary with keys
        """
        old_keys = list(self.dict.keys())
        for key in old_keys:
            if key not in self.keys:
                self.dict.pop(key)

    def update_locs(self,idxs):
        """
        Update location array with indexs
        """
        self.locs = self.locs[idxs]

    def update_relative_seconds(self,idxs):
        """
        Update relative times with indexs
        """
        self.relative_seconds = self.relative_seconds[idxs]
        
    def crop(self,lonmin,lonmax,latmin,latmax):
        """
        Trim the dataset with the boundary conditions
        """
        idxs = np.where((self.locs[:,0]>=lonmin)&(self.locs[:,0]<=lonmax)&\
                        (self.locs[:,1]>=latmin)&(self.locs[:,1]<=latmax))
        self.update_keys(idxs)
        self.update_dict()
        self.update_locs(idxs)
        self.update_relative_seconds(idxs)

    def magsel(self,mag_low,mag_top=10):
        """
        Trim the dataset with the magnitude
        """
        idxs = np.where((self.locs[:,3]>=mag_low)&(self.locs[:,3]<=mag_top))
        self.update_keys(idxs)
        self.update_dict()
        self.update_locs(idxs)
        self.update_relative_seconds(idxs)

    def trim(self,starttime,endtime):
        """
        Trim the dataset with time conditions
        """
        min_reftime = starttime - self.first_time
        max_reftime = endtime - self.first_time
        
        idxs = np.where((self.relative_seconds>=min_reftime)&\
                        (self.relative_seconds<=max_reftime))
        self.update_keys(idxs)
        self.update_dict()
        self.update_locs(idxs)
        self.update_relative_seconds(idxs)

    def sort(self,method="time"):
        idxs = self.relative_seconds.argsort()
        self.update_keys(idxs)
        self.update_dict()
        self.update_locs(idxs)
        self.update_relative_seconds(idxs)

    def hplot(self,
              xlim=[],
              ylim=[],
              figsize=None,
              edgecolor='grey',
              markersize=6,
              size_ratio=1,
              imp_mag=None,
              cmap = None,
              ref_time = UTCDateTime(2019,3,1),
              vmin=0,
              vmax=1,
              unit="day",
              add_section=False,
              alonlat=[104,29],
              blonlat=[105,30],
              section_width=0.05,
              plt_show=True,
              crop=False):
        """
        Map view plot of earthquakes
        """
        if figsize != None:
            plt.figure(figsize=figsize)

        if section_width <=0:
            raise Error("Width <= 0")
        # plot all events
        if add_section==True:
            alon = alonlat[0]; alat = alonlat[1]
            blon = blonlat[0]; blat = blonlat[1]
            print(alon,alat,blon,blat,section_width)
            results = in_rectangle(self.locs,alon,alat,blon,blat,section_width/2)
            jj = np.where(results[:,0]==1)
            if crop == True:
                self.update_keys(jj)
                self.update_relative_seconds(jj)
                self.update_locs(jj)
                self.update_dict()
                
        if cmap == None:
            plt.scatter(self.locs[:,0],
                    self.locs[:,1],
                    (self.locs[:,3]+2)*size_ratio,
                    edgecolors = edgecolor,
                    facecolors='none',
                    marker='o',
                    alpha=1)
        else:
            shift_seconds = ref_time - self.first_time
            times_plot = self.relative_seconds-shift_seconds
            if unit=="day":
                times_plot = times_plot/(24*60*60)
            elif unit=="hour":
                times_plot = times_plot/(60*60)
            elif unit=="minute":
                times_plot = times_plot/60
            plt.scatter(self.locs[:,0],
                    self.locs[:,1],
                    c=times_plot,
                    s=(self.locs[:,3]+2)*size_ratio,
                    cmap = cmap,
                    vmin = vmin,
                    vmax = vmax,
                    marker='o',
                    alpha=1)

        # plot large events
        if imp_mag != None:
            kk = np.where(self.locs[:,3]>=imp_mag)
            if len(kk)>0:                 
                imp = plt.scatter(self.locs[kk,0],
                        self.locs[kk,1],
                        (self.locs[kk,3]+2)*size_ratio*20,
                        edgecolors ='black',
                        facecolors='red',
                        marker='*',
                        alpha=1)
                plt.legend([imp],[f"M$\geq${format(imp_mag,'4.1f')}"])
        
        if add_section == True: # draw cross-section plot
            a1lon,a1lat,b1lon,b1lat = loc_by_width(alonlat[0],
                                                   alonlat[1],
                                                   blonlat[0],
                                                   blonlat[1],
                                                   width=section_width/2,
                                                   direction="right")
            a2lon,a2lat,b2lon,b2lat = loc_by_width(alonlat[0],
                                                   alonlat[1],
                                                   blonlat[0],
                                                   blonlat[1],
                                                   width=section_width/2,
                                                   direction="left")
            plt.plot([a1lon,b1lon,b2lon,a2lon,a1lon],
                     [a1lat,b1lat,b2lat,a2lat,a1lat],
                     linestyle='--',
                     c='darkred')
            plt.plot([alonlat[0],blonlat[0]],[alonlat[1],blonlat[1]],c='darkred')
        # adjust plot parameters
        if len(xlim) != 0:
            plt.xlim(xlim)
        if len(ylim) != 0: 
            plt.ylim(ylim)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.gca().set_aspect("equal")
        if plt_show == True:
            plt.show()
        
    def vplot(self,
              alonlat,
              blonlat,
              edgecolor='grey',
              width=0.1,
              depmin=0,
              depmax=10,
              size_ratio=1,
              imp_mag=None,
              cmap=None,
              ref_time = UTCDateTime(2019,3,1),
              vmin=0,
              vmax=1,
              unit="day",
              aspect="auto"):
        """
        Description
        """
        length_m,_,_ = gps2dist_azimuth(alonlat[1],alonlat[0],blonlat[1],blonlat[0])
        length_km = length_m/1000
        alon = alonlat[0]; alat = alonlat[1]
        blon = blonlat[0]; blat = blonlat[1]
        results = in_rectangle(self.locs,alon,alat,blon,blat,width/2)
        jj = np.where(results[:,0]>0)
        if cmap==None:
            plt.scatter(results[jj,1],
                    self.locs[jj,2],
                    marker='o',
                    edgecolors = edgecolor,
                    facecolors='none',
                    s=(self.locs[jj,3]+2)*size_ratio*5)
        else:
            shift_seconds = ref_time - self.first_time
            times_plot = self.relative_seconds[jj]-shift_seconds
            if unit=="day":
                times_plot = times_plot/(24*60*60)
            elif unit=="hour":
                times_plot = times_plot/(60*60)
            elif unit=="minute":
                times_plot = times_plot/60
            im = plt.scatter(results[jj,1],
                    self.locs[jj,2],
                    c=times_plot,
                    s=(self.locs[jj,3]+2)*size_ratio*5,
                    cmap = cmap,
                    vmin = vmin,
                    vmax = vmax,
                    marker='o',
                    alpha=1)
            cb = plt.colorbar(im)
            cb.set_label(unit)

        tmplocs = self.locs[jj]
        tmpresults = results[jj]
        if imp_mag != None:
            kk = np.where(tmplocs[:,3]>=imp_mag)
            if len(kk)>0:                 
                imp = plt.scatter(tmpresults[kk,1],
                        tmplocs[kk,2],
                        (tmplocs[kk,3]+2)*size_ratio*30,
                        edgecolors ='black',
                        facecolors='red',
                        marker='*',
                        alpha=1)
                plt.legend([imp],[f"M$\geq${format(imp_mag,'4.1f')}"])
        
        plt.ylim([depmax,depmin])
        plt.xlim([0,length_km])
        plt.xlabel("distance (km)")
        plt.ylabel("depth (km)")
        plt.gca().set_aspect(aspect)
        plt.show()
    
    def MT_plot(self,
                ref_time=UTCDateTime(2019,3,1),
                xlim=[],
                ylim=[0,5],
                unit="day",
                cmap=None,
                vmin=0,
                vmax=1,
                mlow=0,
                plt_show=True,
                figsize=(10,5)):
        """
        unit: 'day','hour' or 'second'
        """
        fig,ax = plt.subplots(1,figsize=figsize)
        if unit == "day":
            denominator = (24*60*60)
            plt.xlabel("Time (day)")
        elif unit == "hour":
            denominator = (60*60)
            plt.xlabel("Time (hour)")
        elif unit == "second":
            denominator = 1
            plt.xlabel("Time (second)")
        
        for key in self.keys:
            etime = self.dict[key][4]
            emag = self.dict[key][3]
            diff_seconds = etime - ref_time
            diff_x = diff_seconds/denominator
            if cmap == None:
                plt.plot([diff_x,diff_x],[mlow,emag],c='grey')
            else:
                plt.plot([diff_x,diff_x],[mlow,emag],color=cmap((diff_x-vmin)/(vmax-vmin)))
            plt.plot([diff_x],emag,'x',c='k')
        plt.ylim(ylim)
        if len(xlim)>0:
            plt.xlim(xlim)
        plt.ylabel("Magnitude")
        if plt_show:
            plt.show()

    def dep_dist_plot(self,refid=None,
                  refloc = [],
                  ref_time=UTCDateTime(2019,3,1),
                  xlim=[],
                  deplim=[100,-4],
                  distlim=[],
                  unit="day",
                  cmap=None,
                  vmin=0,
                  vmax=1,
                  figsize=(8,6)):
        fig,axs = plt.subplots(2,1,figsize=figsize)
        if unit == "day":
            denominator = (24*60*60)
            plt.xlabel("Time (day)")
        elif unit == "hour":
            denominator = (60*60)
            plt.xlabel("Time (hour)")
        elif unit == "second":
            denominator = 1
            plt.xlabel("Time (second)")
        if len(xlim)>0:
            axs[0].set_xlim(xlim)
            axs[1].set_xlim(xlim)
        axs[0].set_ylabel("Depth (km)")
        axs[1].set_ylabel("3D-dist (km)")
        axs[0].set_ylim(deplim)
        if len(distlim)>0:
            axs[1].set_ylim(distlim)
        axs[0].grid(axis="y")
        axs[1].grid(axis="y")
        if refid != None:
            reflon = self.dict[refid][0]
            reflat = self.dict[refid][1]
            refdep = self.dict[refid][2]
        if len(refloc)>0:
            reflon,reflat,refdep = refloc
        for evid in self.keys:
            etime =self.dict[evid][4]
            elon = self.dict[evid][0]
            elat = self.dict[evid][1]
            edep = self.dict[evid][2]
            emag = self.dict[evid][3]
            diff_x = (etime-ref_time)/denominator
            dist,_,_ = gps2dist_azimuth(elat,elon,reflat,reflon)
            d3dist = np.sqrt((dist/1000)**2+(edep-refdep)**2)
            if cmap==None:
                axs[0].scatter(diff_x,edep,s=(emag+2)*5,marker='o',c='k')
                axs[1].scatter(diff_x,d3dist,s=(emag+2)*5,marker='o',c='k')
            else:
                axs[0].scatter(diff_x,edep,s=(emag+2)*5,marker='o',color=cmap((diff_x-vmin)/(vmax-vmin)))
                axs[1].scatter(diff_x,d3dist,s=(emag+2)*5,marker='o',color=cmap((diff_x-vmin)/(vmax-vmin)))

        plt.tight_layout()
        plt.show()
        
    def depth_hist(self,mag_threshold=-9,depthmin=0,depthmax=10,gap=0.5):
        bins=np.arange(depthmin,depthmax,gap)
        fig,ax = plt.subplots(1,1,figsize=(6,8))
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("Depth (km)",fontsize=16)
        ax.set_xlabel("Event Qty",fontsize=16)
        kk = np.where(self.locs[:,3]>=mag_threshold)
        hist,bins = np.histogram(self.locs[:,2],bins=bins)
        ax.barh(bins[:-1]+gap/2,hist,height=gap,color='gray',edgecolor='k')
        ax.set_ylim([depthmax,depthmin])
        plt.show()
        
    def day_hist(self,ref_time=UTCDateTime(2019,1,1,0,0,0),xlim=[],ylim=[],color='b',edgecolor='k',plot_months=True,plt_show=True,figsize=None):
        """
        Plot events by day-quantity in a histogram plot.
        Parameters:
            -ref_time: Reference time for plot
        """
        ref_list = []
        time_list = []
        for key in self.dict.keys():
            _,_,_,_,etime = self.dict[key]
            ref_list.append((etime-ref_time)/(24*60*60))

        min_day=floor(min(ref_list))
        max_day=ceil(max(ref_list))
        bins = np.linspace(min_day,max_day,max_day-min_day+1)
        if figsize==None:
            figsize=(8,4)
        fig1 = plt.figure(1,figsize=figsize)
        ax1 = plt.subplot(1,1,1)
        ax1.hist(ref_list,bins,color=color,edgecolor=edgecolor)
        # The bottom x-axis is in days
        ax1.set_xlim([0,max_day])
        # The top x-axis marks year and month in YYYYMM
        tick_list_1 = [] # Store the position number
        tick_list_2 = [] # Store the tick text
        ref_year = ref_time.year
        ref_month = ref_time.month
        ref_day = ref_time.day
        if ref_day == 1:
            tick_list_1.append(0)
            tick_list_2.append(str(ref_year)+str(ref_month).zfill(2))
        status = True # Start to loop month by month
        loop_time = UTCDateTime(ref_year,ref_month,1) # Initiate loop time
        step = 32 #32 > 31. Make sure each step pass to next month
        while status==True:
            loop_time = loop_time + step*24*60*60
            tmp_year = loop_time.year
            tmp_month = loop_time.month
            loop_time = UTCDateTime(tmp_year,tmp_month,1)
            diff_days = (loop_time - ref_time)/(24*60*60)
            if diff_days > (max_day):
                status=False
            else:
                tick_list_1.append(diff_days)
                tick_list_2.append((str(tmp_month).zfill(2)))
        if plot_months:
            ax2 = ax1.twiny()
            ax2.set_xlim([0,max_day])
            ax2.plot(0,0,'k.')
            plt.xticks(tick_list_1,tick_list_2)
            ax2.set_xlabel("date")
        if xlim!=[]:
            ax1.set_xlim(xlim)
            if plot_months:
                ax2.set_xlim(xlim)
        if ylim!=[]:
            plt.ylim(ylim)
        ax1.set_xlabel("Time, days")
        ax1.set_ylabel("event quantity")
        if plt_show:
            plt.show()

    def diffusion_plot(self,refid=None,refloc=[],diff_cfs=[],unit="day",xlim=[],ylim=[],plt_show=True):
        '''
        Parameters:
        refid: reference event id
        refloc: [lon,lat], reference site longitude and latitude, if not provided, use refid
        diff_cfs: diffusion coefficient list, this will draw corresponding lines on the map
        '''
        #from seisloc.plot import diffusion_plot
        
        if refid==None and refloc==[]:
            raise Exception("refid or refloc should be proivded")
        if refloc==[]:
            refloc=[self.dict[refid][0],self.dict[refid][1]]
        dist_list = np.zeros((len(self.keys),1))
        day_list = np.zeros((len(self.keys),1))
        mag_list = np.zeros((len(self.keys),1))
        for i in range(len(self.keys)):
            dist,_,_ = gps2dist_azimuth(self.locs[i,1],\
                                self.locs[i,0],\
                                refloc[1],\
                                refloc[0])
            day_list[i,0] = (self.relative_seconds[i]-np.min(self.relative_seconds))/(24*60*60)
            mag_list[i,0] = self.locs[i,3]
            dist_list[i,0] = dist

        fig1 = plt.figure(1)
        ax1 = plt.subplot(1,1,1)
        if unit=="day":
            x_list = day_list
            plt.xlabel("Time (day)",fontsize=16)
        elif unit == "hour":
            x_list = day_list*24
            plt.xlabel("Time (hour)",fontsize=16)
        else:
            raise Exception("Unit error: 'day' or 'hour'")
        ax1.set_ylabel("Distance (m)",fontsize=16)
        ax1.scatter(x_list,dist_list,(mag_list+2)*3,c='k')
        ax1.set_xlim([0,np.max(x_list)])

        diff_lines = []
        if isinstance(diff_cfs,int) or isinstance(diff_cfs,float):
            diff_cfs=[diff_cfs]
        for diff_cf in diff_cfs:
            if unit=="day":
                x = np.linspace(0,np.max(x_list),int(np.max(x_list)*20)+1)
                y = np.sqrt(4*np.pi*diff_cf*x*24*60*60)
            elif unit=="hour":
                x = np.linspace(0,np.max(x_list),int(np.max(x_list)*20)+1)
                y = np.sqrt(4*np.pi*diff_cf*x*60*60)
            diff_line, = plt.plot(x,y)
            diff_lines.append(diff_line)

        plt.legend(diff_lines,diff_cfs,title="Diffusion Coefficient $m^2/s$")
        if len(xlim)>0:
            plt.xlim(xlim)
        if len(ylim)>0:
            plt.ylim(ylim)
        else:
            plt.ylim(bottom=0)
        if plt_show:
            plt.show()

    def animation(self,
                  incre_hour=2,
                  mb_time=None,
                  me_time=None,
                  xlim=[],
                  ylim=[],
                  geopara=None,
                  cmap=None,
                  vmin=None,
                  vmax=None):
        """
        Generate gif animation file
        increment: Time increased for each plot. Unit: hour
        """
        # Remove previous results
        try:
            shutil.rmtree("dd_animation")
        except:
            pass
        os.makedirs("dd_animation")
        if xlim == []:
            xlim = [np.min(self.locs[:,0]),np.max(self.locs[:,0])]
        if ylim == []:
            ylim = [np.min(self.locs[:,1]),np.max(self.locs[:,1])]    
        min_time = self.first_time+np.min(self.relative_seconds)
        max_time = self.first_time+np.max(self.relative_seconds)
        if mb_time == None:
            mb_time = min_time
        if me_time == None:
            me_time = max_time
        print("Movie start time is: ",mb_time)
        print("  Movie end time is: ",me_time)

        if vmin == None:
            vmin = 0
        if vmax == None:
            vmax = (max_time - min_time)/(24*60*60)

        inc_second = incre_hour*60*60 # Time increment
        loop_time = mb_time
        count = 1
        ref_time = mb_time
        while loop_time <= me_time:
            fig = plt.figure(1,figsize=(8,8))
            ax1 = fig.add_subplot(111)
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax1.set_xlabel("Lon(degree)",fontsize=18)
            ax1.set_ylabel("Lat(degree)",fontsize=18)
            ax1.set_title(f"{str(loop_time)[:19]}",fontsize=16)
            if geopara != None:
                # ----------------Molin faults--------------------------------
                ml_fault = np.array(geopara.dict['ml_fault'])
                ML_fault,=ax1.plot(ml_fault[:,0],ml_fault[:,1],'r-')
                # ----------------Zigong faults-------------------------------
                for key in geopara.dict['zg_faults']:
                    array = np.array(geopara.dict['zg_faults'][key])
                    ax1.plot(array[:,0],array[:,1],'k-',label='Faults')
                #----------------------Wells-----------------------------------
                wells = np.array(geopara.dict['wells'])
                well_lons=[]; well_lats=[]
                for well in geopara.dict['wells']:
                    well_lons.append(well[0])
                    well_lats.append(well[1])
                s_well,=ax1.plot(well_lons,well_lats,'s',c='#1f77b4',markerfacecolor='white',mew=2,markersize=12)
                #----------------------Stations--------------------------------
                sta_lons=[]; sta_lats=[]
                for sta in geopara.dict['sta_locs']:
                    if sta[3]=="SC":
                        sta_lons.append(sta[0]); sta_lats.append(sta[1])
                if len(sta_lons)>0:
                    s_sta=ax1.scatter(sta_lons,sta_lats,marker='^',c='cyan',s=120,edgecolor='k',label='Stations')
            #------------- Events -----------------------------------------
            eve_arr = []
            rela_days = []
            for i,second in enumerate(self.relative_seconds):
                e_time = self.first_time + second
                if e_time<(loop_time+inc_second/2) and e_time>mb_time:
                    eve_arr.append(self.locs[i,:4])
                    rela_days.append((e_time-ref_time)/(24*60*60))
            eve_arr = np.array(eve_arr)
            rela_days = np.array(rela_days)

            if len(eve_arr)>0:
                if cmap != None:
                        s_eve=ax1.scatter(eve_arr[:,0],
                                          eve_arr[:,1],
                                          s=(eve_arr[:,3]+2)*5,
                                          c=rela_days,
                                          cmap=cmap,
                                          vmin=vmin,
                                          vmax=vmax,
                                          label="Events")
                else:
                        s_eve=ax1.scatter(eve_arr[:,0],
                                          eve_arr[:,1],
                                          s=(eve_arr[:,3]+2)*5,
                                          c='k',
                                          label="Events")
            if geopara != None:
                plt.legend([s_well,s_sta,s_eve],\
                       ["Platform",'Station',"Seismicity"],\
                       loc='upper right',
                       fontsize=16)  
            ##------------- save results --------------------------------------------
            plt.savefig(f"dd_animation/{str(count).zfill(3)}.png")
            loop_time = loop_time + inc_second
            count+=1
            plt.close()
        #-------------------- gif -----------------------------
        imgs = []
        for i in range(1,count):
            pic_name = f'dd_animation/{str(i).zfill(3)}.png'
            tmp = Image.open(pic_name)
            imgs.append(tmp)
        imgs[0].save("dd_animation.gif",save_all=True,append_images=imgs,duration=10)
    
    def intervals_plot(self,interval=1,method='month',
                       xlim=[],ylim=[],
                       columns=4,subplotsize=(3,3),
                       marker='o',ms=1,
                       wspace=None,hspace=None):
        
        from seisloc.plot import intervals_plot
        plt.close()

        axs = intervals_plot(xys=self.locs[:,:2],
                rela_secs=self.relative_seconds,
                ref_time=self.first_time,
                interval=interval,method=method,
                xlim=xlim,ylim=ylim,
                columns=columns,subplotsize=subplotsize,
                marker=marker,ms=ms,
                wspace=wspace,hspace=hspace)
        
        return axs
   
    def depths_plot(self,
                deplim=[0,10],interval=1,
                xlim=[],ylim=[],
                columns=4,subplotsize=(3,3),
                marker='o',ms=1,color='k',
                zorder=0,
                wspace=None,hspace=None):
        from seisloc.plot import depths_plot
        axs = depths_plot(xyz=self.locs[:,:3],
                          deplim=deplim,interval=interval,
                          xlim=xlim,ylim=ylim,
                          columns=columns,subplotsize=subplotsize,
                          marker=marker,ms=ms,color=color,
                          zorder=zorder,
                          wspace=None,hspace=None)
        return axs

    def copy(self):
        return copy.deepcopy(self)
    def merge(self,dd2):
        for key in dd2.keys:
            if key in self.keys:
                raise Exception(f"Key error {key}. Please avoid using the same key value")
            self.dict[key]=dd2.dict[key]
        self.init()

    def __repr__(self):
        _qty = f"HypoDD relocation catalog with {len(self.dict.keys())} events\n"
        _time= f"     Time range is: {self.first_time+np.min(self.relative_seconds)} to {self.first_time+np.max(self.relative_seconds)}\n"
        _mag = f" Magnitue range is: {format(np.min(self.locs[:,3]),'4.1f')} to {format(np.max(self.locs[:,3]),'4.1f')}\n"
        _lon = f"Longitude range is: {format(np.min(self.locs[:,0]),'8.3f')} to {format(np.max(self.locs[:,0]),'8.3f')}\n"
        _lat = f" Latitude range is: {format(np.min(self.locs[:,1]),'7.3f')} to {format(np.max(self.locs[:,1]),'7.3f')}\n"
        _dep = f"    Depth range is: {format(np.min(self.locs[:,2]),'4.1f')} to {format(np.max(self.locs[:,2]),'4.1f')}\n"
        return _qty+_time+_mag+_lon+_lat+_dep
    
    def __getitem__(self,key):
        return self.dict[key]
