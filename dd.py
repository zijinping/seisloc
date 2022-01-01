import re
import os
import glob
import shutil
import random
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from seisloc.hypoinv import load_sum_evstr,load_sum_evid
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seisloc.geometry import in_rectangle,loc_by_width
from math import ceil,floor
import multiprocessing as mp
import time
import subprocess
import copy
from PIL import Image
#
def load_PC(catalog="/home/zijinping/Desktop/projects/wy_eq/2018_2019_PC/2018_2019_hypoDD.reloc",
            start_evid=300000):
    """
    Read in Pengcheng's earthquake catalog
    """
    evid = start_evid+1
    catalog_dict = {}
    cont = []
    with open(catalog,'r') as f:
        cont = f.readlines()
    for line in cont:
        _time,_,_lat,_lon,_dep,_mag, = re.split(" +",line.rstrip())[:6]
        etime = UTCDateTime.strptime(_time,'%Y%m%d%H%M%S')
        catalog_dict[evid] = [float(_lon),float(_lat),float(_dep),float(_mag),etime]
        evid += 1
    return catalog_dict

def load_CEDC(catalog="/home/zijinping/Dropbox/resources/catalog/CEDC/20090101_20201231.txt",start_evid=100000):
    """
    Read in China Earthquake Data Center catalog
    """
    evid = start_evid+1
    catalog_dict = {}
    cont = []
    with open(catalog,'r') as f:
        cont = f.readlines()
    for line in cont:
        _time,_lon,_lat,_dep,M,_mag,_,_ = re.split(",",line.rstrip())
        #print(_time,_lon,_lat,_dep,_mag)
        _date,_hr_min = re.split(" ",_time)
        _yr,_mo,_dy = re.split("\/",_date)
        _hr,_min = re.split(":",_hr_min)
        yr = int(_yr); mo = int(_mo); dy = int(_dy)
        hr = int(_hr); minute = int(_min)
        etime = UTCDateTime(yr,mo,dy,hr,minute,0)
        catalog_dict[evid] = [float(_lon),float(_lat),float(_dep),float(_mag),etime]
        evid += 1
    return catalog_dict

class DD():
    def __init__(self,reloc_file="hypoDD.reloc"):
        """
        The programme will read in hypoDD relocation file by default. If no hypoDD
        file provided, it will generate an empty catalog. A user can set up a new 
        catalog by providing a dict in the form:
            dict[evid] = [lon,lat,dep,mag,UTCDateTime]
        example:
        >>> dd = DD()
        >>> dd.dict = cata_dict  #cata_dict is a dictionary following above format
        >>> dd.init()
        >>> print(dd)
        """
        try:
            self.dict,_ = load_DD(reloc_file)
            self.init()
        except:
            print("No hypoDD data read in, an empty catalog generated")
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
              alonlat=[],
              blonlat=[],
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
    
    def MT_plot(self,ref_time=UTCDateTime(2019,3,1),xlim=[],ylim=[0,5],unit="day",cmap=None,vmin=0,vmax=1,mlow=0,plt_show=True):
        """
        unit: 'day','hour' or 'second'
        """
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

    def dep_dist_plot(self,refid,
                  ref_time=UTCDateTime(2019,3,1),
                  xlim=[],
                  deplim=[],
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
        if len(deplim)>0:
            axs[0].set_ylim(deplim)
        if len(distlim)>0:
            axs[1].set_ylim(distlim)
        axs[0].grid(axis="y")
        axs[1].grid(axis="y")
        reflon = self.dict[refid][0]
        reflat = self.dict[refid][1]
        refdep = self.dict[refid][2]
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


def event_sel(evid_list=[],event_dat="event.dat",event_sel="event.sel"):
    '''
    select events in the "event.dat" file and output them into
    the "event.sel" file by the event ID list provided
    '''
    content = []
    # Read in data
    with open(event_dat,'r') as f:
        for line in f:
            line = line.rstrip()
            evid = int(line[-8:])
            if evid in evid_list:
                content.append(line)
    f.close()

    # Output into target file
    with open(event_sel,'w') as f:
        for line in content:
            f.write(line+"\n")
    f.close()

def dtct_sel(evid_list,input_file):
    """
    Output clean dtct file with event id list provided
    """
    out_file = input_file+".sel"
    out_cont = []
    record_status = False # Set initiate value
    with open(input_file,'r') as f:
        for line in f:
            if line[0]=="#": # pair line
                print(re.split(" +",line.rstrip()))
                _,ID1,ID2 = re.split(" +",line.rstrip())
                ID1 = int(ID1)
                ID2 = int(ID2)
                if (ID1 in evid_list) and (ID2 in evid_list):
                    record_status = True
                    out_cont.append(line.rstrip())
                else:
                    record_status = False
            elif record_status==True:
                out_cont.append(line.rstrip())
    f.close()

    with open(out_file,'w') as f:
        for line in out_cont:
            f.write(line+"\n")


def hypoDD_rmdup(in_file="total_hypoDD.reloc"):
    """
    remove duplicated events and take mean values
    """
    count=0
    evid_list = []
    evid_mapper = {}
    with open(in_file,'r') as f:
        for line in f:
            evid = int(line[0:11])
            try:
                evid_mapper[evid].append(line)
                count += 1
            except:
                evid_mapper[evid]=[line]
    f.close()

    evid_list = list(evid_mapper)
    evid_list.sort()

    log_record = []
    f=open(in_file+".rm",'w')
    f.close()
    for evid in evid_list:
        if len(evid_mapper[evid])>1:
            lon_list = []
            lat_list = []
            dep_list = []
            for i in range(len(evid_mapper[evid])):
                lon = float(evid_mapper[evid][i][22:32])
                lon_list.append(lon)
                lat = float(evid_mapper[evid][i][11:20])
                lat_list.append(lat)
                dep = float(evid_mapper[evid][i][36:42])
                dep_list.append(dep)
            lon_mean = np.mean(lon_list)
            lat_mean = np.mean(lat_list)
            dep_mean = np.mean(dep_list)
            lon_str = format(lon_mean,'10.6f')
            lat_str = format(lat_mean,'9.6f')
            dep_str = format(dep_mean,'6.3f')
            log_record.append([evid,lon_str,lat_str,dep_str])
            firstr = evid_mapper[evid][0]  #Use the first record as template
            outstr = firstr.replace(line[22:32],lon_str,1) #Replace lon
            outstr = outstr.replace(line[11:20],lat_str,1) #Replace lat
            outstr = outstr.replace(line[22:32],dep_str,1) #Replace dep
            with open("total_hypoDD.reloc.rm",'a') as f:
                f.write(outstr)
            f.close()
        else:
            with open(in_file+".rm",'a') as f:
                f.write(evid_mapper[evid][0])
            f.close()
    print(log_record)

def gen_dtcc(sta_list,sum_file="out.sum",work_dir="./",cc_threshold=0.7,min_link=4,max_dist=4):
    '''
    This function generate dt.cc.* files from the output of SCC results
    
    Parameters:
        sta_list: list of stations to be processed
        sum_file: summary(*.sum) file generated by HYPOINVERSE
        work_dir: the directory of mp_scc results
        cc_threshold: threshold value of cross_correlation
        min_link: minumum links to form an event pair
        max_dist: maximum distance accepted to form an event pair, unit km
    '''
    sum_rev= load_sum_evstr(sum_file) # dictionary {evid: [e_lon,e_lat,e_dep,e_mag]}
    sum_dict= load_sum_evid(sum_file)    # dictionary {"YYYYmmddHHMMSSff":[e_lon,e_lat,e_dep,e_mag]}
    work_dir = os.path.abspath(work_dir)
    evid_list = []                  # event list included by scc results
    to_cc_list = []                 # event list included in the ouput dt.cc.* results
    
    # Remove existing dt.cc files 
    cc_files = glob.glob(os.path.join(work_dir,"dt.cc*"))
    for cc_file in cc_files:
        os.remove(cc_file)
        
    print(">>> Loading in scc results ...")
    for sta in tqdm(sta_list):                                # Loop for station
        for pha in ["P","S"]:                                 # Loop for phases
            sta_pha = sta+"_"+pha
            globals()[sta_pha+"_cc_dict"] = {}                # Initiate dictionary
            sta_pha_path = os.path.join(work_dir,sta_pha)
            for file in os.listdir(sta_pha_path):
                if file[-3:]!=".xc":                          # none scc results file
                    continue
                with open(os.path.join(sta_pha_path,file),'r') as f:
                    for line in f:
                        line = line.rstrip()
                        path1,arr1,_,path2,arr2,_,cc,aa=re.split(" +",line.rstrip())
                        tmp = os.path.split(path1)[0]
                        eve_folder1 = os.path.split(tmp)[1]
                        evid1 = sum_rev[eve_folder1][0]
                        arr1 =float(arr1)                     # arrival time
                        tmp = os.path.split(path2)[0]
                        eve_folder2 = os.path.split(tmp)[1]
                        evid2 = sum_rev[eve_folder2][0]
                        if evid1 not in evid_list:
                            evid_list.append(evid1)
                        if evid2 not in evid_list:
                            evid_list.append(evid2)
                        arr2 = float(arr2)                    # arrival time
                        cc = float(cc)                        # cross correlation coefficient
                        aa = float(aa)                        # amplitude ratio
                        if cc >=cc_threshold:
                            try:
                                globals()[sta_pha+"_cc_dict"][evid1][evid2]=[arr1,arr2,cc,aa]
                            except:
                                globals()[sta_pha+"_cc_dict"][evid1]={} # Initiation
                                globals()[sta_pha+"_cc_dict"][evid1][evid2]=[arr1,arr2,cc,aa]
    evid_list.sort()
    print("<<< Loading complete! <<<")
    
    print(">>> Preparing dt.cc files ...")
    ##----------------
    for i,evid1 in enumerate(evid_list):
        print(evid1,"  ",end='\r')
        evid1_evlo = sum_dict[evid1][1]
        evid1_evla = sum_dict[evid1][2]
        out_index = int(i/6000)# Every 6k events preserve in a seperate dt.cc.* file.
                               # to avoid extreme large out file size.
        for evid2 in evid_list[i+1:]:
            evid2_evlo = sum_dict[evid2][1]
            evid2_evla = sum_dict[evid2][2]
            dist,_,_ = gps2dist_azimuth(evid1_evla,evid1_evlo,evid2_evla,evid2_evlo)
            if dist/1000>max_dist:                          # discard large distance events
                continue
            link_cc=[]
            for sta in sta_list:                            # Loop for stations
                for pha in ["P","S"]:                       # Loop for phases
                    sta_pha = sta+"_"+pha               
                    try:
                        arr1,arr2,cc,aa = globals()[sta_pha+"_cc_dict"][evid1][evid2]
                        link_cc.append([sta,arr1-arr2,cc,pha])
                    except:
                        continue
            if len(link_cc)>=min_link:
                if evid1 not in to_cc_list:
                    to_cc_list.append(evid1)
                if evid2 not in to_cc_list:
                    to_cc_list.append(evid2)
                cc_file = os.path.join(work_dir,"dt.cc."+f"{out_index}")
                with open(cc_file,'a') as f:                # Write in results
                    f.write(f"# {format(evid1,'5d')} {format(evid2,'5d')} 0\n")
                    for record in link_cc:
                        f.write(f"{record[0][:2]}{record[0]} {format(record[1],'7.4f')} {record[2]} {record[3]}\n")
                f.close()
    print(">>> Number of events in dt.cc is: ",len(to_cc_list))
    
    cont = []
    cc_files = glob.glob(os.path.join(work_dir,"dt.cc*"))
    cc_files.sort()
    for cc_file in cc_files:
        with open(cc_file,'r') as f:
            for line in f:
                cont.append(line)
        f.close()
        os.remove(cc_file)
    with open(os.path.join(work_dir,"dt.cc"),'w') as f:
        for line in cont:
            f.write(line)
    f.close()    
    print("<<< dt.cc files generated! <<<")

def hypoDD_ref_days(reloc_file,ref_time,shift_hours=0):
    """
    Add one column to the last of hypoDD files, calculate the length of time 
    between the referece time and the event time in days.
    The output is a file with the same title with reloc_file and add ".add" as
    suffix.

    Parameters
    ----------
     reloc_file: The hypoDD relocation file.
       ref_time: Reference time in UTCDateTime format
    shift_hours: Used when event time is not in UTC time zone
    """

    new_add=[]
    with open(reloc_file,"r") as f:
        for line in f:
            year = int(re.split(" +",line)[11])
            month = int(re.split(" +",line)[12])
            day = int(re.split(" +",line)[13])
            hour = int(re.split(" +",line)[14])
            minute = int(re.split(" +",line)[15])
            seconds = float(re.split(" +",line)[16])
            eve_time = UTCDateTime(year,month,day,hour,minute)+seconds
            days = (eve_time - ref_time)*1.0/(24*60*60)
            new_line=line[:-1]+" "+format(days,'4.2f')
            new_add.append(new_line)
    f.close()
    with open(reloc_file+".add","w") as f:
        for line in new_add:
            f.write(line+"\n")
    f.close()

def compare_DD(dd1_path,dd2_path):
    dd1,_ = load_DD(dd1_path)
    dd2,_ = load_DD(dd2_path)
    f = open("dd_diff.dat",'w')
    for key in dd1:
        try: 
            lon1 = dd1[key][0] 
            lon2 = dd2[key][0] 
            lat1 = dd1[key][1] 
            lat2 = dd2[key][1] 
            dep1 = dd1[key][2] 
            dep2 = dd2[key][2]
            print("dx,dy,dz:",abs(lon1-lon2)*111*1000,abs(lat1-lat2)*111*1000,abs(dep1-dep2)*1000)
            f.write(f"{abs(lon1-lon2)*111*1000} {abs(lat1-lat2)*111*1000} {abs(dep1-dep2)*1000}\n")
        except: 
            pass 
    f.close()

def hypoDD_mag_mapper(reloc_file,out_sum):
    """
    The output of hypoDD doesn't contain magnitude information.
    This function reads magnitude information from *.sum file, which is the
    output of hyperinverse and provide to hypoDD file.
    
    The results will cover the input reloc_fiie
    """

    #get the magnitude dictionary
    event_mag_list = {}
    with open(out_sum,"r") as f_obj:
        for line in f_obj:
            event_id = int(line[136:146])
            event_mag = int(line[123:126])*0.01
            event_mag_list[event_id]=event_mag
    f_obj.close()
    print(len(event_mag_list.keys()))
    #add in the magnitude
    new_dd = []
    with open(reloc_file,"r") as f_obj:
        for line in f_obj:
            dd_event_id = int(line[0:9])
            dd_event_mag = event_mag_list[dd_event_id]
            new_line=line[:128]+format(dd_event_mag,'5.2f')+line[132:]
            new_dd.append(new_line)
    f_obj.close()
    with open(reloc_file,"w") as f_obj:
        for line in new_dd:
            f_obj.write(line)
    f_obj.close()

def load_DD(reloc_file="hypoDD.reloc",shift_hour=0):
    """
    load results of hypoDD
    return eve_dict, df

    Parameters
    ----------
    If the time of results is not in UTC time zone, a time shift might needed.
    For example, Beijing time zone is 8 hours early than UTC time, 8 hours 
    should be deducted so as to be consistent with UTC time.
    """

    eve_dict={}
    columns = ["ID","LAT","LON","DEPTH","X","Y","Z","EX","EY","EZ",\
           "YR","MO","DY","HR","MI","SC","MAG",\
           "NCCP","NCCS","NCTP","NCTS","RCC","RCT","CID"]
    number = 0

    dataset = np.loadtxt(reloc_file)
    
    if dataset.shape[1] == 24:
        columns = ["ID","LAT","LON","DEPTH","X","Y","Z","EX","EY","EZ",\
                   "YR","MO","DY","HR","MI","SC","MAG",\
                   "NCCP","NCCS","NCTP","NCTS","RCC","RCT","CID"]
    if dataset.shape[1] == 25:
        columns = ["ID","LAT","LON","DEPTH","X","Y","Z","EX","EY","EZ",\
                   "YR","MO","DY","HR","MI","SC","MAG",\
                   "NCCP","NCCS","NCTP","NCTS","RCC","RCT","CID","DAY"]
    
    for i,data in enumerate(dataset):
        eve_id = data[0]
        eve_lat = data[1]
        eve_lon = data[2]
        eve_dep = data[3]
        eve_mag = data[16]
        eve_time = UTCDateTime(int(data[10]),int(data[11]),int(data[12]),int(data[13]),int(data[14]),0)+data[15] - shift_hour*50*60
        eve_dict[int(eve_id)]=[float(eve_lon),float(eve_lat),float(eve_dep),float(eve_mag),eve_time]

    df = pd.DataFrame(data=dataset,columns=columns)
    return eve_dict,df


def run_dd(base_dir="./",work_dir='hypoDD',inp_file="hypoDD.inp"):
    os.chdir(base_dir)
    os.chdir(work_dir)
    subprocess.run(["hypoDD",inp_file])
    os.chdir(base_dir)

def dd_bootstrap(base_folder="hypoDD",times=10,method="event",samp_ratio=0.75,cores=2):
    """
    Randomly run hypoDD with randomly selected events to show the results variation
    Parameters:
        base_folder: the basement folder which should include the material for hypoDD, 
            including dt.ct, dt.cc,hypoDD.inp, event.dat, station.dd
        times: number of hypoDD runs
        method: "event" means sample events; "phase" means sample phases
        samp_ratio: the ratio of events to be relocated in the run
    """
    # Load in event.dat file
    base_dir = os.getcwd()
    e_dat = []
    if method == "event":
        with open(os.path.join(base_folder,"event.dat"),'r') as f:
            for line in f:
                evid = int(line[84:91])
                e_dat.append(line)
        e_qty = len(e_dat)
        s_qty = int(e_qty*samp_ratio) # sample qty

        tar_folders = []
        tasks = []
        # Prepare the subroutine files
        for i in range(1,times+1):
            tar_folder = base_folder + str(i).zfill(3)
            shutil.copytree(base_folder,tar_folder)
            sel_idxs = random.sample(range(e_qty),s_qty)
            with open(os.path.join(tar_folder,'event.sel'),'w') as f:
                for idx in sel_idxs:
                    f.write(e_dat[idx])
            f.close()
            tar_folders.append(tar_folder)
            tasks.append([base_dir,tar_folder])
            
    if method == "phase":
        dtct = []
        out_dtct = []
        with open(os.path.join(base_folder,"dt.ct"),'r') as f:
            for line in f:
                dtct.append(line)        
        len_dtct = len(dtct)
        
        tar_folders = []
        tasks = []
        # Prepare the subroutine files
        for i in range(1,times+1):
            tar_folder = base_folder + str(i).zfill(3)
            shutil.copytree(base_folder,tar_folder)
            with open(os.path.join(tar_folder,'dt.ct'),'w') as f:
                
                for i,line in enumerate(dtct):
                    if line[0] == "#":    # event line
                        f.write(dtct[i])
                        tmp = []
                        j = i+1
                        while j<len_dtct and dtct[j][0]!="#":
                            tmp.append(dtct[j])
                            j=j+1
                        pha_qty = len(tmp)
                        sample_qty = int(pha_qty*samp_ratio+0.5)
                        sel_idxs = random.sample(range(pha_qty),sample_qty)
                        for idx in sel_idxs:
                            f.write(tmp[idx])


            tar_folders.append(tar_folder)
            tasks.append([base_dir,tar_folder])
            
    pool = mp.Pool(processes=cores)
    rs = pool.starmap_async(run_dd,tasks,chunksize=1)
    while True:
        remaining = rs._number_left
        print(f"Finished {len(tasks)-remaining}/{len(tasks)}",end='\r')
        if(rs.ready()):
            break
        time.sleep(0.5)
    print("\nDone!!!")
    
def bootstrap_summary(times,base_folder="hypoDD"):
    """
    """
    rand_dict = {}
    
    with open(os.path.join(base_folder,"event.dat"),'r') as f:
        for line in f:
            evid = int(line[84:91])
            rand_dict[evid] = []

    
    tar_folders = []
    for i in range(1,times+1):
        tar_folder = base_folder + str(i).zfill(3)
        tar_folders.append(tar_folder)

    # Load hypoDD results
    print("Loading results ...")
    for tar_folder in tar_folders:
        reloc_file = os.path.join(tar_folder,'hypoDD.reloc')
        dd_dict,_ = load_DD(reloc_file)
        for key in dd_dict.keys():
            rand_dict[key].append(dd_dict[key][:4])
    
    print("Write out results ... ")
    f = open("hypoDD.rand",'w')
    for key in rand_dict.keys():
        cont = rand_dict[key]
        if len(cont) == 0:
            continue
        else:
            lon_list = []
            lat_list = []
            dep_list = []
            for tmp in cont:
                lon_list.append(tmp[0])
                lat_list.append(tmp[1])
                dep_list.append(tmp[2])
            record_qty = len(cont)
            mean_lon = np.mean(lon_list)
            mean_lat = np.mean(lat_list)
            mean_dep = np.mean(dep_list)
            std_lon = np.std(lon_list)
            std_lat = np.std(lat_list)
            std_herr = np.sqrt(std_lon**2+std_lat**2)
            std_dep = np.std(dep_list)
            f.write(format(key,'7d')+" "+
                format(mean_lon,'8.4f')+" "+
                format(mean_lat,'7.4f')+" "+
                format(mean_dep*1000,'9.3f')+" "+
                format(std_herr*111.1*1000*2,"9.3f")+" "+
                format(std_dep*1000*2,"8.3f")+" "+
                format(record_qty,'3d')+"\n")
    f.close()

def pha_subset(pha_file,loc_filter,obs_filter=8,out_file=None):
    """
    *.pha file is the input file for hypoDD ph2dt, this function subset the
    pha file by the boundary condition and the minimum observation condition.
    The output file is a file with ".st" suffix

    Parameters
    ----------
    pha_file: Str. The input file.
    loc_filter: array in format [lon_min, lon_max, lat_min, lat_max]
    obs_filter: The minimum observation
    out_path: file path for the target file
    """

    lon_min, lon_max, lat_min, lat_max = loc_filter
    if out_file == None:
        out_file = pha_file+".st"
    f = open(out_file,"w")
    f.close()
    pha_content = []
    with open(pha_file,"r") as f:
        for line in f:
            pha_content.append(line.rstrip())
    f.close()
    i = 0
    j = 0
    record_list=[]
    for line in pha_content:
        if line[0]=="#":
            if i>0 and len(record_list) > (obs_filter+1):
                j=j+1
                with open(out_file,"a") as f:
                    for record in record_list:
                        f.write(record+"\n")
                f.close()
                record_list = []
                record_list.append(line)
            else:
                record_list = []
                record_list.append(line)
            i=i+1
            lat = float(re.split(" +",line)[7])
            lon = float(re.split(" +",line)[8])
            if lat>lat_min and lat<lat_max and lon>lon_min and lon<lon_max:
                region_pass = True
            else:
                region_pass = False
        else:
            if region_pass:
                record_list.append(line)
    if i>0 and len(record_list) > (obs_filter+1):
        j=j+1
        with open(out_file,"a") as f:
            for record in record_list:
                f.write(record+"\n")
        f.close()
    print("Event before filtering",i)
    print("Events qty after filtering",j)

def pha_sel(pha_file,e_list=[],remove_net=False):
    '''
    Select phases of events in e_list
    if need to remove net name.
    '''
    out = []   # output
    with open(pha_file,'r') as f:
        for line in f:
            line = line.rstrip()
            if line[0]=="#":
                _evid = re.split(" +",line)[-1]
                evid = int(_evid)
                if evid in e_list or e_list==[]:
                    status = True
                    out.append(line)
                else:
                    status = False
            else:
                if status==True:
                    if remove_net:
                        out.append(line[2:])
                    else:
                        out.append(line)
    f.close()

    with open(pha_file+".sel",'w') as f:
        for line in out:
            f.write(line+'\n')
    f.close()

def inv_dd_compare(inv,dd,keys=[],xlim=[],ylim=[],aspect='auto'):
    inv_locs = []
    dd_locs = []
    for key in keys:
        key = int(key)
        inv_lon = inv[key][1]
        inv_lat = inv[key][2]
        dd_lon = dd[key][0]
        dd_lat = dd[key][1]
        inv_locs.append([inv_lon,inv_lat])
        dd_locs.append([dd_lon,dd_lat])
    inv_locs = np.array(inv_locs)
    dd_locs = np.array(dd_locs)
    plt.plot(inv_locs[:,0],inv_locs[:,1],'kx')
    plt.plot(dd_locs[:,0],dd_locs[:,1],'rv')
    if len(xlim)>0:
        plt.xlim(xlim)
    if len(ylim)>0:
        plt.ylim(ylim)
    plt.gca().set_aspect(aspect)
