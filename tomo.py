import warnings
from seisloc.hypoinv import load_y2000
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from seisloc.utils import read_line_values

def gen_abs_file(arcfile,outfile="Input_Files/absolute.dat",pweight=1,sweight=0.5):
    pweight = 1
    sweight = 0.5
    arcphs = load_y2000(arcfile)
    with open("Input_Files/absolute.dat",'w') as f:
        for key in arcphs.keys():
            evid = arcphs[key]["evid"]
            f.write(f"#             {format(str(evid),'>5s')}\n")
            phases = arcphs[key]["phase"]
            for phase in phases:
                net,sta,phs,abstime = phase
                f.write(f"{format(sta,'<5s')}")
                f.write("      ")
                f.write(f"{format(abstime,'>5.2f')}")
                f.write("       ")
                if phs == "P":
                    f.write(format(pweight,'3.1f'))
                elif phs == "S":
                    f.write(format(sweight,'3.1f'))
                f.write("   ")
                f.write(phs)
                f.write("\n")


def prepMOD(head,lon_list,lat_list,dep_list,vel_list,poisson_list):
    """
    Output MOD file for the tomoDD based on information provided
    Parameters:
    head: bld,nx,ny,nz. bld:resolution; nx/ny/nz: nodes for lon/lat/dep
    vel_list: P wave velocity list
    poisson_list: possion ratio of each layer
    len(lon_list)=nx; len(lat_list)=ny; len(dep_list)=nz;
    len(vel_list)==nz;len(poisson_list)==nz
    """
    f = open("MOD",'w')
    bld = head[0];
    nx = head[1];
    ny = head[2];
    nz = head[3];
    if nx != len(lon_list):
        raise Exception("Wrong longitude list length")
    if ny != len(lat_list):
        raise Exception("Wrong latitude list length")
    if nz != len(dep_list):
        raise Exception("Wrong depth list length")
    if nz != len(vel_list):
        raise Exception("Wrong velocity list length")
    if nz != len(poisson_list):
        raise Exception("Wrong poisson list length")

    if len(lon_list) != len(set(lon_list)):
        raise Exception("Duplicated values in longitude list.")
    if len(lat_list) != len(set(lat_list)):
        raise Exception("Duplicated values in latitude list.")
    if len(dep_list) != len(set(dep_list)):
        raise Exception("Duplicated values in depth list.")

    for i in range(len(lon_list)-1):
        if lon_list[i]>lon_list[i+1]:
            warnings.warn(f"lon_list[{i}]>lon_list[{i+1}]")
    for i in range(len(lat_list)-1):
        if lat_list[i]>lat_list[i+1]:
            warnings.warn(f"lat_list[{i}]>lat_list[{i+1}]")
    for i in range(len(dep_list)-1):
        if dep_list[i]>dep_list[i+1]:
            warnings.warn(f"dep_list[{i}]>dep_list[{i+1}]")

    f.write(f"{bld} {nx} {ny} {nz}\n")
    for i in range(len(lon_list)):
        f.write(str(lon_list[i]))
        if i != len(lon_list):
            f.write(" ")
    f.write("\n")
    for i in range(len(lat_list)):
        f.write(str(lat_list[i]))
        if i != len(lat_list):
            f.write(" ")
    f.write("\n")
    for i in range(len(dep_list)):
        f.write(str(dep_list[i]))
        if i != len(dep_list):
            f.write(" ")
    f.write("\n")
        
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                f.write(format(vel_list[k],'5.3f'))
                if i != (nx-1):
                    f.write(" ")
            f.write("\n")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                f.write(format(poisson_list[k],'5.3f'))
                if i != (nx-1):
                    f.write(" ")
            f.write("\n")
    f.close()

def load_xyz(locfile="tomoDD.reloc"):
    mdat = np.loadtxt(locfile);
    cusp = mdat[:,0];lon=mdat[:,2];lat=mdat[:,1];mag=mdat[:,16];
    x = lon; y=lat; z=mdat[:,3]
    return x,y,z

class MOD():
    def __init__(self,MODfile="MOD"):
        self.load_MOD(MODfile)
        
    def load_MOD(self,MODfile):
        self.lines = []
        with open(MODfile,'r') as f:
            for line in f:
                self.lines.append(line.rstrip())
        self._bld,_nx,_ny,_nz=self.lines[0].split()
        self.bld = float(self._bld)
        self.nx = int(_nx); self.ny = int(_ny); self.nz = int(_nz);
        self.X = np.zeros((1,self.nx))
        _X = self.lines[1].split() # Second line is x list 
        for i in range(len(_X)):
            self.X[0,i]=float(_X[i])
        self.Y = np.zeros((1,self.ny))
        _Y = self.lines[2].split() # Third line is y list
        for i in range(len(_Y)):
            self.Y[0,i]=float(_Y[i])
        self.Z = np.zeros((1,self.nz))
        _Z = self.lines[3].split() # Forth line is z list
        for i in range(len(_Z)):
            self.Z[0,i] = float(_Z[i])
        VpVs = np.loadtxt(MODfile,skiprows=4)    
        self.Vp = np.zeros((self.nx,self.ny,self.nz))
        self.Vs = np.zeros((self.nx,self.ny,self.nz))  

        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    self.Vp[i,j,k] = VpVs[k*self.ny+j,i]
                    self.Vs[i,j,k] = VpVs[k*self.ny+j,i]/VpVs[self.nz*self.ny+k*self.ny+j,i]

def load_tomo_vel(nx,ny,nz,Vpfile="Vp_model.dat",Vsfile="Vs_model.dat"):
    Vp = np.loadtxt(Vpfile)
    Vs = np.loadtxt(Vsfile)
    VEL_P = np.zeros((nx,ny,nz))
    VEL_S = np.zeros((nx,ny,nz))
    POS_RATIO = np.zeros((nx,ny,nz))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                VEL_P[i,j,k] = Vp[k*ny+j,i]
                VEL_S[i,j,k] = Vs[k*ny+j,i]
                POS_RATIO[i,j,k] = Vp[k*ny+j,i]/Vs[k*ny+j,i]
    
    return VEL_P,VEL_S,POS_RATIO

def prep_D3MOD(MODfile="MOD",Vpfile="Output_Files/Vp_model.dat",Vsfile="Output_Files/Vs_model.dat"):
    assert os.path.exists("D3MOD") == False
    mod = MOD(MODfile=MODfile)
    VEL_P,VEL_S,POS_RATIO = load_tomo_vel(mod.nx,mod.ny,mod.nz,Vpfile=Vpfile,Vsfile=Vsfile)
    with open("D3MOD",'w') as f:
        for line in mod.lines[:4]:
            f.write(line+"\n")
        for k in range(mod.nz):
            for j in range(mod.ny):
                for i in range(mod.nx):
                    f.write(format(VEL_P[i,j,k],'5.3f'))
                    if i != (mod.nx-1):
                        f.write(" ")
                f.write("\n")

        for k in range(mod.nz):
            for j in range(mod.ny):
                for i in range(mod.nx):
                    f.write(format(POS_RATIO[i,j,k],'5.3f'))
                    if i != (mod.nx-1):
                        f.write(" ")
                f.write("\n")

def tomo_event_sel(evid_list=[],event_dat="event.dat",event_sel="event.sel"):
    '''
    select events in the "event.dat" file and output them into
    the "event.sel" file by the event ID list provided
    '''
    content = []
    # Read in data
    with open(event_dat,'r') as f:
        for line in f:
            line = line.rstrip()
            evid = int(line[-8:-1])
            if evid in evid_list:
                content.append(line)
    f.close()

    # Output into target file
    with open(event_sel,'w') as f:
        for line in content:
            f.write(line+"\n")
    f.close()

def extract_log_info(damp_log = "tomoDD-SE.log"):
    smooth_damp_list = []    # smooth and damp line
    abs_ccs = []
    wt_ccs = []
    abs_cts = []
    wt_cts = []
    with open(damp_log,'rb') as f:
        for line in f:
            if line[:7]==b" smooth":
                tmp = re.split(":",line.decode("ascii"))[1]
                _,_smooth,_damp,_xnorm,_xnorm_vel,_rnorm,_rnorm_wt = re.split(" +",tmp.rstrip())
                smooth_damp_list.append([float(_smooth),float(_damp),float(_xnorm),float(_xnorm_vel),
                                       float(_rnorm),float(_rnorm_wt)])
            if line[:16]==b" absolute cc rms":
                tmp = re.split("=",line.decode("ascii"))[1]
                _value = re.split(" +",tmp.rstrip())[1]
                abs_ccs.append(float(_value))
            if line[:16]==b" weighted cc rms":
                tmp = re.split("=",line.decode("ascii"))[1]
                _value = re.split(" +",tmp.rstrip())[1]
                wt_ccs.append(float(_value))
            if line[:16]==b" absolute ct rms":
                tmp = re.split("=",line.decode("ascii"))[1]
                _value = re.split(" +",tmp.rstrip())[1]
                abs_cts.append(float(_value))
            if line[:16]==b" weighted ct rms":
                tmp = re.split("=",line.decode("ascii"))[1]
                _value = re.split(" +",tmp.rstrip())[1]
                wt_cts.append(float(_value))
    log_info_dict = {}
    log_info_dict["smooth_damp_list"] = np.array(smooth_damp_list)
    log_info_dict["abs_ccs"] = abs_ccs
    log_info_dict["wt_ccs"] = wt_ccs
    log_info_dict["abs_cts"] = abs_cts
    log_info_dict["wt_cts"] = wt_cts
    return log_info_dict

def log_info_plot(log_info_dict):
    fig,axs = plt.subplots(3,3,figsize=(12,8))
    niter = log_info_dict["smooth_damp_list"].shape[0]
    axs[0,0].plot(np.arange(niter)+1,log_info_dict["smooth_damp_list"][:,2])
    axs[0,0].set_ylabel("xnorm")
    axs[0,1].plot(np.arange(niter)+1,log_info_dict["smooth_damp_list"][:,3])
    axs[0,2].plot(np.arange(niter)+1,log_info_dict["smooth_damp_list"][:,3]/log_info_dict["smooth_damp_list"][:,2])
    axs[0,2].set_ylabel("Ratio")
    axs[0,1].set_ylabel("xnorm_vel")
    axs[1,0].plot(np.arange(niter)+1,log_info_dict["smooth_damp_list"][:,4])
    axs[1,0].set_ylabel("rnorm")
    axs[1,1].plot(np.arange(niter)+1,log_info_dict["smooth_damp_list"][:,5])
    axs[1,1].set_ylabel("rnorm_wt")
    axs[1,2].plot(np.arange(niter)+1,log_info_dict["smooth_damp_list"][:,5]/log_info_dict["smooth_damp_list"][:,4])
    axs[1,2].set_ylabel("Ratio")
    miter = len(log_info_dict["abs_cts"])
    axs[2,0].plot(np.arange(miter)+1,log_info_dict["abs_cts"],'k-',label="abs_cts")
    axs[2,0].plot(np.arange(miter)+1,log_info_dict["wt_cts"],'b-',label="wt_cts")
    axs[2,0].legend()
    axs[2,1].plot(np.arange(miter)+1,log_info_dict["abs_ccs"],'k-',label="abs_ccs")
    axs[2,1].plot(np.arange(miter)+1,log_info_dict["wt_ccs"],'b-',label="wt_ccs")
    axs[2,1].legend()
    plt.tight_layout()
    plt.show()

class Tomo_vel():
    def __init__(self,velfile="tomoDD.vel"):
        with open(velfile,'r') as f:
            self.lines = f.readlines()
        #---------- grids setup -------------------
        self.xs = read_line_values(self.lines[9])
        self.xs = np.array(self.xs)
        self.nx = len(self.xs)
        self.ys = read_line_values(self.lines[12])
        self.ys = np.array(self.ys)
        self.ny = len(self.ys)
        self.zs = read_line_values(self.lines[15])
        self.zs = np.array(self.zs)
        self.nz = len(self.zs)
        #---------- load input velocity -----------
        inp_vel_bidx = 21
        inp_vel_gap = self.ny + 2
        self.inp_vps = np.zeros((self.nx,self.ny,self.nz))
        loop_idx = inp_vel_bidx
        for k in range(self.nz):
            for j in range(self.ny):
                print(self.lines[loop_idx])
                self.inp_vps[:,j,k] = read_line_values(self.lines[loop_idx])
                loop_idx += 1
            loop_idx += 2
        #---------- load Vp/Vs ---------------------
        self.inp_vpvs = np.zeros((self.nx,self.ny,self.nz))
        for k in range(self.nz):
            for j in range(self.ny):
                self.inp_vpvs[:,j,k] = read_line_values(self.lines[loop_idx])
                loop_idx += 1
            loop_idx += 2        
        #---------- load Vs ---------------------
        self.inp_vss = np.zeros((self.nx,self.ny,self.nz))
        for k in range(self.nz):
            for j in range(self.ny):
                self.inp_vss[:,j,k] = read_line_values(self.lines[loop_idx])
                loop_idx += 1
            loop_idx += 2           
        inp_vel_eidx = loop_idx   # end index of the input velocity
        #========== load iteration velocity ==============
        self.iters = {}
        iter_pidxs = []
        iter_ids = []
        loop_idx = inp_vel_eidx
        for line in self.lines[loop_idx:]:
            if line[:29]==" P-wave velocity at iteration":
                _iter_id = re.split(" +",line)[-1]
                iter_id = int(_iter_id)
                iter_ids.append(iter_id)
                iter_pidxs.append(loop_idx)
                self.iters[iter_id] = {}
                self.load_iter_data(loop_idx,iter_id)
            loop_idx += 1

    def load_iter_data(self,loop_pidx,iter_id):
        self.iters[iter_id]["vp"] = np.zeros((self.nx,self.ny,self.nz))
        loop_idx = loop_pidx+1
        #-------- read iter vp ----------------------
        for k in range(self.nz):
            for j in range(self.ny):
                self.iters[iter_id]["vp"][:,j,k] = read_line_values(self.lines[loop_idx])
                loop_idx+=1
        #-------- read iter vs -----------------------
        self.iters[iter_id]["vs"] = np.zeros((self.nx,self.ny,self.nz))
        loop_idx += 2
        for k in range(self.nz):
            for j in range(self.ny):
                self.iters[iter_id]["vs"][:,j,k] = read_line_values(self.lines[loop_idx])
                loop_idx+=1
        #-------- read iter DWS_P -----------------------
        self.iters[iter_id]["DWS_P"] = np.zeros((self.nx,self.ny,self.nz))
        loop_idx += 1
        for k in range(self.nz):
            for j in range(self.ny):
                self.iters[iter_id]["DWS_P"][:,j,k] = read_line_values(self.lines[loop_idx])
                loop_idx+=1
        #-------- read iter DWS_S -----------------------
        self.iters[iter_id]["DWS_S"] = np.zeros((self.nx,self.ny,self.nz))
        loop_idx += 1
        for k in range(self.nz):
            for j in range(self.ny):
                self.iters[iter_id]["DWS_S"][:,j,k] = read_line_values(self.lines[loop_idx])
                loop_idx+=1                
                
    def plot_vel(self,iter_id,sub_figsize=(3,4)):
        fig,axs = plt.subplots(self.nz-2,4,figsize=(4*sub_figsize[1],(self.nz-2)*sub_figsize[0]),
                               sharex=True,sharey=True)
        for k in range(self.nz-2):
            #------------------Vp-------------------------------
            xxs,yys = np.meshgrid(self.xs[1:-1],self.ys[1:-1])
            psm = axs[k,0].pcolormesh(xxs,yys,self.iters[iter_id]["vp"][1:-1,1:-1,k+1].T,cmap="jet_r")
            axs[k,0].set_title(f"Z = {self.zs[k+1]} km; init Vp={self.inp_vps[0,0,k+1]}")
            cb = plt.colorbar(psm,ax=axs[k,0])
            delta_vps = self.iters[iter_id]["vp"][1:-1,1:-1,k+1]-self.inp_vps[1:-1,1:-1,k+1]
            kks = np.where(delta_vps!=0)
            delta_mean = np.mean(delta_vps[kks])
            psm = axs[k,1].pcolormesh(xxs,yys,delta_vps.T,
                               cmap = "jet_r",vmin=-0.5,vmax=0.5)
            cb = plt.colorbar(psm,ax=axs[k,1])
            axs[k,1].set_title(f"Mean $\Delta$ Vp={format(delta_mean,'.3f')}")
            #------------------Vs-------------------------------
            psm = axs[k,2].pcolormesh(xxs,yys,self.iters[iter_id]["vs"][1:-1,1:-1,k+1].T,cmap="jet_r")
            axs[k,2].set_title(f"Vs; init Vp={self.inp_vss[0,0,k+1]}")
            cb = plt.colorbar(psm,ax=axs[k,2])
            delta_vss = self.iters[iter_id]["vs"][1:-1,1:-1,k+1]-self.inp_vss[1:-1,1:-1,k+1]
            kks = np.where(delta_vss!=0)
            delta_mean = np.mean(delta_vss[kks])
            psm = axs[k,3].pcolormesh(xxs,yys,delta_vss.T,
                               cmap = "jet_r",vmin=-0.35,vmax=0.35)
            cb = plt.colorbar(psm,ax=axs[k,3])
            axs[k,3].set_title(f"$\Delta$ Vs={format(delta_mean,'.3f')}")
            
    def plot_DWS(self,iter_id,sub_figsize=(3,4)):
        fig,axs = plt.subplots(self.nz-2,2,figsize=(2*sub_figsize[1],(self.nz-2)*sub_figsize[0]),
                               sharex=True,sharey=True)
        for k in range(self.nz-2):
            #------------------Vp-------------------------------
            xxs,yys = np.meshgrid(self.xs[1:-1],self.ys[1:-1])
            psm = axs[k,0].pcolormesh(xxs,yys,self.iters[iter_id]["DWS_P"][1:-1,1:-1,k+1].T)
            axs[k,0].set_title(f"Z={self.zs[k+1]} km DWS_P")
            cb = plt.colorbar(psm,ax=axs[k,0])
            psm = axs[k,1].pcolormesh(xxs,yys,self.iters[iter_id]["DWS_S"][1:-1,1:-1,k+1].T)
            axs[k,1].set_title(f"DWS_S")
            cb = plt.colorbar(psm,ax=axs[k,1])
            axs[k,1].set_title(f"DWS_S")

    def __repr__(self):
        print("Iteration list: ",end=" ")
        for key in self.iters.keys():
            print(key,end=" ")
        _lons = f"nx: {len(self.xs)}; use *.xs to show the longitude nodes\n"
        _lats = f"ny: {len(self.ys)}; use *.ys to show the latitude nodes\n"
        _deps = f"nz: {len(self.zs)}; use *.zs to show the depth nodes\n"
        return _lons+_lats+_deps
