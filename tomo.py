import warnings
from seisloc.hypoinv import load_y2000
import numpy as np
import os
import re
import matplotlib.pyplot as plt

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

def load_MOD(MODfile):
    MOD = []
    with open(MODfile,'r') as f:
        for line in f:
            MOD.append(line.rstrip())
    _,_nx,_ny,_nz=MOD[0].split()
    global nx,ny,nz
    nx = int(_nx); ny = int(_ny); nz = int(_nz);
    X = np.zeros((1,nx))
    _X = MOD[1].split() # Second line is x list 
    for i in range(len(_X)):
        X[0,i]=float(_X[i])
    Y = np.zeros((1,ny))
    _Y = MOD[2].split() # Third line is y list
    for i in range(len(_Y)):
        Y[0,i]=float(_Y[i])
    Z = np.zeros((1,nz))
    _Z = MOD[3].split() # Forth line is z list
    for i in range(len(_Z)):
        Z[0,i] = float(_Z[i])
    VpVs = np.loadtxt(MODfile,skiprows=4)    
    VEL_P = np.zeros((nx,ny,nz))
    VEL_S = np.zeros((nx,ny,nz))  
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                VEL_P[i,j,k] = VpVs[k*ny+j,i]
                VEL_S[i,j,k] = VpVs[k*ny+j,i]/VpVs[nz*ny+k*ny+j,i]
    return X,Y,Z, VEL_P,VEL_S

class MOD():
    def __init__(self,MODfile="MOD"):
        self.X,self.Y,self.Z,self.Vp, self.Vs = load_MOD(MODfile)
        self.nx = self.X.shape[-1]
        self.ny = self.Y.shape[-1]
        self.nz = self.Z.shape[-1]

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

def prep_D3MOD(MODfile="MOD",Vpfile="Vp_model.dat",Vsfile="Vs_model.dat"):
    assert os.path.exists("D3MOD") == False
    mod = MOD(MODfile=MODfile)
    VEL_P,VEL_S,POS_RATIO = load_velocity(mod.nx,mod.ny,mod.nz,Vpfile=Vpfile,Vsfile=Vsfile)
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
