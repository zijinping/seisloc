import warnings
from seisloc.hypoinv import load_y2000
import numpy as np
import os

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
        self.lines = []
        with open(MODfile,'r') as f:
            for line in f:
                self.lines.append(line.rstrip())

        _,_nx,_ny,_nz=self.lines[0].split()
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

def load_velocity(nx,ny,nz,Vpfile="Vp_model.dat",Vsfile="Vs_model.dat"):
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

