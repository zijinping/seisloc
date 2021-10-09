import matplotlib.pyplot as plt
import re
import numpy as np
from seisloc.utils import draw_vel
from seisloc.sta import load_sta

def load_cnv(cnv_file="velout.cnv"):
    """
    Load VELEST output file and return a dict with event id as key
    the content is [lon,lat,dep,rms]
    """
    cont = []
    with open(cnv_file,'r') as f:
        for line in f:
            cont.append(line.rstrip())
    eve_dict = {}
    e_count = 1
    for line in cont:
        if re.match('\d+',line[:2]):  # event line
            yr = int(line[:2])
            mo = int(line[2:4])
            day = int(line[4:6])
            hr = int(line[7:9])
            minite = int(line[9:11])
            seconds = float(line[12:17])
            
            lat = float(line[18:25])
            if line[25]=="S":
                lat = -lat
            lon = float(line[27:35])
            if line[35]=="W":
                lon = -lon
            dep = float(line[36:42])
            rms = float(line[63:67])

        elif line[:2] == "  " or line=='': # the last line
            if line[:2] == "  ": # line with event id contained
                evid = int(line[2:])
            else:
                evid = e_count
                e_count += 1
            eve_dict[evid] = [lon,lat,dep,rms]
    return eve_dict

def cnv_add_evid(vel_in,vel_out="velout.cnv",output_file="velout.cnv.id"):
    """
    The input VELEST cnv file contains event id information which could be 
    processed by VELEST modified by Hardy ZI.
    The output processed by VELEST contains no event id information,
    needs to be addded.
    """
    with open(vel_in,"r") as f:
        vel_in_cont = f.readlines()
    f.close()
    with open(vel_out,"r") as f:
        vel_out_cont = f.readlines()
    f.close()

    f = open(output_file,'w')

    for i,line in enumerate(vel_out_cont):
        if len(vel_out_cont[i])!=1:
            f.write(vel_out_cont[i])
        else:
            f.write(vel_in_cont[i])
    f.close()

def velestmod2ddinv(in_file="velout.mod"):
    """
    Load velocity structure from VELEST *.mod output file to HYPOINVERSE and 
    HYPODD velocity format
    """
    [P_lays,P_vels,S_lays,S_vels] = load_velest_mod(in_file)

    if P_lays != S_lays:
        print("Warning: Layers of P and S are different")
        print(P_lays)
        print(S_lays)
    
    dd_out_file = in_file+".dd"
    f = open(dd_out_file,'w')

    for lay in P_lays:
        f.write(format(lay,"6.3f")+" ")
    f.write("-9\n")

    for vel in P_vels:
        f.write(format(vel,"6.3f")+" ")
    f.write("-9\n")

    for i in range(len(P_vels)):
        f.write(format(P_vels[i]/S_vels[i],'6.3f')+" ")
    f.write("-9\n")
    f.close()

    inv_P_file = in_file+".Pinv"
    f = open(inv_P_file,'w')
    f.write("Velest generated model\n")
    for i in range(len(P_vels)):
        f.write(format(P_vels[i],"5.2f")+" "+format(P_lays[i],'5.2f')+'\n')
    f.close()
    inv_S_file = in_file+".Sinv"
    f = open(inv_S_file,'w')
    f.write("Velest generated model\n")
    for i in range(len(S_vels)):
        f.write(format(S_vels[i],"5.2f")+" "+format(S_lays[i],'5.2f')+'\n')
    f.close()

def load_velest_mod(in_file):
    """
    Read in velest input and output velocity model
    Return an array containing four lists:
    [P_lays,P_vels,S_lays,S_vels]  
    """
    lines = []
    with open(in_file,'r') as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    f.close()

    P_lays = []
    P_vels = []
    S_lays = []
    S_vels = []

    P_lay_qty = int(lines[1].split()[0])
    for line in lines[2:2+P_lay_qty]:
        _vel,_dep,_damp = re.split(" +",line)[:3]
        vel = float(_vel); dep = float(_dep)
        P_vels.append(vel); P_lays.append(dep)

    S_lay_qty = int(lines[2+P_lay_qty])
    if P_lay_qty != S_lay_qty:
        raise Exception("The qty of P and S layers are different")

    for line in lines[3+P_lay_qty:3+P_lay_qty+S_lay_qty]:
        _vel,_dep,_damp = re.split(" +",line)[:3]
        vel = float(_vel); dep = float(_dep)
        S_vels.append(vel); S_lays.append(dep)

    return [P_lays,P_vels,S_lays,S_vels]

class Vel_iter():
    def __init__(self,iter_name):
        self.name = iter_name
        
    def dep_plot(self,ylim=[10,0]):
        """
        plot depth distribution of iteration
        """
        dep = self.deps
        hist,bins = np.histogram(dep,bins = np.arange(0,10,0.5))
        fig,axs = plt.subplots(1,1,figsize=(4,8))
        axs.xaxis.tick_top()
        axs.xaxis.set_label_position("top")
        axs.set_xlabel("Event quantity")
        cnv_bar = axs.barh(bins[:-1]+0.25,hist,height=0.5,color='green',edgecolor='k',alpha=0.5)
        axs.set_ylim(ylim)
        axs.set_ylabel("Depth (km)")
        axs.set_title("Iteration Result",fontsize=16)
        
    def dd_vpvs_format(self,dd_version="1"):
        """
        output hypoDD format velocity lines.
        """
        if dd_version=="2":
            if len(self.P_deps) != len(self.S_deps):
                raise Exception("The P layers qty not equal the S layers qty")
            if self.P_deps != self.S_deps:
                print("Warning: the P-layer and S-layer depths are different, here P layers are used.")
            for dep in self.P_deps:
                print("%6.3f "%dep,end="")
            print("-9\n",end="")
            for vp in self.P_vels:
                print("%6.3f "%vp,end="")
            print("-9\n",end="")
            for i in range(len(self.P_vels)):
                poisson_ratio = self.P_vels[i]/self.S_vels[i]
                print("%6.3f "%poisson_ratio,end="")
            print("-9\n")
        if dd_version=="1":
            for dep in self.P_deps:
                print("%6.3f "%dep,end="")
            print("\n",end="")
            for vp in self.P_vels:
                print("%6.3f "%vp,end="")
            print("\n",end="")

    def gen_inv_mod_files(self):
        inv_P_file = self.name+".Pinv"
        f = open(inv_P_file,'w')
        f.write("Velest generated model\n")
        for i in range(len(self.P_vels)):
            f.write(format(self.P_vels[i],"5.2f")+" "+format(self.P_deps[i],'5.2f')+'\n')
        f.close()
        inv_S_file = self.name+".Sinv"
        f = open(inv_S_file,'w')
        f.write("Velest generated model\n")
        for i in range(len(self.S_vels)):
            f.write(format(self.S_vels[i],"5.2f")+" "+format(self.S_deps[i],'5.2f')+'\n')
        f.close()
        
    def gen_del_files(self,sta_file="/home/zijinping/Desktop/zijinping/resources/stations/sta_sum_202109.txt"):
        """
        Generate station delay files
        """
        sta_dict = load_sta(sta_file)
        iter_name = self.name
        fp = open("P.dly","w")
        fs = open("S.dly","w")
        for sta in self.P_dly.keys():
            # check whether there are stations in different network with the same name
            count = 0
            for tmp_net in sta_dict.keys():
                for tmp_sta in sta_dict[tmp_net].keys():
                    if tmp_sta == sta:
                        count += 1
                        net = tmp_net
            if count > 1:
                print(f"There are several stations with the same station name: {sta}")
                print("The programme couldn't decide the correct network name.")
                print("Please input the correct network name:")
                net = input()
            if count == 0:
                print(f"Station not included in the station file and couldn't get network name")
                net = "  "
            fp.write(f"{format(sta,'5s')} {format(net,'2s')} {format(self.P_dly[sta],'5.2f')}\n")
            fs.write(f"{format(sta,'5s')} {format(net,'2s')} {format(self.S_dly[sta],'5.2f')}\n")
        fp.close()
        fs.close()
                
class Velest():
    """
    The os.getcwd() should be the same with the velest programme working directory 
    """
    def __init__(self,cmn_file = "velest.cmn"):
        self.cmn_file = cmn_file
        self.o_mod = "velout.mod"
        self.load_parameter()
        self.load_i_mod()
        self.iters = []
        self.load_log()
    def __getitem__(self,index):
        return self.iters[index]
    def load_i_mod(self):
        tmp = load_velest_mod(self.i_mod)
        self.i_P_deps = tmp[0]
        self.i_P_vels = tmp[1]
        self.i_S_deps = tmp[2]
        self.i_S_vels = tmp[3]
    def load_parameter(self):
        cont = []
        line_no = 1
        with open(self.cmn_file,'r') as f:
            for line in f:
                line = line.rstrip()
                if len(line)>0 and line[0]=="*":
                    continue
                cont.append(line)
                line_no += 1
        self.i_mod = cont[11]           # input velocity structure
        self.i_sta = cont[12]           # input station file
        self.i_cnv = cont[18]           # input phase file
        
        if len(cont[20].split()) == 0:  # empty line
            self.log_file = "vel.out"
        else:
            self.log_file = cont[20]
        if len(cont[22].split()) == 0:  # empty line
            self.o_cnv = "velout.cnv"
        else:
            self.o_cnv = cont[22]
        if len(cont[23].split()) == 0:  # empty line
            self.o_sta = "velout.cnv"  
        else:
            self.o_sta = cont[23]
            
    def vel_plot(self,iter_name=None,draw_input=True,xlim=[0,8],ylim=[40,0],figsize=(4,6)):
        """
        if iter_name was set, then plot corresponding iteration velocity,
        otherwise the output velocity strucutre will be plotted.
        draw_input: draw the input velocity in dashed line for comparison
        """
        if iter_name == None:
            P_deps,P_vels,S_deps,S_vels = load_velest_mod(self.o_mod)
        else:
            for i,iteration in enumerate(self.iters):
                if iter_name == iteration.name:
                    P_deps = iteration.P_deps
                    P_vels = iteration.P_vels
                    S_deps = iteration.S_deps
                    S_vels = iteration.S_vels
        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("Depth (km)",fontsize=16)
        ax.set_xlabel("Velocity (km/s)",fontsize=16)
        ax.set_ylim(ylim)
        if draw_input == True:
            draw_vel(ax,self.i_P_deps,self.i_P_vels,linestyle="--")
            draw_vel(ax,self.i_S_deps,self.i_S_vels,linestyle="--")
        draw_vel(ax,P_deps,P_vels,color='b')
        draw_vel(ax,S_deps,S_vels,color="b")
        plt.show()
        
    def res_plot(self):
        """
        Plot residual of each iteration
        """
        name_list = []
        res_list = []
        for iteration in self.iters:
            name_list.append(iteration.name)
            res_list.append(iteration.residual)
        fig,ax = plt.subplots(1,1,figsize=(6,6))
        x_list = np.arange(0,len(self.iters),1)
        ax.plot(x_list,res_list)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")
        gap = int(len(x_list)/6)
        plt.xticks(x_list[::gap],name_list[::gap])
        plt.grid()
        plt.show()
        return name_list,res_list

    def rms_plot(self):
        """
        Plot residual of each iteration
        """
        name_list = []
        rms_list = []
        for iteration in self.iters:
            name_list.append(iteration.name)
            rms_list.append(iteration.rms)
        fig,ax = plt.subplots(1,1,figsize=(6,6))
        x_list = np.arange(0,len(self.iters),1)
        ax.plot(x_list,rms_list)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")
        gap = int(len(x_list)/6)
        plt.xticks(x_list[::gap],name_list[::gap])
        plt.grid()
        plt.show()
        return name_list,rms_list
        
    def load_log(self):
        """
        load information from the log(vel.out) file
        """
        cont = []
        with open(self.log_file,'r') as f:
            for line in f:
                cont.append(line.rstrip())
        f.close()
        i = 0
        while i < len(cont):
            line = cont[i]
            if len(re.split("Iteration",line))>=2:
                line_splits = re.split(" +",line)
                ele_qty = len(line_splits)
                if ele_qty == 5:
                    iter_name = "iter"+line_splits[3]
                if ele_qty == 7:
                    iter_name = "iter"+line_splits[2][:-1]+"B"+line_splits[5]
                new_iter = Vel_iter(iter_name)
                self.iters.append(new_iter)
                residual_line = cont[i+1]
                tmp = residual_line.split()
                new_iter.datavar = float(tmp[1])
                new_iter.residual = float(tmp[5])
                new_iter.rms = float(tmp[8])
                i=i+1
            #-------------------------------------------------------------------------------
            if len(re.split("Velocity model   1",line))>=2:
                j = i+1
                new_iter.P_vels = []
                new_iter.P_deps = []
                while cont[j]!="":
                    _vel,_,_dep = cont[j].split()
                    vel = float(_vel)
                    dep = float(_dep)
                    new_iter.P_vels.append(vel)
                    new_iter.P_deps.append(dep)
                    j=j+1
            #-------------------------------------------------------------------------------
            if len(re.split("Velocity model   2",line))>=2:
                j = i+1
                new_iter.S_vels = []
                new_iter.S_deps = []
                while cont[j]!="":
                    _vel,_,_dep = cont[j].split()[:3]
                    vel = float(_vel)
                    dep = float(_dep)
                    new_iter.S_vels.append(vel)
                    new_iter.S_deps.append(dep)
                    j=j+1 
                i=j
            #-------------------------------------------------------------------------------
            if line == "   stn  ptcor  dpcor":
                j = i+1            # move to the P delay lines
                new_iter.P_dly = {}
                while cont[j]!=" Adjusted station corrections:": # is the start of next part
                    tmp = cont[j].split()
                    sta_qty = len(tmp)/3
                    k=0
                    while k < sta_qty:
                        sta = tmp[k*3+0]
                        delay = float(tmp[k*3+1])
                        new_iter.P_dly[sta]=delay
                        k = k+1
                    j = j+1
            #-------------------------------------------------------------------------------
            if line == "   stn  stcor  dscor":
                j = i+1                            # move to the S delay lines
                new_iter.S_dly = {}
                while cont[j]!="":                 # "" is the end
                    tmp = cont[j].split()          # split values
                    sta_qty = len(tmp)/3           # how many stations
                    k=0                            # denotes the k_th station of line
                    while k < sta_qty:
                        sta = tmp[k*3+0]           # station name
                        delay = float(tmp[k*3+1])  # delay value
                        new_iter.S_dly[sta]=delay  # save value
                        k = k+1                    # move to next station
                    j = j+1                        # move to next line
                i = j                              # reset line indicator
                
            #-------------------------------------------------------------------------------
            if line=="  eq       ot     x      y      z      rms    avres   dot     dx     dy     dz":
                j = i+1
                new_iter.deps=[]
                while cont[j]!="" and cont[j][:5]!="     " :
                    if cont[j][:7]== " ***** ":
                        j = j+1
                        continue
                    tmp = cont[j].split()
                    new_iter.deps.append(float(tmp[4]))
                    j = j + 1
                i = j
            i = i+1
    
    def __len__(self):
        return len(self.iters)

