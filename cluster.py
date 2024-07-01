import pandas as pd
import re
import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import average, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from seisloc.dd import load_DD
from obspy.geodetics import gps2dist_azimuth
import pickle
from seisloc.scc import load_dtcc

class EqCluster():
    def __init__(self,
                 loc_file="hypoDD.reloc",
                 dtcc_file="dt.cc",
                 dtcc_phases=['P','S'],
                 dtcc_minobs=4,
                 dtcc_mean_cc_threshold=0.7,
                 distWt=0):
        self.cc_threshold = dtcc_mean_cc_threshold


        self.pairs = self.dtcc_pairs(dtcc_file,usePhases=dtcc_phases,minObs=dtcc_minobs,ccThred=dtcc_mean_cc_threshold)
        self.dd_dict,_ = load_DD(loc_file)

        self.evids = np.array(list(self.dd_dict.keys())) # all event ids in the reloc file
        self.evids.sort()
        self.evids.astype(int)

        self.cc_evids = []                  # evids constrained by cross-correlation
        self.pairs_in = []
        for pair in self.pairs:
            evid1,evid2,cc = pair
            k1 = np.where(self.evids == evid1)
            k2 = np.where(self.evids == evid2)
            if len(k1[0])==1 and len(k2[0])==1:
                self.cc_evids.append(evid1)
                self.cc_evids.append(evid2)
                self.pairs_in.append(pair)
            if len(k1[0])>1 and len(k2[0])>1:
                raise Exception("There are duplicated event ids")
        self.cc_evids = np.array(list(set(self.cc_evids)))
        self.cc_evids.sort()

        # building cc_matrix
        self.cc_matrix = np.ones((len(self.cc_evids),len(self.cc_evids)))
        self.cluster_matrix = np.ones((len(self.cc_evids),len(self.cc_evids)))
        for i in range(self.cc_matrix.shape[0]):
            self.cc_matrix[i,i] = 0
            self.cluster_matrix[i,i] = 0
        if distWt>0:
            for i in range(len(self.cc_evids)):
                evid1 = self.cc_evids[i]
                for j in range(i,len(self.cc_evids)):
                    evid2 = self.cc_evids[j]
                    loEvid1 = self.dd_dict[evid1][0]
                    laEvid1 = self.dd_dict[evid1][1]
                    dpEvid1 = self.dd_dict[evid1][2]
                    loEvid2 = self.dd_dict[evid2][0]
                    laEvid2 = self.dd_dict[evid2][1]
                    dpEvid2 = self.dd_dict[evid2][2]
                    dist12,_,_ = gps2dist_azimuth(laEvid1,loEvid1,laEvid2,loEvid2)
                    dist3D = np.sqrt((dist12/1000)**2 + (dpEvid1-dpEvid2)**2)
                    self.cluster_matrix[i,j] += distWt*dist3D
                    self.cluster_matrix[j,i] += distWt*dist3D
        for pair in self.pairs_in:
            evid1,evid2,cc = pair
            k1 = np.where(self.cc_evids == evid1)
            k2 = np.where(self.cc_evids == evid2)
            self.cc_matrix[k1[0],k2[0]] -= cc
            self.cc_matrix[k2[0],k1[0]] -= cc
            self.cluster_matrix[k1[0],k2[0]] -= cc
            self.cluster_matrix[k2[0],k1[0]] -= cc

        self.in_matrix_evids = self.cc_evids.copy()


    def dtcc_pairs(self, dtccPth, usePhases=["P", "S"], minObs=4, ccThred=0.7):
        """
        Return a list of pairs with cc value higher than ccThred [evid1, evid2, meanCc]
        """
        pairs = []
        df = load_dtcc(dtccPth)
        evid1s = np.unique(df["evid1"])
        for evid1 in evid1s:
            dfEvid1 = df[df["evid1"] == evid1]
            evid2s = np.unique(dfEvid1["evid2"])
            for evid2 in evid2s:
                ccvs = []
                dfEvid1Evid2 = dfEvid1[dfEvid1["evid2"] == evid2]
                for i, tmpDf in dfEvid1Evid2.iterrows():
                    pha = tmpDf["pha"]
                    if pha.upper() in usePhases or pha.lower() in usePhases:
                        ccvs.append(tmpDf["cc"])
                if len(ccvs) < minObs:
                    continue
                meanCc = np.mean(ccvs)
                if meanCc < ccThred:
                    continue
                pairs.append([evid1, evid2, meanCc])
        return pairs


    def clustering(self,evids=[],tolerance=0,method='average'):
        """
        Parameters:
        method: please refer to scipy.cluster.hierarchy.linkage
        tolerance: Event with inter-event cc pairs qty lower than tolerance will not be included in clustering
        """
        # build cluster matrix
        if len(evids) == 0:
            self.input_evids = self.evids.copy()
        else:
            self.input_evids = evids
            idxs = []
            for evid in self.input_evids:
                idx = np.where(self.in_matrix_evids==evid)
                idxs.append(idx[0][0])
            self.update_cluster(idxs)

        # apply filter with tolerance
        print(">>> Tolerance filter applied")
        tmp = (self.cluster_matrix < 1)  # True and False matrix
        similar_qty_arr = np.sum(tmp,axis=0)
        kk = np.where(similar_qty_arr>=tolerance)
        self.update_cluster(kk[0])
        print(f"len evids is:{len(self.input_evids)}; ",\
              f"len cc_evids is: {len(self.cc_evids)}; "\
              f"len matrix is: {len(self.in_matrix_evids)};",end=' ')
        print(f"len events not in matrix: {len(self.out_matrix_evids)}")

        y = pdist(self.cluster_matrix)
        self.Z = linkage(y,method=method)

    def update_cluster(self,idxs):
        """
        update cluster matrix and cluster evids
        """
        tmp = []
        for idx in idxs:
            tmp.append(self.in_matrix_evids[int(idx)])
        self.in_matrix_evids = tmp
        self.cluster_matrix = self.cluster_matrix[idxs]
        self.cluster_matrix = self.cluster_matrix[:,idxs]
        
        self.out_matrix_evids = []
        for evid in self.input_evids:
            if evid not in self.in_matrix_evids:
                 self.out_matrix_evids.append(evid)
        
        self.in_matrix_evids = np.array(self.in_matrix_evids)
        self.out_matrix_evids = np.array(self.out_matrix_evids)
          
    def dendrogram(self,xlim=[],figsize=(8,6),max_d=None,leaf_rotation=0):
        fig,ax = plt.subplots(1,figsize=figsize)
        if len(xlim)>0:
            ax.set_xlim(xlim)
        self.dn = dendrogram(self.Z,leaf_rotation=leaf_rotation)

        if max_d != None:
            plt.axhline(y=max_d, c='k')

        plt.xlabel("sample index")
        plt.ylabel("distance")

    def fancy_dendrogram(self,
                         truncate_mode='lastp',
                         p=12,
                         show_contracted=True,
                         max_d=None,
                         annotate_above=0,
                         leaf_rotation=0,
                         xytext=(0,-1),
                         no_plot=False):
        """
        https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
        """       
        self.dn2 = dendrogram(self.Z,
                             truncate_mode=truncate_mode,
                             p=p,
                             show_contracted=show_contracted,
                             leaf_rotation=leaf_rotation)
        
        if not no_plot:
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel("sample index")
            plt.ylabel("distance")
            
        if max_d != None:
            plt.axhline(y=max_d, c='k')
            
            for i,d,c in zip(self.dn2['icoord'],self.dn2['dcoord'],self.dn2['color_list']):
                x = 0.5 * sum(i[1:3])      # "x1, x2, x3, x4", here 'x2,x3'
                y = d[1]                   # "y1, y2, y3, y4", here 'y2'
                if y > annotate_above:
                    plt.plot(x,y,'o',c=c)
                    plt.annotate("%.3g"% y, (x,y),xytext=xytext,
                                textcoords='offset points',
                                va = 'top',
                                ha = 'center')
            
    def heatmap(self,source='cluster',xlim=[],ylim=[],figsize=(5,5),showGroups=True):
        """
        Parameters:
        source: 'dd' to show the orignal time-sequence heatmap
                'cluster' to show the clustered heatmap
        xlim: set plt.xlim if not empty
        ylim: set plt.ylim if not empty
        """
        fig,ax = plt.subplots(1,figsize=figsize)
        if source == "dd":
            disp_matrix = self.cc_matrix
            showGroups = False
            
        elif source == "cluster":            
            self.clustered_matrix = self.cluster_matrix.copy()
            self.clustered_matrix = self.clustered_matrix[self.dn['leaves']]
            self.clustered_matrix = self.clustered_matrix[:,self.dn['leaves']]
            disp_matrix=self.clustered_matrix.copy()
        disp_matrix = 1-disp_matrix   #clustering is conducted by 1-cc; reverse back to normal cc value
        for i in range(disp_matrix.shape[0]):
            disp_matrix[i,i] = 0
        if len(xlim)>0:
            plt.xlim(xlim)
        if len(ylim)>0:
            plt.ylim(ylim)
        self.group_ticks = []
        colors = ["lightgrey",'lightgrey','blue','red']
        nodes = [0,self.cc_threshold,self.cc_threshold,1]
        cmap = LinearSegmentedColormap.from_list("mycmap",list(zip(nodes,colors)))
        plt.pcolormesh(disp_matrix,cmap=cmap,vmin=0,vmax=1)
        if showGroups == True:
            tmpSum=0
            for i in range(self.catQty):
                kks = np.where(self.cluster_category==(i+1))
                print(f"*** Cluster {i+1} has {len(kks[0])} events!")
                tmpSum+=len(kks[0])
                self.group_ticks.append(tmpSum)
                plt.plot([tmpSum,tmpSum],[0,tmpSum],'g-',lw=1)
        plt.gca().set_aspect('equal')
        plt.xlabel("Event No")
        plt.ylabel("Event No")
        plt.colorbar()

    def fcluster(self,max_d,criterion='distance'):
        """
        Categorize clusters and generate self.clusters_evids dict
        refer to: scipy.cluster.hierarchy.fcluster¶
        """
        self.cluster_category = fcluster(self.Z,max_d,criterion=criterion)
        self.catQty = len(set(self.cluster_category))
        print("Num of clusters: ",self.catQty)
        self.clusters_evids = {}

        for cluster_id in np.unique(self.cluster_category):
            self.clusters_evids[cluster_id] = self.get_cluster_evids(cluster_id)

    def mapview(self,clusters=[],xlim=[],ylim=[],figsize=(10,5),mag_base=1):
        """
        Showing corresponding map view of earthquake locations
        Parameters:
        |   clusters: empty list for plot all, otherwide plot clusters in the list
        |       xlim: if not empty, set plt.xlim
        |       ylim: if not empty, set plt.ylim
        |   mag_base: parameters control the magnitude size
        """
        fig,ax = plt.subplots(1,figsize=figsize)

        lons = []
        lats = []
        mags = []
               
        for evid in self.in_matrix_evids:
            lons.append(self.dd_dict[evid][0])
            lats.append(self.dd_dict[evid][1])
            mags.append(self.dd_dict[evid][3])
        self.in_matrix_lons = np.array(lons)
        self.in_matrix_lats = np.array(lats)
        self.in_matrix_mags = np.array(mags)
        cmap = matplotlib.cm.get_cmap('tab20',int(np.max(self.cluster_category)))
        
        if len(clusters)==0:
            plt.scatter(self.in_matrix_lons,self.in_matrix_lats,
                        c=self.cluster_category,
                        s=self.in_matrix_mags-np.min(self.in_matrix_mags)+mag_base,
                        cmap=cmap,vmin=1,
                        vmax=int(np.max(self.cluster_category))+1)
        else:
            kks = np.array([],dtype=int)
            for cid in clusters:
                kk = np.where(self.cluster_category==cid)
                kks = np.concatenate((kks,kk[0]))
            plt.scatter(self.in_matrix_lons[kks],self.in_matrix_lats[kks],
                        c=self.cluster_category[kks],
                        s=self.in_matrix_mags[kks]-np.min(self.in_matrix_mags)+mag_base,
                        cmap=cmap,vmin=1,
                        vmax=int(np.max(self.cluster_category))+1)
                
        if len(xlim)>0:
            plt.xlim(xlim)
        if len(ylim)>0:
            plt.ylim(ylim)
        ax.set_aspect('equal')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        cb = plt.colorbar()
        cb.set_label("Cluster")
        
    def get_cluster_evids(self,cluster_id):
        kk = np.where(self.cluster_category==cluster_id)
        sel_cluster_evids = self.in_matrix_evids[kk[0]]   
        
        return sel_cluster_evids
    
    def write_info(self,fileName="evid_cid.EqCluster"):
        """
        Write the cluster information to a file
        """
        with open(fileName,"w") as f:
            f.write("#Cluster info\n")
            f.write(f"#Num of clusters: {self.catQty}\n")
            for i,evid in enumerate(self.in_matrix_evids):
                f.write(f"{format(evid,'6d')} {self.cluster_category[i]}\n")
            for evid in self.out_matrix_evids:
                f.write(f"{format(evid,'6d')} 0\n")
        print("[EqCluster.write_info] output filename: evid_cid.EqCluster") 
    def __repr__(self):
        str1 = f"Eqcluster object with {len(self.evids)} events in hypoDD relocation file\n"
        str2 = f"{len(self.evids)} events constrained by dt.cc file"
        return str1+str2
    
    def copy(self):
        return copy.deepcopy(self)
