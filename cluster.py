import os
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.cluster.hierarchy import average, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from seisloc.dd import load_DD
import pickle
from scipy.cluster.hierarchy import fcluster

def load_dtcc(dtcc="dt.cc",save_pkl=False,phases=['P','S'],minobs=4,mean_cc_threshold=0.7):
    """
    If the provided dtcc file is a pkl file, then pickle.load() will be used for loading data
    
    The programme reads in {dtcc} file and calculates average cc value for designated {phases}.
    The average cc value constrain the similarity of two events.
    
    Parameters:
        if save_pkl: a pkl file will be saved with the name {dtcc}.cluster.pkl
                     please note if dtcc file is a .pkl file, even save_pkl is True, it will not 
                     generate a new pkl file.
    """
    if dtcc[-4:] == ".pkl":                         # .pkl file load by pickle
        f = open(dtcc,'rb')
        pair_list = pickle.load(f)
        
    else:                                           # using normal processing
        pair_dict = {}
        i = 0
        f = open(dtcc,'r')
        for line in f:
            line = line.rstrip()
            if line[0] == "#":
                _,_evid1,_evid2,_ = re.split(" +",line)
                evid1 = int(_evid1); evid2 = int(_evid2)
                evid1 = min([evid1,evid2])
                evid2 = max([evid1,evid2])
            else:
                netsta,_diff,_cc,pha = re.split(" +",line)
                try: 
                    pair_dict[evid1][evid2].append(float(_cc))
                except:
                    try:
                        pair_dict[evid1][evid2]=[]
                        pair_dict[evid1][evid2].append(float(_cc))
                    except:
                        try:
                            pair_dict[evid1] = {}
                            pair_dict[evid1][evid2]=[]
                            pair_dict[evid1][evid2].append(float(_cc))
                        except:
                            pass
                            
        pair_list = []
        for key1 in pair_dict.keys():
            for key2 in pair_dict[key1].keys():
                if len(pair_dict[key1][key2])<minobs:
                    continue
                mean_cc = np.mean(pair_dict[key1][key2])
                if  mean_cc < mean_cc_threshold:
                    continue
                pair_list.append([key1,key2,mean_cc])
                
        return pair_list
        
        if save_pkl:                              # generate pkl file to speed loading in future
            dtcc_pkl = dtcc+".clustering.pkl"
            with open(dtcc_pkl,'wb') as f:
                pickle.dump(pair_list,f)

    return pair_list

class Eqcluster():
    def __init__(self,
                 loc_file="hypoDD.reloc",
                 dtcc_file="dt.cc",
                 dtcc_save_pkl=True,
                 dtcc_phases=['P','S'],
                 dtcc_minobs=4,
                 dtcc_mean_cc_threshold=0.7):
        
        self.pairs = load_dtcc(dtcc_file,
                               save_pkl=dtcc_save_pkl,
                               phases=dtcc_phases,
                               minobs = dtcc_minobs,
                               mean_cc_threshold=dtcc_mean_cc_threshold)
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
        self.cc_matrix = np.zeros((len(self.cc_evids),len(self.cc_evids)))
        for pair in self.pairs_in:
            evid1,evid2,cc = pair
            k1 = np.where(self.cc_evids == evid1)
            k2 = np.where(self.cc_evids == evid2)
            self.cc_matrix[k1[0],k2[0]] = cc
            self.cc_matrix[k2[0],k1[0]] = cc
            
    def clustering(self,evids=[],tolerance=0,method='average'):
        """
        Parameters:
        method: please refer to scipy.cluster.hierarchy.linkage
        tolerance: Event with inter-event cc pairs qty lower than tolerance will not be included in clustering
        """
        # build cluster matrix
        if len(evids) == 0:       
            self.input_evids = self.evids.copy()
            self.cluster_matrix = self.cc_matrix.copy()
            self.in_matrix_evids = self.cc_evids.copy()
        else:
            self.input_evids = evids
            idxs = []
            for evid in self.input_evids:
                idx = np.where(self.cc_evids==evid)
                idxs.append(idx[0][0])
            self.update_cluster(idxs)
        
        # apply filter with tolerance
        print(">>> Tolerance filter applied")
        tmp = (self.cluster_matrix > 0)
        similar_qty_arr = np.sum(tmp,axis=0)
        kk = np.where(similar_qty_arr>=tolerance)
        self.update_cluster(kk[0])
        print(f"len evids is:{len(self.input_evids)}; ",\
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
            tmp.append(self.cc_evids[int(idx)])
        self.in_matrix_evids = tmp
        self.cluster_matrix = self.cc_matrix[idxs]
        self.cluster_matrix = self.cluster_matrix[:,idxs]
        
        self.out_matrix_evids = []
        for evid in self.input_evids:
            if evid not in self.in_matrix_evids:
                 self.out_matrix_evids.append(evid) 
          
    def dendrogram(self,xlim=[],figsize=(8,6)):
        fig,ax = plt.subplots(1,figsize=figsize)
        if len(xlim)>0:
            ax.set_xlim(xlim)
        self.dn = dendrogram(self.Z)
        
    def fancy_dendrogram(self,
                         truncate_mode='lastp',
                         p=12,
                         show_contracted=True,
                         max_d=None,
                         annotate_above=0,
                         xytext=(0,-1),
                         no_plot=False):
        """
        https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
        """       
        self.dn = dendrogram(self.Z,
                             truncate_mode=truncate_mode,
                             p=p,
                             show_contracted=show_contracted)
        
        if not no_plot:
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel("sample index or (cluster size)")
            plt.ylabel("distance")
            
            for i,d,c in zip(self.dn['icoord'],self.dn['dcoord'],self.dn['color_list']):
                x = 0.5 * sum(i[1:3])      # "x1, x2, x3, x4", here 'x2,x3'
                y = d[1]                   # "y1, y2, y3, y4", here 'y2'
                if y > annotate_above:
                    plt.plot(x,y,'o',c=c)
                    plt.annotate("%.3g"% y, (x,y),xytext=xytext,
                                textcoords='offset points',
                                va = 'top',
                                ha = 'center')
        
    def heatmap(self,source='cluster',xlim=[],ylim=[],figsize=(5,5)):
        fig,ax = plt.subplots(1,figsize=figsize)
        if source == "dd":
            disp_matrix = self.cc_matrix
            for i in range(disp_matrix.shape[0]):
                disp_matrix[i,i] = 1
            
        elif source == "cluster":            
            disp_matrix = self.cluster_matrix.copy()
            for i in range(disp_matrix.shape[0]):
                disp_matrix[i,i] = 1
            disp_matrix = disp_matrix[self.dn['leaves']]
            disp_matrix = disp_matrix[:,self.dn['leaves']]
            
        if len(xlim)>0:
            plt.xlim(ylim)
        if len(ylim)>0:
            plt.ylim(ylim)
        plt.pcolormesh(disp_matrix)
        plt.gca().set_aspect('equal')
        plt.xlabel("Event No")
        plt.ylabel("Event No")

    def mapview(self,max_d,criterion='distance',xlim=[],ylim=[],figsize=(10,5),mag_base=1):
        """
        Showing corresponding map view of earthquake locations
        """
        fig,ax = plt.subplots(1,figsize=figsize)
        clusters = fcluster(self.Z,max_d,criterion=criterion)
        print("Num of clusters: ",len(set(clusters)))
        lons = []
        lats = []
        mags = []
        for evid in self.in_matrix_evids:
            lons.append(self.dd_dict[evid][0])
            lats.append(self.dd_dict[evid][1])
            mags.append(self.dd_dict[evid][3])
        mags = np.array(mags)
        plt.scatter(lons,lats,c=clusters,s=mags-np.min(mags)+mag_base,cmap='tab20')
        if len(xlim)>0:
            plt.xlim(xlim)
        if len(ylim)>0:
            plt.ylim(ylim)
        ax.set_aspect('equal')
