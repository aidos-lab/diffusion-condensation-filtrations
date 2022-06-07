# Import Modules

import numpy as np
import pandas as pd
import scipy as sp
import dionysus as d
import matplotlib.pyplot as plt
#import pecan as pc
import sys
import os


#from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


from sklearn.metrics.pairwise import euclidean_distances

import networkx as nx

import matplotlib.pyplot as plt
import argparse
import sys

import matplotlib.collections
import matplotlib.lines
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import gudhi as gd  

import time

from basic_functions import *
from visualizations import *



class OperatorPowerFiltrationHomogeneous:
    """
    Class to generate a filtration over the power tau of a diffusion operator
    """
    def __init__(self, 
                experiment_path,
                experiment_name,
                mode,
                kernel,
                e, 
                b,
                t):
        """
        Args:
            experiment_path [string]: path to the PECAN .npz file.
            experiment_name [string]: name of the PECAN .npz file .
            mode [string]: resting / non_resting. - resting or non-resting random walk
            kernel [string]: 'gaussian_aniso' / 'gaussian' (gaussian)
            e [float]: kernel bandwith 
            b [float]: gaussian_aniso exp
            t [int]: diffusion condensation time

        """
        # load diffused data sets
        self.data=np.load(experiment_path+experiment_name+'.npz')
        parsed_keys = parse_keys(self.data)
        assert 'data' in parsed_keys
        self.X = make_tensor(self.data, parsed_keys['data'])
        T = self.X.shape[-1]
        N=self.X.shape[0]
        name_t=str('data_t_'+str(t))
        self.X_t=self.data[name_t]
        # create diffusion operator
        if mode=='resting':
            K=kernel_gaussian_resting(self.X_t,e)
        if mode=='non_resting':
            K=kernel_gaussian_nonresting(self.X_t,e)
        if kernel== 'gaussian_aniso':
            K_=kernel_gaussian_aniso_norm(K,b)
            K=K_
        if kernel== 'gaussian':
            K=K
        self.P_t=diffusion_operator(K)
        self.N=len(self.P_t)
        self.experiment_path=experiment_path
        self.experiment_name=experiment_name
        self.mode=mode
        self.kernel=kernel
        self.e=e
        self.b=b
        self.t=t
        
    def compute_edge_weights(self, tau_max):
        """
        Function to compute all N_E=(N c 2) edge weights for all steps tau of the filtration. Result stored in
        tau_max x N_E matrix W. Order of elements in a row of W according to upper triangle matrix of P_t. Edge weights
        are mean of both corresponding elements of P_t^tau: 0.5*(P_t[i][j]+P_t[j][i])
        Args:
            tau_max [int]: filtration from tau=1 to tau=tau_max
        """
        P_t_tau_flat=upper_triu(self.P_t).flatten()
        P_t_tau_flat_nonzero=P_t_tau_flat[P_t_tau_flat>0]
        N_E=len(P_t_tau_flat_nonzero)

        self.W_tensor=np.zeros((tau_max,N_E))
        self.W_tensor[0]=P_t_tau_flat_nonzero
        P_t_tau=self.P_t
        for tau in range(1,tau_max):
            P_t_tau=P_t_tau@self.P_t
            P_t_tau_mean=0.5*(P_t_tau+P_t_tau.T)
            P_t_tau=P_t_tau_mean
            P_t_tau_flat=upper_triu(P_t_tau).flatten()
            P_t_tau_flat_nonzero=P_t_tau_flat[P_t_tau_flat>0]
            self.W_tensor[tau]=P_t_tau_flat_nonzero
        return self.W_tensor
    

    def compute_filtration(self,weight_threshold,dist_threshold,tau_max, max_dimension,plot_embedding='fix',merge=False,show_fig=True,save_fig=True,save_path=''):
        
        """
        Args:
            weight_threshold (float): minimum edge weight
            dist_threshold (float): distance merge threshold 
            tau_max [int]: filtration from tau=1 to tau=tau_max
            max_dimension (int): graph expansion until this given dimension.
            plot_embedding (string): visualization of data: fix - at the fix diffsion time t choosen above / var - time variable
            merge : False - do not merge data points closter than weight_threshold / True - do not merge data points closter than weight_threshold
            save_path (string): path to the storage of the visualizations of the simplicial complices
        """
        
        P_t_tau=self.P_t
        betti_tensor=np.zeros(3)
        simplices_tensor=np.zeros(3)
        simplex_list=[]
        time_list=[]
        for f in range(1,tau_max+1):
            if f==1:
                P_t_tau=self.P_t
            else:
                P_t_tau=P_t_tau@self.P_t
            
            P_t_tau_mean=0.5*(P_t_tau+P_t_tau.T)
            P_t_tau=P_t_tau_mean
            if merge==True:
                name=str('data_t_'+str(f-1))
                X_plot=self.data[name]
                dist=sp.spatial.distance.cdist(X_plot,X_plot)
                tf_dist=dist<=dist_threshold
                tf_weight=P_t_tau>=weight_threshold
                tf=np.logical_or(tf_dist,tf_weight)
            if merge==False:
                tf=P_t_tau>=weight_threshold
            A_list=(np.where(tf))   
            A_bol_1=np.zeros((self.N,self.N))
            for v in range(0,len(A_list[0])):
                A_bol_1[A_list[0][v]][A_list[1][v]]=1
            np.fill_diagonal(A_bol_1,0)
            G_f = nx.from_numpy_matrix(A_bol_1)
            T_f=find_triangles(A_bol_1)
            G_f_list=list(G_f.edges())
            st = gd.SimplexTree()
            for r in range(0,self.N):
                st.insert([r],0)  #-1
            for u in range(0,len(G_f_list)):
                G_u=G_f_list[u]
                st.insert([G_u[0],G_u[1]])
            st.expansion(max_dimension)
            sim_0=0
            if f==1:
                st_gen = st.get_skeleton(2) 
                for splx in st_gen :
                    simplex_list.append(list(sorted(splx[0])))
                    time_list.append([1])
                    if len(splx[0])==1:
                        sim_0=sim_0+1
                        
            else:
                simplex_t=[]
                st_gen = st.get_skeleton(2) 
                for splx in st_gen :
                    simplex_t.append(list(sorted(splx[0])))
                    if len(splx[0])==1:
                        sim_0=sim_0+1
                simplex_list,time_list=BarCodesUpdate(simplex_list,time_list,simplex_t,f)
                
            BarCodes=st.persistence(min_persistence=-1,persistence_dim_max=True)
            betti_f=(st.betti_numbers())
            if len (betti_f)==3:
                betti_tensor=np.vstack((betti_tensor,betti_f))
            if len (betti_f)==2:
                betti_tensor=np.vstack((betti_tensor,np.array([betti_f[0],betti_f[1],0])))
            if len (betti_f)==1:
                betti_tensor=np.vstack((betti_tensor,np.array([betti_f[0],0,0])))
            simplices_tensor=np.vstack((simplices_tensor,np.array([len(G_f.nodes()),len(G_f.edges()),len(T_f)])))
            print('τ=',f)
            if show_fig == True:
                if plot_embedding=='fix':
                    X_plot=self.X_t
                if plot_embedding=='var':
                    name=str('data_t_'+str(f-1))
                    X_plot=self.data[name]
                    
                plt.figure(figsize=(8,8))
                plt.xlim(-1.5,1.5)
                plt.ylim(-1.5,1.5)
                
                for z in range(0,len(T_f),1):  
                    plt.fill(*X_plot[get_tf(T_f[z],self.N)].T,alpha=0.1,c='r')
                nx.draw(G_f, X_plot,node_size=15,width=0.5,edge_color='k',node_color='k')
                if save_fig == True:
                    if not os.path.exists(save_path+"/"+self.experiment_name):
                        os.mkdir(save_path+"/"+self.experiment_name)
                    if not os.path.exists(save_path+"/"+self.experiment_name+"/homogeneous/"):
                        os.mkdir(save_path+"/"+self.experiment_name+"/homogeneous/")
                    if not os.path.exists(save_path+"/"+self.experiment_name+"/homogeneous/simplicial_complex/"):
                        os.mkdir(save_path+"/"+self.experiment_name+"/homogeneous/simplicial_complex/")   
                    if not os.path.exists(save_path+"/"+self.experiment_name+"/homogeneous/simplicial_complex/"+str(weight_threshold).replace(".", "_")):
                        os.mkdir(save_path+"/"+self.experiment_name+"/homogeneous/simplicial_complex/"+str(weight_threshold).replace(".", "_")) 
         
                        
                    if merge == True:
                        plt.savefig(save_path+"/"+self.experiment_name+"/homogeneous/simplicial_complex/"+str(weight_threshold).replace(".", "_")+'/simplicialcomplex_homo_'+self.experiment_name+'_e_'+str(self.e).replace(".", "_")+'_b_'+str(self.b).replace(".", "_")+'_kernel_'+str(self.kernel)+'_mode_'+str(self.mode)+'_t_'+str(self.t)+'_thres_'+str(weight_threshold).replace(".", "_")+'_tau_'+str(f+1)+'_merge_'+str(merge)+'_'+str(dist_threshold).replace(".", "_")+'_embedding_'+plot_embedding+'.png')
                    if merge == False:
                        plt.savefig(save_path+"/"+self.experiment_name+"/homogeneous/simplicial_complex/"+str(weight_threshold).replace(".", "_")+'/simplicialcomplex_homo_'+self.experiment_name+'_e_'+str(self.e).replace(".", "_")+'_b_'+str(self.b).replace(".", "_")+'_kernel_'+str(self.kernel)+'_mode_'+str(self.mode)+'_t_'+str(self.t)+'_thres_'+str(weight_threshold).replace(".", "_")+'_tau_'+str(f+1)+'_merge_'+str(merge)+'_embedding_'+plot_embedding+'.png')

                plt.show()
                if len(betti_f) >1:
                    print( 'β0='+str(betti_f[0]),'' ,'β1='+str(betti_f[1]))
                if len(betti_f) ==1:
                    print( 'β0='+str(betti_f[0]))
                print()
        f = d.Filtration(simplex_list)
        zz, dgms, cells = d.zigzag_homology_persistence(f, time_list)
        BarCodes_=[]
        for i,dgm in enumerate(dgms):
            for p in dgm:
                if i <2:
                    BarCodes_.append((i,(p.birth,p.death)))
        return betti_tensor[1:], simplices_tensor[1:],BarCodes_  
                
    def plot_betti(self,betti_tensor,merge,weight_threshold,dist_threshold,scale='linear',show_fig=True,save_fig=True,save_path=''):
        plt.figure(figsize=(10,5))
        for z in range(0,len(betti_tensor[0])):
            if z==0:
                plt.plot(betti_tensor[:,0],c='r',label=r'$\beta_{0}$')
                plt.plot(betti_tensor[:,1],c='g',label=r'$\beta_{1}$')
            else:
                plt.plot(betti_tensor[:,0],c='r')
                plt.plot(betti_tensor[:,1],c='g')
        plt.grid()
        plt.yscale(scale)
        plt.legend()
        plt.xlabel(r'$\tau$', fontsize=20)
        plt.ylabel('#', fontsize=20)
        if save_fig==True:
            if not os.path.exists(save_path+"/"+self.experiment_name):
                os.mkdir(save_path+"/"+self.experiment_name)
            if not os.path.exists(save_path+"/"+self.experiment_name+"/homogeneous/"):
                os.mkdir(save_path+"/"+self.experiment_name+"/homogeneous/")
            if not os.path.exists(save_path+"/"+self.experiment_name+"/homogeneous/features/"):
                os.mkdir(save_path+"/"+self.experiment_name+"/homogeneous/features/")   
            if not os.path.exists(save_path+"/"+self.experiment_name+"/homogeneous/features/betti_numbers/"):
                os.mkdir(save_path+"/"+self.experiment_name+"/homogeneous/features/betti_numbers/")             
  
            if merge == True:
                plt.savefig(save_path+"/"+self.experiment_name+"/homogeneous/features/betti_numbers/"+'/bettinumbers_homo_'+self.experiment_name+'_e_'+str(self.e).replace(".", "_")+'_b_'+str(self.b).replace(".", "_")+'_kernel_'+str(self.kernel)+'_mode_'+str(self.mode)+'_t_'+str(self.t)+'_thres_'+str(weight_threshold).replace(".", "_")+'_merge_'+str(merge)+'_'+str(dist_threshold).replace(".", "_")+'_'+scale+'.png')
            if merge == False:
                plt.savefig(save_path+"/"+self.experiment_name+"/homogeneous/features/betti_numbers/"+'/bettinumbers_homo_'+self.experiment_name+'_e_'+str(self.e).replace(".", "_")+'_b_'+str(self.b).replace(".", "_")+'_kernel_'+str(self.kernel)+'_mode_'+str(self.mode)+'_t_'+str(self.t)+'_thres_'+str(weight_threshold).replace(".", "_")+'_merge_'+str(merge)+'_'+scale+'.png')
        if show_fig==True:
            plt.show() 
            
    def plot_num_simplices(self,simplices_tensor,merge,weight_threshold,dist_threshold,scale='log',show_fig=True,save_fig=True,save_path=''):
        plt.figure(figsize=(10,5))
        for z in range(0,len(simplices_tensor[0])):
            if z==0:
                plt.plot(simplices_tensor[:,0],c='r',label='0-simplices')
                plt.plot(simplices_tensor[:,1],c='g',label='1-simplices')
                plt.plot(simplices_tensor[:,2],c='b',label='2-simplices')
            else:
                plt.plot(simplices_tensor[:,0],c='r')
                plt.plot(simplices_tensor[:,1],c='g')
                plt.plot(simplices_tensor[:,2],c='b')
        plt.grid()
        plt.yscale(scale)
        plt.legend()
        plt.xlabel(r'$\tau$', fontsize=20)
        plt.ylabel('#', fontsize=20)
        if save_fig==True:
            if not os.path.exists(save_path+"/"+self.experiment_name):
                os.mkdir(save_path+"/"+self.experiment_name)
            if not os.path.exists(save_path+"/"+self.experiment_name+"/homogeneous/"):
                os.mkdir(save_path+"/"+self.experiment_name+"/homogeneous/")
            if not os.path.exists(save_path+"/"+self.experiment_name+"/homogeneous/features/"):
                os.mkdir(save_path+"/"+self.experiment_name+"/homogeneous/features/")   
            if not os.path.exists(save_path+"/"+self.experiment_name+"/homogeneous/features/num_simplices/"):
                os.mkdir(save_path+"/"+self.experiment_name+"/homogeneous/features/num_simplices/")                
            
            if merge == True:
                plt.savefig(save_path+"/"+self.experiment_name+"/homogeneous/features/num_simplices/"+'/numsimplices_homo_'+self.experiment_name+'_e_'+str(self.e).replace(".", "_")+'_b_'+str(self.b).replace(".", "_")+'_kernel_'+str(self.kernel)+'_mode_'+str(self.mode)+'_t_'+str(self.t)+'_thres_'+str(weight_threshold).replace(".", "_")+'_merge_'+str(merge)+'_'+str(dist_threshold).replace(".", "_")+'_'+scale+'.png')
            if merge == False:
                plt.savefig(save_path+"/"+self.experiment_name+"/homogeneous/features/num_simplices/"+'/numsimplices_homo_'+self.experiment_name+'_e_'+str(self.e).replace(".", "_")+'_b_'+str(self.b).replace(".", "_")+'_kernel_'+str(self.kernel)+'_mode_'+str(self.mode)+'_t_'+str(self.t)+'_thres_'+str(weight_threshold).replace(".", "_")+'_merge_'+str(merge)+'_'+scale+'.png')
        if show_fig==True:
            plt.show() 
        
        
     
            