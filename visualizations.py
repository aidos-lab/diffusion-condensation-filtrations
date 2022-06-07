# Import Modules
import numpy as np
import pandas as pd
import scipy as sp
import dionysus as d
import matplotlib.pyplot as plt
#import pecan as pc
import sys
import collections
import re
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

def plot_weights_during_filtration(W_1,experiment_name,e,b,kernel,mode,version,t,W_2=None,W_3=None,scale='log', thresholds=None,save=False,save_path='',label_1='',label_2='',label_3=''):
    plt.figure(figsize=(15,10))
    N_E_1=len(W_1[0])
    s=1
    for z in range(0,N_E_1,1):
        if z ==0:
            plt.plot(np.arange(1,1+len(W_1[:,z])),W_1[:,z],linewidth=0.01,color='k',label=label_1)
        else:
            plt.plot(np.arange(1,1+len(W_1[:,z])),W_1[:,z],linewidth=0.01,color='k')
    if W_2 is not None:
        s=s+1
        N_E_2=len(W_2[0])
        for z in range(0,N_E_2,1):
            if z ==0:
                plt.plot(np.arange(1,1+len(W_2[:,z])),W_2[:,z],linewidth=0.01,color='r',label=label_2)
            else:
                plt.plot(np.arange(1,1+len(W_2[:,z])),W_2[:,z],linewidth=0.01,color='r')
    if W_3 is not None:
        s=s+1
        N_E_3=len(W_3[0])
        for z in range(0,N_E_3,1):
            if z ==0:
                plt.plot(np.arange(1,1+len(W_3[:,z])),W_3[:,z],linewidth=0.01,color='g',label=label_3)
            else:
                plt.plot(np.arange(1,1+len(W_3[:,z])),W_3[:,z],linewidth=0.01,color='g')
                
    if thresholds is not None:
        color_array=['m','gold','indigo', 'lawngreen','aqua','peru']
        for u in range(0,len(thresholds)):
            threshold=thresholds[u]
            plt.plot(np.arange(1,1+len(W_1[:,z])),threshold*np.ones(len(np.arange(1,1+len(W_1[:,z])))),linewidth=0.75,color=color_array[u],label=str(u+1))
        

    plt.xlabel(r'$\tau$', fontsize=20)
    plt.ylabel('$p_{i}$', fontsize=20)
    plt.grid()
    plt.legend()
    plt.yscale(scale)
    if save==True:
        if version =='homogeneous':
            folder='/homogeneous'
        if version =='inhomogeneous':
            folder='/inhomogeneous'
        if not os.path.exists(save_path+"/"+experiment_name):
            os.mkdir(save_path+"/"+experiment_name)
        if not os.path.exists(save_path+"/"+experiment_name+folder+"/"):
            os.mkdir(save_path+"/"+experiment_name+folder+"/")
        if not os.path.exists(save_path+"/"+experiment_name+folder+"/weights/"):
            os.mkdir(save_path+"/"+experiment_name+folder+"/weights/") 
        path_=save_path+"/"+experiment_name+folder+"/weights/"        

        plt.savefig(path_+'weights_'+experiment_name+'_e_'+str(e).replace(".", "_")+'_b_'+str(b).replace(".", "_")+'_kernel_'+str(kernel)+'_mode_'+str(mode)+'_t_'+str(t).replace(".", "_")+'_scale_'+scale+'_'+str(s).replace(".", "_")+'.png')
    plt.show()
    
def persistence(BarCodes,experiment_name,e,b,t,kernel,mode,version,merge,dist_threshold,weight_threshold,save_path,save=True):

    if save==True:
        if version =='homogeneous':
            folder='/homogeneous'
        if version =='inhomogeneous':
            folder='/inhomogeneous'
        if not os.path.exists(save_path+"/"+experiment_name):
            os.mkdir(save_path+"/"+experiment_name)
        if not os.path.exists(save_path+"/"+experiment_name+folder+"/"):
            os.mkdir(save_path+"/"+experiment_name+folder+"/")
        if not os.path.exists(save_path+"/"+experiment_name+folder+"/persistence/"):
            os.mkdir(save_path+"/"+experiment_name+folder+"/persistence/") 
        path_=save_path+"/"+experiment_name+folder+"/persistence/"
     
        gd.plot_persistence_diagram(BarCodes,legend=True);
        if save==True:
            plt.savefig(path_+'persistence_diagram_'+experiment_name+'_e_'+str(e).replace(".", "_")+'_b_'+str(b).replace(".", "_")+'_kernel_'+str(kernel)+'_mode_'+str(mode)+'_t_'+str(t).replace(".", "_")+'_thres_'+str(weight_threshold).replace(".", "_")+'_merge_'+str(merge)+'_'+str(dist_threshold).replace(".", "_")+'_'+version+'.png')
        plt.show()
        gd.plot_persistence_barcode(BarCodes[0:-1],legend=True);
        if save==True:
            plt.savefig(path_+'persistence_diagram_'+experiment_name+'_e_'+str(e).replace(".", "_")+'_b_'+str(b).replace(".", "_")+'_kernel_'+str(kernel)+'_mode_'+str(mode)+'_t_'+str(t).replace(".", "_")+'_thres_'+str(weight_threshold).replace(".", "_")+'_merge_'+str(merge)+'_'+str(dist_threshold).replace(".", "_")+'_'+version+'.png')
        plt.show()
        
   
    
def get_simplex(simplex_tree,f):
    """
    Function that gives the simplicial complex (graph and triangles) and betti numbers at filtration step f 
    """
    T = np.zeros(3)
    G = nx.Graph()
    st = gd.SimplexTree()
    for filtered_value in simplex_tree.get_filtration():
        if filtered_value[1]<=f:
            simplex_=filtered_value[0]
            simplex_dim=len(simplex_)
            if simplex_dim==1:
                G.add_node(simplex_[0])
                st.insert(simplex_)
            if simplex_dim==2:
                G.add_edge(simplex_[0], simplex_[1])
                st.insert(simplex_)
            if simplex_dim==3:
                T=np.vstack((T,simplex_))
                st.insert(simplex_)
    BarCodes=st.persistence(min_persistence=-1,persistence_dim_max=True)
    betti_array=(st.betti_numbers())
    return st,G,T[1:],betti_array

def bifiltration_plot_betti(betti_tensor_filtration,w_min,w_max,version,experiment_name,e,b,t,kernel,mode,weight_threshold_steps,merge,dist_threshold,save_fig=True,save_path=''):
    for u in range(0,2):
        fig=plt.figure(figsize=(12,12))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
        plt.imshow(betti_tensor_filtration[:,:,u])
        ax.set_xticks(np.arange(0,len(betti_tensor_filtration[0])))
        ax.set_xticklabels([str(int) for int in np.arange(1,1+len(betti_tensor_filtration[0]))])
        ax.set_yticks(np.arange(0,len(weight_threshold_steps)))
        ax.set_yticklabels([str(int) for int in weight_threshold_steps])
        plt.colorbar()
        plt.xlabel(r'$\tau$', fontsize=20)
        plt.ylabel(r'$p_{max}$', fontsize=20)
        plt.grid()
        plt.title(r'$\beta_{'+str(u)+'}$', fontsize=25)
        
        if save_fig==True:
            if version =='homogeneous':
                folder='/homogeneous'
            if version =='inhomogeneous':
                folder='/inhomogeneous'
            if not os.path.exists(save_path+"/"+experiment_name):
                os.mkdir(save_path+"/"+experiment_name)
            if not os.path.exists(save_path+"/"+experiment_name+folder+"/"):
                os.mkdir(save_path+"/"+experiment_name+folder+"/")
            if not os.path.exists(save_path+"/"+experiment_name+folder+"/bifiltration/"):
                os.mkdir(save_path+"/"+experiment_name+folder+"/bifiltration/") 
            if not os.path.exists(save_path+"/"+experiment_name+folder+"/bifiltration/betti_numbers/"):
                os.mkdir(save_path+"/"+experiment_name+folder+"/bifiltration/betti_numbers/")   
            path_=save_path+"/"+experiment_name+folder+"/bifiltration/betti_numbers/"            
            
            if merge==True:
                plt.savefig(path_+'/bifiltration_betti_'+str(u)+'_'+version+'_'+experiment_name+'_e_'+str(e).replace(".", "_")+'_b_'+str(b).replace(".", "_")+'_kernel_'+str(kernel)+'_mode_'+str(mode)+'_t_'+str(t)+'_merge_'+str(merge)+'_'+str(dist_threshold).replace(".", "_")+'_from'+str(w_min).replace(".", "_")+'to_'+str(w_max).replace(".", "_")+'.png', bbox_inches='tight')
            if merge==False:
                plt.savefig(path_+'/bifiltration_betti_'+str(u)+'_'+version+'_'+experiment_name+'_e_'+str(e).replace(".", "_")+'_b_'+str(b).replace(".", "_")+'_kernel_'+str(kernel)+'_mode_'+str(mode)+'_t_'+str(t)+'_merge_'+str(merge)+'_from'+str(w_min).replace(".", "_")+'to_'+str(w_max).replace(".", "_")+'.png', bbox_inches='tight')
                
        plt.show()

def bifiltration_plot_simplices(simplices_tensor_filtration,w_min,w_max,version,experiment_name,e,b,t,kernel,mode,weight_threshold_steps,merge,dist_threshold,save_fig=True,save_path=''):
    for u in range(0,3):
        fig=plt.figure(figsize=(12,12))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
        plt.imshow(simplices_tensor_filtration[:,:,u])
        ax.set_xticks(np.arange(0,len(simplices_tensor_filtration[0])))
        ax.set_xticklabels([str(int) for int in np.arange(1,1+len(simplices_tensor_filtration[0]))])
        ax.set_yticks(np.arange(0,len(weight_threshold_steps)))
        ax.set_yticklabels([str(int) for int in weight_threshold_steps])
        plt.colorbar()
        plt.xlabel(r'$\tau$', fontsize=20)
        plt.ylabel(r'$p_{max}$', fontsize=20)
        plt.grid()
        plt.title(str(u)+'-simplices', fontsize=25)
        
        if save_fig==True:
            if version =='homogeneous':
                folder='/homogeneous'
            if version =='inhomogeneous':
                folder='/inhomogeneous'
            if not os.path.exists(save_path+"/"+experiment_name):
                os.mkdir(save_path+"/"+experiment_name)
            if not os.path.exists(save_path+"/"+experiment_name+folder+"/"):
                os.mkdir(save_path+"/"+experiment_name+folder+"/")
            if not os.path.exists(save_path+"/"+experiment_name+folder+"/bifiltration/"):
                os.mkdir(save_path+"/"+experiment_name+folder+"/bifiltration/") 
            if not os.path.exists(save_path+"/"+experiment_name+folder+"/bifiltration/num_simplices/"):
                os.mkdir(save_path+"/"+experiment_name+folder+"/bifiltration/num_simplices/")   
            path_=save_path+"/"+experiment_name+folder+"/bifiltration/num_simplices/"                
            
            if merge==True:
                plt.savefig(path_+'/bifiltration_simplices_'+str(u)+'_'+version+'_'+experiment_name+'_e_'+str(e).replace(".", "_")+'_b_'+str(b).replace(".", "_")+'_kernel_'+str(kernel)+'_mode_'+str(mode)+'_t_'+str(t)+'_thres_'+'_merge_'+str(merge)+'_'+str(dist_threshold).replace(".", "_")+'_from'+str(w_min).replace(".", "_")+'to_'+str(w_max).replace(".", "_")+'.png', bbox_inches='tight')
            if merge==False:
                plt.savefig(path_+'/bifiltration_simplices_'+str(u)+'_'+version+'_'+experiment_name+'_e_'+str(e).replace(".", "_")+'_b_'+str(b).replace(".", "_")+'_kernel_'+str(kernel)+'_mode_'+str(mode)+'_t_'+str(t)+'_thres_'+'_merge_'+str(merge)+'_from'+str(w_min).replace(".", "_")+'to_'+str(w_max).replace(".", "_")+'.png', bbox_inches='tight')
        plt.show()