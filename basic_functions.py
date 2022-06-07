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

def parse_keys(data):
    """Extract keys from a set of matrices.

    The goal of this function is to parse the keys of a set of matrices
    and return them for subsequent processing. Keys will be checked for
    being time-varying. If so, the time steps will be extracted as well
    and corresponding tuples will be returned.

    The output of this function is dictionary mapping the name of a key
    to a list of instances of the key, plus optional time steps::

        {
            # Time-varying
            'data': [
                ('data_t_0', 0),
                ('data_t_1', 1),
                ('data_t_2', 2),
            ],

            # Static
            'diffusion_homology_pairs': [
                ('diffusion_homology_pairs', None)
            ]
        }

    Parameters
    ----------
    data : `dict` of `np.array`
        A sequence of matrices, typically originating from an `.npz`
        file that was loaded.

    Returns
    -------
    Dictionary with parsed keys, as described in the example above.
    """
    # Parses a time-varying key. If a string matches this regular
    # expression, it is time-varying.
    re_time = r'(.*)_t_(\d+)$'

    parsed_keys = collections.defaultdict(list)

    for key in data.keys():
        m = re.match(re_time, key)
        if m:
            name = m.group(1)
            time = int(m.group(2))

            parsed_keys[name].append((key, time))
        else:
            parsed_keys[key].append((key, None))

    return parsed_keys

def make_tensor(data, parsed_keys):
    """Create a tensor from a time-varying data set.

    This function takes a time-varying data set of the same (!) shape
    and turns it into a tensor whose last axis denotes the time steps
    of the process.

    Parameters
    ----------
    data : `dict` of `np.array`
        A sequence of matrices, typically originating from an `.npz`
        file that was loaded.

    parsed_keys : list of tuples
        List of `(key, t)` tuples, where `key` indicates the
        corresponding key and `t` the time step.

    Returns
    -------
    Tensor comprising all data arrays that match the supplied key, with
    an additional axis (the last one) representing time.
    """
    T = len(parsed_keys)

    if not T:
        return None

    shape = data[parsed_keys[0][0]].shape
    X = np.empty(shape=(*shape, T))

    for key, t in parsed_keys:
        X[..., t] = data[key]

    return X

def kernel_gaussian_resting(X,e):
    dist=sp.spatial.distance.cdist(X,X)
    K=np.exp(-dist**2/e)
    return K
    
def kernel_gaussian_nonresting(X,e): 
    dist=sp.spatial.distance.cdist(X,X)
    K=np.exp(-dist**2/e)
    np.fill_diagonal(K,0)
    return K

def kernel_gaussian_aniso_norm(K,b):
    d_array=np.sum(K,axis=1)
    D_inv=np.diag(d_array**(-b))
    K_n=D_inv@K@D_inv
    return K_n
    
def diffusion_operator(K):
    d=np.sum(K,axis=1)
    D_inv=np.diag(d**(-1))
    return D_inv@K

def diag_0(A):
    np.fill_diagonal(A,0)
    return A

def upper_triu(A):
    A_upper=np.triu(A,1)
    return A_upper

def lower_triu(A):
    A_upper=np.tril(A,-1)
    return A_upper

def triangles_walk(A):
    A_upper=upper_triu(A)
    A_lower=lower_triu(A)
    T=A_upper@A_upper@A_lower
    return T

def find_triangles(A):
    A_upper=upper_triu(A)
    A_lower=lower_triu(A)
    A_2=A_upper@A_upper
    A_3=A_2@A_lower
    N_A=len(A)
    
    diag_A_3=np.diagonal(A_3)
    num_triangles=int(np.sum(diag_A_3))
    T=np.zeros((num_triangles,3))
    t_=0
    triangle_start=np.arange(0,N_A)[diag_A_3>0]
    for r in range(0,len(triangle_start)):
        t_r=triangle_start[r]
        num_r=diag_A_3[t_r]
        n_=0
        A_2_t=A_2[t_r]
        A_2_t_index=np.arange(0,N_A)[A_2_t>0]
        for z in range(0,len(A_2_t_index)):
            t_z=A_2_t_index[z]
            if A[t_z][t_r]==1:
                s=A_upper[t_r]
                e=A_upper[:,t_z]
                t_m_array=np.arange(0,N_A)[(s*e)==1]
                for f in range(0,len(t_m_array)):
                    t_m=t_m_array[f]
                    T[t_]=[t_r,t_m,t_z]
                    t_=t_+1
    return (T)

def get_tf(index,N):
    tf=np.zeros(N,dtype=bool)
    tf[int(index[0])]=True
    tf[int(index[1])]=True
    tf[int(index[2])]=True
    return tf

def BarCodesUpdate(simplex_list,time_list,simplex_t,time_t):
    index_list=[]
    N_l=len(simplex_list)
    N_t=len(simplex_t)
    for z in range(0,N_t):
        if simplex_t[z] not in simplex_list:
            simplex_list.append(simplex_t[z])
            time_list.append([time_t])
        else:
            index_=simplex_list.index(simplex_t[z])
            index_list.append(index_)
            time_=time_list[index_]
            if len(time_)%2==0:
                time_list_new=time_list[index_]
                time_list_new.append(time_t)
                time_list[index_]=time_list_new
    for v in range(0,N_l):
        if v not in index_list:
            time_=time_list[v]
            if len(time_)%2!=0:
                time_list_new=time_list[v]
                time_list_new.append(time_t-1)
                time_list[v]=time_list_new
    return simplex_list,time_list
             
from sklearn import cluster, datasets, mixture



