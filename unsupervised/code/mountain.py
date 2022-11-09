import scipy
import numpy as np
import math as m
from metrics import euclidean_distance
import pandas as pd
import matplotlib.pyplot as plt




def mountain(v,x,sigma):
    return np.sum(np.exp(-euclidean_distance(v,x)**(2) /2*sigma**2))

def mountain_clustering(n: int, gr: int, X: np.ndarray, sigma: np.array, kn: int):
    '''Where n refers to de dimension, gr the grid divisions, X is the raw data
    to be clustered, sigma is an array of deviations for the mountain density
    function, and kn is the number of centers to be found'''
#Set up grid matrix of n-dimensions (V)

    v_dim=gr*np.ones([1,n])
    M = np.zeros([int(x) for x in v_dim[0]])
    M_r=M.reshape(1,-1)[0]
    cur=np.ones([1,n])
    for i in range(0,n):
        for j in range(0,i+1):
            cur[:,i]=cur[:,i]*v_dim[:,j]
    # max_m=[] #greatest density value
    # max_v=[] #Cluster center position
    center=[]
    max_idx=[]

    for k in range(0,kn):
        max_m = 0
        max_v = 0
        max_i=i
        for i in range(0,int(cur[:,-1][0])):
            #Calculate the vector indexes
            idx=i+1
            dim=np.zeros(len(range(n,0,-1))).tolist()
            for j in range(n-1,0,-1):
                dim[j]=(m.ceil(idx/cur[:,j-1]))
                idx=int(idx-cur[:,j-1]*(dim[j]-1))
            dim[0]=idx
            #Dim is holing the current point index vector
            #but needs to be normalized to the range [0,1]
            v=[d /gr for d in dim]
            #calculate the density of the current point
            if k==0:
                M_r[i]=mountain(v,X,sigma[k])
            else:
                M_r[i]=M_r[i]-M_r[max_idx[k-1]]*np.exp(-euclidean_distance(np.array(v),np.array(center[k-1]))**(2) /2*sigma[k]**2)

            #update the max density and the max density point
            if M_r[i]>max_m:
                max_m=M_r[i]
                max_v=v
                max_i=i
        center.append(max_v)
        max_idx.append(max_i)
        print('Cluster ',k+1,' center: ',max_v)
    distances=scipy.spatial.distance.cdist(center, X).T
    #Asigning a cluster to each point with the minimum distance
    clusters=np.argmin(distances,axis=1)
    return clusters, center


def plot_mountain1(df_data: pd.DataFrame, k:int, center: list,clusters:np.ndarray):
    centers = pd.DataFrame(center)
    #Plot the data points
    plt.scatter(df_data.iloc[:,2], df_data.iloc[:, 3], c=clusters, s=40,alpha=0.5)
    #Plot the cluster centers
    plt.scatter(centers.iloc[:, 2], centers.iloc[:, 3], marker='o', c=np.arange(k), s=500,alpha=0.5)
    plt.ylabel(df_data.columns[3])
    plt.xlabel(df_data.columns[2])
    plt.show()

def plot_mountain(df_data: pd.DataFrame, k:int, center: list,clusters:np.ndarray):
    centers = pd.DataFrame(center)
    #Plot the data points
    plt.scatter(df_data.iloc[:,0], df_data.iloc[:, 1], c=clusters, s=40,alpha=0.5)
    #Plot the cluster centers
    plt.scatter(centers.iloc[:, 0], centers.iloc[:, 1], marker='o', c=np.arange(k), s=500,alpha=0.5)
    plt.ylabel(df_data.columns[1])
    plt.xlabel(df_data.columns[0])
    plt.show()