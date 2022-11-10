import scipy
import numpy as np
import math as m
from metrics import euclidean_distance
import pandas as pd
import matplotlib.pyplot as plt




def mountain_height(prototype,data,variance):
    return np.sum(np.exp(-euclidean_distance(prototype,data)**(2) /2*variance**2))

def mountain_clustering(n: int, grid: int, data: np.ndarray, variance: np.array, number_clusters: int):
    '''Performs mountain clustering on the data in df, with the number of clusters defined by number_clusters.
       The radius of the neighborhood of each cluster is given by variance.
    
    Args:
        n: number to be used in the grid dimension
        grid: grid dimension
        data: data to be clustered
        variance: variance of the gaussian function
        number_clusters: number of clusters to be found

    Output:
        center: list of cluster centers
        clusters: array of cluster assignments
    '''
#Set up grid matrix of n-dimensions (prototype)

    proto_dimension=grid*np.ones([1,n])
    mountain = np.zeros([int(data) for data in proto_dimension[0]])
    mountain_reshaped=mountain.reshape(1,-1)[0]
    current=np.ones([1,n])
    for i in range(0,n):
        for j in range(0,i+1):
            current[:,i]=current[:,i]*proto_dimension[:,j]
    # max_mountain=[] #greatest density value
    # max_prototype=[] #Cluster center position
    center=[]
    max_index=[]

    for k in range(0,number_clusters):
        max_mountain = 0
        max_prototype = 0
        max_i=i
        for i in range(0,int(current[:,-1][0])):
            #Calculate the vector indexes
            index=i+1
            dim=np.zeros(len(range(n,0,-1))).tolist()
            for j in range(n-1,0,-1):
                dim[j]=(m.ceil(index/current[:,j-1]))
                index=int(index-current[:,j-1]*(dim[j]-1))
            dim[0]=index
            #Dim is holing the current point index vector
            #but needs to be normalized to the range [0,1]
            prototype=[d /grid for d in dim]
            #calculate the density of the current point
            if k==0:
                mountain_reshaped[i]=mountain_height(prototype,data,variance[k])
            else:
                mountain_reshaped[i]=mountain_reshaped[i]-mountain_reshaped[max_index[k-1]]*np.exp(-euclidean_distance(np.array(prototype),np.array(center[k-1]))**(2) /2*variance[k]**2)

            #update the max density and the max density point
            if mountain_reshaped[i]>max_mountain:
                max_mountain=mountain_reshaped[i]
                max_prototype=prototype
                max_i=i
        center.append(max_prototype)
        max_index.append(max_i)
        print('Cluster ',k+1,' center: ',max_prototype)
    distances=scipy.spatial.distance.cdist(center, data).T
    #Asigning a cluster to each point with the minimum distance
    clusters=np.argmin(distances,axis=1)
    return clusters, center


def plot_mountain1(df_data: pd.DataFrame, k:int, center: list,clusters:np.ndarray):
    '''Plots the data points and the cluster centers.
    
    Args:
        df_data: data to be plotted
        k: number of clusters
        center: list of cluster centers
        clusters: array of cluster assignments
    '''
    centers = pd.DataFrame(center)
    #Plot the data points
    plt.scatter(df_data.iloc[:,2], df_data.iloc[:, 3], c=clusters, s=40,alpha=0.5)
    #Plot the cluster centers
    plt.scatter(centers.iloc[:, 2], centers.iloc[:, 3], marker='o', c=np.arange(k), s=500,alpha=0.5)
    plt.ylabel(df_data.columns[3])
    plt.xlabel(df_data.columns[2])
    plt.show()

def plot_mountain(df_data: pd.DataFrame, k:int, center: list,clusters:np.ndarray):
    '''Plots the data points and the cluster centers.
    
    Args:
        df_data: data to be plotted
        k: number of clusters
        center: list of cluster centers
        clusters: array of cluster assignments
    '''
    centers = pd.DataFrame(center)
    #Plot the data points
    plt.scatter(df_data.iloc[:,0], df_data.iloc[:, 1], c=clusters, s=40,alpha=0.5)
    #Plot the cluster centers
    plt.scatter(centers.iloc[:, 0], centers.iloc[:, 1], marker='o', c=np.arange(k), s=500,alpha=0.5)
    plt.ylabel(df_data.columns[1])
    plt.xlabel(df_data.columns[0])
    plt.show()