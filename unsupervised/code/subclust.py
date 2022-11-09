import numpy as np
import matplotlib.pyplot as plt
from metrics import euclidean_distance
import scipy
import pandas as pd



def subtractive_clustering(r_a: float, r_b: float, df, kn:int):
    '''Where r_a s the radium  for the first cluster, and r_b the second radio '''
    #Density matrix
    D = np.zeros(len(df.values))

    for i in range(0,len(df.values)):
        for j in range(0,len(df.values)):
            if i!=j:
                D[i]=D[i]+np.exp(-euclidean_distance(df.values[i],df.values[j])**(2) /(2*r_a)**2)

    #Plotting the density with sns
    # sns.scatterplot(x=df.values[:,0],y=df.values[:,1],hue=D)
    # plt.show()

    #Selecting the cluster centers
    #The cluster centers are the points with the greatest density
    centers=[]

    for k in range(0,kn):
        # sc= plt.scatter(df.values[:,0],df.values[:,1],c=D)
        # plt.colorbar(sc)
        # plt.show()

        if k==0:
            centers.append(np.argmax(D))
        else:
            #Update the density matrix
            for i in range(0,len(df.values)):
                D[i]=D[i]-D[centers[k-1]]*np.exp(-euclidean_distance(df.values[i],df.values[centers[k-1]])**(2) /(2*r_b)**2)
            centers.append(np.argmax(D))

    centers_cords=df.values[centers]
    distances=scipy.spatial.distance.cdist(centers_cords, df.values).T
    #Asigning a cluster to each point with the minimum distance
    clusters=np.argmin(distances,axis=1)

    return clusters, centers_cords


def plot_subtractive(df_data: pd.DataFrame, k:int, center: list,clusters:np.ndarray):
    centers = pd.DataFrame(center)
    #Plot the data points
    plt.scatter(df_data.iloc[:,0], df_data.iloc[:, 1], c=clusters, s=40,alpha=0.5)
    #Plot the cluster centers
    plt.scatter(centers.iloc[:, 0], centers.iloc[:, 1], marker='o', c=np.arange(k), s=500,alpha=0.5)
    plt.ylabel(df_data.columns[1])
    plt.xlabel(df_data.columns[0])
    plt.show()

def plot_subtractive1(df_data: pd.DataFrame, k:int, center: list,clusters:np.ndarray):
    centers = pd.DataFrame(center)
    #Plot the data points
    plt.scatter(df_data.iloc[:,2], df_data.iloc[:, -1], c=clusters, s=40,alpha=0.5)
    #Plot the cluster centers
    plt.scatter(centers.iloc[:, 2], centers.iloc[:, -1], marker='o', c=np.arange(k), s=500,alpha=0.5)
    plt.ylabel(df_data.columns[-1])
    plt.xlabel(df_data.columns[2])
    plt.show()