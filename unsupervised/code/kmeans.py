import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

def initialize_centroids(df_data: pd.DataFrame, k:int) -> pd.DataFrame:
    """
    Initialize centroids randomly
    """
    centroids = df_data.sample(n=k)
    return centroids

def assign_clusters(df_data: pd.DataFrame, centroids: pd.DataFrame,metric: str = 'euclidean') -> pd.Series:
    """
    Assign clusters to each data point
    """
    # calculate pairwise distances between data points and centroids
    distances = cdist(df_data,centroids,metric='euclidean')
    # assign cluster to each data point based on the closest centroid
    clusters = np.argmin(distances, axis=1)
    return clusters

def update_centroids(df_data: pd.DataFrame, centroids: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """
    Update centroids based on the new clusters
    """
    new_centroids = df_data.groupby(clusters).mean()
    return new_centroids

def cost_function(df_data: pd.DataFrame, centroids:  pd.DataFrame,metric: str = 'euclidean') -> float:
    """
    Calculate the cost function
    """
    distances = cdist(df_data,centroids,metric)
    cost = np.sum(np.min(distances, axis=1))
    return cost

def stop_criteria(old_centroids: pd.DataFrame, new_centroids: pd.DataFrame, iterations: int, max_iter: int = 1000) -> bool:
    """
    Check if the algorithm has converged
    """
    if iterations > max_iter:
        return True
    return np.all(old_centroids.reset_index(drop=True) == new_centroids.reset_index(drop=True))

def kmeans_clustering(df_data: pd.DataFrame, k: int, metric: str = 'euclidean'):
    """
    K-means clustering algorithm
    """
    # initialize centroids
    centroids = initialize_centroids(df_data, k)
    # initialize bookkeeping vars
    iterations = 0
    old_centroids = centroids
    # run the main k-means algorithm
    clusters = assign_clusters(df_data, centroids,metric)
    while not stop_criteria(old_centroids, centroids, iterations):
        # save old centroids for convergence test
        old_centroids = centroids
        # assign clusters based on centroid positions
        clusters = assign_clusters(df_data, centroids,metric)
        # calculate new centroids
        centroids = update_centroids(df_data, centroids, clusters)
        # count iterations
        iterations += 1
    return centroids, clusters

def plot_kmeans(df_data: pd.DataFrame, k: int, centroids: pd.DataFrame, clusters: pd.Series):
    #Plot the data points
    plt.scatter(df_data.iloc[:,0], df_data.iloc[:, 1], c=clusters, s=40,alpha=0.5)
    #Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', c=np.arange(k), s=500,alpha=0.5)
    plt.ylabel(df_data.columns[1])
    plt.xlabel(df_data.columns[0])
    plt.show()
