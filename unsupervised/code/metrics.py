import numpy as np

def euclidean_distance(x:np.array, y:np.array) -> float:
    """Compute the euclidean distance between two vectors x and y.
    
    Args:
        x: A vector of shape (n,).
        y: A vector of shape (n,).  
    """
    return np.sqrt(np.sum((x - y)**2))


def cosine_similarity(x:np.array, y:np.array) -> float:
    """Compute the cosine similarity between two vectors x and y.

     Args:
        x: A vector of shape (n,).
        y: A vector of shape (n,).  
    
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def manhattan_distance(x:np.array, y:np.array) -> float:
    """Compute the manhattan distance between two vectors x and y.
    
     Args:
        x: A vector of shape (n,).
        y: A vector of shape (n,).  
    
    """
    return np.sum(np.abs(x - y))

def mahalanobis_distance(x:np.array, y:np.array, cov:np.array) -> float:
    """Compute the mahalanobis distance between two vectors x and y.
    
     Args:
        x: A vector of shape (n,).
        y: A vector of shape (n,).  
        cov: A covariance matrix of shape (n, n).
    """
    return np.sqrt(np.dot(np.dot((x - y), np.linalg.inv(cov)), (x - y).T))

def jaccard_similarity(x:np.array, y:np.array) -> float:
    """Compute the jaccard similarity between two vectors x and y.
    
     Args:
        x: A vector of shape (n,).
        y: A vector of shape (n,).  
    """
    return np.sum(np.minimum(x, y)) / np.sum(np.maximum(x, y))

def jaccard_distance(x:np.array, y:np.array) -> float:
    """Compute the jaccard distance between two vectors x and y.
    
     Args:
        x: A vector of shape (n,).
        y: A vector of shape (n,).  
    """
    return 1 - jaccard_similarity(x, y)

def similarity(x:np.array, y:np.array, metric:str) -> float:
    """Compute the similarity between two vectors x and y.
    
     Args:
        x: A vector of shape (n,).
        y: A vector of shape (n,).  
        metric: The similarity metric to use. 
    """
    if metric == 'euclidean':
        return euclidean_distance(x, y)
    elif metric == 'cosine':
        return cosine_similarity(x, y)
    elif metric == 'manhattan':
        return manhattan_distance(x, y)
    elif metric == 'jaccard':
        return jaccard_similarity(x, y)
    elif metric == 'mahalanobis':
        return mahalanobis_distance(x, y)
    elif metric == 'jaccard_distance':
        return jaccard_distance(x, y)
    else:
        raise ValueError('Unknown metric {}'.format(metric))