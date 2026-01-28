import numpy as np

def total_variation_distance(p:float, q: float) -> float:
    """
    Computes the Total Variation (TV) distance between two probability distributions
    TV distance = 0.5 * sum(|p - q|)
    """
    return 0.5 * np.sum(np.abs(p - q))

def aitchison_mean(list_of_lists):
    """
    Aitchison (compositional) mean for a list of compositions
    """
    X = np.asarray(list_of_lists, dtype=float)          
                           
    X = X / X.sum(axis=1, keepdims=True)               

    logX = np.log(X)
    clrX = logX - logX.mean(axis=1, keepdims=True)      

    mean_clr = clrX.mean(axis=0)                      
    mu = np.exp(mean_clr)                             
    return mu / mu.sum()