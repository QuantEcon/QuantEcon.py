"""
Source: QEwP by John Stachurski and Thomas J. Sargent
Date: May 2013
Filename: mc_stationary3.py
"""

import numpy as np
from mc_sample import sample_path

def stationary3(p, n=1000, lae=False): 
    """ 
    Computes the stationary distribution via the LLN for stable
    Markov chains.

    Parameters: 

        * p is a 2D NumPy array
        * n is a positive integer giving the sample size
        * lae is a flag indicating whether to use the look-ahead method

    Returns: An array containing the estimated stationary distribution
    """
    N = len(p)
    X = sample_path(p, sample_size=n)    # Sample path starting from X = 0
    if lae:
        solution = [np.mean(p[X,y]) for y in range(N)]
    else:
        solution = [np.mean(X == y) for y in range(N)]
    return np.array(solution)

