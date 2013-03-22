"""
File:   mc_sample.py
Author: John Stachurski, with Thomas J. Sargent
Date:   2/2013
"""

import numpy as np
from discreterv import discreteRV

def sample_path(p, init=0, sample_size=1000): 
    """
    A function that generates sample paths of a finite Markov chain with 
    kernel p on state space S = [0,...,N-1], starting from state init.

    Parameters: 

        * p is a 2D NumPy array, nonnegative, rows sum to 1
        * init is an integer in S
        * sample_size is an integer

    Returns: A flat NumPy array containing the sample
    """
    N = len(p)
    # Let P[x] be the distribution corresponding to p[x,:]
    P = [discreteRV(p[x,:]) for x in range(N)]
    X = np.empty(sample_size, dtype=int)
    X[0] = init
    for t in range(sample_size - 1):
        X[t+1] = P[X[t]].draw()
    return X
