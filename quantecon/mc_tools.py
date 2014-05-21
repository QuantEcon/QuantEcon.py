"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: mc_tools.py
Authors: John Stachurski and Thomas J. Sargent
LastModified: 11/08/2013
"""

import numpy as np
from discrete_rv import discreteRV

def compute_stationary(P):
    """
    Computes the stationary distribution of Markov matrix P.

    Parameters: 
    
        * P is a square 2D NumPy array

    Returns: A flat array giving the stationary distribution
    """
    n = len(P)                               # P is n x n
    I = np.identity(n)                       # Identity matrix
    B, b = np.ones((n, n)), np.ones((n, 1))  # Matrix and vector of ones
    A = np.transpose(I - P + B) 
    solution = np.linalg.solve(A, b)
    return solution.flatten()                # Return a flat array



def sample_path(P, init=0, sample_size=1000): 
    """
    Generates one sample path from a finite Markov chain with (n x n) Markov
    matrix P on state space S = {0,...,n-1}. 

    Parameters: 

        * P is a nonnegative 2D NumPy array with rows that sum to 1
        * init is either an integer in S or a nonnegative array of length n
            with elements that sum to 1
        * sample_size is an integer

    If init is an integer, the integer is treated as the determinstic initial
    condition.  If init is a distribution on S, then X_0 is drawn from this
    distribution.

    Returns: A NumPy array containing the sample path
    """
    # === set up array to store output === #
    X = np.empty(sample_size, dtype=int)
    if isinstance(init, int):
        X[0] = init
    else:
        X[0] = discreteRV(init).draw()

    # === turn each row into a distribution === #
    # In particular, let P_dist[i] be the distribution corresponding to the
    # i-th row P[i,:]
    n = len(P)
    P_dist = [discreteRV(P[i,:]) for i in range(n)]

    # === generate the sample path === #
    for t in range(sample_size - 1):
        X[t+1] = P_dist[X[t]].draw()
    return X

