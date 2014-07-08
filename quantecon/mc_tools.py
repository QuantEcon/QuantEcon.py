"""
Filename: mc_tools.py
Authors: Thomas J. Sargent, John Stachurski
"""

import numpy as np
from discrete_rv import DiscreteRV

def mc_compute_stationary(P):
    """
    Computes the stationary distribution of Markov matrix P.

    Parameters 
    ===========
        P : a square 2D NumPy array

    Returns: A flat array giving the stationary distribution
    """
    n = len(P)                               # P is n x n
    I = np.identity(n)                       # Identity matrix
    B, b = np.ones((n, n)), np.ones((n, 1))  # Matrix and vector of ones
    A = np.transpose(I - P + B) 
    solution = np.linalg.solve(A, b)
    return solution.flatten()                # Return a flat array



def mc_sample_path(P, init=0, sample_size=1000): 
    """
    Generates one sample path from a finite Markov chain with (n x n) Markov
    matrix P on state space S = {0,...,n-1}. 

    Parameters 
    ==========
        P : A nonnegative 2D NumPy array with rows that sum to 1

        init : Either an integer in S or a nonnegative array of length n
                with elements that sum to 1

        sample_size : int

    If init is an integer, the integer is treated as the determinstic initial
    condition.  If init is a distribution on S, then X_0 is drawn from this
    distribution.

    Returns
    ========
        A NumPy array containing the sample path

    """
    # === set up array to store output === #
    X = np.empty(sample_size, dtype=int)
    if isinstance(init, int):
        X[0] = init
    else:
        X[0] = DiscreteRV(init).draw()

    # === turn each row into a distribution === #
    # In particular, let P_dist[i] be the distribution corresponding to the
    # i-th row P[i,:]
    n = len(P)
    P_dist = [DiscreteRV(P[i,:]) for i in range(n)]

    # === generate the sample path === #
    for t in range(sample_size - 1):
        X[t+1] = P_dist[X[t]].draw()
    return X

