"""
Origin: QEwP by John Stachurski and Thomas J. Sargent
Date:   6/2013
File:   mc_stationary1.py

"""

import numpy as np

def stationary1(P):
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

