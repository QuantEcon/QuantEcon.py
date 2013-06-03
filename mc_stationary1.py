## Filename: mc_stationary1.py
## Author: John Stachurski

import numpy as np

def stationary1(p):
    """
    Parameters: 
    
        * p is a 2D NumPy array, assumed to be stationary

    Returns: A flat array giving the stationary distribution
    """
    N = len(p)                               # p is N x N
    I = np.identity(N)                       # Identity matrix
    B, b = np.ones((N, N)), np.ones((N, 1))  # Matrix and vector of ones
    A = np.transpose(I - p + B) 
    solution = np.linalg.solve(A, b)
    return solution.flatten()                # Return a flat array

