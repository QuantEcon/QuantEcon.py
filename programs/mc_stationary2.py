## Filename: mc_stationary2.py
## Author: John Stachurski

import numpy as np

# Import the sample_path function defined above
from mc_sample import sample_path

def stationary2(p, n=1000): 
    """ 
    Computes the stationary distribution via the LLN for stable
    Markov chains.

    Parameters: 

        * p is a 2D NumPy array
        * n is a positive integer giving the sample size

    Returns: An array containing the estimated stationary distribution
    """
    N = len(p)
    X = sample_path(p, sample_size=n)    # Sample path starting from X = 0
    q = [np.mean(X == y) for y in range(N)]
    return np.array(q)

