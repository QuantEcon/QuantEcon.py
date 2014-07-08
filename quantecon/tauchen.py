"""
Filename: tauchen.py
Authors: Thomas Sargent, John Stachurski

Discretizes Gaussian linear AR(1) processes via Tauchen's method

"""

import numpy as np
from scipy.stats import norm

def approx_markov(rho, sigma_u, m=3, n=7):
    """
    Computes the Markov matrix associated with a discretized version of
    the linear Gaussian AR(1) process 

        y_{t+1} = rho * y_t + u_{t+1}

    according to Tauchen's method.  Here {u_t} is an iid Gaussian process with
    zero mean.

    Parameters:

        * rho is the correlation coefficient
        * sigma_u is the standard deviation of u
        * m parameterizes the width of the state space
        * n is the number of states
    
    Returns:

        * x, the state space, as a NumPy array
        * a matrix P, where P[i,j] is the probability of transitioning from
            x[i] to x[j]

    """
    F = norm(loc=0, scale=sigma_u).cdf
    std_y = np.sqrt(sigma_u**2 / (1-rho**2))  # standard deviation of y_t
    x_max = m * std_y                         # top of discrete state space
    x_min = - x_max                           # bottom of discrete state space
    x = np.linspace(x_min, x_max, n)          # discretized state space
    step = (x_max - x_min) / (n - 1)
    half_step = 0.5 * step
    P = np.empty((n, n))

    for i in range(n):
        P[i, 0] = F(x[0]-rho * x[i] + half_step)
        P[i, n-1] = 1 - F(x[n-1] - rho * x[i] - half_step)
        for j in range(1, n-1):
            z = x[j] - rho * x[i]
            P[i, j] = F(z + half_step) - F(z - half_step)

    return x, P
