"""
Origin: QEwP by John Stachurski and Thomas J. Sargent
Date: 3/2013
File: tauchen.py

Discretizes Gaussian linear AR(1) processes via Tauchen's method

"""

import numpy as np
from scipy.stats import norm

def approx_markov(rho, sigma, m=3, N=7):
    """
    Computes the Markov matrix associated with a discretized version of
    the linear Gaussian AR(1) process 

        y_{t+1} = rho * y_t + u_{t+1}

    according to Tauchen's method.  Here {u_t} is an iid Gaussian process with
    zero mean.

    Parameters

        * rho is the correlation coefficient
        * sigma is the standard deviation of u
        * m parameterizes the width of the state space
        * N is the number of states

    """
    F = norm(loc=0, scale=sigma).cdf
    std_y = np.sqrt(sigma**2 / (1-rho**2))  # standard deviation of y_t
    ymax = m * std_y                        # top of discrete state space
    ymin = - ymax                           # bottom of discrete state space
    S = np.linspace(ymin, ymax, N)          # discretized state space
    step = (ymax - ymin) / (N - 1)
    half_step = 0.5 * step
    P = np.empty((N, N))

    for j in range(N):
        P[j, 0] = F(S[0]-rho * S[j] + half_step)
        P[j, N-1] = 1 - F(S[N-1] - rho * S[j] - half_step)
        for k in range(1, N-1):
            z = S[k] - rho * S[j]
            P[j, k] = F(z + half_step) - F(z - half_step)

    return S, P
