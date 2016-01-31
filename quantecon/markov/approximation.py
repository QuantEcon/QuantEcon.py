"""
Filename: approximation.py

Authors: Thomas Sargent, John Stachurski

tauchen
-------
Discretizes Gaussian linear AR(1) processes via Tauchen's method

"""

from math import erfc, sqrt

import numpy as np
from numba import njit


def tauchen(rho, sigma_u, m=3, n=7):
    """
    Computes the Markov matrix associated with a discretized version of
    the linear Gaussian AR(1) process

        y_{t+1} = rho * y_t + u_{t+1}

    according to Tauchen's method.  Here {u_t} is an iid Gaussian
    process with zero mean.

    Parameters
    ----------
    rho : scalar(float)
        The autocorrelation coefficient
    sigma_u : scalar(float)
        The standard deviation of the random process
    m : scalar(int), optional(default=3)
        The number of standard deviations to approximate out to
    n : scalar(int), optional(default=7)
        The number of states to use in the approximation

    Returns
    -------

    x : array_like(float, ndim=1)
        The state space of the discretized process
    P : array_like(float, ndim=2)
        The Markov transition matrix where P[i, j] is the probability
        of transitioning from x[i] to x[j]

    """

    # standard deviation of y_t
    std_y = np.sqrt(sigma_u**2 / (1 - rho**2))

    # top of discrete state space
    x_max = m * std_y

    # bottom of discrete state space
    x_min = -x_max

    # discretized state space
    x = np.linspace(x_min, x_max, n)

    step = (x_max - x_min) / (n - 1)
    half_step = 0.5 * step
    P = np.empty((n, n))

    _fill_tauchen(x, P, n, rho, sigma_u, half_step)

    return x, P


@njit
def std_norm_cdf(x):
    return 0.5 * erfc(-x / sqrt(2))


@njit
def _fill_tauchen(x, P, n, rho, sigma, half_step):
    for i in range(n):
        P[i, 0] = std_norm_cdf((x[0] - rho * x[i] + half_step) / sigma)
        P[i, n - 1] = 1 - \
            std_norm_cdf((x[n - 1] - rho * x[i] - half_step) / sigma)
        for j in range(1, n - 1):
            z = x[j] - rho * x[i]
            P[i, j] = (std_norm_cdf((z + half_step) / sigma) - std_norm_cdf(
                (z - half_step) / sigma))
