"""
tauchen
-------
Discretizes Gaussian linear AR(1) processes via Tauchen's method

"""

from math import erfc, sqrt
from .core import MarkovChain

import numpy as np
from numba import njit


def rouwenhorst(n, ybar, sigma, rho):
    r"""
    Takes as inputs n, p, q, psi. It will then construct a markov chain
    that estimates an AR(1) process of:
    :math:`y_t = \bar{y} + \rho y_{t-1} + \varepsilon_t`
    where :math:`\varepsilon_t` is i.i.d. normal of mean 0, std dev of sigma

    The Rouwenhorst approximation uses the following recursive defintion
    for approximating a distribution:

    .. math::

        \theta_2 =
        \begin{bmatrix}
        p     &  1 - p \\
        1 - q &  q     \\
        \end{bmatrix}

    .. math::

        \theta_{n+1} =
        p
        \begin{bmatrix}
        \theta_n & 0   \\
        0        & 0   \\
        \end{bmatrix}
        + (1 - p)
        \begin{bmatrix}
        0  & \theta_n  \\
        0  &  0        \\
        \end{bmatrix}
        + q
        \begin{bmatrix}
        0        & 0   \\
        \theta_n & 0   \\
        \end{bmatrix}
        + (1 - q)
        \begin{bmatrix}
        0  &  0        \\
        0  & \theta_n  \\
        \end{bmatrix}


    Parameters
    ----------
    n : int
        The number of points to approximate the distribution

    ybar : float
        The value :math:`\bar{y}` in the process.  Note that the mean of this
        AR(1) process, :math:`y`, is simply :math:`\bar{y}/(1 - \rho)`

    sigma : float
        The value of the standard deviation of the :math:`\varepsilon` process

    rho : float
        By default this will be 0, but if you are approximating an AR(1)
        process then this is the autocorrelation across periods

    Returns
    -------

    mc : MarkovChain
        An instance of the MarkovChain class that stores the transition
        matrix and state values returned by the discretization method

    """

    # Get the standard deviation of y
    y_sd = sqrt(sigma**2 / (1 - rho**2))

    # Given the moments of our process we can find the right values
    # for p, q, psi because there are analytical solutions as shown in
    # Gianluca Violante's notes on computational methods
    p = (1 + rho) / 2
    q = p
    psi = y_sd * np.sqrt(n - 1)

    # Find the states
    ubar = psi
    lbar = -ubar

    bar = np.linspace(lbar, ubar, n)

    def row_build_mat(n, p, q):
        """
        This method uses the values of p and q to build the transition
        matrix for the rouwenhorst method

        """

        if n == 2:
            theta = np.array([[p, 1 - p], [1 - q, q]])

        elif n > 2:
            p1 = np.zeros((n, n))
            p2 = np.zeros((n, n))
            p3 = np.zeros((n, n))
            p4 = np.zeros((n, n))

            new_mat = row_build_mat(n - 1, p, q)

            p1[:n - 1, :n - 1] = p * new_mat
            p2[:n - 1, 1:] = (1 - p) * new_mat
            p3[1:, :-1] = (1 - q) * new_mat
            p4[1:, 1:] = q * new_mat

            theta = p1 + p2 + p3 + p4
            theta[1:n - 1, :] = theta[1:n - 1, :] / 2

        else:
            raise ValueError("The number of states must be positive " +
                             "and greater than or equal to 2")

        return theta

    theta = row_build_mat(n, p, q)

    bar += ybar / (1 - rho)

    return MarkovChain(theta, bar)


def tauchen(rho, sigma_u, b=0., m=3, n=7):
    r"""
    Computes a Markov chain associated with a discretized version of
    the linear Gaussian AR(1) process

    .. math::

        y_{t+1} = b + \rho y_t + u_{t+1}

    using Tauchen's method. Here :math:`{u_t}` is an i.i.d. Gaussian process
    with zero mean.

    Parameters
    ----------
    b : scalar(float)
        The constant term of {y_t}
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

    mc : MarkovChain
        An instance of the MarkovChain class that stores the transition
        matrix and state values returned by the discretization method

    """

    # standard deviation of demeaned y_t
    std_y = np.sqrt(sigma_u**2 / (1 - rho**2))

    # top of discrete state space for demeaned y_t
    x_max = m * std_y

    # bottom of discrete state space for demeaned y_t
    x_min = -x_max

    # discretized state space for demeaned y_t
    x = np.linspace(x_min, x_max, n)

    step = (x_max - x_min) / (n - 1)
    half_step = 0.5 * step
    P = np.empty((n, n))

    # approximate Markov transition matrix for
    # demeaned y_t
    _fill_tauchen(x, P, n, rho, sigma_u, half_step)

    # shifts the state values by the long run mean of y_t
    mu = b / (1 - rho)

    mc = MarkovChain(P, state_values=x+mu)

    return mc


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
            P[i, j] = (std_norm_cdf((z + half_step) / sigma) -
                       std_norm_cdf((z - half_step) / sigma))
