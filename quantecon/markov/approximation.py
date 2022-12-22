"""
tauchen
-------
Discretizes Gaussian linear AR(1) processes via Tauchen's method

"""

from math import erfc, sqrt
from .core import MarkovChain

import warnings
import numpy as np
from numba import njit


def rouwenhorst(n, rho, sigma, mu=0.):
    r"""
    Takes as inputs n, mu, sigma, rho. It will then construct a markov chain
    that estimates an AR(1) process of:
    :math:`y_t = \mu + \rho y_{t-1} + \varepsilon_t`
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

    where :math:`{p = q = \frac{(1 + \rho)}{2}}`

    Parameters
    ----------
    n : int
        The number of points to approximate the distribution

    rho : float
        Persistence parameter in AR(1) process, if you are approximating
        an AR(1) process then this is the autocorrelation across periods.

    sigma : float
        The value of the standard deviation of the :math:`\varepsilon` process

    mu : float, optional(default=0.0)
        The value :math:`\mu` in the process.  Note that the mean of this
        AR(1) process, :math:`y`, is simply :math:`\mu/(1 - \rho)`

    Returns
    -------

    mc : MarkovChain
        An instance of the MarkovChain class that stores the transition
        matrix and state values returned by the discretization method

    Note
    ----

    UserWarning: The API of `rouwenhorst` was changed from
    `rouwenhorst(n, ybar, sigma, rho)` to
    `rouwenhorst(n, rho, sigma, mu=0.)` in version 0.6.0.

    """

    warnings.warn("The API of rouwenhorst has changed from `rouwenhorst(n, ybar, sigma, rho)`"
                  " to `rouwenhorst(n, rho, sigma, mu=0.)`. To find more details please visit:"
                  " https://github.com/QuantEcon/QuantEcon.py/issues/663.",
                  UserWarning, stacklevel=2)
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

    bar += mu / (1 - rho)

    return MarkovChain(theta, bar)


def tauchen(n, rho, sigma, mu=0., n_std=3):
    r"""
    Computes a Markov chain associated with a discretized version of
    the linear Gaussian AR(1) process

    .. math::

        y_t = \mu + \rho y_{t-1} + \epsilon_t

    using Tauchen's method. Here :math:`{\epsilon_t}` is an i.i.d. Gaussian process
    with zero mean.

    Parameters
    ----------

    n : scalar(int)
        The number of states to use in the approximation
    rho : scalar(float)
        The autocorrelation coefficient, Persistence parameter in AR(1) process
    sigma : scalar(float)
        The standard deviation of the random process
    mu : scalar(float), optional(default=0.0)
        The value :math:`\mu` in the process.  Note that the mean of this
        AR(1) process, :math:`y`, is simply :math:`\mu/(1 - \rho)`
    n_std : scalar(int), optional(default=3)
        The number of standard deviations to approximate out to

    Returns
    -------

    mc : MarkovChain
        An instance of the MarkovChain class that stores the transition
        matrix and state values returned by the discretization method

    Note
    ----

    UserWarning: The API of `tauchen` was changed from
    `tauchen(rho, sigma_u, b=0., m=3, n=7)` to
    `tauchen(n, rho, sigma, mu=0., n_std=3)` in version 0.6.0.

    """
    warnings.warn("The API of tauchen has changed from `tauchen(rho, sigma_u, b=0., m=3, n=7)`"
                  " to `tauchen(n, rho, sigma, mu=0., n_std=3)`. To find more details please visit:"
                  " https://github.com/QuantEcon/QuantEcon.py/issues/663.",
                  UserWarning, stacklevel=2)

    # standard deviation of demeaned y_t
    std_y = np.sqrt(sigma**2 / (1 - rho**2))

    # top of discrete state space for demeaned y_t
    x_max = n_std * std_y

    # bottom of discrete state space for demeaned y_t
    x_min = -x_max

    # discretized state space for demeaned y_t
    x = np.linspace(x_min, x_max, n)

    step = (x_max - x_min) / (n - 1)
    half_step = 0.5 * step
    P = np.empty((n, n))

    # approximate Markov transition matrix for
    # demeaned y_t
    _fill_tauchen(x, P, n, rho, sigma, half_step)

    # shifts the state values by the long run mean of y_t
    mu = mu / (1 - rho)

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
