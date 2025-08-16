"""
Collection of functions to approximate a continuous random process by a
discrete Markov chain.

"""

from math import erfc, sqrt
from .core import MarkovChain
from .estimate import fit_discrete_mc
from .._lss import simulate_linear_model
from .._matrix_eqn import solve_discrete_lyapunov
from ..util import check_random_state

import warnings
import numpy as np
import numbers
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

    Notes
    -----

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

    Notes
    -----

    UserWarning: The API of `tauchen` was changed from
    `tauchen(rho, sigma_u, b=0., m=3, n=7)` to
    `tauchen(n, rho, sigma, mu=0., n_std=3)` in version 0.6.0.

    """

    if not isinstance(n, numbers.Integral):
        warnings.warn(
            "The API of tauchen has changed from `tauchen(rho, sigma_u, b=0., m=3, n=7)`"
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


def discrete_var(A,
                 C,
                 grid_sizes=None,
                 std_devs=np.sqrt(10),
                 sim_length=1_000_000,
                 rv=None,
                 order='C',
                 random_state=None):
    r"""
    Generate an `MarkovChain` instance that discretizes a multivariate
    autorregressive process by a simulation of the process.

    This function discretizes a VAR(1) process of the form:

    .. math::

        x_t = A x_{t-1} + C u_t

    where :math:`{u_t}` is drawn iid from a distribution with mean 0 and
    unit standard deviaiton; and :math:`{C}` is a volatility matrix.
    Internally, from this process a sample time series of length
    `sim_length` is produced, and with a cartesian grid as specified by
    `grid_sizes` and `std_devs` a Markov chain is estimated that fits
    the time series, where the states that are never visited under the
    simulation are removed.

    For a mathematical derivation check *Finite-State Approximation Of
    VAR Processes: A Simulation Approach* by Stephanie Schmitt-Grohé and
    Martín Uribe, July 11, 2010. In particular, we follow Schmitt-Grohé
    and Uribe's method in contructing the grid for approximation.

    Parameters
    ----------
    A : array_like(float, ndim=2)
        An m x m matrix containing the process' autocorrelation
        parameters. Its eigenvalues are assumed to have moduli 
        bounded by unity.
    C : array_like(float, ndim=2)
        An m x r volatility matrix
    grid_sizes : array_like(int, ndim=1), optional(default=None)
        An m-vector containing the number of grid points in the
        discretization of each dimension of x_t. If None, then set to
        (10, ..., 10).
    std_devs : float, optional(default=np.sqrt(10))
        The number of standard deviations the grid should stretch in
        each dimension, where standard deviations are measured under the
        stationary distribution.
    sim_length : int, optional(default=1_000_000)
        The length of the simulated time series.
    rv : optional(default=None)
        Object that represents the disturbance term u_t. If None, then
        standard normal distribution from numpy.random is used.
        Alternatively, one can pass a "frozen" object of a multivariate
        distribution from `scipy.stats`. It must have a zero mean and
        unit standard deviation (of dimension r).
    order : str, optional(default='C')
        ('C' or 'F') order in which the states in the cartesian grid are
        enumerated.
    random_state : int or np.random.RandomState/Generator, optional
        Random seed (integer) or np.random.RandomState or Generator
        instance to set the initial state of the random number generator
        for reproducibility. If None, a randomly initialized RandomState
        is used.

    Returns
    -------
    mc : MarkovChain
        An instance of the MarkovChain class that stores the transition
        matrix and state values returned by the discretization method,
        in the following attributes:

            `P` : ndarray(float, ndim=2)
                A 2-dim array containing the transition probability
                matrix over the discretized states.

            `state_values` : ndarray(float, ndim=2)
                A 2-dim array containing the state vectors (of dimension
                m) as rows, which are ordered according to the `order`
                option.

    Examples
    --------
    This example discretizes the stochastic process used to calibrate
    the economic model included in "Downward Nominal Wage Rigidity,
    Currency Pegs, and Involuntary Unemployment" by Stephanie
    Schmitt-Grohé and Martín Uribe, Journal of Political Economy 124,
    October 2016, 1466-1514.

    >>> rng = np.random.default_rng(12345)
    >>> A = [[0.7901, -1.3570],
    ...      [-0.0104, 0.8638]]
    >>> Omega = [[0.0012346, -0.0000776],
    ...          [-0.0000776, 0.0000401]]
    >>> C = scipy.linalg.sqrtm(Omega)
    >>> grid_sizes = [21, 11]
    >>> mc = discrete_var(A, C, grid_sizes, random_state=rng)
    >>> mc.P.shape
    (145, 145)
    >>> mc.state_values.shape
    (145, 2)
    >>> mc.state_values[:10]  # First 10 states
    array([[-0.38556417,  0.02155098],
           [-0.38556417,  0.03232648],
           [-0.38556417,  0.04310197],
           [-0.38556417,  0.05387746],
           [-0.34700776,  0.01077549],
           [-0.34700776,  0.02155098],
           [-0.34700776,  0.03232648],
           [-0.34700776,  0.04310197],
           [-0.34700776,  0.05387746],
           [-0.30845134,  0.        ]])
    >>> mc.simulate(10, random_state=rng)
    array([[ 0.11566925, -0.01077549],
           [ 0.11566925, -0.01077549],
           [ 0.15422567,  0.        ],
           [ 0.15422567,  0.        ],
           [ 0.15422567, -0.01077549],
           [ 0.11566925, -0.02155098],
           [ 0.11566925, -0.03232648],
           [ 0.15422567, -0.03232648],
           [ 0.15422567, -0.03232648],
           [ 0.19278209, -0.03232648]])

    The simulation below uses the same parameters with :math:`{u_t}`
    drawn from a multivariate t-distribution

    >>> df = 100
    >>> Sigma = np.diag(np.tile((df-2)/df, 2))
    >>> mc = discrete_var(A, C, grid_sizes, 
    ...              rv=scipy.stats.multivariate_t(shape=Sigma, df=df), 
    ...              random_state=rng)
    >>> mc.P.shape
    (146, 146)
    >>> mc.state_values.shape
    (146, 2)
    >>> mc.state_values[:10]
    array([[-0.38556417,  0.02155098],
           [-0.38556417,  0.03232648],
           [-0.38556417,  0.04310197],
           [-0.38556417,  0.05387746],
           [-0.34700776,  0.01077549],
           [-0.34700776,  0.02155098],
           [-0.34700776,  0.03232648],
           [-0.34700776,  0.04310197],
           [-0.34700776,  0.05387746],
           [-0.30845134,  0.        ]])
    >>> mc.simulate(10, random_state=rng)
    array([[-0.03855642, -0.02155098],
           [ 0.03855642, -0.03232648],
           [ 0.07711283, -0.03232648],
           [ 0.15422567, -0.03232648],
           [ 0.15422567, -0.04310197],
           [ 0.15422567, -0.03232648],
           [ 0.15422567, -0.03232648],
           [ 0.2313385 , -0.04310197],
           [ 0.2313385 , -0.03232648],
           [ 0.26989492, -0.03232648]])
    """
    A = np.asarray(A)
    C = np.asarray(C)
    m, r = C.shape

    # Run simulation to compute transition probabilities
    random_state = check_random_state(random_state)

    if rv is None:
        u = random_state.standard_normal(size=(sim_length-1, r))
    else:
        u = rv.rvs(size=sim_length-1, random_state=random_state)
    
    v = C @ u.T
    x0 = np.zeros(m)
    X = simulate_linear_model(A, x0, v, ts_length=sim_length)

    # Compute stationary variance-covariance matrix of AR process and use
    # it to obtain grid bounds.
    Sigma = solve_discrete_lyapunov(A, C @ C.T)
    sigma_vector = np.sqrt(np.diagonal(Sigma))    # Stationary std dev
    upper_bounds = std_devs * sigma_vector

    # Build the individual grids along each dimension
    if grid_sizes is None:
        # Set the size of every grid to default_grid_size
        default_grid_size = 10
        grid_sizes = np.full(m, default_grid_size)

    V = [np.linspace(-upper_bounds[i], upper_bounds[i], grid_sizes[i])
         for i in range(m)]

    # Fit the Markov chain
    mc = fit_discrete_mc(X.T, V, order=order)

    return mc
