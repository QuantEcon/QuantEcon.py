"""
Implements inequality and segregation measures such as Gini, Lorenz Curve

"""

import numpy as np
from numba import njit, prange


@njit
def lorenz_curve(y):
    """
    Calculates the Lorenz Curve, a graphical representation of
    the distribution of income or wealth.

    It returns the cumulative share of people (x-axis) and
    the cumulative share of income earned.

    Parameters
    ----------
    y : array_like(float or int, ndim=1)
        Array of income/wealth for each individual.
        Unordered or ordered is fine.

    Returns
    -------
    cum_people : array_like(float, ndim=1)
        Cumulative share of people for each person index (i/n)
    cum_income : array_like(float, ndim=1)
        Cumulative share of income for each person index


    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Lorenz_curve

    Examples
    --------
    >>> a_val, n = 3, 10_000
    >>> y = np.random.pareto(a_val, size=n)
    >>> f_vals, l_vals = lorenz(y)

    """

    n = y.shape[0]
    y = np.sort(y)
    _zero = np.zeros(1)
    s = np.concatenate((_zero, np.cumsum(y)))
    _cum_p = np.arange(1, n + 1) / n
    cum_income = s / s[n]
    cum_people = np.concatenate((_zero, _cum_p))
    return cum_people, cum_income


@njit(parallel=True)
def gini_coefficient(y):
    r"""
    Implements the Gini inequality index

    Parameters
    ----------
    y : array_like(float)
        Array of income/wealth for each individual.
        Ordered or unordered is fine

    Returns
    -------
    Gini index: float
        The gini index describing the inequality of the array of income/wealth

    References
    ----------

    https://en.wikipedia.org/wiki/Gini_coefficient
    """
    n = len(y)
    i_sum = 0
    t_sum = 0
    for i in prange(n):
        i_sum += np.sum(np.abs(y[i] - y))
        t_sum += y[i]
    return i_sum / (2 * n * t_sum)


def shorrocks_index(A):
    r"""
    Implements Shorrocks mobility index

    Parameters
    ----------
    A : array_like(float)
        Square matrix with transition probabilities (mobility matrix) of
        dimension m

    Returns
    -------
    Shorrocks index: float
        The Shorrocks mobility index calculated as

        .. math::

            s(A) = \frac{m - \sum_j a_{jj} }{m - 1} \in (0, 1)

        An index equal to 0 indicates complete immobility.

    References
    ----------
    .. [1] Wealth distribution and social mobility in the US:
       A quantitative approach (Benhabib, Bisin, Luo, 2017).
       https://www.econ.nyu.edu/user/bisina/RevisionAugust.pdf
    """

    A = np.asarray(A)  # Convert to array if not already
    m, n = A.shape

    if m != n:
        raise ValueError('A must be a square matrix')

    diag_sum = np.diag(A).sum()

    return (m - diag_sum) / (m - 1)


def rank_size(data, c=1.0):
    """
    Generate rank-size data corresponding to distribution data.

    Examples
    --------
    >>> y = np.exp(np.random.randn(1000))  # simulate data
    >>> rank_data, size_data = rank_size(y, c=0.85)

    Parameters
    ----------
    data : array_like
        the set of observations
    c : int or float
        restrict plot to top (c x 100)% of the distribution

    Returns
    -------
    rank_data : array_like(float, ndim=1)
        Location in the population when sorted from smallest to largest
    size_data : array_like(float, ndim=1)
        Size data for top (c x 100)% of the observations
    """
    w = - np.sort(- data)                  # Reverse sort
    w = w[:int(len(w) * c)]                # extract top (c * 100)%
    rank_data = np.arange(len(w)) + 1
    size_data = w
    return rank_data, size_data
