"""
Implements inequality and segregation measures such as Gini, Lorenz Curve

"""

import numpy as np
from numba import njit, prange


@njit
def lorenz_curve(y):
    """
    Calculates the Lorenz Curve, a graphical representation of the distribution of income
    or wealth.

    It returns the cumulative share of people (x-axis) and the cumulative share of income earned

    Parameters
    ----------
    y : array_like(float or int, ndim=1)
        Array of income/wealth for each individual. Unordered or ordered is fine.

    Returns
    -------
    cum_people : array_like(float, ndim=1)
        Cumulative share of people for each person index (i/n)
    cum_income : array_like(float, ndim=1)
        Cumulative share of income for each person index


    References
    ----------
    https://en.wikipedia.org/wiki/Lorenz_curve

    Examples
    --------
    a_val, n = 3, 10_000
    y = np.random.pareto(a_val, size=n)
    f_vals, l_vals = lorenz(y)
    #Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(f_vals, l_vals, label="Pareto with a={0}".format(a_val))
    fig.suptitle("Pareto distribution with a={0}".format(a_val))

    """

    n = len(y)
    y = np.sort(y)
    s = np.zeros(n + 1)
    s[1:] = np.cumsum(y)
    cum_people = np.zeros(n + 1)
    cum_income = np.zeros(n + 1)
    for i in range(1, n + 1):
        cum_people[i] = i / n
        cum_income[i] = s[i] / s[n]
    return cum_people, cum_income


@njit(parallel=True)
def gini_coefficient(y):
    r"""
    Implements the Gini inequality index

    Parameters
    -----------
    y : array_like(float)
        Array of income/wealth for each individual. Ordered or unordered is fine

    Returns
    -------
    Gini index: float
        The gini index describing the inequality of the array of income/wealth

    References
    ----------

    https://en.wikipedia.org/wiki/Gini_coefficient
    """
    n = len(y)
    i_sum = np.zeros(n)
    for i in prange(n):
        for j in range(n):
            i_sum[i] += abs(y[i] - y[j])
    return np.sum(i_sum) / (2 * n * np.sum(y))


