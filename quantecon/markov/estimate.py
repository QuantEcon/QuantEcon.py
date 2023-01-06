import numpy as np
from numba import njit
from .core import MarkovChain
from .._gridtools import cartesian_nearest_index, cartesian


def estimate_mc(X):
    r"""
    Estimate the Markov chain associated with a time series :math:`X =
    (X_0, \ldots, X_{T-1})` assuming that the state space is the finite
    set :math:`\{X_0, \ldots, X_{T-1}\}` (duplicates removed). The
    estimation is by maximum likelihood. The estimated transition
    probabilities are given by the matrix :math:`P` such that
    :math:`P[i, j] = N_{ij} / N_i`, where :math:`N_{ij} =
    \sum_{t=0}^{T-1} 1_{\{X_t=s_i, X_{t+1}=s_j\}}`, the number of
    transitions from state :math:`s_i` to state :math:`s_j`, while
    :math:`N_i` is the total number of visits to :math:`s_i`. The result
    is returned as a `MarkovChain` instance.

    Parameters
    ----------
    X : array_like
        A time series of state values, from which the transition matrix
        will be estimated, where `X[t]` contains the t-th observation.

    Returns
    -------
    mc : MarkovChain
        A MarkovChain instance where `mc.P` is a stochastic matrix
        estimated from the data `X` and `mc.state_values` is an array of
        values that appear in `X` (sorted in ascending order).

    """
    X = np.asarray(X)
    axis = 0 if X.ndim > 1 else None
    state_values, indices = np.unique(X, return_inverse=True, axis=axis)

    n = len(state_values)
    P = np.zeros((n, n))  # dtype=float to modify in place upon normalization
    P = _count_transition_frequencies(indices, P)
    P /= P.sum(1)[:, np.newaxis]

    mc = MarkovChain(P, state_values=state_values)
    return mc


@njit(cache=True)
def _count_transition_frequencies(index_series, trans_counter):
    T = len(index_series)
    i = index_series[0]
    for t in range(1, T):
        j = index_series[t]
        trans_counter[i, j] += 1
        i = j
    return trans_counter


def fit_discrete_mc(X, grids, order='C'):
    r"""
    Function that takes an arbitrary time series :math: `(X_t)_{t=0}^{T-1}` in
    :math: `\mathbb R^n` plus a set of grid points in each dimension and converts
    it to a MarkovChain by first applying discretization onto the grid
    and then estimation of the Markov chain.

    Parameters
    ----------

    X: array_like(ndim=2)
        Time-series such that the t-th row is :math:`x_t`.
        It should be of the shape T x n, where n is the number of dimensions.

    grids: array_like(array_like(ndim=1))
        Array of `n` sorted arrays. Set of grid points in each dimension

    Examples
    --------

    >>> grids = (np.arange(3), np.arange(2))
    >>> X = [(-0.1, 1.2), (2, 0), (0.6, 0.4), (1.0, 0.1)]
    >>> mc = fit_discrete_mc(X, grids)
    >>> mc.state_values
    array([[0, 1],
           [1, 0],
           [2, 0]])
    >>> mc.P
    array([[0., 0., 1.],
           [0., 1., 0.],
           [0., 1., 0.]])

    Returns
    -------

    mc: MarkovChain
        An instance of the MarkovChain class constructed after discretization
        onto the grid.
    """
    X_indices = cartesian_nearest_index(X, grids, order=order)
    mc = estimate_mc(X_indices)
    # Assign the visited states in the cartesian product as the state values
    prod = cartesian(grids, order=order)
    mc.state_values = prod[mc.state_values]
    return mc
