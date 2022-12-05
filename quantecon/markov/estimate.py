import numpy as np
from numba import njit
from .core import MarkovChain


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
