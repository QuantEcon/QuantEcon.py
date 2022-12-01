import numpy as np
from numba import njit
from .core import MarkovChain

def estimate_mc_discrete(index_series, state_values=None):
    r"""
    Estimates the Markov chain associated with discrete-valued
    time series data :math:`(X_0, ..., X_{k-1})` that is assumed to be
    Markovian.  The estimation is by maximum likelihood. Transition
    probabilities are returned as a matrix :math:`P` where :math:`P[i, j] =
    N_{ij} / N_i`.  Here :math:`N_{ij}` is :math:`\sum_{t=0}^{k-1} 1\{X_t=i,\;
    X_{t+1}=j\}`, the number of transitions from i to j, while :math:`N_i` is
    total number of visits to i.


    Parameters
    ----------

    index_series : array_like (int, ndim=1)
        A time series in index form, from which the transition matrix
        will be estimated.  A sequence (i_0, i_1, ..., i_{k-1}) where
        element i_t indicates that the Markov chain is in state
        state_values[i_t] at time t.

    state_values : array_like (default=None, ndim=1)
        A flat array_like of length n containing the values associated with the
        states.   If state_values is None, then state_values is set to the
        integer sequence (0, ..., n-1), where n = max(index_series).

    Returns
    -------

    mc : MarkovChain
        A MarkovChain object where mc.P is a stochastic matrix estimated from
        the index_series data and mc.state_values is the states of the Markov
        chain.

    """

    if state_values is None:
        state_values = np.arange(max(index_series)+1)

    k = len(index_series)
    n = len(state_values)

    state_values = np.asarray(state_values)

    # Compute the number of visits to each state and the frequency of
    # transiting from state i to state
    state_counter = np.zeros(n, dtype=int)
    trans_counter = np.zeros((n, n), dtype=int)
    _fill_counters(state_counter, trans_counter, index_series)

    # Cut states where the column sum of trans_counter is zero (i.e.,
    # inaccesible states according to the simulation)
    zero_index = np.where(np.sum(trans_counter, axis=0) == 0)
    trans_counter = np.delete(trans_counter, zero_index, axis=0)
    trans_counter = np.delete(trans_counter, zero_index, axis=1)
    state_values = np.delete(state_values, zero_index)

    P = trans_counter / trans_counter.sum(1)[:, np.newaxis]

    mc = MarkovChain(P, state_values=state_values)
    return mc

@njit
def _fill_counters(state_counter, trans_counter, index_series):
    i = index_series[0]
    for j in index_series[1:]:
        state_counter[i] += 1
        trans_counter[i, j] += 1
        i = j
