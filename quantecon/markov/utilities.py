"""
Utility routines for the markov submodule

"""
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def sa_indices(num_states, num_actions):
    """
    Generate `s_indices` and `a_indices` for `DiscreteDP`, for the case
    where all the actions are feasible at every state.

    Parameters
    ----------
    num_states : scalar(int)
        Number of states.

    num_actions : scalar(int)
        Number of actions.

    Returns
    -------
    s_indices : ndarray(int, ndim=1)
        Array containing the state indices.

    a_indices : ndarray(int, ndim=1)
        Array containing the action indices.

    Examples
    --------
    >>> s_indices, a_indices = qe.markov.sa_indices(4, 3)
    >>> s_indices
    array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    >>> a_indices
    array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    """
    L = num_states * num_actions
    dtype = np.int_
    s_indices = np.empty(L, dtype=dtype)
    a_indices = np.empty(L, dtype=dtype)

    i = 0
    for s in range(num_states):
        for a in range(num_actions):
            s_indices[i] = s
            a_indices[i] = a
            i += 1

    return s_indices, a_indices


@jit(nopython=True, cache=True)
def _fill_dense_Q(s_indices, a_indices, Q_in, Q_out):
    L = Q_in.shape[0]
    for i in range(L):
        Q_out[s_indices[i], a_indices[i], :] = Q_in[i, :]

    return Q_out


@jit(nopython=True, cache=True)
def _s_wise_max_argmax(a_indices, a_indptr, vals, out_max, out_argmax):
    n = len(out_max)
    for i in range(n):
        if a_indptr[i] != a_indptr[i+1]:
            m = a_indptr[i]
            for j in range(a_indptr[i]+1, a_indptr[i+1]):
                if vals[j] > vals[m]:
                    m = j
            out_max[i] = vals[m]
            out_argmax[i] = a_indices[m]


@jit(nopython=True, cache=True)
def _s_wise_max(a_indices, a_indptr, vals, out_max):
    n = len(out_max)
    for i in range(n):
        if a_indptr[i] != a_indptr[i+1]:
            m = a_indptr[i]
            for j in range(a_indptr[i]+1, a_indptr[i+1]):
                if vals[j] > vals[m]:
                    m = j
            out_max[i] = vals[m]


@jit(nopython=True, cache=True)
def _find_indices(a_indices, a_indptr, sigma, out):
    n = len(sigma)
    for i in range(n):
        for j in range(a_indptr[i], a_indptr[i+1]):
            if sigma[i] == a_indices[j]:
                out[i] = j


@jit(nopython=True, cache=True)
def _has_sorted_sa_indices(s_indices, a_indices):
    """
    Check whether `s_indices` and `a_indices` are sorted in
    lexicographic order.

    Parameters
    ----------
    s_indices, a_indices : ndarray(ndim=1)

    Returns
    -------
    bool
        Whether `s_indices` and `a_indices` are sorted.

    """
    L = len(s_indices)
    for i in range(L-1):
        if s_indices[i] > s_indices[i+1]:
            return False
        if s_indices[i] == s_indices[i+1]:
            if a_indices[i] >= a_indices[i+1]:
                return False
    return True


@jit(nopython=True, cache=True)
def _generate_a_indptr(num_states, s_indices, out):
    """
    Generate `a_indptr`; stored in `out`. `s_indices` is assumed to be
    in sorted order.

    Parameters
    ----------
    num_states : scalar(int)

    s_indices : ndarray(int, ndim=1)

    out : ndarray(int, ndim=1)
        Length must be num_states+1.

    """
    idx = 0
    out[0] = 0
    for s in range(num_states-1):
        while(s_indices[idx] == s):
            idx += 1
        out[s+1] = idx
    out[num_states] = len(s_indices)
