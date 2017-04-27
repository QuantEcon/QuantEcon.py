"""
Utility routines for the markov submodule

"""
import numpy as np
from numba import jit

@jit(nopython=True)
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
