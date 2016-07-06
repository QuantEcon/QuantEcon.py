"""
Utilities to Support Random Operations and Generating Vectors and Matrices
"""

import numpy as np
from numba import jit, guvectorize

from ..util import check_random_state


# Generating Arrays and Vectors #

def probvec(m, k, random_state=None, parallel=True):
    """
    Return m randomly sampled probability vectors of dimension k.

    Parameters
    ----------
    m : scalar(int)
        Number of probability vectors.

    k : scalar(int)
        Dimension of each probability vectors.

    random_state : scalar(int) or np.random.RandomState,
                   optional(default=None)
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    parallel : bool(default=True)
        Whether to use multi-core CPU (parallel=True) or single-threaded
        CPU (parallel=False). (Internally the code is executed through
        Numba.guvectorize.)

    Returns
    -------
    x : ndarray(float, ndim=2)
        Array of shape (m, k) containing probability vectors as rows.

    Examples
    --------
    >>> qe.random.probvec(2, 3, random_state=1234)
    array([[ 0.19151945,  0.43058932,  0.37789123],
           [ 0.43772774,  0.34763084,  0.21464142]])

    """
    if k == 1:
        return np.ones((m, k))

    # if k >= 2
    random_state = check_random_state(random_state)
    r = random_state.random_sample(size=(m, k-1))
    x = np.empty((m, k))

    # Parse Parallel Option #
    if parallel:
        _probvec_parallel(r, x)
    else:
        _probvec_cpu(r, x)

    return x


def _probvec(r, out):
    """
    Fill `out` with randomly sampled probability vectors as rows.

    To be complied as a ufunc by guvectorize of Numba. The inputs must
    have the same shape except the last axis; the length of the last
    axis of `r` must be that of `out` minus 1, i.e., if out.shape[-1] is
    k, then r.shape[-1] must be k-1.

    Parameters
    ----------
    r : ndarray(float)
        Array containing random values in [0, 1).

    out : ndarray(float)
        Output array.

    """
    n = r.shape[0]
    r.sort()
    out[0] = r[0]
    for i in range(1, n):
        out[i] = r[i] - r[i-1]
    out[n] = 1 - r[n-1]

_probvec_parallel = guvectorize(
    ['(f8[:], f8[:])'], '(n), (k)', nopython=True, target='parallel'
    )(_probvec)
_probvec_cpu = guvectorize(
    ['(f8[:], f8[:])'], '(n), (k)', nopython=True, target='cpu'
    )(_probvec)


@jit
def sample_without_replacement(n, k, num_trials=None, random_state=None):
    """
    Randomly choose k integers without replacement from 0, ..., n-1.

    Parameters
    ----------
    n : scalar(int)
        Number of integers, 0, ..., n-1, to sample from.

    k : scalar(int)
        Number of integers to sample.

    num_trials : scalar(int), optional(default=None)
        Number of trials.

    random_state : scalar(int) or np.random.RandomState,
                   optional(default=None)
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    result : ndarray(int, ndim=1 or 2)
        Array of shape (k,) if num_trials is None, or of shape
        (num_trials, k) otherwise, (each row of) which contains k unique
        random elements chosen from 0, ..., n-1.

    Examples
    --------
    >>> qe.random.sample_without_replacement(5, 3, random_state=1234)
    array([0, 2, 1])
    >>> qe.random.sample_without_replacement(5, 3, num_trials=4,
    ...                                      random_state=1234)
    array([[0, 2, 1],
           [3, 4, 0],
           [1, 3, 2],
           [4, 1, 3]])

    """
    if n <= 0:
        raise ValueError('n must be greater than 0')
    if k > n:
        raise ValueError('k must be smaller than or equal to n')

    m = 1 if num_trials is None else num_trials

    random_state = check_random_state(random_state)
    r = random_state.random_sample(size=(m, k))

    # Logic taken from random.sample in the standard library
    result = np.empty((m, k), dtype=int)
    pool = np.empty(n, dtype=int)
    for i in range(m):
        for j in range(n):
            pool[j] = j
        for j in range(k):
            idx = int(np.floor(r[i, j] * (n-j)))  # np.floor returns a float
            result[i, j] = pool[idx]
            pool[idx] = pool[n-j-1]

    if num_trials is None:
        return result[0]
    else:
        return result
