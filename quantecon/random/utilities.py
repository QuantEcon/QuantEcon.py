"""
Utilities to Support Random Operations and Generating Vectors and Matrices

"""

import numpy as np
from numba import guvectorize, types
from numba.extending import overload
from ..util import check_random_state, searchsorted


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

    random_state : int or np.random.RandomState/Generator, optional
        Random seed (integer) or np.random.RandomState or Generator
        instance to set the initial state of the random number generator
        for reproducibility. If None, a randomly initialized RandomState
        is used.

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
    r = random_state.random(size=(m, k-1))
    x = np.empty((m, k))

    # Parse Parallel Option #
    if parallel:
        _probvec_parallel(r, x)
    else:
        _probvec_cpu(r, x)

    return x


def _probvec(r, out):  # pragma: no cover
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
    ['(f8[:], f8[:])'], '(n), (k)', nopython=True, target='parallel',
    cache=True
    )(_probvec)
_probvec_cpu = guvectorize(
    ['(f8[:], f8[:])'], '(n), (k)', nopython=True, target='cpu',
    cache=True
    )(_probvec)


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

    random_state : int or np.random.RandomState/Generator, optional
        Random seed (integer) or np.random.RandomState or Generator
        instance to set the initial state of the random number generator
        for reproducibility. If None, a randomly initialized RandomState
        is used.

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

    size = k if num_trials is None else (num_trials, k)

    random_state = check_random_state(random_state)
    r = random_state.random(size=size)
    result = _sample_without_replacement(n, r)

    return result


@guvectorize(['(i8, f8[:], i8[:])'], '(),(k)->(k)', nopython=True, cache=True)
def _sample_without_replacement(n, r, out):
    """
    Main body of `sample_without_replacement`. To be complied as a ufunc
    by guvectorize of Numba.

    """
    k = r.shape[0]

    # Logic taken from random.sample in the standard library
    pool = np.arange(n)
    for j in range(k):
        idx = np.intp(np.floor(r[j] * (n-j)))  # np.floor returns a float
        out[j] = pool[idx]
        pool[idx] = pool[n-j-1]


# Pure python implementation that will run if the JIT compiler is disabled
def draw(cdf, size=None):
    """
    Generate a random sample according to the cumulative distribution
    given by `cdf`. Jit-complied by Numba in nopython mode.

    Parameters
    ----------
    cdf : array_like(float, ndim=1)
        Array containing the cumulative distribution.

    size : scalar(int), optional(default=None)
        Size of the sample. If an integer is supplied, an ndarray of
        `size` independent draws is returned; otherwise, a single draw
        is returned as a scalar.

    Returns
    -------
    scalar(int) or ndarray(int, ndim=1)

    Examples
    --------
    >>> cdf = np.cumsum([0.4, 0.6])
    >>> qe.random.draw(cdf)
    1
    >>> qe.random.draw(cdf, 10)
    array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])

    """
    if isinstance(size, int):
        rs = np.random.random(size)
        out = np.empty(size, dtype=np.int_)
        for i in range(size):
            out[i] = searchsorted(cdf, rs[i])
        return out
    else:
        r = np.random.random()
        return searchsorted(cdf, r)


# Overload for the `draw` function
@overload(draw)
def ol_draw(cdf, size=None):
    if isinstance(size, types.Integer):
        def draw_impl(cdf, size=None):
            rs = np.random.random(size)
            out = np.empty(size, dtype=np.int_)
            for i in range(size):
                out[i] = searchsorted(cdf, rs[i])
            return out
    else:
        def draw_impl(cdf, size=None):
            r = np.random.random()
            return searchsorted(cdf, r)
    return draw_impl
