"""
Utilities to Support Random Operations and Generating Vectors and Matrices

"""

import numpy as np
from numba import guvectorize, types
from numba import TypingError
from numba.extending import overload
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

    random_state : int or np.random.RandomState/Generator, optional
        Random seed (integer) or np.random.RandomState or Generator
        instance to set the initial state of the random number generator
        for reproducibility. If None, a randomly initialized RandomState
        is used.

    parallel : bool(default=True)
        Whether to use multi-core CPU (parallel=True) or single-threaded
        CPU (parallel=False). (Internally the code is executed through
        Numba.guvectorize.) On Emscripten (JupyterLite), parallel=True
        executes serially via the Numba emscripten-forge patch 0007.

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
def draw(cdf, size=None, rng=None):
    """
    Generate a random sample according to the cumulative distribution
    given by `cdf`. JIT-compiled by Numba in nopython mode.

    Parameters
    ----------
    cdf : array_like(float, ndim=1)
        Array containing the cumulative distribution.

    size : scalar(int), optional(default=None)
        Size of the sample. If an integer is supplied, an ndarray of
        `size` independent draws is returned; otherwise, a single draw
        is returned as a scalar.

    rng : np.random.Generator, optional(default=None)
        Random number generator to draw from. Must be a
        `np.random.Generator` or None; in particular, integer seeds and
        `np.random.RandomState` are not accepted. If None, the global
        random state is used (see Notes).

    Returns
    -------
    scalar(int) or ndarray(int, ndim=1)

    Notes
    -----
    `draw` is intended primarily for use inside jit-compiled functions.
    Pass a `np.random.Generator` as `rng` whenever the draws should be
    reproducible; the generator is consumed in place, so its state
    advances across successive calls.

    `rng` accepts only None or a `Generator` because Numba's nopython
    mode can neither construct a generator from a seed nor represent
    `np.random.RandomState`. A jit-compiled caller passing anything
    else gets a compile-time `numba.TypingError`. Build a generator
    outside any jit-compiled function with `np.random.default_rng(seed)`
    and pass it in.

    If `rng` is None, a jit-compiled caller draws from Numba's own
    internal random state, which is seeded by calling `np.random.seed`
    inside a jit-compiled function; seeding NumPy's global random state
    from Python has no effect on it.

    A single `Generator` must not be shared across the iterations of a
    `numba.prange` loop. Its state is updated in place, so concurrent
    iterations race and can consume the same underlying random number;
    the draws are then neither reproducible nor independent. Give each
    iteration its own generator, spawned with `Generator.spawn` outside
    the jit-compiled function -- `spawn` itself cannot be called in
    nopython mode.

    Examples
    --------
    >>> import numpy as np
    >>> import quantecon as qe
    >>> cdf = np.cumsum([0.4, 0.6])
    >>> rng = np.random.default_rng(1234)
    >>> qe.random.draw(cdf, 10, rng=rng)
    array([1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
    >>> qe.random.draw(cdf, rng=np.random.default_rng(1234))
    np.int64(1)

    Inside a jit-compiled function:

    >>> from numba import njit
    >>> @njit
    ... def draw_jitted(cdf, size, rng):
    ...     return qe.random.draw(cdf, size, rng)
    >>> draw_jitted(cdf, 5, np.random.default_rng(1234))
    array([1, 0, 1, 0, 0])

    """
    if rng is None:
        rng = np.random
    if isinstance(size, int):
        rs = rng.random(size)
        out = np.searchsorted(cdf, rs, side='right')
        return out
    else:
        r = rng.random()
        return np.searchsorted(cdf, r, side='right')


def _is_no_rng(numba_type):
    """
    Return True if `numba_type`, as seen from inside the `draw`
    overload, means that `rng` was not supplied.

    Numba spells "no value" in more than one way depending on the call
    site, and each must be recognised: an omitted argument, as in
    ``draw(cdf, 10)``, arrives as the *Python* object `None`, while an
    explicit `None`, as in ``draw(cdf, 10, None)``, and a `None` default
    forwarded by a jit-compiled caller both arrive as
    `numba.types.none`. `numba.types.Omitted` is not currently observed
    in an `@overload` body but is handled for the benefit of other Numba
    typing templates.

    """
    return (numba_type is None or
            isinstance(numba_type, types.NoneType) or
            (isinstance(numba_type, types.Omitted) and
             numba_type.value is None))


# Overload for the `draw` function
#
# The implementations below must return the same values as the pure
# Python body above: the same random numbers drawn in the same order,
# and the same `searchsorted` result. The sized branches keep the
# hand-written loop rather than the array `np.searchsorted` used in the
# Python body -- Numba supports both and they agree exactly, but the
# loop is measurably faster. `TestDraw.test_python_jitted_agree` is what
# holds the two paths together.
@overload(draw)
def ol_draw(cdf, size=None, rng=None):
    if isinstance(rng, types.NumPyRandomGeneratorType):
        if isinstance(size, types.Integer):
            def draw_impl(cdf, size=None, rng=None):
                rs = rng.random(size)
                out = np.empty(size, dtype=np.int_)
                for i in range(size):
                    out[i] = np.searchsorted(cdf, rs[i], side='right')
                return out
        else:
            def draw_impl(cdf, size=None, rng=None):
                r = rng.random()
                return np.searchsorted(cdf, r, side='right')
    elif _is_no_rng(rng):
        if isinstance(size, types.Integer):
            def draw_impl(cdf, size=None, rng=None):
                rs = np.random.random(size)
                out = np.empty(size, dtype=np.int_)
                for i in range(size):
                    out[i] = np.searchsorted(cdf, rs[i], side='right')
                return out
        else:
            def draw_impl(cdf, size=None, rng=None):
                r = np.random.random()
                return np.searchsorted(cdf, r, side='right')
    else:
        if (isinstance(rng, types.Optional) and
                isinstance(rng.type, types.NumPyRandomGeneratorType)):
            hint = ('Numba could not prove that `rng` is a Generator '
                    'rather than None at this call site; hoist the None '
                    'case out of the branch that reaches `draw`.')
        else:
            hint = ('Integer seeds and np.random.RandomState are not '
                    'accepted, because nopython mode cannot construct a '
                    'generator. Build one outside the jit-compiled '
                    'function with np.random.default_rng(seed) and pass '
                    'that in.')
        raise TypingError(
            'quantecon.random.draw: `rng` must be None or an '
            f'np.random.Generator; got {rng}. {hint}'
        )
    return draw_impl
