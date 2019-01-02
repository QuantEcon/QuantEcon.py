"""
Array Utilities
===============

Array
-----
searchsorted

"""

from numba import jit

# ----------------- #
# -ARRAY UTILITIES- #
# ----------------- #

@jit(nopython=True)
def searchsorted(a, v):
    """
    Custom version of np.searchsorted. Return the largest index `i` such
    that `a[i-1] <= v < a[i]` (for `i = 0`, `v < a[0]`); if `v[n-1] <=
    v`, return `n`, where `n = len(a)`.

    Parameters
    ----------
    a : ndarray(float, ndim=1)
        Input array. Must be sorted in ascending order.

    v : scalar(float)
        Value to be compared with elements of `a`.

    Returns
    -------
    scalar(int)
        Largest index `i` such that `a[i-1] <= v < a[i]`, or len(a) if
        no such index exists.

    Notes
    -----
    This routine is jit-complied if the module Numba is vailable; if
    not, it is an alias of np.searchsorted(a, v, side='right').

    Examples
    --------
    >>> a = np.array([0.2, 0.4, 1.0])
    >>> searchsorted(a, 0.1)
    0
    >>> searchsorted(a, 0.4)
    2
    >>> searchsorted(a, 2)
    3

    """
    lo = -1
    hi = len(a)
    while(lo < hi-1):
        m = (lo + hi) // 2
        if v < a[m]:
            hi = m
        else:
            lo = m
    return hi
