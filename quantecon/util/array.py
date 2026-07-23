"""
Array Utilities
===============

Array
-----
searchsorted

"""

import warnings
from numba import jit, objmode

# ----------------- #
# -ARRAY UTILITIES- #
# ----------------- #

@jit(nopython=True)
def searchsorted(a, v):
    """
    Custom version of np.searchsorted. Return the largest index `i` such
    that `a[i-1] <= v < a[i]` (for `i = 0`, `v < a[0]`); if `a[n-1] <=
    v`, return `n`, where `n = len(a)`.

    .. deprecated:: 0.11.0

        Use `np.searchsorted(a, v, side='right')` instead.

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
    This routine is jit-compiled by Numba in nopython mode.

    Examples
    --------
    >>> import numpy as np
    >>> from quantecon.util import searchsorted
    >>> a = np.array([0.2, 0.4, 1.0])
    >>> searchsorted(a, 0.1)
    0
    >>> searchsorted(a, 0.4)
    2
    >>> searchsorted(a, 2)
    3

    """
    with objmode():
        warnings.warn(
            "`searchsorted(a, v)` is deprecated. "
            "Use `np.searchsorted(a, v, side='right')` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    lo = -1
    hi = len(a)
    while(lo < hi-1):
        m = (lo + hi) // 2
        if v < a[m]:
            hi = m
        else:
            lo = m
    return hi
