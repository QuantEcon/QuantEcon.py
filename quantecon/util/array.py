"""
Array Utilities
===============

Array
-----
searchsorted (deprecated - use np.searchsorted with side='right' instead)

"""

import warnings
import numpy as np

# ----------------- #
# -ARRAY UTILITIES- #
# ----------------- #

def searchsorted(a, v):
    """
    .. deprecated:: 0.10.2
        `searchsorted` is deprecated and will be removed in a future version.
        Use `np.searchsorted(a, v, side='right')` instead.

    Return the largest index `i` such that `a[i-1] <= v < a[i]` (for
    `i = 0`, `v < a[0]`); if `a[n-1] <= v`, return `n`, where `n =
    len(a)`.

    This function is now a thin wrapper around `np.searchsorted(a, v,
    side='right')` and emits a deprecation warning when called.

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
    This routine was originally jit-compiled when Numba did not support
    the `side` keyword argument for `np.searchsorted`. Now that Numba
    supports this feature, this function is deprecated in favor of using
    `np.searchsorted(a, v, side='right')` directly.

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
    warnings.warn(
        "searchsorted is deprecated and will be removed in a future version. "
        "Use np.searchsorted(a, v, side='right') instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return np.searchsorted(a, v, side='right')
