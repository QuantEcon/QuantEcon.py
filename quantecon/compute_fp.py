"""
Filename: compute_fp.py
Authors: Thomas Sargent, John Stachurski

Compute the fixed point of a given operator T, starting from
specified initial condition v.

"""

import numpy as np


def compute_fixed_point(T, v, error_tol=1e-3, max_iter=50, verbose=1, *args,
                        **kwargs):
    """
    Computes and returns :math:`T^k v`, an approximate fixed point.

    Here T is an operator, v is an initial condition and k is the number
    of iterates. Provided that T is a contraction mapping or similar,
    :math:`T^k v` will be an approximation to the fixed point.

    Parameters
    ----------
    T : callable
        A callable object (e.g., function) that acts on v
    v : object
        An object such that T(v) is defined
    error_tol : scalar(float), optional(default=1e-3)
        Error tolerance
    max_iter : scalar(int), optional(default=50)
        Maximum number of iterations
    verbose : bool, optional(default=True)
        If True then print current error at each iterate.
    args, kwargs :
        Other arguments and keyword arguments that are passed directly
        to  the function T each time it is called

    Returns
    -------
    v : object
        The approximate fixed point

    """
    iterate = 0
    error = error_tol + 1
    while iterate < max_iter and error > error_tol:
        new_v = T(v, *args, **kwargs)
        iterate += 1
        error = np.max(np.abs(new_v - v))
        if verbose:
            print("Computed iterate %d with error %f" % (iterate, error))
        try:
            v[:] = new_v
        except TypeError:
            v = new_v
    return v
