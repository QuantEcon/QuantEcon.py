"""
Filename: compute_fp.py
Authors: Thomas Sargent, John Stachurski

Compute the fixed point of a given operator T, starting from
specified initial condition v.

"""
import time
import numpy as np


def _print_after_skip(skip, it=None, dist=None, etime=None):
    if it is None:
        # print initial header
        msg = "{i:<13}{d:<15}{t:<17}".format(i="Iteration",
                                             d="Distance",
                                             t="Elapsed (seconds)")
        print(msg)
        print("-" * len(msg))

        return

    if it % skip == 0:
        if etime is None:
            print("After {it} iterations dist is {d}".format(it=it, d=dist))

        else:
            # leave 4 spaces between columns if we have %3.3e in d, t
            msg = "{i:<13}{d:<15.3e}{t:<18.3e}"
            print(msg.format(i=it, d=dist, t=etime))

    return


def compute_fixed_point(T, v, error_tol=1e-3, max_iter=50, verbose=1,
                        print_skip=5, *args, **kwargs):
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

    if verbose:
        start_time = time.time()
        _print_after_skip(print_skip, it=None)

    while iterate < max_iter and error > error_tol:
        new_v = T(v, *args, **kwargs)
        iterate += 1
        error = np.max(np.abs(new_v - v))

        if verbose:
            etime = time.time() - start_time
            _print_after_skip(print_skip, iterate, error, etime)

        try:
            v[:] = new_v
        except TypeError:
            v = new_v
    return v
