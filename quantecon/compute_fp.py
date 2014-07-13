"""
Filename: compute_fp.py
Authors: Thomas Sargent, John Stachurski 

Compute the fixed point of a given operator T, starting from 
specified initial condition v.
"""

import numpy as np

def compute_fixed_point(T, v, error_tol=1e-3, max_iter=50, verbose=1):
    """
    Computes and returns T^k v, an approximate fixed point.
    
    Here T is an operator, v is an initial condition and k is the number of
    iterates. Provided that T is a contraction mapping or similar, 
    T^k v will be an approximation to the fixed point.

    Parameters
    ----------
        T : callable
            A callable object (e.g., function) that acts on v
        v : object
            An object such that T(v) is defined 
        error_tol : float, optional
            Error tolerance
        max_iter : int, optional
            Maximum number of iterations
        verbose : bool, optional
            If true then print current error at each iterate

    Returns
    -------
        v : object
            The approximate fixed point

    """
    iterate = 0 
    error = error_tol + 1
    while iterate < max_iter and error > error_tol:
        new_v = T(v)
        iterate += 1
        error = np.max(np.abs(new_v - v))
        if verbose:
            print "Computed iterate %d with error %f" % (iterate, error)
        v = new_v
    return v

