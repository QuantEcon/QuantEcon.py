"""
Origin: QEwP by John Stachurski and Thomas J. Sargent
Filename: linapprox.py
Authors: John Stachurski, Thomas J. Sargent
LastModified: 11/08/2013

"""

from __future__ import division  # Omit if using Python 3.x

def linapprox(f, a, b, n, x):
    """
    Evaluates the piecewise linear interpolant of f at x on the interval 
    [a, b], with n evenly spaced grid points.

    Parameters 
    ===========
        f : function
            The function to approximate

        x, a, b : scalars (floats or integers) 
            Evaluation point and endpoints, with a <= x <= b

        n : integer
            Number of grid points

    Returns
    =========
        A float. The interpolant evaluated at x

    """
    length_of_interval = b - a
    num_subintervals = n - 1
    step = length_of_interval / num_subintervals  

    # === find first grid point larger than x === #
    point = a
    while point <= x:
        point += step

    # === x must lie between the gridpoints (point - step) and point === #
    u, v = point - step, point  

    return f(u) + (x - u) * (f(v) - f(u)) / (v - u)
