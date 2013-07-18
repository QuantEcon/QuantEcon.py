from __future__ import division  # Omit if using Python 3.x

def linapprox(f, a, b, n, x):
    """
    Evaluates the piecewise linear interpolant of f at x on the interval [a,
    b], with n evenly spaced grid points.

    Parameters: 

        f is a function
        x, a and b are numbers with a <= x <= b
        n is an integer.

    Returns: 

        A number (the interpolant evaluated at x)

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
