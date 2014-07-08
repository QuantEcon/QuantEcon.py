"""
Filename: riccati.py
Authors: Thomas Sargent, John Stachurski

Solves the discrete-time algebraic Riccati equation 
"""

import numpy as np
from numpy import dot
from numpy.linalg import solve

def dare(A, B, R, Q, tolerance=1e-10, max_iter=150):
    """
    Solves the discrete-time algebraic Riccati equation 
    
        X = A'XA - A'XB(B'XB + R)^{-1}B'XA + Q  

    via the doubling algorithm.  An explanation of the algorithm can be found
    in "Optimal Filtering" by B.D.O. Anderson and J.B. Moore (Dover
    Publications, 2005, p. 159).

    Parameters
    ============
    All arguments should be NumPy ndarrays.

        * A is k x k
        * B is k x n
        * Q is k x k, symmetric and nonnegative definite
        * R is n x n, symmetric and positive definite

    Returns
    ========
    X : a  k x k numpy.ndarray representing the approximate solution

    """
    # == Set up == #
    error = tolerance + 1
    fail_msg = "Convergence failed after {} iterations."
    # == Make sure that all arrays are two-dimensional == #
    A, B, Q, R = map(np.atleast_2d, (A, B, Q, R))
    k = Q.shape[0]
    I = np.identity(k)

    # == Initial conditions == #
    a0 = A
    b0 = dot(B, solve(R, B.T))
    g0 = Q
    i = 1

    # == Main loop == #
    while error > tolerance:

        if i > max_iter:
            raise ValueError(fail_msg.format(i))

        else:

            a1 = dot(a0, solve(I + dot(b0, g0), a0))
            b1 = b0 + dot(a0, solve(I + dot(b0, g0), dot(b0, a0.T)))
            g1 = g0 + dot(dot(a0.T, g0), solve(I + dot(b0, g0), a0))

            error = np.max(np.abs(g1 - g0))

            a0 = a1
            b0 = b1
            g0 = g1

            i += 1

    return g1  # Return X


if __name__ == '__main__': ## Example of useage

    a = np.array([[0.1, 0.1, 0.0],
                  [0.1, 0.0, 0.1],
                  [0.0, 0.4, 0.0]])
                       
    b = np.array([[1.0, 0.0], 
                  [0.0, 0.0], 
                  [0.0, 1.0]])
                       
    r = np.array([[0.5, 0.0], 
                  [0.0, 1.0]])
                       
    q = np.array([[1, 0.0, 0.0],
                  [0.0, 1, 0.0],
                  [0.0, 0.0, 10.0]])

    x = dare(a, b, r, q)
    print x
