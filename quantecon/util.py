"""
Filename: util.py

This files holds several useful functions that are needed for the
quantecon library

"""
from __future__ import division
import numpy as np


def solve_discrete_lyapunov(A, B, max_it=50):
    r"""
    Computes the solution to the discrete lyapunov equation

    .. math::

        AXA' - X + B = 0

    X is computed by using a doubling algorithm. In particular, we
    iterate to convergence on X_j with the following recursions for j =
    1, 2,... starting from X_0 = B, a_0 = A:

    .. math::

        a_j = a_{j-1} a_{j-1}

        X_j = X_{j-1} + a_{j-1} X_{j-1} a_{j-1}'

    Parameters
    ----------
    A : array_like(float, ndim=2)
        An n x n matrix as described above.  We assume in order for
        convergence that the eigenvalues of A have moduli bounded by
        unity
    B : array_like(float, ndim=2)
        An n x n matrix as described above.  We assume in order for
        convergence that the eigenvalues of A have moduli bounded by
        unity
    max_it : scalar(int), optional(default=50)
        The maximum number of iterations

    Returns
    ========
    gamma1: array_like(float, ndim=2)
        Represents the value V

    """
    A, B = list(map(np.atleast_2d, [A, B]))
    alpha0 = A
    gamma0 = B

    diff = 5
    n_its = 1

    while diff > 1e-15:

        alpha1 = alpha0.dot(alpha0)
        gamma1 = gamma0 + np.dot(alpha0.dot(gamma0), alpha0.T)

        diff = np.max(np.abs(gamma1 - gamma0))
        alpha0 = alpha1
        gamma0 = gamma1

        n_its += 1

        if n_its > max_it:
            raise ValueError('Exceeded maximum iterations of %i.' % (max_it) +
                             ' Check your input matrices')

    return gamma1