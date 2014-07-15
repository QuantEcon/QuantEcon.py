"""
Filename: quadsums.py
Authors: Thomas Sargent,  John Stachurski

This module provides functions to compute quadratic sums of the form described
in the docstrings.

"""


import numpy as np
from numpy import sqrt, dot
import scipy.linalg


def var_quadratic_sum(A, C, H, beta, x0):
    r"""
    Computes the expected discounted quadratic sum

    .. math::

        q(x_0) := E \sum_{t=0}^{\infty} \beta^t x_t' H x_t

    Here {x_t} is the VAR process x_{t+1} = A x_t + C w_t with {w_t}
    standard normal and x_0 the initial condition.

    Parameters
    ----------
    A : array_like(float, ndim=2)
        The matrix described above in description.  Should be n x n
    C : array_like(float, ndim=2)
        The matrix described above in description.  Should be n x n
    H : array_like(float, ndim=2)
        The matrix described above in description.  Should be n x n
    beta: scalar(float)
        Should take a value in (0, 1)
    x_0: array_like(float, ndim=1)
        The initial condtion. A conformable array (of length n, or with
        n rows)

    Returns
    -------
    q0: scalar(float)
        Represents the value q(x_0)

    Remarks: The formula for computing q(x_0) is q(x_0) = x_0' Q x_0 + v
    where

        Q is the solution to Q = H + beta A' Q A and
        v = \trace(C' Q C) \beta / (1 - \beta)

    """
    # == Make sure that A, C, H and x0 are array_like == #
    A, C, H, x0 = map(np.asarray, (A, C, H, x0))
    # == Start computations == #
    Q = scipy.linalg.solve_discrete_lyapunov(sqrt(beta) * A.T, H)
    cq = dot(dot(C.T, Q), C)
    v = np.trace(cq) * beta / (1 - beta)
    q0 = dot(dot(x0.T, Q), x0) + v

    return q0


def m_quadratic_sum(A, B, max_it=50):
    """
    Computes the quadratic sum

        V = \sum_{j=0}^{\infty} A^j B A^j'

    V is computed by using a doubling algorithm. In particular, we
    iterate to convergence on V_j with the following recursions for j =
    1, 2,... starting from V_0 = B, a_0 = A:

        a_j = a_{j-1} a_{j-1}

        V_j = V_{j-1} + a_{j-1} V_{j-1} a_{j-1}'

    Parameters
    ----------
    A : array_like(float, ndim=1)
        An n x n matrix as described above.  We assume in order for
        convergence that the eigenvalues of A have moduli bounded by
        unity
    B : array_like(float, ndim=1)
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
