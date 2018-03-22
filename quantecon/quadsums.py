"""
This module provides functions to compute quadratic sums of the form described
in the docstrings.

"""


import numpy as np
from numpy import sqrt, dot
import scipy.linalg
from .matrix_eqn import solve_discrete_lyapunov


def var_quadratic_sum(A, C, H, beta, x0):
    r"""
    Computes the expected discounted quadratic sum

    .. math::

        q(x_0) = \mathbb{E} \Big[ \sum_{t=0}^{\infty} \beta^t x_t' H x_t \Big]

    Here :math:`{x_t}` is the VAR process :math:`x_{t+1} = A x_t + C w_t`
    with :math:`{x_t}` standard normal and :math:`x_0` the initial condition.

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
        Represents the value :math:`q(x_0)`

    Remarks: The formula for computing :math:`q(x_0)` is
    :math:`q(x_0) = x_0' Q x_0 + v`
    where

        * :math:`Q` is the solution to :math:`Q = H + \beta A' Q A`, and
        * :math:`v = \frac{trace(C' Q C) \beta}{(1 - \beta)}`

    """
    # == Make sure that A, C, H and x0 are array_like == #

    A, C, H = list(map(np.atleast_2d, (A, C, H)))
    x0 = np.atleast_1d(x0)
    # == Start computations == #
    Q = scipy.linalg.solve_discrete_lyapunov(sqrt(beta) * A.T, H)
    cq = dot(dot(C.T, Q), C)
    v = np.trace(cq) * beta / (1 - beta)
    q0 = dot(dot(x0.T, Q), x0) + v

    return q0


def m_quadratic_sum(A, B, max_it=50):
    r"""
    Computes the quadratic sum

    .. math::

        V = \sum_{j=0}^{\infty} A^j B A^{j'}

    V is computed by solving the corresponding discrete lyapunov
    equation using the doubling algorithm.  See the documentation of
    `util.solve_discrete_lyapunov` for more information.

    Parameters
    ----------
    A : array_like(float, ndim=2)
        An n x n matrix as described above.  We assume in order for
        convergence that the eigenvalues of :math:`A` have moduli bounded by
        unity
    B : array_like(float, ndim=2)
        An n x n matrix as described above.  We assume in order for
        convergence that the eigenvalues of :math:`A` have moduli bounded by
        unity
    max_it : scalar(int), optional(default=50)
        The maximum number of iterations

    Returns
    ========
    gamma1: array_like(float, ndim=2)
        Represents the value :math:`V`

    """

    gamma1 = solve_discrete_lyapunov(A, B, max_it)

    return gamma1
