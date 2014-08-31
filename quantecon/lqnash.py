from __future__ import division, print_function
from numbers import Number
from math import sqrt
import numpy as np
from numpy import dot, eye
from scipy.linalg import solve, eig


def nnash(a, b1, b2, r1, r2, q1, q2, s1, s2, w1, w2, m1, m2,
          tol=1e-8, max_iter=1000):
    """
    Compute the limit of a Nash linear quadratic dynamic game. In this
    problem, player i maximizes

    .. math::
        - \\sum_{t=0}^{\\infty} \\left\\{x_t' r_i x_t + 2 x_t' w_i
        u_{it} +u_{it}' q_i u_{it} + u_{jt}' s_i u_{jt} + 2 u_{jt}'
        m_i u_{it} \\right\\}

    subject to the law of motion

    .. math::
        x_{t+1} = a x_t + b_1 u_{1t} + b_2 u_{2t}

    and a perceived control law :math:`u_j(t) = - f_j x_t` for the other
    player.

    The solution computed in this routine is the :math:`f_i` and
    :math:`p_i` of the associated double optimal linear regulator
    problem.

    Parameters
    ----------
    a : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (n, n)
    b1 : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (n, k_1)
    b2 : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (n, k_2)
    r1 : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (n, n)
    r2 : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (n, n)
    q1 : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (k_1, k_1)
    q2 : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (k_2, k_2)
    s1 : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (k_1, k_1)
    s2 : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (k_2, k_2)
    w1 : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (n, k_1)
    w2 : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (n, k_2)
    m1 : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (k_2, k_1)
    m2 : scalar(float) or array_like(float)
        This is a matrix that corresponds to the above equation and
        should be of size (k_1, k_2)
    tol : scalar(float), optional(default=1e-8)
        This is the tolerance level for convergence
    max_iter : scalar(int), optional(default=1000)
        This is the maximum number of iteratiosn allowed

    Returns
    -------
    f_1 : array_like, dtype=float, shape=(k_1, n)
        Feedback law for agent 1
    f_2 : array_like, dtype=float, shape=(k_2, n)
        Feedback law for agent 2
    p_1 : array_like, dtype=float, shape=(n, n)
        The steady-state solution to the associated discrete matrix
        Riccati equation for agent 1
    p_2 : array_like, dtype=float, shape=(n, n)
        The steady-state solution to the associated discrete matrix
        Riccati equation for agent 2

    """
    # Unload parameters and make sure everything is an array
    a, b1, b2, r1, r2, q1, q2, s1, s2, w1, w2, m1, m2 = map(np.asarray,
                                                            [a, b1, b2, r1, r2,
                                                             q1, q2, s1, s2,
                                                             w1, w2, m1, m2])

    n = a.shape[0]

    if b1.ndim == 1:
        k_1 = 1
        b1 = np.reshape(b1, (n, 1))
    else:
        k_1 = b1.shape[1]

    if b2.ndim == 1:
        k_2 = 1
        b2 = np.reshape(b2, (n, 1))
    else:
        k_2 = b2.shape[1]

    v1 = eye(k_1)
    v2 = eye(k_2)
    p1 = np.zeros((n, n))
    p2 = np.zeros((n, n))
    f1 = np.random.randn(k_1, n)
    f2 = np.random.randn(k_2, n)

    for it in range(max_iter):
        # update
        f10 = f1
        f20 = f2

        g2 = solve(dot(b2.T, p2.dot(b2))+q2, v2)
        g1 = solve(dot(b1.T, p1.dot(b1))+q1, v1)
        h2 = dot(g2, b2.T.dot(p2))
        h1 = dot(g1, b1.T.dot(p1))

        # break up the computation of f1, f2
        f_1_left = v1 - dot(h1.dot(b2)+g1.dot(m1.T),
                            h2.dot(b1)+g2.dot(m2.T))
        f_1_right = h1.dot(a)+g1.dot(w1.T) - dot(h1.dot(b2)+g1.dot(m1.T),
                                                 h2.dot(a)+g2.dot(w2.T))
        f1 = solve(f_1_left, f_1_right)
        f2 = h2.dot(a)+g2.dot(w2.T) - dot(h2.dot(b1)+g2.dot(m2.T), f1)

        a2 = a - b2.dot(f2)
        a1 = a - b1.dot(f1)

        p1 = (dot(a2.T, p1.dot(a2)) + r1 + dot(f2.T, s1.dot(f2)) -
              dot(dot(a2.T, p1.dot(b1)) + w1 - f2.T.dot(m1), f1))
        p2 = (dot(a1.T, p2.dot(a1)) + r2 + dot(f1.T, s2.dot(f1)) -
              dot(dot(a1.T, p2.dot(b2)) + w2 - f1.T.dot(m2), f2))

        dd = np.max(np.abs(f10 - f1)) + np.max(np.abs(f20 - f2))

        if dd < tol:  # success!
            break

    else:
        msg = 'No convergence: Iteration limit of {0} reached in nnash'
        raise ValueError(msg.format(max_iter))

    return f1, f2, p1, p2