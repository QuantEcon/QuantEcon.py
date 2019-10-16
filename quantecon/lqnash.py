import numpy as np
from numpy import dot, eye
from scipy.linalg import solve
from .util import check_random_state


def nnash(A, B1, B2, R1, R2, Q1, Q2, S1, S2, W1, W2, M1, M2,
          beta=1.0, tol=1e-8, max_iter=1000, random_state=None):
    r"""
    Compute the limit of a Nash linear quadratic dynamic game. In this
    problem, player i minimizes

    .. math::
        \sum_{t=0}^{\infty}
        \left\{
            x_t' r_i x_t + 2 x_t' w_i
            u_{it} +u_{it}' q_i u_{it} + u_{jt}' s_i u_{jt} + 2 u_{jt}'
            m_i u_{it}
        \right\}

    subject to the law of motion

    .. math::
        x_{t+1} = A x_t + b_1 u_{1t} + b_2 u_{2t}

    and a perceived control law :math:`u_j(t) = - f_j x_t` for the other
    player.

    The solution computed in this routine is the :math:`f_i` and
    :math:`p_i` of the associated double optimal linear regulator
    problem.

    Parameters
    ----------
    A : scalar(float) or array_like(float)
        Corresponds to the above equation, should be of size (n, n)
    B1 : scalar(float) or array_like(float)
        As above, size (n, k_1)
    B2 : scalar(float) or array_like(float)
        As above, size (n, k_2)
    R1 : scalar(float) or array_like(float)
        As above, size (n, n)
    R2 : scalar(float) or array_like(float)
        As above, size (n, n)
    Q1 : scalar(float) or array_like(float)
        As above, size (k_1, k_1)
    Q2 : scalar(float) or array_like(float)
        As above, size (k_2, k_2)
    S1 : scalar(float) or array_like(float)
        As above, size (k_1, k_1)
    S2 : scalar(float) or array_like(float)
        As above, size (k_2, k_2)
    W1 : scalar(float) or array_like(float)
        As above, size (n, k_1)
    W2 : scalar(float) or array_like(float)
        As above, size (n, k_2)
    M1 : scalar(float) or array_like(float)
        As above, size (k_2, k_1)
    M2 : scalar(float) or array_like(float)
        As above, size (k_1, k_2)
    beta : scalar(float), optional(default=1.0)
        Discount rate
    tol : scalar(float), optional(default=1e-8)
        This is the tolerance level for convergence
    max_iter : scalar(int), optional(default=1000)
        This is the maximum number of iteratiosn allowed
    random_state : int or np.random.RandomState, optional
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    F1 : array_like, dtype=float, shape=(k_1, n)
        Feedback law for agent 1
    F2 : array_like, dtype=float, shape=(k_2, n)
        Feedback law for agent 2
    P1 : array_like, dtype=float, shape=(n, n)
        The steady-state solution to the associated discrete matrix
        Riccati equation for agent 1
    P2 : array_like, dtype=float, shape=(n, n)
        The steady-state solution to the associated discrete matrix
        Riccati equation for agent 2

    """
    # == Unload parameters and make sure everything is an array == #
    params = A, B1, B2, R1, R2, Q1, Q2, S1, S2, W1, W2, M1, M2
    params = map(np.asarray, params)
    A, B1, B2, R1, R2, Q1, Q2, S1, S2, W1, W2, M1, M2 = params

    # == Multiply A, B1, B2 by sqrt(beta) to enforce discounting == #
    A, B1, B2 = [np.sqrt(beta) * x for x in (A, B1, B2)]

    n = A.shape[0]

    if B1.ndim == 1:
        k_1 = 1
        B1 = np.reshape(B1, (n, 1))
    else:
        k_1 = B1.shape[1]

    if B2.ndim == 1:
        k_2 = 1
        B2 = np.reshape(B2, (n, 1))
    else:
        k_2 = B2.shape[1]

    random_state = check_random_state(random_state)
    v1 = eye(k_1)
    v2 = eye(k_2)
    P1 = np.zeros((n, n))
    P2 = np.zeros((n, n))
    F1 = random_state.randn(k_1, n)
    F2 = random_state.randn(k_2, n)

    for it in range(max_iter):
        # update
        F10 = F1
        F20 = F2

        G2 = solve(dot(B2.T, P2.dot(B2))+Q2, v2)
        G1 = solve(dot(B1.T, P1.dot(B1))+Q1, v1)
        H2 = dot(G2, B2.T.dot(P2))
        H1 = dot(G1, B1.T.dot(P1))

        # break up the computation of F1, F2
        F1_left = v1 - dot(H1.dot(B2)+G1.dot(M1.T),
                           H2.dot(B1)+G2.dot(M2.T))
        F1_right = H1.dot(A)+G1.dot(W1.T) - dot(H1.dot(B2)+G1.dot(M1.T),
                                                H2.dot(A)+G2.dot(W2.T))
        F1 = solve(F1_left, F1_right)
        F2 = H2.dot(A)+G2.dot(W2.T) - dot(H2.dot(B1)+G2.dot(M2.T), F1)

        Lambda1 = A - B2.dot(F2)
        Lambda2 = A - B1.dot(F1)
        Pi1 = R1 + dot(F2.T, S1.dot(F2))
        Pi2 = R2 + dot(F1.T, S2.dot(F1))

        P1 = dot(Lambda1.T, P1.dot(Lambda1)) + Pi1 - \
             dot(dot(Lambda1.T, P1.dot(B1)) + W1 - F2.T.dot(M1), F1)
        P2 = dot(Lambda2.T, P2.dot(Lambda2)) + Pi2 - \
             dot(dot(Lambda2.T, P2.dot(B2)) + W2 - F1.T.dot(M2), F2)

        dd = np.max(np.abs(F10 - F1)) + np.max(np.abs(F20 - F2))

        if dd < tol:  # success!
            break

    else:
        msg = 'No convergence: Iteration limit of {0} reached in nnash'
        raise ValueError(msg.format(max_iter))

    return F1, F2, P1, P2
