"""
Filename: matrix_eqn.py

This files holds several functions that are used to solve matrix
equations.  Currently has functionality to solve:

* Lyapunov Equations
* Ricatti Equations

TODO: 1. See issue 47 on github repository, should add support for
      Sylvester equations
      2. Fix warnings from checking conditioning of matrices
"""
from __future__ import division
import numpy as np
from numpy import dot
from numpy.linalg import solve
from scipy.linalg import solve_discrete_lyapunov as sp_solve_discrete_lyapunov


def solve_discrete_lyapunov(A, B, max_it=50, method="doubling"):
    r"""
    Computes the solution to the discrete lyapunov equation

    .. math::

        AXA' - X + B = 0

    :math:`X` is computed by using a doubling algorithm. In particular, we
    iterate to convergence on :math:`X_j` with the following recursions for
    :math:`j = 1, 2, \dots` starting from :math:`X_0 = B`, :math:`a_0 = A`:

    .. math::

        a_j = a_{j-1} a_{j-1}

    .. math::

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
    method : string, optional(default="doubling")
        Describes the solution method to use.  If it is "doubling" then
        uses the doubling algorithm to solve, if it is "bartels-stewart"
        then it uses scipy's implementation of the Bartels-Stewart
        approach.

    Returns
    ========
    gamma1: array_like(float, ndim=2)
        Represents the value :math:`V`

    """
    if method == "doubling":
        A, B = list(map(np.atleast_2d, [A, B]))
        alpha0 = A
        gamma0 = B

        diff = 5
        n_its = 1

        while diff > 1e-15:

            alpha1 = alpha0.dot(alpha0)
            gamma1 = gamma0 + np.dot(alpha0.dot(gamma0), alpha0.conjugate().T)

            diff = np.max(np.abs(gamma1 - gamma0))
            alpha0 = alpha1
            gamma0 = gamma1

            n_its += 1

            if n_its > max_it:
                msg = "Exceeded maximum iterations {}, check input matrics"
                raise ValueError(msg.format(n_its))

    elif method == "bartels-stewart":
        gamma1 = sp_solve_discrete_lyapunov(A, B)

    else:
        msg = "Check your method input. Should be doubling or bartels-stewart"
        raise ValueError(msg)

    return gamma1


def solve_discrete_riccati(A, B, Q, R, N=None, tolerance=1e-10, max_iter=500):
    """
    Solves the discrete-time algebraic Riccati equation

    .. math::

        X = A'XA - (N + B'XA)'(B'XB + R)^{-1}(N + B'XA) + Q

    via a modified structured doubling algorithm. An explanation of the
    algorithm can be found in the reference below.

    Note that SciPy also has a discrete riccati equation solver. However it
    cannot handle the case where :math:`R` is not invertible, or when :math:`N`
    is nonzero. Both of these cases can be handled in the algorithm implemented
    below.

    Parameters
    ----------
    A : array_like(float, ndim=2)
        k x k array.
    B : array_like(float, ndim=2)
        k x n array
    Q : array_like(float, ndim=2)
        k x k, should be symmetric and non-negative definite
    R : array_like(float, ndim=2)
        n x n, should be symmetric and positive definite
    N : array_like(float, ndim=2)
        n x k array
    tolerance : scalar(float), optional(default=1e-10)
        The tolerance level for convergence
    max_iter : scalar(int), optional(default=500)
        The maximum number of iterations allowed

    Returns
    -------
    X : array_like(float, ndim=2)
        The fixed point of the Riccati equation; a  k x k array
        representing the approximate solution

    References
    ----------
    Chiang, Chun-Yueh, Hung-Yuan Fan, and Wen-Wei Lin. "STRUCTURED DOUBLING
    ALGORITHM FOR DISCRETE-TIME ALGEBRAIC RICCATI EQUATIONS WITH SINGULAR
    CONTROL WEIGHTING MATRICES." Taiwanese Journal of Mathematics 14, no. 3A
    (2010): pp-935.

    """
    # == Set up == #
    error = tolerance + 1
    fail_msg = "Convergence failed after {} iterations."

    # == Make sure that all array_likes are np arrays, two-dimensional == #
    A, B, Q, R = np.atleast_2d(A, B, Q, R)
    n, k = R.shape[0], Q.shape[0]
    I = np.identity(k)
    if N is None:
        N = np.zeros((n, k))
    else:
        N = np.atleast_2d(N)

    # == Choose optimal value of gamma in R_hat = R + gamma B'B == #
    current_min = np.inf
    candidates = (0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0, 100.0, 10e5)
    BB = dot(B.T, B)
    BTA = dot(B.T, A)
    for gamma in candidates:
        Z = R + gamma * BB
        cn = np.linalg.cond(Z)
        if np.isfinite(cn):
            Q_tilde = - Q + dot(N.T, solve(Z, N + gamma * BTA)) + gamma * I
            G0 = dot(B, solve(Z, B.T))
            A0 = dot(I - gamma * G0, A) - dot(B, solve(Z, N))
            H0 = gamma * dot(A.T, A0) - Q_tilde
            f1 = np.linalg.cond(Z, np.inf)
            f2 = gamma * f1
            f3 = np.linalg.cond(I + dot(G0, H0))
            f_gamma = max(f1, f2, f3)
            if f_gamma < current_min:
                best_gamma = gamma
                current_min = f_gamma

    # == If no candidate successful then fail == #
    if current_min == np.inf:
        msg = "Unable to initialize routine due to ill conditioned arguments"
        raise ValueError(msg)

    gamma = best_gamma
    R_hat = R + gamma * BB

    # == Initial conditions == #
    Q_tilde = - Q + dot(N.T, solve(R_hat, N + gamma * BTA)) + gamma * I
    G0 = dot(B, solve(R_hat, B.T))
    A0 = dot(I - gamma * G0, A) - dot(B, solve(R_hat, N))
    H0 = gamma * dot(A.T, A0) - Q_tilde
    i = 1

    # == Main loop == #
    while error > tolerance:

        if i > max_iter:
            raise ValueError(fail_msg.format(i))

        else:
            A1 = dot(A0, solve(I + dot(G0, H0), A0))
            G1 = G0 + dot(dot(A0, G0), solve(I + dot(H0, G0), A0.T))
            H1 = H0 + dot(A0.T, solve(I + dot(H0, G0), dot(H0, A0)))

            error = np.max(np.abs(H1 - H0))
            A0 = A1
            G0 = G1
            H0 = H1
            i += 1

    return H1 + gamma * I  # Return X
