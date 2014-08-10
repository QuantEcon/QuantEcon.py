"""
Filename: riccati.py

Authors: Thomas Sargent, John Stachurski

Solves the discrete-time algebraic Riccati equation

"""

import numpy as np
from numpy import dot
from numpy.linalg import solve
import warnings

# == Suppress warnings from checking conditioning of matrices == #
warnings.simplefilter("ignore", RuntimeWarning)


def dare(A, B, Q, R, C=None, tolerance=1e-10, max_iter=500):
    """
    Solves the discrete-time algebraic Riccati equation

        X = A'XA - (C + B'XA)'(B'XB + R)^{-1}(C + B'XA) + Q

    via a modified structured doubling algorithm.  An explanation of the
    algorithm can be found in the reference below.

    Parameters
    ----------
    A : array_like(float, ndim=2)
        k x k array.
    B : array_like(float, ndim=2)
        k x n array
    C : array_like(float, ndim=2)
        n x k array
    R : array_like(float, ndim=2)
        n x n, should be symmetric and positive definite
    Q : array_like(float, ndim=2)
        k x k, should be symmetric and non-negative definite
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
    if C == None:
        C = np.zeros((n, k))
    else:
        C = np.atleast_2d(C)

    # == Choose optimal value of gamma in R_hat = R + gamma B'B == #
    current_min = np.inf
    candidates = (0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0, 100.0, 10e5)
    BB = dot(B.T, B)
    BTA = dot(B.T, A)
    for gamma in candidates:
        Z = R + gamma * BB
        cn = np.linalg.cond(Z)
        if np.isfinite(cn):
            Q_tilde = - Q + dot(C.T, solve(Z, C + gamma * BTA)) + gamma * I
            G0 = dot(B, solve(Z, B.T))
            A0 = dot(I - gamma * G0, A) - dot(B, solve(Z, C))
            H0 = gamma * dot(A.T, A0) - Q_tilde
            f1 = np.linalg.cond(Z, np.inf)
            f2 = gamma * f1
            f3 = np.linalg.cond(I + dot(G0, H0))
            f_gamma = max(f1, f2, f3)
            if  f_gamma < current_min:
                best_gamma = gamma
                current_min = f_gamma

    # == If no candidate successful then fail == #
    if current_min == np.inf:
        msg = "Unable to initialize routine due to ill conditioned arguments"
        raise ValueError(msg)

    gamma = best_gamma
    R_hat = R + gamma * BB



    # == Initial conditions == #
    Q_tilde = - Q + dot(C.T, solve(R_hat, C + gamma * BTA)) + gamma * I
    G0 = dot(B, solve(R_hat, B.T))
    A0 = dot(I - gamma * G0, A) - dot(B, solve(R_hat, C))
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

