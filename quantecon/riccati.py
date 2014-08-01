"""
Filename: riccati.py

Authors: Thomas Sargent, John Stachurski

Solves the discrete-time algebraic Riccati equation

"""

import sys
import numpy as np
from numpy import dot
from numpy.linalg import solve
import warnings

# == Suppress warnings from checking conditioning of matrices == #
warnings.simplefilter("ignore", RuntimeWarning)



def dare(A, B, Q, R, C=None, tolerance=1e-10, max_iter=150):
    """
    Solves the discrete-time algebraic Riccati equation

        X = A'XA - (C + B'XA)'(B'XB + R)^{-1}(C + B'XA) + Q

    via a modified structured doubling algorithm.  
    
    An explanation of the algorithm can be
    found in "Optimal Filtering" by B.D.O. Anderson and J.B. Moore
    (Dover Publications, 2005, p. 159).

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
    max_iter : scalar(int), optional(default=150)
        The maximum number of iterations allowed

    Returns
    -------
    X : array_like(float, ndim=2)
        The fixed point of the Riccati equation; a  k x k array
        representing the approximate solution

    """
    # == Set up == #
    error = tolerance + 1
    fail_msg = "Convergence failed after {} iterations."

    # == Make sure that all array_likes are np arrays, two-dimensional == #
    A, B, Q, R = list(map(np.atleast_2d, (A, B, Q, R)))
    n, k = R.shape[0], Q.shape[0]
    I = np.identity(k)
    if C == None:
        C = np.zeros((n, k))
    else:
        C = np.atleast_2d(C)

    # == Choose gamma to make R_hat = R + gamma B'B invertible == #
    R_hat = None
    current_min = np.inf
    candidates = (0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0, 100.0, 10e5)
    BB = dot(B.T, B)
    BTA = dot(B.T, A)
    for gamma in candidates:
        Z = R + gamma * BB
        if np.linalg.cond(Z) > 1 / sys.float_info.epsilon:
            pass
        else:
            Q_tilde = - Q + dot(C.T, solve(Z, C + gamma * BTA)) + gamma * I
            G0 = dot(B, solve(Z, B.T))
            A0 = dot(I - gamma * G0, A) - dot(B, solve(Z, C))
            H0 = gamma * dot(A.T, A0) - Q_tilde
            f1 = np.linalg.cond(Z, np.inf)
            f2 = gamma * f1
            f3 = np.linalg.cond(I + dot(G0, H0), np.inf)
            f_gamma = max(f1, f2, f3)
            print f_gamma
            if  f_gamma < current_min:
                best_gamma = gamma
                current_min = f_gamma

    # == If no candidate successful then fail == #
    if current_min == np.inf:
        msg = "Unable to initialize routine due to ill conditioned arguments"
        raise ValueError(msg)

    gamma = best_gamma
    print "final gamma", gamma
    R_hat = R + gamma * BB

    Q_tilde = - Q + dot(C.T, solve(R_hat, C + gamma * BTA)) + gamma * I
    G0 = dot(B, solve(R_hat, B.T))
    A0 = dot(I - gamma * G0, A) - dot(B, solve(R_hat, C))
    H0 = gamma * dot(A.T, A0) - Q_tilde


    # == Initial conditions == #

    A0 = dot(I - gamma * G0, A) - dot(B, solve(R_hat, C))
    i = 1

    # == Main loop == #
    while error > tolerance:

        if i > max_iter:
            raise ValueError(fail_msg.format(i))

        else:
            A1 = dot(A0, solve(I + dot(G0, H0), A0))
            G1 = G0 + dot(dot(A0, G0), solve(I + dot(G0, H0), A0.T))
            H1 = H0 + dot(A0.T, solve(I + dot(G0, H0), dot(H0, A0)))

            error = np.max(np.abs(H1 - H0))
            A0 = A1
            G0 = G1
            H0 = H1
            i += 1

    return H1 + gamma * I  # Return X


def dare_backup(A, B, R, Q, tolerance=1e-10, max_iter=150):
    """
    An older routine that solves the discrete-time algebraic Riccati equation

        X = A'XA - A'XB(B'XB + R)^{-1}B'XA + Q

    via the doubling algorithm.  Note however that this routine fails if R is
    not invertible.  An explanation of the algorithm can be found in "Optimal
    Filtering" by B.D.O. Anderson and J.B. Moore (Dover Publications, 2005, p.
    159).

    Parameters
    ----------
    A : array_like(float, ndim=2)
        k x k array.
    B : array_like(float, ndim=2)
        k x n array
    R : array_like(float, ndim=2)
        n x n, should be symmetric and positive definite
    Q : array_like(float, ndim=2)
        k x k, should be symmetric and non-negative definite
    tolerance : scalar(int), optional(default=1e-10)
        The tolerance level for convergence
    max_iter : scalar(float), optional(default=150)
        The maximum number of iterations allowed

    Returns
    -------
    X : array_like(float, ndim=2)
        The fixed point of the Riccati equation; a  k x k array
        representing the approximate solution

    """
    # == Set up == #
    error = tolerance + 1
    fail_msg = "Convergence failed after {} iterations."
    # == Make sure that all arrays are two-dimensional == #
    A, B, Q, R = list(map(np.atleast_2d, (A, B, Q, R)))
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



def test_1():
    A = [[0.0, 0.1, 0.0],
         [0.0, 0.0, 0.1],
         [0.0, 0.0, 0.0]]

    B = [[1.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0]]

    Q = [[10**5, 0.0, 0.0],
         [0.0, 10**3, 0.0],
         [0.0, 0.0, -10.0]]

    R = [[0.0, 0.0],
         [0.0, 1.0]]

    X = dare(A, B, Q, R)
    Y = np.diag((10**5, 10**3, 0))
    print(X)
    print(Y)


def test_1_dot_5():

    A = [[0, -1],
         [0, 2]] 

    B = [[1, 0],
         [1, 1]]

    Q = [[1, 0],
         [0, 0]]

    R = [[4, 2],
         [2, 1]]

    X = dare(A, B, Q, R)
    Y = np.zeros((2, 2))
    Y[0, 0] = 1
    print(X)
    print(Y)


def test_2():

    r = 0.5
    I = np.identity(2)

    A = [[2 + r**2, 0],
         [0,        0]] 
    A = np.array(A)

    B = I

    R = [[1, r],
         [r, r*r]] 

    Q = I - np.dot(A.T, A) + np.dot(A.T, np.linalg.solve(R + I, A))

    X = dare(A, B, Q, R)
    Y = np.identity(2)
    print(X)
    print(Y)
