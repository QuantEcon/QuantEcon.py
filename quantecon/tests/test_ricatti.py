"""
Filename: test_tauchen.py
Authors: Chase Coleman, John Stachurski
Date: 07/22/2014

Tests for ricatti.py file

"""
import numpy as np
from numpy.testing import assert_allclose
from quantecon.matrix_eqn import solve_discrete_riccati

def dare_test_golden_num_float():
    val = solve_discrete_riccati(1.0, 1.0, 1.0, 1.0)
    gold_ratio = (1 + np.sqrt(5)) / 2.
    assert_allclose(val, gold_ratio)

def dare_test_golden_num_2d():
    A, B, R, Q = np.eye(2), np.eye(2), np.eye(2), np.eye(2)
    gold_diag = np.eye(2) * (1 + np.sqrt(5)) / 2.
    val = solve_discrete_riccati(A, B, R, Q)
    assert_allclose(val, gold_diag)

def dare_test_tjm_1():
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
    X = solve_discrete_riccati(A, B, Q, R)
    Y = np.diag((1e5, 1e3, 0.0))
    assert_allclose(X, Y, atol=1e-07)


def dare_test_tjm_2():
    A = [[0, -1],
         [0, 2]]
    B = [[1, 0],
         [1, 1]]
    Q = [[1, 0],
         [0, 0]]
    R = [[4, 2],
         [2, 1]]
    X = solve_discrete_riccati(A, B, Q, R)
    Y = np.zeros((2, 2))
    Y[0, 0] = 1
    assert_allclose(X, Y, atol=1e-07)


def dare_test_tjm_3():
    r = 0.5
    I = np.identity(2)
    A = [[2 + r**2, 0],
         [0,        0]]
    A = np.array(A)
    B = I
    R = [[1, r],
         [r, r*r]]
    Q = I - np.dot(A.T, A) + np.dot(A.T, np.linalg.solve(R + I, A))
    X = solve_discrete_riccati(A, B, Q, R)
    Y = np.identity(2)
    assert_allclose(X, Y, atol=1e-07)
