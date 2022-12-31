"""
Tests for quantecon.util

"""
import numpy as np
from numpy.testing import assert_allclose
from quantecon import matrix_eqn as qme


def test_solve_discrete_lyapunov_zero():
    'Simple test where X is all zeros'
    A = np.eye(4) * .95
    B = np.zeros((4, 4))

    X = qme.solve_discrete_lyapunov(A, B)

    assert_allclose(X, np.zeros((4, 4)))


def test_solve_discrete_lyapunov_B():
    'Simple test where X is same as B'
    A = np.ones((2, 2)) * .5
    B = np.array([[.5, -.5], [-.5, .5]])

    X = qme.solve_discrete_lyapunov(A, B)

    assert_allclose(B, X)

def test_solve_discrete_lyapunov_complex():
    'Complex test, A is companion matrix'
    A = np.array([[0.5 + 0.3j, 0.1 + 0.1j],
                  [         1,          0]])
    B = np.eye(2)

    X = qme.solve_discrete_lyapunov(A, B)

    assert_allclose(np.dot(np.dot(A, X), A.conj().transpose()) - X, -B,
                    atol=1e-15)

