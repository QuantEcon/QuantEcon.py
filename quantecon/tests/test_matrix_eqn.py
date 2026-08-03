"""
Tests for quantecon.util

"""
import numpy as np
from numpy.testing import assert_allclose, assert_raises
from quantecon import _matrix_eqn as qme


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


def test_solve_discrete_riccati_system_beta_one_failure_message():
    Π = np.ones((1, 1))
    As = np.ones((1, 1, 1))
    Bs = np.zeros((1, 1, 1))
    Cs = np.zeros((1, 1, 1))
    Qs = np.ones((1, 1, 1))
    Rs = np.ones((1, 1, 1))
    Ns = np.zeros((1, 1, 1))

    with assert_raises(ValueError) as excinfo:
        qme.solve_discrete_riccati_system(
            Π, As, Bs, Cs, Qs, Rs, Ns, beta=1, max_iter=0
        )

    message = str(excinfo.exception)
    expected_parts = (
        "beta=1",
        "strict contraction",
        "may be very slow",
        "fail to reach the requested tolerance",
        "try increasing max_iter",
    )
    for expected in expected_parts:
        assert expected in message
