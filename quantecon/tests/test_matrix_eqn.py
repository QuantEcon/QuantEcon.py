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

    assert_allclose(A @ X @ A.conj().T - X, -B, atol=1e-15)


def test_solve_discrete_riccati_system_beta_ge_one_failure_message():
    Π = np.ones((1, 1))
    As = np.ones((1, 1, 1))
    Bs = np.zeros((1, 1, 1))
    Cs = np.zeros((1, 1, 1))
    Qs = np.ones((1, 1, 1))
    Rs = np.ones((1, 1, 1))
    Ns = np.zeros((1, 1, 1))

    for beta in (1.0, 1.05):
        with assert_raises(ValueError) as excinfo:
            qme.solve_discrete_riccati_system(
                Π, As, Bs, Cs, Qs, Rs, Ns,
                beta=beta, max_iter=0
            )

        message = str(excinfo.exception)
        expected_parts = (
            "beta>=1",
            "stationary solution",
            "stabilized without discounting",
        )
        for expected in expected_parts:
            assert expected in message

        assert "max_iter" not in message


def test_solve_discrete_riccati_system_overflow_raises():
    # x' = 2x, B = 0: P grows 4x per iteration and overflows before max_iter
    one, zero = np.ones((1, 1, 1)), np.zeros((1, 1, 1))
    with np.errstate(over='ignore', invalid='ignore'):
        with assert_raises(ValueError) as excinfo:
            qme.solve_discrete_riccati_system(
                np.ones((1, 1)), 2 * one, zero, zero, one, one, zero,
                beta=1.0
            )
    assert "stabilized without discounting" in str(excinfo.exception)


def test_solve_discrete_riccati_overflow_raises():
    # Same unstabilizable system for the doubling solver
    with np.errstate(over='ignore', invalid='ignore'):
        with assert_raises(ValueError) as excinfo:
            qme.solve_discrete_riccati(2., 0., 1., 1.)
    assert "Convergence failed" in str(excinfo.exception)
