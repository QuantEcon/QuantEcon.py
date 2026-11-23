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


def test_discrete_lyapunov_doubling_small_rhs():
    from fractions import Fraction

    a = .9999
    a_exact = Fraction.from_float(a)
    for b in (1e-20, np.ldexp(1.0, -1000), np.ldexp(1.0, 900), 1.0):
        expected = float(Fraction.from_float(b) / (1 - a_exact**2))
        solved = qme.solve_discrete_lyapunov(a, b)
        assert_allclose(solved, [[expected]], rtol=3e-12, atol=0)


def test_discrete_lyapunov_doubling_delayed_nonnormal_term():
    # A**4 == 0, so the exact solution is the sum of four powers.
    # Two initially tiny terms must not hide the unit fourth term.
    a = np.zeros((4, 4))
    a[1, 0], a[2, 1], a[3, 2] = 2.0**-30, 2.0**-30, 2.0**60
    b = np.diag([1.0, 0, 0, 0])
    solution = qme.solve_discrete_lyapunov(a, b)
    assert_allclose(solution, np.diag([1.0, 2.0**-60, 2.0**-120, 1.0]),
                    rtol=0, atol=0)
    assert_allclose(a @ solution @ a.T - solution + b, 0, rtol=0, atol=0)


def test_discrete_lyapunov_doubling_tiny_rhs_does_not_waive_max_it():
    with assert_raises(ValueError) as excinfo:
        qme.solve_discrete_lyapunov(.9999, 1e-20, max_it=2)
    assert "Exceeded maximum iterations" in str(excinfo.exception)


def test_discrete_lyapunov_doubling_nonfinite_iterate():
    # An unstable case is outside the doubling domain and must not return
    # a non-finite result as if the iteration succeeded.
    with np.errstate(over="ignore", invalid="ignore"):
        with assert_raises(ValueError) as excinfo:
            qme.solve_discrete_lyapunov(2.0, 1.0)
    assert "Non-finite values" in str(excinfo.exception)


def test_discrete_lyapunov_doubling_keeps_representable_forcing():
    # Do not normalize a small forcing to one before a huge transition:
    # the unscaled iterate is representable, but the normalized one is not.
    a = np.array([[0.0, 0.0], [1e200, 0.0]])
    b = np.diag([1e-200, 0.0])
    assert_allclose(qme.solve_discrete_lyapunov(a, b),
                    np.diag([1e-200, 1e200]), rtol=3e-15, atol=0)


def test_discrete_lyapunov_bartels_stewart_is_unchanged():
    # The algebraic solution can exist outside the doubling domain.
    x = qme.solve_discrete_lyapunov([[1.25]], [[2.0]],
                                    method="bartels-stewart")
    assert_allclose(x, [[-32 / 9]], rtol=1e-15, atol=0)


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
