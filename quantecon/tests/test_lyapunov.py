"""
Tests for ricatti.py

"""
import numpy as np
from numpy.testing import assert_allclose
from quantecon.matrix_eqn import solve_discrete_lyapunov


def test_dlyap_simple_ones():
    A = np.zeros((4, 4))
    B = np.ones((4, 4))

    sol = solve_discrete_lyapunov(A, B)

    assert_allclose(sol, np.ones((4, 4)))


def test_dlyap_scalar():
    a = .5
    b = .75

    sol = solve_discrete_lyapunov(a, b)

    assert_allclose(sol, np.ones((1, 1)))
