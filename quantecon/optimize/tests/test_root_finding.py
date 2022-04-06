import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from numba import njit

from quantecon.optimize import (
    newton, newton_halley, newton_secant, bisect, brentq
)


@njit
def func(x):
    """
    Function for testing on.
    """
    return (x**3 - 1)


@njit
def func_prime(x):
    """
    Derivative for func.
    """
    return (3*x**2)


@njit
def func_prime2(x):
    """
    Second order derivative for func.
    """
    return 6*x


@njit
def func_two(x):
    """
    Harder function for testing on.
    """
    return np.sin(4 * (x - 1/4)) + x + x**20 - 1


@njit
def func_two_prime(x):
    """
    Derivative for func_two.
    """
    return 4*np.cos(4*(x - 1/4)) + 20*x**19 + 1


@njit
def func_two_prime2(x):
    """
    Second order derivative for func_two
    """
    return 380*x**18 - 16*np.sin(4*(x - 1/4))


def test_newton_basic():
    """
    Uses the function f defined above to test the scalar maximization
    routine.
    """
    true_fval = 1.0
    fval = newton(func, 5, func_prime)
    assert_almost_equal(true_fval, fval.root, decimal=4)


def test_newton_basic_two():
    """
    Uses the function f defined above to test the scalar maximization
    routine.
    """
    true_fval = 1.0
    fval = newton(func, 5, func_prime)
    assert_allclose(true_fval, fval.root, rtol=1e-5, atol=0)


def test_newton_hard():
    """
    Harder test for convergence.
    """
    true_fval = 0.408
    fval = newton(func_two, 0.4, func_two_prime)
    assert_allclose(true_fval, fval.root, rtol=1e-5, atol=0.01)


def test_halley_basic():
    """
    Basic test for halley method
    """
    true_fval = 1.0
    fval = newton_halley(func, 5, func_prime, func_prime2)
    assert_almost_equal(true_fval, fval.root, decimal=4)


def test_halley_hard():
    """
    Harder test for halley method
    """
    true_fval = 0.408
    fval = newton_halley(func_two, 0.4, func_two_prime, func_two_prime2)
    assert_allclose(true_fval, fval.root, rtol=1e-5, atol=0.01)


def test_secant_basic():
    """
    Basic test for secant option.
    """
    true_fval = 1.0
    fval = newton_secant(func, 5)
    assert_allclose(true_fval, fval.root, rtol=1e-5, atol=0.001)


def test_secant_hard():
    """
    Harder test for convergence for secant function.
    """
    true_fval = 0.408
    fval = newton_secant(func_two, 0.4)
    assert_allclose(true_fval, fval.root, rtol=1e-5, atol=0.01)


def run_check(method, name):
    a = -1
    b = np.sqrt(3)
    true_fval = 0.408
    r = method(func_two, a, b)
    assert_allclose(true_fval, r.root, atol=0.01, rtol=1e-5,
                    err_msg='method %s' % name)


def test_bisect_basic():
    run_check(bisect, 'bisect')


def test_brentq_basic():
    run_check(brentq, 'brentq')
