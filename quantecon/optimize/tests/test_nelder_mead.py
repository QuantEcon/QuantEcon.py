"""
Tests for `multivar_maximization.py`

"""

import numpy as np
from numba import njit
from numpy.testing import assert_allclose, assert_raises

from quantecon.optimize import nelder_mead
from ..nelder_mead import _nelder_mead_algorithm


@njit
def rosenbrock(x):
    # Rosenbrock (1960)
    f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0])**2
    return -f


@njit
def powell(x):
    # Powell (1962)
    f = (x[0] + 10 * x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2 + \
        (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4
    return -f


@njit
def mccormick(x):
    # https://www.sfu.ca/~ssurjano/mccorm.html
    f = np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + \
             2.5 * x[1] + 1
    return -f


@njit
def bohachevsky(x):
    # https://www.sfu.ca/~ssurjano/boha.html
    f = x[0] ** 2 + x[1] ** 2 - 0.3 * np.cos(3 * np.pi * x[0]) - \
        0.4 * np.cos(4 * np.pi * x[1]) + 0.7
    return -f


@njit
def easom(x):
    # https://www.sfu.ca/~ssurjano/easom.html
    f = -(np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi) ** 2 -
          (x[1] - np.pi) ** 2))
    return -f


@njit
def perm_function(x, β):
    # https://www.sfu.ca/~ssurjano/perm0db.html
    d = x.size
    f = 0
    for i in range(1, d+1):
        temp = 0
        for j in range(1, d+1):
            temp += (j + β) * (x[j-1] ** i - 1 / (j ** i))
        f += temp ** 2

    return -f


@njit
def rotated_hyper_ellipsoid(x):
    # https://www.sfu.ca/~ssurjano/rothyp.html
    d = x.size
    f = 0
    for i in range(1, d+1):
        for j in range(i):
            f += x[j-1] ** 2

    return -f


@njit
def booth(x):
    # https://www.sfu.ca/~ssurjano/booth.html
    f = (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
    return -f


@njit
def zakharov(x):
    # https://www.sfu.ca/~ssurjano/zakharov.html
    d = x.size
    _range = np.arange(1, d+1)
    f = (x ** 2).sum() + (0.5 * x * _range).sum() ** 2 + \
        (0.5 * x * _range).sum() ** 4
    return -f


@njit
def colville(x):
    # https://www.sfu.ca/~ssurjano/colville.html
    f = 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2 + (x[2] - 1) ** 2 + \
        90 * (x[2] ** 2 - x[3]) ** 2 + 10.1 * \
        ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1)
    return -f


@njit
def styblinski_tang(x):
    # https://www.sfu.ca/~ssurjano/stybtang.html
    f = 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x).sum()
    return -f


@njit
def goldstein_price(x):
    # https://www.sfu.ca/~ssurjano/goldpr.html
    p1 = (x[0] + x[1] + 1) ** 2
    p2 = 19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + \
        3 * x[1] ** 2
    p3 = (2 * x[0] - 3 * x[1]) ** 2
    p4 = 18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + \
        27 * x[1] ** 2

    f = (1 + p1 * p2) * (30 + p3 * p4)
    return -f


@njit
def sum_squared(x):
    return - (x ** 2).sum()


@njit
def f(x):
    return -(x[0]**2 + x[0])


@njit
def g(x):
    if x[0] < 1:
        return -(0.75 * x[0]**2 - x[0] + 2)
    else:
        return -(0.5 * x[0] ** 2 - x[0] + 1)


@njit
def h(x):
    return -(abs(x[0]) + abs(x[1]))


class TestMaximization():
    def test_rosenbrock(self):
        sol = np.array([1., 1.])
        fun = 0.

        x0 = np.array([-2, 1])

        results = nelder_mead(rosenbrock, x0, tol_x=1e-20, tol_f=1e-20)

        assert_allclose(results.x, sol, atol=1e-4)
        assert_allclose(results.fun, fun, atol=1e-4)

    def test_powell(self):
        sol = np.zeros(4)
        fun = 0.

        x0 = np.array([3., -1., 0., 1.])

        results = nelder_mead(powell, x0, tol_x=1e-20, tol_f=1e-20)

        assert_allclose(results.x, sol, atol=1e-4)
        assert_allclose(results.fun, fun, atol=1e-4)

    def test_mccormick(self):
        sol = np.array([-0.54719, -1.54719])
        fun = 1.9133

        x0 = np.array([-1., -1.5])
        bounds = np.array([[-1.5, 4.],
                           [-3., 4.]])

        results = nelder_mead(mccormick, x0, bounds=bounds)

        assert_allclose(results.x, sol, rtol=1e-3)
        assert_allclose(results.fun, fun, rtol=1e-3)

    def test_bohachevsky(self):
        sol = np.array([0., 0.])
        fun = 0.

        # Starting point makes significant difference
        x0 = np.array([np.pi, -np.pi])

        results = nelder_mead(bohachevsky, x0)

        assert_allclose(results.x, sol, atol=1e-4)
        assert_allclose(results.fun, fun, atol=1e-4)

    def test_easom(self):
        sol = np.array([np.pi, np.pi])
        fun = 1.

        x0 = np.array([5, -1])

        results = nelder_mead(easom, x0, tol_x=1e-20, tol_f=1e-20)

        assert_allclose(results.x, sol, atol=1e-4)
        assert_allclose(results.fun, fun, atol=1e-4)

    def test_perm_function(self):
        d = 4.
        x0 = np.ones(int(d))
        bounds = np.array([[-d, d]] * int(d))

        sol = np.array([1 / d for d in range(1, int(d)+1)])
        fun = 0.

        results = nelder_mead(perm_function, x0, bounds=bounds, args=(1., ),
                              tol_x=1e-30, tol_f=1e-30)

        assert_allclose(results.x, sol, atol=1e-7)
        assert_allclose(results.fun, fun, atol=1e-7)

    def test_rotated_hyper_ellipsoid(self):
        d = 5
        x0 = np.random.normal(size=d)
        bounds = np.array([[-65.536, 65.536]] * d)

        sol = np.zeros(d)
        fun = 0.

        results = nelder_mead(rotated_hyper_ellipsoid, x0, bounds=bounds,
                              tol_x=1e-30, tol_f=1e-30)

        assert_allclose(results.x, sol, atol=1e-6)
        assert_allclose(results.fun, fun, atol=1e-7)

    def test_booth(self):
        x0 = np.array([0., 0.])

        sol = np.array([1., 3.])
        fun = 0.

        results = nelder_mead(booth, x0, tol_x=1e-20, tol_f=1e-20)

        assert_allclose(results.x, sol, atol=1e-7)
        assert_allclose(results.fun, fun, atol=1e-7)

    def test_zakharov(self):
        x0 = np.array([-3., 8., 1., 3., -0.5])
        bounds = np.array([[-5., 10.]] * 5)

        sol = np.zeros(5)
        fun = 0.

        results = nelder_mead(zakharov, x0, bounds=bounds, tol_f=1e-30,
                              tol_x=1e-30)

        assert_allclose(results.x, sol, atol=1e-7)
        assert_allclose(results.fun, fun, atol=1e-7)

    def test_colville(self):
        x0 = np.array([-3.5, 9., 0.25, -1.])
        bounds = np.array([[-10., 10.]] * 4)

        sol = np.ones(4)
        fun = 0.

        results = nelder_mead(colville, x0, bounds=bounds, tol_f=1e-35,
                              tol_x=1e-35)

        assert_allclose(results.x, sol)
        assert_allclose(results.fun, fun, atol=1e-7)

    def test_styblinski_tang(self):
        d = 8
        x0 = -np.ones(d)
        bounds = np.array([[-5., 5.]] * d)

        sol = np.array([-2.903534] * d)
        fun = 39.16599 * d

        results = nelder_mead(styblinski_tang, x0, bounds=bounds, tol_f=1e-35,
                              tol_x=1e-35)

        assert_allclose(results.x, sol, rtol=1e-4)
        assert_allclose(results.fun, fun, rtol=1e-5)

    def test_goldstein_price(self):
        x0 = np.array([-1.5, 0.5])

        results = nelder_mead(goldstein_price, x0)

        sol = np.array([0., -1.])
        fun = -3.

        assert_allclose(results.x, sol, atol=1e-5)
        assert_allclose(results.fun, fun)

    def test_sum_squared(self):
        x0 = np.array([0.5, -np.pi, np.pi])

        sol = np.zeros(3)
        fun = 0.

        results = nelder_mead(sum_squared, x0, tol_f=1e-50, tol_x=1e-50)
        assert_allclose(results.x, sol, atol=1e-5)
        assert_allclose(results.fun, fun, atol=1e-5)

    def test_corner_sol(self):
        sol = np.array([0.])
        fun = 0.

        x0 = np.array([10.])
        bounds = np.array([[0., np.inf]])

        results = nelder_mead(f, x0, bounds=bounds, tol_f=1e-20)

        assert_allclose(results.x, sol)
        assert_allclose(results.fun, fun)

    def test_discontinuous(self):
        sol = np.array([1.])
        fun = -0.5

        x0 = np.array([-10.])

        results = nelder_mead(g, x0)

        assert_allclose(results.x, sol)
        assert_allclose(results.fun, fun)


def test_invalid_bounds_1():
    x0 = np.array([-2., 1.])
    bounds = np.array([[10., -10.], [10., -10.]])
    assert_raises(ValueError, nelder_mead, rosenbrock, x0, bounds=bounds)


def test_invalid_bounds_2():
    x0 = np.array([-2., 1.])
    bounds = np.array([[10., -10., 10., -10.]])
    assert_raises(ValueError, nelder_mead, rosenbrock, x0, bounds=bounds)


def test_invalid_ρ():
    vertices = np.array([[-2., 1.],
                         [1.05 * -2., 1.],
                         [-2., 1.05 * 1.]])
    invalid_ρ = -1.
    assert_raises(ValueError, _nelder_mead_algorithm, rosenbrock,
                  vertices, ρ=invalid_ρ)


def test_invalid_χ():
    vertices = np.array([[-2., 1.],
                         [1.05 * -2., 1.],
                         [-2., 1.05 * 1.]])
    invalid_χ = 0.5
    assert_raises(ValueError, _nelder_mead_algorithm, rosenbrock,
                  vertices, χ=invalid_χ)


def test_invalid_ρχ():
    vertices = np.array([[-2., 1.],
                         [1.05 * -2., 1.],
                         [-2., 1.05 * 1.]])
    ρ = 2
    χ = 1.5
    assert_raises(ValueError, _nelder_mead_algorithm, rosenbrock,
                  vertices, ρ=ρ, χ=χ)


def test_invalid_γ():
    vertices = np.array([[-2., 1.],
                         [1.05 * -2., 1.],
                         [-2., 1.05 * 1.]])
    invalid_γ = -1e-7
    assert_raises(ValueError, _nelder_mead_algorithm, rosenbrock,
                  vertices, γ=invalid_γ)


def test_invalid_σ():
    vertices = np.array([[-2., 1.],
                         [1.05 * -2., 1.],
                         [-2., 1.05 * 1.]])
    invalid_σ = 1. + 1e-7
    assert_raises(ValueError, _nelder_mead_algorithm, rosenbrock,
                  vertices, σ=invalid_σ)
