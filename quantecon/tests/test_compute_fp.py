"""
Tests for compute_fp.py

References
----------

https://www.math.ucdavis.edu/~hunter/book/ch3.pdf

TODO: add multivariate case

"""
import warnings
import numpy as np
from numpy.testing import (
    assert_, assert_allclose, assert_array_equal, assert_raises
)
from quantecon import compute_fixed_point
import quantecon._compute_fp
from quantecon._compute_fp import _initialize_tableau_ig, _update_distances


class TestFPLogisticEquation():

    @classmethod
    def setup_method(cls):
        cls.mu_1 = 0.2  # 0 is unique fixed point forall x_0 \in [0, 1]

        # (4mu - 1)/(4mu) is a fixed point forall x_0 \in [0, 1]
        cls.mu_2 = 0.3

        # starting points on (0, 1)
        cls.unit_inverval = [0.1, 0.3, 0.6, 0.9]

        # arguments for compute_fixed_point
        cls.kwargs = {"error_tol": 1e-5, "max_iter": 200, "verbose": 0}

    def T(self, x, mu):
        return 4.0 * mu * x * (1.0 - x)

    def test_contraction_1(self):
        "compute_fp: convergence inside interval of convergence"
        f = lambda x: self.T(x, self.mu_1)
        for i in self.unit_inverval:
            # should have fixed point of 0.0
            assert_(abs(compute_fixed_point(f, i, **self.kwargs)) < 1e-4)

    def test_not_contraction_2(self):
        "compute_fp: no convergence outside interval of convergence"
        f = lambda x: self.T(x, self.mu_2)
        for i in self.unit_inverval:
            # This shouldn't converge to 0.0
            assert_(not (abs(compute_fixed_point(f, i, **self.kwargs)) < 1e-4))

    def test_contraction_2(self):
        "compute_fp: convergence inside interval of convergence"
        f = lambda x: self.T(x, self.mu_2)
        fp = (4 * self.mu_2 - 1) / (4 * self.mu_2)
        for i in self.unit_inverval:
            # This should converge to fp
            assert_(abs(compute_fixed_point(f, i, **self.kwargs)-fp) < 1e-4)

    def test_not_contraction_1(self):
        "compute_fp: no convergence outside interval of convergence"
        f = lambda x: self.T(x, self.mu_1)
        fp = (4 * self.mu_1 - 1) / (4 * self.mu_1)
        for i in self.unit_inverval:
            # This should not converge  (b/c unique fp is 0.0)
            assert_(not (abs(compute_fixed_point(f, i, **self.kwargs)-fp)
                    < 1e-4))

    def test_imitation_game_method(self):
        "compute_fp: Test imitation game method"
        method = 'imitation_game'
        error_tol = self.kwargs['error_tol']

        for mu in [self.mu_1, self.mu_2]:
            for i in self.unit_inverval:
                fp_computed = compute_fixed_point(self.T, i, method=method,
                                                  mu=mu, **self.kwargs)
                assert_(
                    abs(self.T(fp_computed, mu=mu) - fp_computed) <= error_tol
                )

            # numpy array input
            i = np.asarray(self.unit_inverval)
            fp_computed = compute_fixed_point(self.T, i, method=method, mu=mu,
                                              **self.kwargs)
            assert_(
                abs(self.T(fp_computed, mu=mu) - fp_computed).max() <=
                error_tol
            )


class TestComputeFPContraction():
    def setup_method(self):
        self.coeff = 0.5
        self.methods = ['iteration', 'imitation_game']

    def f(self, x):
        return self.coeff * x

    def test_num_iter_one(self):
        init = 1.
        error_tol = self.coeff

        for method in self.methods:
            fp_computed = compute_fixed_point(self.f, init,
                                              error_tol=error_tol,
                                              method=method)
            assert_(fp_computed <= error_tol * 2)

    def test_num_iter_large(self):
        init = 1.
        buff_size = 2**8  # buff_size in 'imitation_game'
        max_iter = buff_size + 2
        error_tol = self.coeff**max_iter

        for method in self.methods:
            fp_computed = compute_fixed_point(self.f, init,
                                              error_tol=error_tol,
                                              max_iter=max_iter, method=method,
                                              print_skip=max_iter)
            assert_(fp_computed <= error_tol * 2)

    def test_2d_input(self):
        error_tol = self.coeff**4

        for method in self.methods:
            init = np.array([[-1, 0.5], [-1/3, 0.1]])
            fp_computed = compute_fixed_point(self.f, init,
                                              error_tol=error_tol,
                                              method=method)
            assert_((fp_computed <= error_tol * 2).all())


def test_raises_value_error_nonpositive_max_iter():
    f = lambda x: 0.5*x
    init = 1.
    max_iter = 0
    assert_raises(ValueError, compute_fixed_point, f, init, max_iter=max_iter)


def test_imitation_game_numerical_breakdown():
    # Values of order 1e7: the Lemke-Howson pivoting on the imitation
    # game breaks down numerically (the tableaux hold no equilibrium);
    # the routine must stop and report non-convergence, with a specific
    # warning, instead of forming the next iterate from the tableaux
    X = np.array([-43904436.689917795, -58130594.69589446,
                  -51017515.692906134])
    Y = np.array([-58130594.69589446, 73372551.04835357,
                  45264022.10520259])
    max_iter = 10
    evaluations = []

    def T(x):  # Map reproducing the sequence: nearest recorded point
        evaluations.append(x)
        return Y[np.argmin(np.abs(X - x))]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        x = compute_fixed_point(T, X[0], error_tol=1e-3, max_iter=max_iter,
                                verbose=1, method='imitation_game')
    messages = [str(wi.message) for wi in w]
    assert_(any('numerical breakdown' in msg for msg in messages))
    assert_(not any('max_iter' in msg for msg in messages))
    assert_(len(evaluations) < max_iter)  # Stopped at the breakdown
    assert_(np.isfinite(x))


def test_initialize_tableau_ig_layout():
    # [I | shifted payoffs | 1] with the slacks basic: the mover's payoff
    # from point i against point j is -||X[i] - Y[j]||^2, shifted column
    # by column so that each column has minimum 1
    rng = np.random.RandomState(0)
    m, n = 4, 3
    X, Y = rng.standard_normal((m, n)), rng.standard_normal((m, n))
    D = np.empty((m, m))
    for k in range(m):  # One point at a time, as in the iteration
        _update_distances(X, Y, k, D)
    assert_allclose(D, ((X[:, np.newaxis, :] - Y[np.newaxis, :, :])**2)
                    .sum(axis=-1))

    tableau = np.empty((m, 2*m+1))
    basis = np.empty(m, dtype=int)
    _initialize_tableau_ig(D, tableau, basis)

    assert_array_equal(tableau[:, :m], np.eye(m))
    assert_allclose(tableau[:, m:2*m], D.max(axis=0) - D + 1)
    assert_array_equal(tableau[:, 2*m], np.ones(m))
    assert_array_equal(basis, np.arange(m))


def test_update_distances_scalar_points():
    X = np.array([0., 1., 3.])
    Y = np.array([1., 1., 0.])
    D = np.empty((3, 3))
    for k in range(3):
        _update_distances(X, Y, k, D)
    assert_array_equal(D, (X[:, np.newaxis] - Y[np.newaxis, :])**2)


def test_imitation_game_buffer_growth(monkeypatch):
    # The buffers for the points, the distances and the tableau grow by
    # doubling; the iterates must not depend on the initial buffer size.
    # The chaotic logistic map never converges, so the iteration runs to
    # max_iter and goes through several doublings from a tiny buffer
    T = lambda x: 4 * x * (1 - x)
    kwargs = dict(error_tol=0, max_iter=60, verbose=0,
                  method='imitation_game')
    x_ref = compute_fixed_point(T, 0.3, **kwargs)
    monkeypatch.setattr(quantecon._compute_fp, '_BUFF_SIZE', 3)
    x_small = compute_fixed_point(T, 0.3, **kwargs)
    assert_array_equal(x_small, x_ref)
