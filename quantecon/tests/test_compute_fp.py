"""
Tests for compute_fp.py

References
----------

https://www.math.ucdavis.edu/~hunter/book/ch3.pdf

TODO: add multivariate case

"""
import unittest
import numpy as np
from nose.tools import ok_, raises
from quantecon import compute_fixed_point


class TestFPLogisticEquation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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
            self.assertTrue(abs(compute_fixed_point(f, i, **self.kwargs))
                            < 1e-4)

    def test_not_contraction_2(self):
        "compute_fp: no convergence outside interval of convergence"
        f = lambda x: self.T(x, self.mu_2)
        for i in self.unit_inverval:
            # This shouldn't converge to 0.0
            self.assertFalse(abs(compute_fixed_point(f, i, **self.kwargs))
                             < 1e-4)

    def test_contraction_2(self):
        "compute_fp: convergence inside interval of convergence"
        f = lambda x: self.T(x, self.mu_2)
        fp = (4 * self.mu_2 - 1) / (4 * self.mu_2)
        for i in self.unit_inverval:
            # This should converge to fp
            self.assertTrue(abs(compute_fixed_point(f, i, **self.kwargs)-fp)
                            < 1e-4)

    def test_not_contraction_1(self):
        "compute_fp: no convergence outside interval of convergence"
        f = lambda x: self.T(x, self.mu_1)
        fp = (4 * self.mu_1 - 1) / (4 * self.mu_1)
        for i in self.unit_inverval:
            # This should not converge  (b/c unique fp is 0.0)
            self.assertFalse(abs(compute_fixed_point(f, i, **self.kwargs)-fp)
                             < 1e-4)

    def test_imitation_game_method(self):
        "compute_fp: Test imitation game method"
        method = 'imitation_game'
        error_tol = self.kwargs['error_tol']

        for mu in [self.mu_1, self.mu_2]:
            for i in self.unit_inverval:
                fp_computed = compute_fixed_point(self.T, i, method=method,
                                                  mu=mu, **self.kwargs)
                self.assertTrue(
                    abs(self.T(fp_computed, mu=mu) - fp_computed) <= error_tol
                )

            # numpy array input
            i = np.asarray(self.unit_inverval)
            fp_computed = compute_fixed_point(self.T, i, method=method, mu=mu,
                                              **self.kwargs)
            self.assertTrue(
                abs(self.T(fp_computed, mu=mu) - fp_computed).max() <=
                error_tol
            )


class TestComputeFPContraction():
    def setUp(self):
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
            ok_(fp_computed <= error_tol * 2)

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
            ok_(fp_computed <= error_tol * 2)

    def test_2d_input(self):
        error_tol = self.coeff**4

        for method in self.methods:
            init = np.array([[-1, 0.5], [-1/3, 0.1]])
            fp_computed = compute_fixed_point(self.f, init,
                                              error_tol=error_tol,
                                              method=method)
            ok_((fp_computed <= error_tol * 2).all())


@raises(ValueError)
def test_raises_value_error_nonpositive_max_iter():
    f = lambda x: 0.5*x
    init = 1.
    max_iter = 0
    compute_fixed_point(f, init, max_iter=max_iter)
