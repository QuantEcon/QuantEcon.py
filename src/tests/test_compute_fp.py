"""
tests for quantecon.compute_fp module

@author : Spencer Lyon
@date : 2014-07-31

References
----------

https://www.math.ucdavis.edu/~hunter/book/ch3.pdf

TODO: add multivariate case

"""
from __future__ import division
import unittest
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
