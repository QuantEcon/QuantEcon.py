"""
Author: Chase Coleman
Filename: test_lqcontrol

Tests for lqcontrol.py file

"""
import sys
import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.lqcontrol import LQ


class TestLQControl(unittest.TestCase):

    def setUp(self):
        # Initial Values
        self.q = 1.
        self.r = 1.
        self.rf = 1.
        self.a = .95
        self.b = -1.
        self.c = .05
        self.beta = .95
        self.T = 2

        self.lq_scalar = LQ(q, r, a, b, c, beta, T, rf)

    def tearDown(self):
        del self.lq_scalar
        del self.lq_mat


    def test_stationarity(self):
        vals = self.ss.stationary_distributions(max_iter=1000, tol=1e-9)
        ssmux, ssmuy, sssigx, sssigy = vals

        self.assertTrue(abs(ssmux - ssmuy) < 2e-8)
        self.assertTrue(abs(sssigx - sssigy) < 2e-8)
        self.assertTrue(abs(ssmux) < 2e-8)
        self.assertTrue(abs(sssigx - self.ss.C/(1 - self.ss.A**2)))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLQControl)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
