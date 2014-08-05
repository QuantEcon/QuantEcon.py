"""
Filename: test_lss.py
Authors: Chase Coleman
Date: 07/24/2014

Tests for lss.py file

"""
import sys
import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.lss import LSS


class TestLinearStateSpace(unittest.TestCase):

    def setUp(self):
        # Initial Values
        A = .95
        C = .05
        G = 1.
        mu_0 = .75

        self.ss = LSS(A, C, G, mu_0)

    def tearDown(self):
        del self.ss

    def test_stationarity(self):
        vals = self.ss.stationary_distributions(max_iter=1000, tol=1e-9)
        ssmux, ssmuy, sssigx, sssigy = vals

        self.assertTrue(abs(ssmux - ssmuy) < 2e-8)
        self.assertTrue(abs(sssigx - sssigy) < 2e-8)
        self.assertTrue(abs(ssmux) < 2e-8)
        self.assertTrue(abs(sssigx - self.ss.C/(1 - self.ss.A**2)))

    def test_replicate(self):
        xval, yval = self.ss.replicate(T=100, num_reps=5000)

        assert_allclose(xval, yval)
        self.assertEqual(xval.size, 5000)
        self.assertLessEqual(abs(np.mean(xval)), .01)

    # def test_


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinearStateSpace)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)

