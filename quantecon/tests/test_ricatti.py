"""
Filename: test_tauchen.py
Authors: Chase Coleman
Date: 07/22/2014

Tests for ricatti.py file

"""
import sys
import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.riccati import dare


class TestDoubling(unittest.TestCase):

    def setUp(self):
        self.A, self.B, self.R, self.Q = 1., 1., 1., 1.

    def tearDown(self):
        del self.A
        del self.B
        del self.R
        del self.Q

    def testGoldenNumberfloat(self):
        val = dare(self.A, self.B, self.R, self.Q)
        gold_ratio = (1 + np.sqrt(5)) / 2.
        self.assertTrue( abs(val - gold_ratio) < 1e-12)

    def testGoldenNumber2d(self):
        A, B, R, Q = np.eye(2), np.eye(2), np.eye(2), np.eye(2)
        gold_diag = np.eye(2) * (1 + np.sqrt(5)) / 2.
        val = dare(A, B, R, Q)

        self.assertTrue(np.allclose(val, gold_diag))

    def testSingularR(self):
        # Need to fix this in the algorithm before we test it
        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDoubling)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
