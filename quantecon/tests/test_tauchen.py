"""
Filename: test_tauchen.py
Authors: Chase Coleman, Spencer Lyon
Date: 07/18/2014

Tests for tauchen.py file

"""

import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.tauchen import approx_markov


class TestApproxMarkov(unittest.TestCase):

    def setUp(self):
        self.rho, self.sigma_u = .9, 1.
        self.m, self.n = 10, 3

        self.x, self.P = approx_markov(self.rho, self.sigma_u, self.m, self.n)

    def tearDown(self):
        del self.x
        del self.P

    def testShape(self):
        i, j = self.P.shape

        self.assertTrue(i == j)
    def testSomething(self):
        self.assertTrue(False)

    def testDim(self):
        dim_x = self.x.ndim
        dim_P = self.P.ndim

        self.assertTrue(dim_x == 1 and dim_P == 2)




# def test_shape():
#     x, P = approx_markov(.9, 1., 10, 3)

#     n, m = P.shape
#     j = x.size

#     assert(n == m and n == j)

# def test_dims():
#     x, P = a



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestApproxMarkov)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
