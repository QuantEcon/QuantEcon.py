"""
Tests for lss.py

"""
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.lss import LinearStateSpace
from nose.tools import raises


class TestLinearStateSpace(unittest.TestCase):

    def setUp(self):
        # Initial Values
        A = .95
        C = .05
        G = 1.
        mu_0 = .75

        self.ss = LinearStateSpace(A, C, G, mu_0=mu_0)

    def tearDown(self):
        del self.ss

    def test_stationarity(self):
        vals = self.ss.stationary_distributions(max_iter=1000, tol=1e-9)
        ssmux, ssmuy, sssigx, sssigy = vals

        self.assertTrue(abs(ssmux - ssmuy) < 2e-8)
        self.assertTrue(abs(sssigx - sssigy) < 2e-8)
        self.assertTrue(abs(ssmux) < 2e-8)
        self.assertTrue(abs(sssigx - self.ss.C**2/(1 - self.ss.A**2)) < 2e-8)

    def test_simulate(self):
        ss = self.ss

        sim = ss.simulate(ts_length=250)
        for arr in sim:
            self.assertTrue(len(arr[0])==250)

    def test_simulate_with_seed(self):
        ss = self.ss

        xval, yval = ss.simulate(ts_length=5, random_state=5)
        expected_output = np.array([0.75 , 0.73456137, 0.6812898, 0.76876387,
                                    .71772107])

        assert_allclose(xval[0], expected_output)
        assert_allclose(yval[0], expected_output)

    def test_replicate(self):
        xval, yval = self.ss.replicate(T=100, num_reps=5000)

        assert_allclose(xval, yval)
        self.assertEqual(xval.size, 5000)
        self.assertLessEqual(abs(np.mean(xval)), .05)

    def test_replicate_with_seed(self):
        xval, yval = self.ss.replicate(T=100, num_reps=5, random_state=5)
        expected_output = np.array([0.06871204, 0.06937119, -0.1478022,
                                    0.23841252, -0.06823762])

        assert_allclose(xval[0], expected_output)
        assert_allclose(yval[0], expected_output)


@raises(ValueError)
def test_non_square_A():
    A = np.zeros((1, 2))
    C = np.zeros((1, 1))
    G = np.zeros((1, 1))

    LinearStateSpace(A, C, G)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinearStateSpace)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)

