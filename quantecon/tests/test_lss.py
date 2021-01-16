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
        # Example 1
        A = .95
        C = .05
        G = 1.
        mu_0 = .75

        self.ss1 = LinearStateSpace(A, C, G, mu_0=mu_0)

        # Example 2
        ρ1 = 0.5
        ρ2 = 0.3
        α = 0.5

        A = np.array([[ρ1, ρ2, α], [1, 0, 0], [0, 0, 1]])
        C = np.array([[1], [0], [0]])
        G = np.array([[1, 0, 0]])
        mu_0 = [0.5, 0.5, 1]

        self.ss2 = LinearStateSpace(A, C, G, mu_0=mu_0)

    def tearDown(self):
        del self.ss1
        del self.ss2

    def test_stationarity(self):
        vals = self.ss1.stationary_distributions()
        ssmux, ssmuy, sssigx, sssigy, sssigyx = vals

        self.assertTrue(abs(ssmux - ssmuy) < 2e-8)
        self.assertTrue(abs(sssigx - sssigy) < 2e-8)
        self.assertTrue(abs(ssmux) < 2e-8)
        self.assertTrue(abs(sssigx - self.ss1.C**2/(1 - self.ss1.A**2)) < 2e-8)
        self.assertTrue(abs(sssigyx - self.ss1.G @ sssigx) < 2e-8)

        vals = self.ss2.stationary_distributions()
        ssmux, ssmuy, sssigx, sssigy, sssigyx = vals

        assert_allclose(ssmux.flatten(), np.array([2.5, 2.5, 1]))
        assert_allclose(ssmuy.flatten(), np.array([2.5]))
        assert_allclose(sssigx, self.ss2.A @ sssigx @ self.ss2.A.T + self.ss2.C @ self.ss2.C.T)
        assert_allclose(sssigy, self.ss2.G @ sssigx @ self.ss2.G.T)
        assert_allclose(sssigyx, self.ss2.G @ sssigx)

    def test_simulate(self):
        ss = self.ss1

        sim = ss.simulate(ts_length=250)
        for arr in sim:
            self.assertTrue(len(arr[0])==250)

    def test_simulate_with_seed(self):
        ss = self.ss1

        xval, yval = ss.simulate(ts_length=5, random_state=5)
        expected_output = np.array([0.75 , 0.73456137, 0.6812898, 0.76876387,
                                    .71772107])

        assert_allclose(xval[0], expected_output)
        assert_allclose(yval[0], expected_output)

    def test_replicate(self):
        xval, yval = self.ss1.replicate(T=100, num_reps=5000)

        assert_allclose(xval, yval)
        self.assertEqual(xval.size, 5000)
        self.assertLessEqual(abs(np.mean(xval)), .05)

    def test_replicate_with_seed(self):
        xval, yval = self.ss1.replicate(T=100, num_reps=5, random_state=5)
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

