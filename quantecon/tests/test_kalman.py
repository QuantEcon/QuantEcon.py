"""
Tests for the kalman.py

"""
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.lss import LinearStateSpace
from quantecon.kalman import Kalman


class TestKalman(unittest.TestCase):

    def setUp(self):
        # Initial Values
        self.A = np.array([[.95, 0], [0., .95]])
        self.C = np.eye(2) * np.sqrt(0.5)
        self.G = np.eye(2) * .5
        self.H = np.eye(2) * np.sqrt(0.2)

        self.Q = np.dot(self.C, self.C.T)
        self.R = np.dot(self.H, self.H.T)

        ss = LinearStateSpace(self.A, self.C, self.G, self.H)

        self.kf = Kalman(ss)

        self.methods = ['doubling', 'qz']


    def tearDown(self):
        del self.kf


    def test_stationarity(self):
        A, Q, G, R = self.A, self.Q, self.G, self.R
        kf = self.kf

        for method in self.methods:
            sig_inf, kal_gain = kf.stationary_values(method=method)

            mat_inv = np.linalg.inv(G.dot(sig_inf).dot(G.T) + R)

            # Compute the kalmain gain and sigma infinity according to the
            # recursive equations and compare
            kal_recursion = np.dot(A, sig_inf).dot(G.T).dot(mat_inv)
            sig_recursion = (A.dot(sig_inf).dot(A.T) -
                                kal_recursion.dot(G).dot(sig_inf).dot(A.T) + Q)

            assert_allclose(kal_gain, kal_recursion, rtol=1e-4, atol=1e-2)
            assert_allclose(sig_inf, sig_recursion, rtol=1e-4, atol=1e-2)


    def test_update_using_stationary(self):
        kf = self.kf

        for method in self.methods:
            sig_inf, kal_gain = kf.stationary_values(method=method)

            kf.set_state(np.zeros((2, 1)), sig_inf)

            kf.update(np.zeros((2, 1)))

            assert_allclose(kf.Sigma, sig_inf, rtol=1e-4, atol=1e-2)
            assert_allclose(kf.x_hat.squeeze(), np.zeros(2),
                            rtol=1e-4, atol=1e-2)


    def test_update_nonstationary(self):
        A, Q, G, R = self.A, self.Q, self.G, self.R
        kf = self.kf

        curr_x, curr_sigma = np.ones((2, 1)), np.eye(2) * .75
        y_observed = np.ones((2, 1)) * .75

        kf.set_state(curr_x, curr_sigma)
        kf.update(y_observed)

        mat_inv = np.linalg.inv(G.dot(curr_sigma).dot(G.T) + R)
        curr_k = np.dot(A, curr_sigma).dot(G.T).dot(mat_inv)
        new_sigma = (A.dot(curr_sigma).dot(A.T) -
                    curr_k.dot(G).dot(curr_sigma).dot(A.T) + Q)

        new_xhat = A.dot(curr_x) + curr_k.dot(y_observed - G.dot(curr_x))

        assert_allclose(kf.Sigma, new_sigma, rtol=1e-4, atol=1e-2)
        assert_allclose(kf.x_hat, new_xhat, rtol=1e-4, atol=1e-2)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKalman)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
