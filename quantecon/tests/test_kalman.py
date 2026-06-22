"""
Tests for the kalman.py

"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from quantecon import LinearStateSpace
from quantecon import Kalman


class TestKalman:

    def setup_method(self):
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


    def teardown_method(self):
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


class TestKalmanStationaryCoefficients:

    def setup_method(self):
        self.A = np.array([[.95, 0], [0., .95]])
        self.C = np.eye(2) * np.sqrt(0.5)
        self.G = np.eye(2) * .5
        self.H = np.eye(2) * np.sqrt(0.2)
        ss = LinearStateSpace(self.A, self.C, self.G, self.H)
        self.kf = Kalman(ss)
        self.methods = ['doubling', 'qz']

    @staticmethod
    def _expected_stationary_coefficients(kf, j, coeff_type):
        """Closed-form reference using matrix powers."""
        A, G = kf.ss.A, kf.ss.G
        K = kf.K_infinity
        if coeff_type == 'ma':
            coeffs = [np.identity(kf.ss.k)]
            coeffs.extend(
                G @ np.linalg.matrix_power(A, i) @ K for i in range(j)
            )
        elif coeff_type == 'var':
            phi = A - K @ G
            coeffs = [G @ K]
            coeffs.extend(
                G @ np.linalg.matrix_power(phi, i) @ K for i in range(1, j + 1)
            )
        else:
            raise ValueError("Unknown coefficient type")
        return coeffs

    @staticmethod
    def _assert_coeff_lists_equal(actual, expected):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a.shape == e.shape
            assert_allclose(a, e, rtol=1e-10, atol=1e-10)

    def test_stationary_coefficients_ma(self):
        kf = self.kf
        for method in self.methods:
            kf.stationary_values(method=method)
            for j in (0, 1, 3):
                actual = kf.stationary_coefficients(j, coeff_type='ma')
                expected = self._expected_stationary_coefficients(kf, j, 'ma')
                self._assert_coeff_lists_equal(actual, expected)

    def test_stationary_coefficients_var(self):
        kf = self.kf
        for method in self.methods:
            kf.stationary_values(method=method)
            for j in (0, 1, 3):
                actual = kf.stationary_coefficients(j, coeff_type='var')
                expected = self._expected_stationary_coefficients(kf, j, 'var')
                self._assert_coeff_lists_equal(actual, expected)

    def test_stationary_coefficients_invalid_type(self):
        kf = self.kf
        kf.stationary_values()
        with pytest.raises(ValueError, match="Unknown coefficient type"):
            kf.stationary_coefficients(1, coeff_type='invalid')
