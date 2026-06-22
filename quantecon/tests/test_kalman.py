"""
Tests for the kalman.py

"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from quantecon import LinearStateSpace
from quantecon import Kalman
from quantecon import solve_discrete_riccati


RICCATI_METHODS = ['doubling', 'qz']


def _symmetric_kalman():
    """Simple diagonal 2D model used in the original Kalman tests."""
    A = np.array([[.95, 0], [0., .95]])
    C = np.eye(2) * np.sqrt(0.5)
    G = np.eye(2) * .5
    H = np.eye(2) * np.sqrt(0.2)
    return Kalman(LinearStateSpace(A, C, G, H))


def _two_noisy_signals_kalman(rho=0.8, sigma_v=0.5, sigma_e=0.6):
    """
    Two-noisy-signals filter from section 37.5.3 of the QuantEcon lecture
    "Knowing the Forecasts of Others":
    https://python-advanced.quantecon.org/knowing_forecasts_of_others.html#two-noisy-signals

    State: theta_{t+1} = rho theta_t + v_t,  Var(v_t) = sigma_v^2
    Observations: w_t = [1, 1]' theta_t + [e_1t, e_2t]'
    """
    A = np.array([[rho]])
    C = np.array([[sigma_v]])
    G = np.array([[1.], [1.]])
    H = np.eye(2) * sigma_e
    return Kalman(LinearStateSpace(A, C, G, H))


def _lecture_two_signal_kalman_gain(rho, sigma_v, sigma_e):
    """
    Stationary Kalman gain from lecture eqs. (37.25)-(37.26), independent of
    Kalman.stationary_values.
    """
    p = solve_discrete_riccati(
        np.array([[rho]]),
        np.array([[np.sqrt(2.)]]),
        np.array([[sigma_v ** 2]]),
        np.array([[sigma_e ** 2]]),
        np.zeros((1, 1)),
    ).item()
    kappa = rho * p / (2 * p + sigma_e ** 2)
    return np.array([[kappa, kappa]])


def _expected_stationary_coefficients(A, G, K, k, j, coeff_type):
    """Closed-form coefficients using matrix powers and a supplied gain K."""
    if coeff_type == 'ma':
        coeffs = [np.identity(k)]
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


def _assert_coeff_lists_equal(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a.shape == e.shape
        assert_allclose(a, e, rtol=1e-10, atol=1e-10)


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

        self.methods = RICCATI_METHODS


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
        self.kf = _symmetric_kalman()
        self.methods = RICCATI_METHODS

    def test_stationary_coefficients_ma(self):
        kf = self.kf
        for method in self.methods:
            kf.stationary_values(method=method)
            A, G, K, k = kf.ss.A, kf.ss.G, kf.K_infinity, kf.ss.k
            for j in (0, 1, 3):
                actual = kf.stationary_coefficients(j, coeff_type='ma')
                expected = _expected_stationary_coefficients(
                    A, G, K, k, j, 'ma')
                _assert_coeff_lists_equal(actual, expected)

    def test_stationary_coefficients_var(self):
        kf = self.kf
        for method in self.methods:
            kf.stationary_values(method=method)
            A, G, K, k = kf.ss.A, kf.ss.G, kf.K_infinity, kf.ss.k
            for j in (0, 1, 3):
                actual = kf.stationary_coefficients(j, coeff_type='var')
                expected = _expected_stationary_coefficients(
                    A, G, K, k, j, 'var')
                _assert_coeff_lists_equal(actual, expected)

    def test_stationary_coefficients_invalid_type(self):
        kf = self.kf
        kf.stationary_values()
        with pytest.raises(ValueError, match="Unknown coefficient type"):
            kf.stationary_coefficients(1, coeff_type='invalid')


class TestKalmanStationaryCoefficientsTwoNoisySignals:

    rho, sigma_v, sigma_e = 0.8, 0.5, 0.6

    def setup_method(self):
        self.kf = _two_noisy_signals_kalman(
            self.rho, self.sigma_v, self.sigma_e)
        self.methods = RICCATI_METHODS
        self.K_ref = _lecture_two_signal_kalman_gain(
            self.rho, self.sigma_v, self.sigma_e)

    def test_stationary_kalman_gain_matches_lecture(self):
        kf = self.kf
        for method in self.methods:
            kf.stationary_values(method=method)
            assert_allclose(kf.K_infinity, self.K_ref, rtol=1e-10, atol=1e-10)

    def test_stationary_coefficients_ma(self):
        kf = self.kf
        A, G, k = kf.ss.A, kf.ss.G, kf.ss.k
        for method in self.methods:
            kf.stationary_values(method=method)
            for j in (0, 1, 3):
                actual = kf.stationary_coefficients(j, coeff_type='ma')
                expected = _expected_stationary_coefficients(
                    A, G, self.K_ref, k, j, 'ma')
                _assert_coeff_lists_equal(actual, expected)

    def test_stationary_coefficients_var(self):
        kf = self.kf
        A, G, k = kf.ss.A, kf.ss.G, kf.ss.k
        for method in self.methods:
            kf.stationary_values(method=method)
            for j in (0, 1, 3):
                actual = kf.stationary_coefficients(j, coeff_type='var')
                expected = _expected_stationary_coefficients(
                    A, G, self.K_ref, k, j, 'var')
                _assert_coeff_lists_equal(actual, expected)

    def test_coefficients_are_not_diagonal(self):
        """Sanity check: (n, k) = (1, 2) model yields non-diagonal MA coefficients."""
        expected = _expected_stationary_coefficients(
            self.kf.ss.A, self.kf.ss.G, self.K_ref, self.kf.ss.k, 1, 'ma')
        psi_1 = expected[1]
        assert psi_1.shape == (2, 2)
        assert not np.allclose(psi_1, np.diag(np.diag(psi_1)))
