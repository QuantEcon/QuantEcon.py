"""
Tests for the kalman.py

"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from quantecon import LinearStateSpace
from quantecon import Kalman


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


def _general_kalman():
    """
    Stable model with n = 3 states, m = 1 shock, k = 2 observables and
    l = 4 measurement shocks. The four dimensions are pairwise different
    (asserted below) and A, G, H are non-symmetric, so a transposed block,
    or a block built from the wrong dimension (n where l is meant, say),
    changes the shape or the values of the result.
    """
    A = np.array([[0.8, 0.2, 0.0], [0.1, 0.5, 0.3], [0.0, 0.2, 0.6]])
    C = np.array([[1.0], [0.3], [0.5]])
    G = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.7]])
    H = np.array([[0.7, 0.1, 0.0, 0.3], [0.0, 0.5, 0.2, 0.1]])
    ss = LinearStateSpace(A, C, G, H)
    assert len({ss.n, ss.m, ss.k, ss.l}) == 4
    return Kalman(ss)


def _lecture_two_signal_posterior_variance(rho, sigma_v, sigma_e):
    """
    Solve lecture eq. (37.26) for p.

    The Riccati equation reduces to
    2 p^2 + p (sigma_e^2 - 2 sigma_v^2 - rho^2 sigma_e^2) - sigma_v^2 sigma_e^2 = 0.
    """
    a = 2.0
    b = sigma_e ** 2 - 2 * sigma_v ** 2 - rho ** 2 * sigma_e ** 2
    c = -sigma_v ** 2 * sigma_e ** 2
    discriminant = b ** 2 - 4 * a * c
    return (-b + np.sqrt(discriminant)) / (2 * a)


def _lecture_two_signal_kalman_gain(rho, sigma_v, sigma_e):
    """
    Stationary Kalman gain from lecture eqs. (37.25)-(37.26), independent of
    Kalman.stationary_values.
    """
    p = _lecture_two_signal_posterior_variance(rho, sigma_v, sigma_e)
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

        self.Q = self.C @ self.C.T
        self.R = self.H @ self.H.T

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

            mat_inv = np.linalg.inv(G @ sig_inf @ G.T + R)

            # Compute the kalmain gain and sigma infinity according to the
            # recursive equations and compare
            kal_recursion = A @ sig_inf @ G.T @ mat_inv
            sig_recursion = (A @ sig_inf @ A.T -
                             kal_recursion @ G @ sig_inf @ A.T + Q)

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

        mat_inv = np.linalg.inv(G @ curr_sigma @ G.T + R)
        curr_k = A @ curr_sigma @ G.T @ mat_inv
        new_sigma = A @ curr_sigma @ A.T - curr_k @ G @ curr_sigma @ A.T + Q

        new_xhat = A @ curr_x + curr_k @ (y_observed - G @ curr_x)

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


class TestKalmanInfinityAndInnovationTwoNoisySignals:
    """
    The properties Sigma_infinity / K_infinity and
    stationary_innovation_covar, checked against the closed form of the
    two-noisy-signals model (see _two_noisy_signals_kalman).
    """

    rho, sigma_v, sigma_e = 0.8, 0.5, 0.6

    def setup_method(self):
        self.kf = _two_noisy_signals_kalman(
            self.rho, self.sigma_v, self.sigma_e)
        # p solves the Riccati equation in closed form (lecture eq. 37.26),
        # so p = Sigma_infinity
        self.p = _lecture_two_signal_posterior_variance(
            self.rho, self.sigma_v, self.sigma_e)

    def test_Sigma_infinity_computed_on_first_access(self):
        kf = self.kf

        # No call to stationary_values() beforehand: the property solves
        # the Riccati equation itself
        assert_allclose(kf.Sigma_infinity, [[self.p]], rtol=1e-10)

    def test_K_infinity_computed_on_first_access(self):
        kf = self.kf
        K_ref = _lecture_two_signal_kalman_gain(
            self.rho, self.sigma_v, self.sigma_e)

        assert_allclose(kf.K_infinity, K_ref, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("method", RICCATI_METHODS)
    def test_stationary_innovation_covar(self, method):
        kf = self.kf
        kf.stationary_values(method=method)
        # G = [1, 1]' and R = sigma_e^2 I, so G Sigma G' + R is
        # p [[1, 1], [1, 1]] + sigma_e^2 I
        expected = self.p * np.ones((2, 2)) + self.sigma_e ** 2 * np.eye(2)

        actual = kf.stationary_innovation_covar()

        assert actual.shape == (2, 2)
        assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


class TestKalmanGeneralModel:
    """
    Tests on a model whose four dimensions all differ (see
    _general_kalman). The steady-state filter is checked through
    fixed-point identities, since the Riccati solution has no closed form
    here.
    """

    def setup_method(self):
        self.kf = _general_kalman()

    def test_default_prior(self):
        # Documented defaults: x_hat = 0 (n x 1) and Sigma = I (n x n)
        kf = self.kf
        n = kf.ss.n

        assert_allclose(kf.x_hat, np.zeros((n, 1)))
        assert_allclose(kf.Sigma, np.identity(n))

    def test_update_matches_filter_recursion(self):
        kf = self.kf
        A, C, G, H = kf.ss.A, kf.ss.C, kf.ss.G, kf.ss.H
        x_hat = np.array([[1.0], [-0.5], [0.25]])
        Sigma = np.array([[1.0, 0.2, 0.0], [0.2, 0.8, 0.1], [0.0, 0.1, 0.6]])
        y = np.array([[0.4], [-0.3]])
        # Flat inputs are reshaped to columns
        kf.set_state(x_hat.ravel(), Sigma)

        kf.update(y.ravel())

        # One step of the recursion, with F = G Sigma G' + R and the gain
        # K = A Sigma G' F^{-1}: the new prior mean is
        # A x_hat + K (y - G x_hat) and the new prior covariance is
        # A Sigma A' - K F K' + C C'
        F = G @ Sigma @ G.T + H @ H.T
        K = A @ Sigma @ G.T @ np.linalg.inv(F)
        assert_allclose(kf.x_hat, A @ x_hat + K @ (y - G @ x_hat),
                        rtol=1e-10, atol=1e-12)
        assert_allclose(kf.Sigma, A @ Sigma @ A.T - K @ F @ K.T + C @ C.T,
                        rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("coeff_type", ['ma', 'var'])
    def test_stationary_coefficients(self, coeff_type):
        kf = self.kf
        A, G, k = kf.ss.A, kf.ss.G, kf.ss.k

        actual = kf.stationary_coefficients(3, coeff_type=coeff_type)
        expected = _expected_stationary_coefficients(
            A, G, kf.K_infinity, k, 3, coeff_type)

        _assert_coeff_lists_equal(actual, expected)

    @pytest.mark.parametrize("method", RICCATI_METHODS)
    def test_stationary_innovation_covar_fixed_point(self, method):
        kf = self.kf
        A, C, G = kf.ss.A, kf.ss.C, kf.ss.G
        Sigma, K = kf.stationary_values(method=method)

        F = kf.stationary_innovation_covar()

        assert F.shape == (kf.ss.k, kf.ss.k)
        # The gain is K = A Sigma G' F^{-1}, i.e. K F = A Sigma G' ...
        assert_allclose(K @ F, A @ Sigma @ G.T, rtol=1e-8, atol=1e-8)
        # ... and the Riccati equation in innovations form is
        # Sigma = A Sigma A' - K F K' + C C'
        assert_allclose(Sigma, A @ Sigma @ A.T - K @ F @ K.T + C @ C.T,
                        rtol=1e-8, atol=1e-8)

    def test_whitener_lss_matrices(self):
        kf = self.kf
        A, C, G, H = kf.ss.A, kf.ss.C, kf.ss.G, kf.ss.H
        n, m, ell = kf.ss.n, kf.ss.m, kf.ss.l
        K = kf.K_infinity

        whitened = kf.whitener_lss()

        # Block matrices as given in the docstring of whitener_lss
        Atil = np.block([
            [A, np.zeros((n, n)), np.zeros((n, ell))],
            [K @ G, A - K @ G, K @ H],
            [np.zeros((ell, n)), np.zeros((ell, n)), np.zeros((ell, ell))],
        ])
        Ctil = np.block([
            [C, np.zeros((n, ell))],
            [np.zeros((n, m)), np.zeros((n, ell))],
            [np.zeros((ell, m)), np.eye(ell)],
        ])
        Gtil = np.hstack([G, -G, H])
        assert isinstance(whitened, LinearStateSpace)
        assert whitened.n == 2 * n + ell
        assert whitened.m == m + ell
        assert whitened.k == kf.ss.k
        assert_allclose(whitened.A, Atil, rtol=1e-10, atol=1e-12)
        assert_allclose(whitened.C, Ctil, rtol=1e-10, atol=1e-12)
        assert_allclose(whitened.G, Gtil, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("method", RICCATI_METHODS)
    def test_whitener_lss_output_is_white_innovation(self, method):
        kf = self.kf
        kf.stationary_values(method=method)
        F = kf.stationary_innovation_covar()

        whitened = kf.whitener_lss()
        A_til, G_til = whitened.A, whitened.G
        _, mu_a, Sigma_x, Sigma_a, _ = whitened.stationary_distributions()

        # The output is the innovation y_t - G x_hat_t: mean zero, with
        # the steady-state innovation covariance G Sigma G' + R ...
        assert_allclose(mu_a, 0., atol=1e-10)
        assert_allclose(Sigma_a, F, rtol=1e-8, atol=1e-8)
        # ... and serially uncorrelated: Cov(a_{t+j}, a_t) = G A^j Sigma_x G'
        for j in (1, 2, 3):
            autocov = G_til @ np.linalg.matrix_power(A_til, j) \
                @ Sigma_x @ G_til.T
            assert_allclose(autocov, 0., atol=1e-8)
