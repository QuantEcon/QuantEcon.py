"""
Tests for lss.py

"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_, assert_raises
from quantecon import LinearStateSpace
from quantecon._lss import simulate_linear_model


class TestLinearStateSpace:

    def setup_method(self):
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

    def teardown_method(self):
        del self.ss1
        del self.ss2

    def test_stationarity(self):
        vals = self.ss1.stationary_distributions()
        ssmux, ssmuy, sssigx, sssigy, sssigyx = vals

        assert_(abs(ssmux - ssmuy) < 2e-8)
        assert_(abs(sssigx - sssigy) < 2e-8)
        assert_(abs(ssmux) < 2e-8)
        assert_(abs(sssigx - self.ss1.C**2/(1 - self.ss1.A**2)) < 2e-8)
        assert_(abs(sssigyx - self.ss1.G @ sssigx) < 2e-8)

        vals = self.ss2.stationary_distributions()
        ssmux, ssmuy, sssigx, sssigy, sssigyx = vals

        assert_allclose(ssmux.flatten(), np.array([2.5, 2.5, 1]))
        assert_allclose(ssmuy.flatten(), np.array([2.5]))
        assert_allclose(
            sssigx,
            self.ss2.A @ sssigx @ self.ss2.A.T + self.ss2.C @ self.ss2.C.T
        )
        assert_allclose(sssigy, self.ss2.G @ sssigx @ self.ss2.G.T)
        assert_allclose(sssigyx, self.ss2.G @ sssigx)

    def test_simulate(self):
        ss = self.ss1

        sim = ss.simulate(ts_length=250)
        for arr in sim:
            assert_(len(arr[0]) == 250)

    def test_simulate_with_seed(self):
        ss = self.ss1

        xval, yval = ss.simulate(ts_length=5, random_state=5)
        expected_output = np.array([0.75, 0.69595649, 0.78269723, 0.73095776,
                                    0.69989036])

        assert_allclose(xval[0], expected_output)
        assert_allclose(yval[0], expected_output)

    def test_replicate(self):
        xval, yval = self.ss1.replicate(T=100, num_reps=5000)

        assert_allclose(xval, yval)
        assert_(xval.size == 5000)
        assert_(abs(np.mean(xval)) <= .05)

    def test_replicate_with_seed(self):
        xval, yval = self.ss1.replicate(T=100, num_reps=5, random_state=5)
        expected_output = np.array([0.10498898, 0.02892168, 0.04915998,
                                    0.18568489, 0.04541764])

        assert_allclose(xval[0], expected_output)
        assert_allclose(yval[0], expected_output)


def test_non_square_A():
    A = np.zeros((1, 2))
    C = np.zeros((1, 1))
    G = np.zeros((1, 1))

    assert_raises(ValueError, LinearStateSpace, A, C, G)


def test_C_incompatible_with_A():
    A = np.zeros((2, 2))
    C = np.zeros((3, 1))  # should have n = 2 rows
    G = np.zeros((1, 2))

    with pytest.raises(ValueError, match="Matrix C"):
        LinearStateSpace(A, C, G)


def test_G_incompatible_with_A():
    A = np.zeros((2, 2))
    C = np.zeros((2, 1))
    G = np.zeros((1, 3))  # should have n = 2 columns

    with pytest.raises(ValueError, match="Matrix G"):
        LinearStateSpace(A, C, G)


def test_init_converts_array_like_inputs():
    # Lists and scalars become float 2-D arrays; mu_0 becomes a column.
    ss = LinearStateSpace([[0.5, 0.1], [0., 0.4]], [[1.], [0.]], [1, 0],
                          H=2, mu_0=[1, 2], Sigma_0=[[1, 0], [0, 2]])

    assert (ss.n, ss.m, ss.k, ss.l) == (2, 1, 1, 1)
    for arr, shape in [(ss.A, (2, 2)), (ss.C, (2, 1)), (ss.G, (1, 2)),
                       (ss.H, (1, 1)), (ss.mu_0, (2, 1)),
                       (ss.Sigma_0, (2, 2))]:
        assert arr.shape == shape
        assert arr.dtype == float
    assert_allclose(ss.mu_0, [[1.], [2.]])
    assert_allclose(ss.Sigma_0, [[1., 0.], [0., 2.]])


def test_simulate_linear_model_matches_recursion():
    # x_{t+1} = A x_t + v_t unrolls to
    #   x_t = A^t x_0 + sum_{s=0}^{t-1} A^(t-1-s) v_s
    A = np.array([[0.5, 0.2], [0.1, 0.7]])
    x0 = np.array([1., -0.5])
    ts_length = 8
    v = np.random.default_rng(0).standard_normal((2, ts_length - 1))

    x = simulate_linear_model(A, x0, v, ts_length)

    assert x.shape == (2, ts_length)
    for t in range(ts_length):
        expected = np.linalg.matrix_power(A, t) @ x0
        for s in range(t):
            expected += np.linalg.matrix_power(A, t - 1 - s) @ v[:, s]
        assert_allclose(x[:, t], expected, rtol=1e-10, atol=1e-12)


def test_geometric_sums_scalar():
    # With scalar A = a and G = g:
    #   S_x = x_t / (1 - beta a),  S_y = g S_x
    a, g, beta, x_t = 0.95, 2., 0.9, 3.
    ss = LinearStateSpace(a, 0.05, g)

    S_x, S_y = ss.geometric_sums(beta, x_t)

    assert_allclose(S_x, x_t / (1 - beta * a))
    assert_allclose(S_y, g * x_t / (1 - beta * a))


class TestLinearStateSpaceGeneral:
    """
    A stable model whose dimensions and matrices are chosen so that a
    transposed or mis-ordered matrix product, or one built from the wrong
    dimension (n where l is meant, say), changes the shape or the values
    of the answer: n = 2 states, m = 3 shocks, k = 4 observables and
    l = 5 measurement shocks (all different), with a non-symmetric A and
    non-square C, G and H.
    """

    def setup_method(self):
        self.A = np.array([[0.5, 0.2], [0.1, 0.7]])
        self.C = np.array([[1.0, 0.0, 0.3], [0.0, 0.8, 0.2]])
        self.G = np.array([[1.0, 0.5], [0.0, 1.0], [0.4, -0.2], [0.3, 0.6]])
        self.H = np.array([[0.8, 0.0, 0.3, 0.0, 0.2],
                           [0.0, 0.7, 0.0, 0.4, 0.1],
                           [0.3, 0.0, 0.6, 0.2, 0.0],
                           [0.0, 0.2, 0.0, 0.5, 0.7]])
        self.mu_0 = [1.0, -0.5]
        self.Sigma_0 = np.array([[0.5, 0.1], [0.1, 0.3]])

        self.ss = LinearStateSpace(self.A, self.C, self.G, self.H,
                                   mu_0=self.mu_0, Sigma_0=self.Sigma_0)

    def test_dimensions(self):
        # The dimensions are read off the shapes of A, C, G and H; they are
        # all different, which the other tests of this class rely on
        ss = self.ss
        assert (ss.n, ss.m, ss.k, ss.l) == (2, 3, 4, 5)

    def test_default_initial_conditions(self):
        # mu_0 is an n x 1 column and Sigma_0 an n x n matrix
        ss = LinearStateSpace(self.A, self.C, self.G, self.H)

        assert ss.mu_0.shape == (ss.n, 1)
        assert ss.Sigma_0.shape == (ss.n, ss.n)

    @pytest.mark.parametrize("with_H", [True, False])
    def test_moment_sequence(self, with_H):
        H = self.H if with_H else None
        ss = LinearStateSpace(self.A, self.C, self.G, H,
                              mu_0=self.mu_0, Sigma_0=self.Sigma_0)
        A, C, G = ss.A, ss.C, ss.G
        R = self.H @ self.H.T if with_H else 0.

        moments = ss.moment_sequence()
        for t in range(6):
            mu_x, mu_y, Sigma_x, Sigma_y = next(moments)

            # x_t = A^t x_0 + sum_{s=0}^{t-1} A^s C w_{t-s}, so
            At = np.linalg.matrix_power(A, t)
            mu_x_t = At @ ss.mu_0
            Sigma_x_t = At @ ss.Sigma_0 @ At.T
            for s in range(t):
                As = np.linalg.matrix_power(A, s)
                Sigma_x_t = Sigma_x_t + As @ C @ C.T @ As.T

            assert_allclose(mu_x, mu_x_t, rtol=1e-10, atol=1e-12)
            assert_allclose(Sigma_x, Sigma_x_t, rtol=1e-10, atol=1e-12)
            assert_allclose(mu_y, G @ mu_x_t, rtol=1e-10, atol=1e-12)
            assert_allclose(Sigma_y, G @ Sigma_x_t @ G.T + R,
                            rtol=1e-10, atol=1e-12)

    def test_stationary_distributions_with_H(self):
        ss = self.ss
        A, C, G, H = ss.A, ss.C, ss.G, ss.H

        mu_x, mu_y, Sigma_x, Sigma_y, Sigma_yx = ss.stationary_distributions()

        # No constant states and A is stable, so the stationary mean is zero
        assert_allclose(mu_x, 0., atol=1e-12)
        assert_allclose(mu_y, 0., atol=1e-12)
        # Discrete Lyapunov equation for Sigma_x
        assert_allclose(Sigma_x, A @ Sigma_x @ A.T + C @ C.T,
                        rtol=1e-10, atol=1e-12)
        # Measurement noise H v_t adds H H' to the variance of y_t
        assert_allclose(Sigma_y, G @ Sigma_x @ G.T + H @ H.T,
                        rtol=1e-10, atol=1e-12)
        assert_allclose(Sigma_yx, G @ Sigma_x, rtol=1e-10, atol=1e-12)

    def test_moment_sequence_converges_to_stationary(self):
        ss = self.ss
        stationary = ss.stationary_distributions()

        moments = ss.moment_sequence()
        for _ in range(200):
            mu_x, mu_y, Sigma_x, Sigma_y = next(moments)

        for actual, expected in zip((mu_x, mu_y, Sigma_x, Sigma_y),
                                    stationary[:4]):
            assert_allclose(actual, expected, atol=1e-10)

    def test_simulate_measurement_noise(self):
        ss = self.ss
        ts_length = 20000

        x, y = ss.simulate(ts_length=ts_length, random_state=0)

        assert x.shape == (ss.n, ts_length)
        assert y.shape == (ss.k, ts_length)
        # y_t - G x_t = H v_t with v_t iid N(0, I): the sample covariance
        # should be near H H' (loose tolerance, the seed fixes the draws)
        assert_allclose(np.cov(y - ss.G @ x), ss.H @ ss.H.T, atol=0.05)

    def test_replicate_measurement_noise(self):
        ss = self.ss
        num_reps = 3000

        x, y = ss.replicate(T=3, num_reps=num_reps, random_state=0)

        assert x.shape == (ss.n, num_reps)
        assert y.shape == (ss.k, num_reps)
        # As in test_simulate_measurement_noise: y - G x = H v
        assert_allclose(np.cov(y - ss.G @ x), ss.H @ ss.H.T, atol=0.15)

    @pytest.mark.parametrize("beta", [0., 0.5, 0.9])
    def test_geometric_sums(self, beta):
        ss = self.ss
        x_t = np.array([[1.], [2.]])

        S_x, S_y = ss.geometric_sums(beta, x_t)

        # Truncated series sum_j beta^j A^j x_t; its terms shrink like
        # (beta * 0.78)^j (0.78 ~ spectral radius of A), so 200 terms are
        # far below the tolerance
        series = sum(beta**j * np.linalg.matrix_power(ss.A, j) @ x_t
                     for j in range(200))
        assert S_x.shape == (ss.n, 1)
        assert S_y.shape == (ss.k, 1)
        assert_allclose(S_x, series, rtol=1e-10, atol=1e-12)
        assert_allclose(S_y, ss.G @ series, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("j", [0, 1, 4])
    def test_impulse_response(self, j):
        ss = self.ss

        xcoef, ycoef = ss.impulse_response(j)

        # j + 1 coefficients, for lags 0 through j
        assert len(xcoef) == len(ycoef) == j + 1
        for lag in range(j + 1):
            Aj_C = np.linalg.matrix_power(ss.A, lag) @ ss.C
            assert xcoef[lag].shape == (ss.n, ss.m)
            assert ycoef[lag].shape == (ss.k, ss.m)
            assert_allclose(xcoef[lag], Aj_C, rtol=1e-10, atol=1e-12)
            assert_allclose(ycoef[lag], ss.G @ Aj_C, rtol=1e-10, atol=1e-12)

    def test_impulse_response_default_lags(self):
        xcoef, ycoef = self.ss.impulse_response()

        assert len(xcoef) == len(ycoef) == 6
