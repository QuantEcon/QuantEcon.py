"""
Tests for arma.py file.  Most of this testing can be considered
covered by the numpy tests since we rely on much of their code.

"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_
from quantecon import ARMA


class TestARMA():
    def setup_method(self):
        # Initial Values
        phi = np.array([.95, -.4, -.4])
        theta = np.zeros(3)
        sigma = .15


        self.lp = ARMA(phi, theta, sigma)

    def teardown_method(self):
        del self.lp

    def test_simulate(self):
        lp = self.lp

        sim = lp.simulation(ts_length=250)

        assert_(sim.size == 250)

    def test_simulate_with_seed(self):
        lp = self.lp
        seed = 5
        sim0 = lp.simulation(ts_length=10, random_state=seed)
        sim1 = lp.simulation(ts_length=10, random_state=seed)

        assert_array_equal(sim0, sim1)

    def test_impulse_response(self):
        lp = self.lp

        imp_resp = lp.impulse_response(impulse_length=75)

        assert_(imp_resp.size == 75)


def _ma_coefficients(phi, theta, n):
    # psi_j = theta_j + sum_i phi_i psi_{j-i}, with psi_0 = 1
    phi, theta = np.atleast_1d(phi), np.atleast_1d(theta)
    psi = np.zeros(n)
    for j in range(n):
        psi[j] = 1.0 if j == 0 else (theta[j - 1] if j <= len(theta) else 0.0)
        for i in range(1, min(j, len(phi)) + 1):
            psi[j] += phi[i - 1] * psi[j - i]
    return psi


ORDERS = [
    (0.9, 0),                          # AR(1)
    ([0.5, -0.2], 0),                  # AR(2), default theta
    ([0.5, 0.1, 0.05], 0),             # AR(3)
    ([0.5, -0.2], [0.3]),              # ARMA(2, 1)
    ([0.5], [0.3, 0.2]),               # ARMA(1, 2)
    ([0.95, -0.4, -0.4], np.zeros(3)),
]


def test_impulse_response_matches_ma_representation():
    for phi, theta in ORDERS:
        imp_resp = ARMA(phi, theta).impulse_response(impulse_length=20)
        assert_allclose(imp_resp, _ma_coefficients(phi, theta, 20))


def test_simulation_matches_recursion():
    for phi, theta in ORDERS:
        sigma, seed, n = 0.5, 1234, 50
        sim = ARMA(phi, theta, sigma).simulation(ts_length=n,
                                                 random_state=seed)
        u = np.random.RandomState(seed).standard_normal(n) * sigma
        # With zero initial conditions, X_t = sum_j psi_j u_{t-j}
        psi = _ma_coefficients(phi, theta, n)
        expected = np.array([psi[:t + 1] @ u[t::-1] for t in range(n)])
        assert_allclose(sim, expected)
