"""
Tests for approximation.py file (i.e. tauchen)

"""
import numpy as np
import pytest
from quantecon.markov import tauchen, rouwenhorst
from numpy.testing import assert_, assert_allclose

#from quantecon.markov.approximation import rouwenhorst


class TestTauchen:

    def setup_method(self):
        self.rho, self.sigma = np.random.rand(2)
        self.n = np.random.randint(3, 25)
        self.n_std = np.random.randint(5)
        self.tol = 1e-12
        self.mu = 0.

        with pytest.warns(UserWarning):
            mc = tauchen(self.n, self.rho, self.sigma, self.mu, self.n_std)
        self.x, self.P = mc.state_values, mc.P

    def teardown_method(self):
        del self.x
        del self.P

    def testStateCenter(self):
        for mu in [0., 1., -1.]:
            mu_expect = mu / (1 - self.rho)
            with pytest.warns(UserWarning):
                mc = tauchen(self.n, self.rho, self.sigma, mu, self.n_std)
            assert_allclose(mu_expect, np.mean(mc.state_values), atol=self.tol)

    def testShape(self):
        i, j = self.P.shape

        assert_(i == j)

    def testDim(self):
        dim_x = self.x.ndim
        dim_P = self.P.ndim

        assert_(dim_x == 1 and dim_P == 2)

    def test_transition_mat_row_sum_1(self):
        assert_allclose(np.sum(self.P, axis=1), 1, atol=self.tol)

    def test_positive_probs(self):
        assert_(np.all(self.P > -self.tol))

    def test_states_sum_0(self):
        assert_(abs(np.sum(self.x)) < self.tol)


class TestRouwenhorst:

    def setup_method(self):
        self.rho, self.sigma = np.random.uniform(0, 1, size=2)
        self.n = np.random.randint(3, 26)
        self.mu = np.random.randint(0, 11)
        self.tol = 1e-10

        with pytest.warns(UserWarning):
            mc = rouwenhorst(self.n, self.rho, self.sigma, self.mu)
        self.x, self.P = mc.state_values, mc.P

    def teardown_method(self):
        del self.x
        del self.P

    def testShape(self):
        i, j = self.P.shape

        assert_(i == j)

    def testDim(self):
        dim_x = self.x.ndim
        dim_P = self.P.ndim
        assert_(dim_x == 1 and dim_P == 2)

    def test_transition_mat_row_sum_1(self):
        assert_allclose(np.sum(self.P, axis=1), 1, atol=self.tol)

    def test_positive_probs(self):
        assert_(np.all(self.P > -self.tol))

    def test_states_sum_0(self):
        tol = self.tol + self.n*(self.mu/(1 - self.rho))
        assert_(abs(np.sum(self.x)) < tol)

    def test_control_case(self):
        n = 3; mu = 1; sigma = 0.5; rho = 0.8;
        with pytest.warns(UserWarning):
            mc_rouwenhorst = rouwenhorst(n, rho, sigma, mu)
        mc_rouwenhorst.x, mc_rouwenhorst.P = mc_rouwenhorst.state_values, mc_rouwenhorst.P
        sigma_y = np.sqrt(sigma**2 / (1-rho**2))
        psi = sigma_y * np.sqrt(n-1)
        known_x = np.array([-psi+5.0, 5., psi+5.0])
        known_P = np.array(
            [[0.81, 0.18, 0.01], [0.09, 0.82, 0.09], [0.01, 0.18, 0.81]])
        assert_(np.sum(mc_rouwenhorst.x - known_x) < self.tol and
                np.sum(mc_rouwenhorst.P - known_P) < self.tol)
