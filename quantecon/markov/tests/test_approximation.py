"""
Tests for approximation.py file (i.e. tauchen)

"""
import numpy as np
from quantecon.markov import tauchen, rouwenhorst
from numpy.testing import assert_, assert_allclose

#from quantecon.markov.approximation import rouwenhorst


class TestTauchen:

    def setup(self):
        self.rho, self.sigma_u = np.random.rand(2)
        self.n = np.random.randint(3, 25)
        self.m = np.random.randint(5)
        self.tol = 1e-12
        self.b = 0.

        mc = tauchen(self.rho, self.sigma_u, self.b, self.m, self.n)
        self.x, self.P = mc.state_values, mc.P

    def tearDown(self):
        del self.x
        del self.P

    def testStateCenter(self):
        for b in [0., 1., -1.]:
            mu = b / (1 - self.rho)
            mc = tauchen(self.rho, self.sigma_u, b, self.m, self.n)
            assert_allclose(mu, np.mean(mc.state_values), atol=self.tol)

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

    def setup(self):
        self.rho, self.sigma = np.random.uniform(0, 1, size=2)
        self.n = np.random.randint(3, 26)
        self.ybar = np.random.randint(0, 11)
        self.tol = 1e-10

        mc = rouwenhorst(self.n, self.ybar, self.sigma, self.rho)
        self.x, self.P = mc.state_values, mc.P

    def tearDown(self):
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
        tol = self.tol + self.n*(self.ybar/(1 - self.rho))
        assert_(abs(np.sum(self.x)) < tol)

    def test_control_case(self):
        n = 3; ybar = 1; sigma = 0.5; rho = 0.8;
        mc_rouwenhorst = rouwenhorst(n, ybar, sigma, rho)
        mc_rouwenhorst.x, mc_rouwenhorst.P = mc_rouwenhorst.state_values, mc_rouwenhorst.P
        sigma_y = np.sqrt(sigma**2 / (1-rho**2))
        psi = sigma_y * np.sqrt(n-1)
        known_x = np.array([-psi+5.0, 5., psi+5.0])
        known_P = np.array(
            [[0.81, 0.18, 0.01], [0.09, 0.82, 0.09], [0.01, 0.18, 0.81]])
        assert_(np.sum(mc_rouwenhorst.x - known_x) < self.tol and
                np.sum(mc_rouwenhorst.P - known_P) < self.tol)
