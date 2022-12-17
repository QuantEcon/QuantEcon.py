"""
Tests for approximation.py file (i.e. tauchen)

"""
import numpy as np
import pytest
from quantecon.markov import tauchen, rouwenhorst, discrete_var
from numpy.testing import assert_, assert_allclose


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


class TestDiscreteVar:

    def setup_method(self):
        self.T, self.burn_in  = 1_000_000, 100_000

        self.A = [[0.7901, -1.3570],
             [-0.0104, 0.8638]]
        self.Omega = [[0.0012346, -0.0000776],
                 [-0.0000776, 0.0000401]]

        self.sizes = np.array((2, 3))

        # Expected outputs
        self.S_out = [[-0.38556417, -0.05387746],
                     [-0.38556417,  0.        ],
                     [-0.38556417,  0.05387746],
                     [ 0.38556417, -0.05387746],
                     [ 0.38556417,  0.        ],
                     [ 0.38556417,  0.05387746]]

        self.P_out = [[8.06451613e-02, 1.93548387e-01, 0.00000000e+00, 3.54838710e-01,
              3.70967742e-01, 0.00000000e+00],
             [8.39514352e-05, 8.50806955e-01, 3.79097454e-02, 4.96901738e-04,
              1.10647992e-01, 5.44549850e-05],
             [0.00000000e+00, 3.01394785e-01, 6.97927443e-01, 0.00000000e+00,
              5.70755895e-04, 1.07016730e-04],
             [1.03646634e-04, 5.52782048e-04, 0.00000000e+00, 6.99407487e-01,
              2.99936085e-01, 0.00000000e+00],
             [3.14483775e-05, 1.09581871e-01, 4.53755161e-04, 3.85467256e-02,
              8.51289608e-01, 9.65914451e-05],
             [0.00000000e+00, 3.63636364e-01, 3.37662338e-01, 0.00000000e+00,
              2.46753247e-01, 5.19480519e-02]]

        self.A, self.Omega, self.S_out, self.P_out = map(np.array,
                                (self.A, self.Omega, self.S_out, self.P_out))

    def teardown_method(self):
        del self.A, self.Omega, self.S_out, self.P_out

    def test_discretization(self):
        mc = discrete_var(
                self.A, self.Omega, grid_sizes=self.sizes,
                sim_length=self.T, burn_in=self.burn_in)
        assert_allclose(mc.state_values, self.S_out)
        assert_allclose(mc.P, self.P_out)
