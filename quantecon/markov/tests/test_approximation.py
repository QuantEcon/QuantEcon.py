"""
Tests for approximation.py file (i.e. tauchen)

"""
import numpy as np
import pytest
from quantecon.markov import tauchen, rouwenhorst, discrete_var
from numpy.testing import assert_, assert_allclose, assert_raises
import scipy as sp

class TestTauchen:

    def setup_method(self):
        self.rho, self.sigma = np.random.rand(2)
        self.n = np.random.randint(3, 25)
        self.n_std = np.random.randint(5)
        self.tol = 1e-12
        self.mu = 0.

        mc = tauchen(self.n, self.rho, self.sigma, self.mu, self.n_std)
        self.x, self.P = mc.state_values, mc.P

    def teardown_method(self):
        del self.x
        del self.P

    def testStateCenter(self):
        for mu in [0., 1., -1.]:
            mu_expect = mu / (1 - self.rho)
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

    def test_old_tauchen_api_warning(self):
        # Test the warning
        with pytest.warns(UserWarning):
            # This will raise an error because `n` must be an int
            assert_raises(TypeError, tauchen, 4.0, self.rho, self.sigma,
                                self.mu, self.n_std)

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
        self.random_state = np.random.RandomState(0)

        Omega = np.array([[0.0012346, -0.0000776],
                          [-0.0000776, 0.0000401]])
        self.A = [[0.7901, -1.3570], 
                  [-0.0104, 0.8638]]
        self.C = sp.linalg.sqrtm(Omega)
        self.T= 1_000_000
        self.sizes = np.array((2, 3))

        # Expected outputs
        self.S_out = [
                     [-0.38556417, -0.05387746],
                     [-0.38556417,  0.        ],
                     [-0.38556417,  0.05387746],
                     [ 0.38556417, -0.05387746],
                     [ 0.38556417,  0.        ],
                     [ 0.38556417,  0.05387746]]

        self.S_out_orderF = [
                     [-0.38556417, -0.05387746],
                     [ 0.38556417, -0.05387746],
                     [-0.38556417,  0.        ],
                     [ 0.38556417,  0.        ],
                     [-0.38556417,  0.05387746],
                     [ 0.38556417,  0.05387746]]
            
        self.P_out = [
            [1.61290323e-02, 1.12903226e-01, 0.00000000e+00, 3.70967742e-01, 
            5.00000000e-01, 0.00000000e+00],
            [1.00964548e-04, 8.51048124e-01, 3.82857566e-02, 4.03858192e-04,
            1.10111936e-01, 4.93604457e-05],
            [0.00000000e+00, 3.02295449e-01, 6.97266822e-01, 0.00000000e+00,
            3.85201268e-04, 5.25274456e-05],
            [3.60600761e-05, 4.86811027e-04, 0.00000000e+00, 6.97473992e-01,
            3.02003137e-01, 0.00000000e+00],
            [3.17039037e-05, 1.11090478e-01, 4.55177474e-04, 3.75374219e-02,
            8.50778784e-01, 1.06434534e-04],
            [0.00000000e+00, 4.45945946e-01, 3.37837838e-01, 0.00000000e+00,
            1.89189189e-01, 2.70270270e-02]]

        self.P_out_orderF =[
            [1.61290323e-02, 3.70967742e-01, 1.12903226e-01, 5.00000000e-01, 
            0.00000000e+00, 0.00000000e+00],
            [3.60600761e-05, 6.97473992e-01, 4.86811027e-04, 3.02003137e-01, 
            0.00000000e+00, 0.00000000e+00],
            [1.00964548e-04, 4.03858192e-04, 8.51048124e-01, 1.10111936e-01, 
            3.82857566e-02, 4.93604457e-05],
            [3.17039037e-05, 3.75374219e-02, 1.11090478e-01, 8.50778784e-01, 
            4.55177474e-04, 1.06434534e-04],
            [0.00000000e+00, 0.00000000e+00, 3.02295449e-01, 3.85201268e-04, 
            6.97266822e-01, 5.25274456e-05],
            [0.00000000e+00, 0.00000000e+00, 4.45945946e-01, 1.89189189e-01, 
            3.37837838e-01, 2.70270270e-02]]
        
        self.P_out_non_square = [
            [3.70370370e-02, 2.22222222e-01, 0.00000000e+00, 4.25925926e-01, 
            3.14814815e-01, 0.00000000e+00],
            [6.92865939e-05, 8.50870664e-01, 3.83177215e-02, 4.17954615e-04, 
            1.10275202e-01, 4.91711312e-05],
            [0.00000000e+00, 3.08160000e-01, 6.91360000e-01, 0.00000000e+00, 
            3.91111111e-04, 8.88888889e-05],
            [1.08405001e-04, 5.05890005e-04, 0.00000000e+00, 6.95707162e-01, 
            3.03678543e-01, 0.00000000e+00],
            [3.40242525e-05, 1.11864937e-01, 4.44583566e-04, 3.77260912e-02, 
            8.49844169e-01, 8.61947730e-05],
            [0.00000000e+00, 4.70588235e-01, 3.08823529e-01, 0.00000000e+00, 
            1.76470588e-01, 4.41176471e-02]]

        self.A, self.C, self.S_out, self.P_out, self.S_out_orderF,\
        self.P_out_orderF, self.P_out_non_square \
            = map(np.array,(self.A, self.C, self.S_out, self.P_out, 
                                self.S_out_orderF, self.P_out_orderF, 
                                self.P_out_non_square))

    def teardown_method(self):
        del self.A, self.C, self.S_out, self.P_out

    def test_discretization(self):
        mc = discrete_var(
                self.A, self.C, grid_sizes=self.sizes,
                sim_length=self.T, random_state=self.random_state)
        assert_allclose(mc.state_values, self.S_out)
        assert_allclose(mc.P, self.P_out)

    def test_sp_distributions(self):
        mc = discrete_var(
                self.A, self.C, 
                grid_sizes=self.sizes,
                sim_length=self.T, 
                rv=sp.stats.multivariate_normal(cov=np.identity(2)),
                random_state=self.random_state)
        assert_allclose(mc.state_values, self.S_out)
        assert_allclose(mc.P, self.P_out)

    def test_order_F(self):
        mc = discrete_var(
                self.A, self.C, 
                grid_sizes=self.sizes,
                sim_length=self.T, 
                order='F',
                random_state=self.random_state)
        assert_allclose(mc.state_values, self.S_out_orderF)
        assert_allclose(mc.P, self.P_out_orderF)

    def test_order_non_squareC(self):
        new_col = np.array([0, 0])
        self.C = np.insert(self.C, 2, new_col, axis=1)
        mc = discrete_var(
            self.A, self.C, 
            grid_sizes=self.sizes,
            sim_length=self.T, 
            order='C',
            random_state=self.random_state)
        assert_allclose(mc.state_values, self.S_out)
        assert_allclose(mc.P, self.P_out_non_square)
