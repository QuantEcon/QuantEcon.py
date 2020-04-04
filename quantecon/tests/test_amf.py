"""
Tests for amf.py

"""

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from quantecon import (AMF_LSS_VAR, pth_order_to_stacked_1st_order,
                       compute_BQ_restricted_B_0)
from scipy.stats import multivariate_normal as mvn
from nose.tools import assert_raises


class TestAMFLSSVAR:
    def setUp(self):
        ϕ_1, ϕ_2, ϕ_3, ϕ_4 = 0.5, -0.2, 0, 0.5
        σ = 0.01
        self.ν = np.array([[0.01]])  # Growth rate

        # A matrix should be n x n
        self.A = np.array([[ϕ_1, ϕ_2, ϕ_3, ϕ_4],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])

        # B matrix should be n x k
        self.B = np.array([[σ, 0, 0, 0]]).T

        self.D = np.array([[1, 0, 0, 0]]) @ self.A
        self.F = np.array([[1, 0, 0, 0]]) @ self.B

        self.amf = AMF_LSS_VAR(self.A, self.B, self.D, self.F, self.ν)

    def test__construct_x0(self):
        ny0r = np.ones(2)
        nx0r = 2. * np.ones(3)
        x0 = self.amf._construct_x0(nx0r, ny0r)

        x0_sol = np.array([1., 0., 2., 2., 2., 1., 1., 1., 1.])

        assert_array_equal(x0, x0_sol)

    def test__construct_A_bar(self):
        x0 = np.ones(2 + self.amf.nx + 2 * self.amf.ny)
        nx0c = 3 * np.ones((self.amf.nx, 1))
        nyx0m = 4 * np.ones_like(self.D)
        ny0c = 5 * np.ones((self.amf.ny, 1))
        ny1m = 6 * np.eye(self.amf.ny)
        ny0m = 7 * np.ones((self.amf.ny, self.amf.ny))

        A_bar = self.amf._construct_A_bar(x0, nx0c, nyx0m, ny0c, ny1m, ny0m)

        A1_2_sol = np.array([[1., 1., 1., 1., 1., 1., 1., 1.],
                             [1., 1., 1., 1., 1., 1., 1., 1.]])

        A3_sol = np.hstack([[[3.], [3.], [3.], [3.]],
                            [[3.], [3.], [3.], [3.]],
                            self.A,
                            [[4.], [4.], [4.], [4.]],
                            [[4.], [4.], [4.], [4.]]])

        A4_sol = np.hstack([self.ν, [[5.]], self.D, [[6.]], [[7.]]])

        A5_sol = np.array([5., 5., 4., 4., 4., 4., 7., 6.])

        A_bar_sol = np.vstack([A1_2_sol, A3_sol, A4_sol, A5_sol])

        assert_array_equal(A_bar, A_bar_sol)

    def test__construct_B_bar(self):
        nk0 = np.ones(self.amf.nk)
        H = 2 * np.ones((self.amf.nk, self.amf.nk))
        B_bar = self.amf._construct_B_bar(nk0, H)

        B_bar_sol = np.vstack([nk0, nk0, self.B, self.F, H])

        assert_array_equal(B_bar, B_bar_sol)

    def test__construct_G_bar(self):
        nx0c = np.ones((self.amf.nx, 1))
        nyx0m = 2 * np.ones_like(self.D)
        ny0c = 3 * np.ones((self.amf.ny, 1))
        ny1m = 4 * np.eye(self.amf.ny)
        ny0m = 5 * np.ones((self.amf.ny, self.amf.ny))
        g = self.amf.additive_decomp[2]

        G_bar = self.amf._construct_G_bar(nx0c, self.amf.nx, nyx0m, ny0c,
                                          ny1m, ny0m, g)

        G_1_2_3_sol = np.array([[1., 1., 1., 0., 0., 0., 2., 2.],
                                [1., 1., 0., 1., 0., 0., 2., 2.],
                                [1., 1., 0., 0., 1., 0., 2., 2.],
                                [1., 1., 0., 0., 0., 1., 2., 2.],
                                [3., 3., 2., 2., 2., 2., 4., 5.],
                                [3., 3., 2., 2., 2., 2., 5., 4.]])

        G_4_sol = np.hstack([[[3.]], [[3.]], -g, [[5.]], [[5.]]])

        G_5_sol = np.hstack([[[3.]], self.ν, [[2., 2., 2., 2.]], [[5.]],
                             [[5.]]])

        G_bar_sol = np.vstack([G_1_2_3_sol, G_4_sol, G_5_sol])

        assert_array_equal(G_bar, G_bar_sol)

    def test__construct_H_bar(self):
        nx, ny, nk = 2, 3, 5
        H_bar = self.amf._construct_H_bar(nx, ny, nk)

        H_bar_sol = np.array([[0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.]])

        assert_array_equal(H_bar, H_bar_sol)

    def test__construct_Sigma_0(self):
        x0 = np.array([1., 2., 3.])
        Sigma_0 = self.amf._construct_Sigma_0(x0)
        Sigma_0_sol = np.array([[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.]])

        assert_array_equal(Sigma_0, Sigma_0_sol)

    def test_invalid_dimensions(self):
        inputs = (self.A, self.B, self.D, self.F, self.ν)
        inputs_passed = list(inputs)
        invalid_inputs = [[], np.array([]), ()]

        for invalid_input in invalid_inputs:
            for i in range(len(inputs)):
                inputs_passed[i] = invalid_input  # Set input i to be invalid
                with assert_raises(ValueError):
                    AMF_LSS_VAR(*inputs_passed)

                inputs_passed[i] = inputs[i]  # Restore original input

    def test_invalid_shape(self):
        inputs = (self.A, self.B, self.D, self.F, self.ν)
        inputs_passed = list(inputs)
        invalid_input = np.eye(10)

        for i in range(len(inputs)):
            inputs_passed[i] = invalid_input  # Set input i to be invalid
            with assert_raises(ValueError):
                AMF_LSS_VAR(*inputs_passed)

            inputs_passed[i] = inputs[i]  # Restore original input

    def test_non_square_A(self):
        A = np.zeros((1, 3))
        B = np.zeros((1, 4))
        D = np.zeros((2, 3))
        F = np.zeros((2, 4))
        ν = np.zeros((2, 1))

        with assert_raises(ValueError):
            AMF_LSS_VAR(A, B, D, F, ν)

    def test_loglikelihood(self):
        x = np.random.rand(4, 10) * 0.005
        y = np.random.rand(1, 10) * 0.005

        temp = y[:, 1:] - y[:, :-1] - self.D @ x[:, :-1]

        llh = self.amf.loglikelihood_path(x, y)

        cov = self.F @ self.F.T

        llh_sol_scipy = np.cumsum(np.log([mvn.pdf(obs, mean=0, cov=cov)
                                          for obs in temp]))

        assert_allclose(llh, llh_sol_scipy)


def test_pth_order_to_stacked_1st_order():
    # First test
    n = 2
    p = 5

    ζ_hat = np.array([1., 2.])

    A_hats = ([(i + 1) * np.eye(n) for i in range(p)])

    ζ, A = pth_order_to_stacked_1st_order(ζ_hat, A_hats)

    A_sol = np.array([[1., 0., 2., 0., 3., 0., 4., 0., 5., 0.],
                      [0., 1., 0., 2., 0., 3., 0., 4., 0., 5.],
                      [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])

    ζ_sol = np.array([1., 2., 0., 0., 0., 0., 0., 0., 0., 0.])

    assert_array_equal(ζ, ζ_sol)
    assert_array_equal(A, A_sol)

    # Second test
    A_hats = (np.array([[.1, -.07, .03], [.4, .01, -.05], [.01, -.1, .6]]),
              np.array([[-.8, .2, .02], [-.3, -.07, 0.05], [-.02, .1, -.09]]))

    ζ_hat = np.array([0.01, 0.02, 0.03])

    ζ, A = pth_order_to_stacked_1st_order(ζ_hat, A_hats)

    ζ_sol = np.array([0.01, 0.02, 0.03, 0., 0., 0.])

    A_sol = np.array([[.1, -.07, .03, -.8, .2, .02],
                      [.4, .01, -.05, -.3, -.07, 0.05],
                      [.01, -.1, .6, -.02, .1, -.09],
                      [1., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0.]])

    assert_array_equal(ζ, ζ_sol)
    assert_array_equal(A, A_sol)


def test_compute_BQ_restricted_B_0():
    A = np.array([[0.9, -0.2],
                  [0.3, 0.6]])

    Ω_hat = np.array([[0.001, -0.0005],
                      [-0.0005, 0.001]])

    A_hats = (A, )

    B_0 = compute_BQ_restricted_B_0(A_hats, Ω_hat)

    B_0_sol = np.array([[-0.02192645, 0.02278664],
                        [0.03069703, 0.00759555]])

    assert_allclose(B_0, B_0_sol, rtol=1e-6)
