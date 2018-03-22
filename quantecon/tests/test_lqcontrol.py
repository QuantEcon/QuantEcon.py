"""
Tests for lqcontrol.py file

"""
import sys
import os
import unittest
import numpy as np
from scipy.linalg import LinAlgError
from numpy.testing import assert_allclose
from numpy import dot
from quantecon.lqcontrol import LQ


class TestLQControl(unittest.TestCase):

    def setUp(self):
        # Initial Values
        q = 1.
        r = 1.
        rf = 1.
        a = .95
        b = -1.
        c = .05
        beta = .95
        T = 1

        self.lq_scalar = LQ(q, r, a, b, C=c, beta=beta, T=T, Rf=rf)

        Q = np.array([[0., 0.], [0., 1]])
        R = np.array([[1., 0.], [0., 0]])
        RF = np.eye(2) * 100
        A = np.ones((2, 2)) * .95
        B = np.ones((2, 2)) * -1

        self.lq_mat = LQ(Q, R, A, B, beta=beta, T=T, Rf=RF)

        self.methods = ['doubling', 'qz']

    def tearDown(self):
        del self.lq_scalar
        del self.lq_mat

    def test_scalar_sequences(self):

        lq_scalar = self.lq_scalar
        x0 = 2

        x_seq, u_seq, w_seq = lq_scalar.compute_sequence(x0)

        # Solution found by hand
        u_0 = (-2*lq_scalar.A*lq_scalar.B*lq_scalar.beta*lq_scalar.Rf) / \
            (2*lq_scalar.Q+lq_scalar.beta*lq_scalar.Rf*2*lq_scalar.B**2) \
            * x0
        x_1 = lq_scalar.A * x0 + lq_scalar.B * u_0 + \
            dot(lq_scalar.C, w_seq[0, -1])

        assert_allclose(u_0, u_seq, rtol=1e-4)
        assert_allclose(x_1, x_seq[0, -1], rtol=1e-4)

    def test_scalar_sequences_with_seed(self):
        lq_scalar = self.lq_scalar
        x0 = 2
        x_seq, u_seq, w_seq = \
            lq_scalar.compute_sequence(x0, 10, random_state=5)

        expected_output = np.array([[ 0.44122749, -0.33087015]])

        assert_allclose(w_seq, expected_output)

    def test_mat_sequences(self):

        lq_mat = self.lq_mat
        x0 = np.random.randn(2) * 25

        x_seq, u_seq, w_seq = lq_mat.compute_sequence(x0)

        assert_allclose(np.sum(u_seq), .95 * np.sum(x0), atol=1e-3)
        assert_allclose(x_seq[:, -1], np.zeros_like(x0), atol=1e-3)

    def test_stationary_mat(self):
        x0 = np.random.randn(2) * 25
        lq_mat = self.lq_mat

        f_answer = np.array([[-.95, -.95], [0., 0.]])
        p_answer = np.array([[1., 0], [0., 0.]])
        val_func_answer = x0[0]**2

        for method in self.methods:
            P, F, d = lq_mat.stationary_values(method=method)
            val_func_lq = np.dot(x0, P).dot(x0)

            assert_allclose(f_answer, F, atol=1e-3)
            assert_allclose(val_func_lq, val_func_answer, atol=1e-3)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLQControl)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
