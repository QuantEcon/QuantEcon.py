"""
Tests for lqcontrol.py file

"""
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose
from numpy import dot
from quantecon.lqcontrol import LQ, LQMarkov


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
        val_func_answer = x0[0]**2

        for method in self.methods:
            P, F, d = lq_mat.stationary_values(method=method)
            val_func_lq = np.dot(x0, P).dot(x0)

            assert_allclose(f_answer, F, atol=1e-3)
            assert_allclose(val_func_lq, val_func_answer, atol=1e-3)


class TestLQMarkov(unittest.TestCase):

    def setUp(self):

        # Markov chain transition matrix
        Π = np.array([[0.8, 0.2],
                      [0.2, 0.8]])

        # discount rate
        beta = .95

        # scalar case
        q1, q2 = 1., .5
        r1, r2 = 1., .5
        a1, a2 = .95, .9
        b1, b2 = -1., -.5

        self.lq_markov_scalar = LQMarkov(Π, [q1, q2], [r1, r2], [a1, a2],
                                         [b1, b2], beta=beta)
        # matrix case
        Π = np.array([[0.8, 0.2],
                      [0.2, 0.8]])

        Qs = np.array([[[0.9409]], [[0.870489]]])
        Rs = np.array([[[1., 0., 1.],
                        [0., 0., 0.],
                        [1., 0., 1.]],
                       [[1., 0., 1.],
                        [0., 0., 0.],
                        [1., 0., 1.]]])
        Ns = np.array([[[-0.97, 0., -0.97]],
                       [[-0.933, 0., -0.933]]])
        As = np.array([[[0., 0., 0.],
                        [0., 1., 0.],
                        [0., 5., 0.8]],
                       [[0., 0., 0.],
                        [0., 1., 0.],
                        [0., 5., 0.8]]])
        B = np.array([[1., 0., 0.]]).T
        Bs = [B, B]
        C = np.array([[0., 0., 1.]]).T
        Cs = [C, C]

        self.lq_markov_mat1 = LQMarkov(Π, Qs, Rs, As, Bs,
                                       Cs=Cs, Ns=Ns, beta=0.95)
        self.lq_markov_mat2 = LQMarkov(Π, Qs, Rs, As, Bs,
                                       Cs=Cs, Ns=Ns, beta=1.05)

    def tearDown(self):
        del self.lq_markov_scalar
        del self.lq_markov_mat1
        del self.lq_markov_mat2

    def test_print(self):
        print(self.lq_markov_scalar)
        print(self.lq_markov_mat1)

    def test_scalar_sequences_with_seed(self):

        lq_markov_scalar = self.lq_markov_scalar
        x0 = 2

        expected_x_seq = np.array([[2., 1.15977567, 0.6725398]])
        expected_u_seq = np.array([[1.28044866, 0.7425166]])
        expected_w_seq = np.array([[1.3486939, 0.55721062, 0.53423587]])
        expected_state = np.array([1, 1, 1])

        x_seq, u_seq, w_seq, state = \
            lq_markov_scalar.compute_sequence(x0, ts_length=2,
                                              random_state=1234)

        assert_allclose(x_seq, expected_x_seq, atol=1e-6)
        assert_allclose(u_seq, expected_u_seq, atol=1e-6)
        assert_allclose(w_seq, expected_w_seq, atol=1e-6)
        assert_allclose(state, expected_state, atol=1e-6)

    def test_stationary_scalar(self):

        lq_markov_scalar = self.lq_markov_scalar

        P_answer = np.array([[[1.51741465]],
                             [[1.07334181]]])
        d_answer = np.array([0., 0.])
        F_answer = np.array([[[-0.54697435]],
                             [[-0.64022433]]])

        Ps, ds, Fs = lq_markov_scalar.stationary_values()

        assert_allclose(F_answer, Fs, atol=1e-6)
        assert_allclose(P_answer, Ps, atol=1e-6)
        assert_allclose(d_answer, ds, atol=1e-6)

    def test_mat_sequences(self):

        lq_markov_mat = self.lq_markov_mat1
        x0 = np.array([[1000, 1, 25]])

        expected_x_seq = np.array([[1.00000000e+03, 1.01372101e+03],
                                   [1.00000000e+00, 1.00000000e+00],
                                   [2.50000000e+01, 2.61845443e+01]])
        expected_u_seq = np.array([[1013.72101253]])
        expected_w_seq = np.array([[0.41782708, 1.18454431]])
        expected_state = np.array([1, 1])

        x_seq, u_seq, w_seq, state = \
            lq_markov_mat.compute_sequence(x0, ts_length=1, random_state=1234)

        assert_allclose(x_seq, expected_x_seq, atol=1e-6)
        assert_allclose(u_seq, expected_u_seq, atol=1e-6)
        assert_allclose(w_seq, expected_w_seq, atol=1e-6)
        assert_allclose(state, expected_state, atol=1e-6)

    def test_stationary_mat(self):
        lq_markov_mat = self.lq_markov_mat1

        d_answer = np.array([16.2474886, 16.31935939])
        P_answer = np.array([[[4.5144056e-02, 1.8627227e+01, 1.9348906e-01],
                              [1.8627227e+01, 7.9343733e+03, 8.0055130e+01],
                              [1.9348906e-01, 8.0055130e+01, 8.2991316e-01]],
                             [[5.3606323e-02, 2.0136657e+01, 2.1763323e-01],
                              [2.0136657e+01, 7.8100167e+03, 8.1960509e+01],
                              [2.1763323e-01, 8.1960509e+01, 8.8413147e-01]]])
        F_answer = np.array([[[-0.98437714, 19.2051657, -0.83142157]],
                             [[-1.01434303, 21.58480004, -0.83851124]]])

        Ps, ds, Fs = lq_markov_mat.stationary_values()

        assert_allclose(F_answer, Fs, atol=1e-6)
        assert_allclose(P_answer, Ps, atol=1e-6)
        assert_allclose(d_answer, ds, atol=1e-6)

    def test_raise_error(self):
        # test raising error for not converging
        lq_markov_mat = self.lq_markov_mat2

        self.assertRaises(ValueError, lq_markov_mat.stationary_values)

if __name__ == '__main__':
    for Test in [TestLQControl, TestLQMarkov]:
        suite = unittest.TestLoader().loadTestsFromTestCase(Test)
        unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
