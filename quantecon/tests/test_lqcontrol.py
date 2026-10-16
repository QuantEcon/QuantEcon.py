"""
Tests for lqcontrol.py file

"""
import numpy as np
from numpy.testing import assert_allclose, assert_raises
from quantecon import LQ, LQMarkov


class TestLQControl:

    def setup_method(self):
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

    def teardown_method(self):
        del self.lq_scalar
        del self.lq_mat

    def test_default_beta_one_constructs(self):
        lq = LQ(1., 1., .95, -1.)

        assert lq.beta == 1

    def test_scalar_sequences(self):

        lq_scalar = self.lq_scalar
        x0 = 2

        x_seq, u_seq, w_seq = lq_scalar.compute_sequence(x0)

        # Solution found by hand
        u_0 = (-2*lq_scalar.A*lq_scalar.B*lq_scalar.beta*lq_scalar.Rf) / \
            (2*lq_scalar.Q+lq_scalar.beta*lq_scalar.Rf*2*lq_scalar.B**2) \
            * x0
        # C is (1, 1) and w_seq[0, -1] is 0-d, so this term is * not @:
        # @ rejects 0-d operands, while np.dot (and *) multiply.
        x_1 = lq_scalar.A * x0 + lq_scalar.B * u_0 + \
            lq_scalar.C * w_seq[0, -1]

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
            val_func_lq = x0 @ P @ x0

            assert_allclose(f_answer, F, atol=1e-3)
            assert_allclose(val_func_lq, val_func_answer, atol=1e-3)


def _lq_markov_policy_cost(lq, Fs):
    """
    Return the matrices V(s) such that, when it is finite, the expected
    discounted cost of following u = -F(s) x from (x, s) in `lq` is
    x' V(s) x plus a term that does not depend on x. They solve the stacked
    linear system

        V(i) = R_F(i) + beta A_F(i)' [sum_j Π[i, j] V(j)] A_F(i)

    where A_F(i) = A(i) - B(i) F(i) and
    R_F(i) = R(i) + F(i)' Q(i) F(i) - F(i)' N(i) - N(i)' F(i).

    """
    m, n = lq.m, lq.n
    M = np.identity(m * n**2)
    b = np.empty(m * n**2)
    for i in range(m):
        A_F = lq.As[i] - lq.Bs[i] @ Fs[i]
        rows = slice(i * n**2, (i+1) * n**2)
        for j in range(m):
            # vec(A_F' V A_F) = kron(A_F', A_F') vec(V)
            M[rows, j * n**2:(j+1) * n**2] -= \
                lq.beta * lq.Π[i, j] * np.kron(A_F.T, A_F.T)
        R_F = lq.Rs[i] + Fs[i].T @ lq.Qs[i] @ Fs[i] - \
            Fs[i].T @ lq.Ns[i] - lq.Ns[i].T @ Fs[i]
        b[rows] = R_F.ravel()
    return np.linalg.solve(M, b).reshape(m, n, n)


class TestLQMarkov:

    def setup_method(self):

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

    def teardown_method(self):
        del self.lq_markov_scalar
        del self.lq_markov_mat1

    def test_print(self):
        print(self.lq_markov_scalar)
        print(self.lq_markov_mat1)

    def test_scalar_sequences_with_seed(self):

        lq_markov_scalar = self.lq_markov_scalar
        x0 = 2

        expected_x_seq = np.array([[2., 1.15843266, 0.67098311]])
        expected_u_seq = np.array([[1.28313468, 0.74321256]])
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

        P_answer = np.array([[[1.52007965]],
                             [[1.07741061]]])
        d_answer = np.array([0., 0.])
        F_answer = np.array([[[-0.54745226]],
                             [[-0.64156734]]])

        Ps, ds, Fs = lq_markov_scalar.stationary_values()

        assert_allclose(F_answer, Fs, atol=1e-6)
        assert_allclose(P_answer, Ps, atol=1e-6)
        assert_allclose(d_answer, ds, atol=1e-6)

    def test_mat_sequences(self):

        lq_markov_mat = self.lq_markov_mat1
        x0 = np.array([[1000, 1, 25]])

        expected_x_seq = np.array([[1.00000000e+03, 1.01342472e+03],
                                   [1.00000000e+00, 1.00000000e+00],
                                   [2.50000000e+01, 2.61845443e+01]])
        expected_u_seq = np.array([[1013.42471946]])
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

        d_answer = np.array([16.31235346, 16.38442572])
        P_answer = np.array([[[4.5345640e-02, 1.8660614e+01, 1.9430712e-01],
                              [1.8660614e+01, 7.9402383e+03, 8.0190872e+01],
                              [1.9430712e-01, 8.0190872e+01, 8.3325640e-01]],
                             [[5.3840146e-02, 2.0171229e+01, 2.1853447e-01],
                              [2.0171229e+01, 7.8154682e+03, 8.2093814e+01],
                              [2.1853447e-01, 8.2093814e+01, 8.8762671e-01]]])
        F_answer = np.array([[[-0.98417975, 19.23774624, -0.83061122]],
                             [[-1.01410488, 21.6197524, -0.83758363]]])

        Ps, ds, Fs = lq_markov_mat.stationary_values()

        assert_allclose(F_answer, Fs, atol=1e-6)
        assert_allclose(P_answer, Ps, atol=1e-6)
        assert_allclose(d_answer, ds, atol=1e-6)

    def test_stationary_values_are_policy_costs(self):
        # x' P(s) x must be the x-dependent part of the discounted cost of
        # following u = -F(s) x from (x, s). This fails if the expectation
        # over next period's Markov state is taken outside the inverse in
        # the Riccati recursion (#982).
        Π = np.array([[0.5, 0.5],
                      [0.3, 0.7]])
        As = [[[1., .1], [0., .9]], [[.9, .3], [.2, 1.]]]
        Bs = [[[1.], [0.]], [[.2], [1.]]]
        Rs = [np.eye(2), 2 * np.eye(2)]
        lq_markov_mat2 = LQMarkov(Π, [.1, 5.], Rs, As, Bs, beta=.95)

        for lq in (self.lq_markov_scalar, self.lq_markov_mat1,
                   lq_markov_mat2):
            Ps, ds, Fs = lq.stationary_values()

            assert_allclose(Ps, _lq_markov_policy_cost(lq, Fs), rtol=1e-6)

    def test_identical_regimes_match_lq(self):
        # With one regime, or identical regimes, LQMarkov solves a standard
        # LQ problem. A is unstable, so this also fails if the Riccati
        # iteration lets rounding asymmetry in Ps grow.
        A = np.array([[1.5, .3], [.1, 1.5]])
        B = np.array([[1.], [1.]])
        P, F, d = LQ(1., np.eye(2), A, B, beta=.99).stationary_values()

        for Π in ([[1.]], [[.9, .1], [.2, .8]]):
            m = len(Π)
            lq = LQMarkov(Π, [1.] * m, [np.eye(2)] * m, [A] * m, [B] * m,
                          beta=.99)
            Ps, ds, Fs = lq.stationary_values()

            for i in range(m):
                assert_allclose(Ps[i], P, rtol=1e-8)
                assert_allclose(Fs[i], F, rtol=1e-8)

    def test_beta_one_without_shocks_one_state(self):
        lq = LQMarkov([[1.]], [1.], [1.], [.5], [0.], beta=1)

        Ps, ds, Fs = lq.stationary_values()

        assert_allclose(ds, np.zeros(1))

    def test_beta_one_without_shocks_two_states(self):
        lq = LQMarkov([[.8, .2], [.3, .7]],
                      [1., 1.], [1., 1.], [.5, .4], [0., 0.], beta=1)

        Ps, ds, Fs = lq.stationary_values()

        assert_allclose(ds, np.zeros(2))

    def test_beta_one_with_partial_shocks_raises(self):
        with assert_raises(ValueError) as excinfo:
            LQMarkov([[.8, .2], [.3, .7]],
                     [1., 1.], [1., 1.], [.5, .4], [0., 0.],
                     Cs=[0., .1], beta=1)

        assert str(excinfo.exception) == (
            'beta must be strictly smaller than 1 if C != 0'
        )

    def test_beta_greater_than_one_with_partial_shocks_raises(self):
        assert_raises(ValueError, LQMarkov,
                      [[.8, .2], [.3, .7]],
                      [1., 1.], [1., 1.], [.5, .4], [0., 0.],
                      Cs=[0., .1], beta=1.05)

    def test_beta_greater_than_one_without_shocks_constructs(self):
        lq = LQMarkov([[1.]], [1.], [1.], [.5], [0.], beta=1.05)

        assert lq.beta == 1.05
