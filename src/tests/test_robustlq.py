"""
Author: Chase Coleman
Filename: test_lqcontrol

Tests for lqcontrol.py file

"""
import sys
import os
import unittest
import numpy as np
from scipy.linalg import LinAlgError
from numpy.testing import assert_allclose
from quantecon.lqcontrol import LQ
from quantecon.robustlq import RBLQ


class TestRBLQControl(unittest.TestCase):

    def setUp(self):
        # Initial Values
        a_0     = 100
        a_1     = 0.5
        rho     = 0.9
        sigma_d = 0.05
        beta    = 0.95
        c       = 2
        gamma   = 50.0
        theta = 0.002
        ac    = (a_0 - c) / 2.0

        R = np.array([[0,  ac,    0],
                      [ac, -a_1, 0.5],
                      [0., 0.5,  0]])

        R = -R
        Q = gamma / 2

        A = np.array([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., rho]])
        B = np.array([[0.],
                      [1.],
                      [0.]])
        C = np.array([[0.],
                      [0.],
                      [sigma_d]])


        self.rblq_test = RBLQ(Q, R, A, B, C, beta, theta)
        self.lq_test = LQ(Q, R, A, B, C, beta)

        self.Fr, self.Kr, self.Pr = self.rblq_test.robust_rule()

    def tearDown(self):
        del self.rblq_test

    def test_robust_rule_vs_simple(self):
        rblq = self.rblq_test
        Fr, Kr, Pr = self.Fr, self.Kr, self.Pr

        Fs, Ks, Ps = rblq.robust_rule_simple(P_init=Pr, tol=1e-12)

        assert_allclose(Fr, Fs, rtol=1e-4)
        assert_allclose(Kr, Ks, rtol=1e-4)
        assert_allclose(Pr, Ps, rtol=1e-4)

    def test_f2k_and_k2f(self):
        rblq = self.rblq_test
        Fr, Kr, Pr = self.Fr, self.Kr, self.Pr

        K_f2k, P_f2k = rblq.F_to_K(Fr)
        F_k2f, P_k2f = rblq.K_to_F(Kr)

        assert_allclose(K_f2k, Kr, rtol=1e-4)
        assert_allclose(F_k2f, Fr, rtol=1e-4)
        assert_allclose(P_f2k, P_k2f, rtol=1e-4)

    def test_evaluate_F(self):
        rblq = self.rblq_test
        Fr, Kr, Pr = self.Fr, self.Kr, self.Pr

        Kf, Pf, df, Of, of = rblq.evaluate_F(Fr)

        # In the future if we wanted, we could check more things, but I
        # think the other pieces are basically just plugging these into
        # equations so if these hold then the others should be correct
        # as well.
        assert_allclose(Pf, Pr)
        assert_allclose(Kf, Kr)








if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRBLQControl)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
