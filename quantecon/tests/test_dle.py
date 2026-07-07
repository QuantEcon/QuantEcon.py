"""
Tests for dle.py file
"""

import numpy as np
from numpy.testing import assert_allclose
from quantecon import DLE

ATOL = 1e-10


class TestDLE:

    def setup_method(self):
        """
        Given LQ control is tested we will test the transformation
        to alter the problem into a form suitable to solve using LQ
        """
        # Initial Values
        gam = 0
        gamma = np.array([[gam], [0]])
        phic = np.array([[1], [0]])
        phig = np.array([[0], [1]])
        phi1 = 1e-4
        phii = np.array([[0], [-phi1]])
        deltak = np.array([[.95]])
        thetak = np.array([[1]])
        beta = np.array([[1 / 1.05]])
        ud = np.array([[5, 1, 0], [0, 0, 0]])
        a22 = np.array([[1, 0, 0], [0, 0.8, 0], [0, 0, 0.5]])
        c2 = np.array([[0, 1, 0], [0, 0, 1]]).T
        llambda = np.array([[0]])
        pih = np.array([[1]])
        deltah = np.array([[.9]])
        thetah = np.array([[1]]) - deltah
        ub = np.array([[30, 0, 0]])

        information = (a22, c2, ub, ud)
        technology = (phic, phig, phii, gamma, deltak, thetak)
        preferences = (beta, llambda, pih, deltah, thetah)

        self.dle = DLE(information, technology, preferences)

    def teardown_method(self):
        del self.dle

    def test_transformation_Q(self):
        Q_solution = np.array([[5.e-09]])
        assert_allclose(Q_solution, self.dle.Q)

    def test_transformation_R(self):
        R_solution = np.array([[0.,   0.,   0.,   0.,   0.],
                               [0.,   0.,   0.,   0.,   0.],
                               [0.,   0., 312.5, -12.5,   0.],
                               [0.,   0., -12.5,   0.5,   0.],
                               [0.,   0.,   0.,   0.,   0.]])
        assert_allclose(R_solution, self.dle.R)

    def test_transformation_A(self):
        A_solution = np.array([[0.9, 0., 0.5, 0.1, 0.],
                               [0., 0.95, 0., 0., 0.],
                               [0., 0., 1., 0., 0.],
                               [0., 0., 0., 0.8, 0.],
                               [0., 0., 0., 0., 0.5]])
        assert_allclose(A_solution, self.dle.A)

    def test_transformation_B(self):
        B_solution = np.array([[-0.],
                               [1.],
                               [0.],
                               [0.],
                               [0.]])
        assert_allclose(B_solution, self.dle.B)

    def test_transformation_C(self):
        C_solution = np.array([[0., 0.],
                               [0., 0.],
                               [0., 0.],
                               [1., 0.],
                               [0., 1.]])
        assert_allclose(C_solution, self.dle.C)

    def test_transformation_W(self):
        W_solution = np.array([[0., 0., 0., 0., 0.]])
        assert_allclose(W_solution, self.dle.W)

    def test_compute_steadystate(self):
        solutions = {
            'css' : np.array([[5.]]),
            'sss' : np.array([[5.]]),
            'iss' : np.array([[0.]]),
            'dss' : np.array([[5.], [0.]]),
            'bss' : np.array([[30.]]),
            'kss' : np.array([[0.]]),
            'hss' : np.array([[5.]]),
        }
        self.dle.compute_steadystate()
        for item in solutions.keys():
            assert_allclose(self.dle.__dict__[
                            item], solutions[item], atol=ATOL)

    def test_canonical(self):
        solutions = {
            'pihat': np.array([[1.]]),
            'llambdahat': np.array([[-1.48690584e-19]]),
            'ubhat': np.array([[30., -0., -0.]])
        }
        self.dle.canonical()
        for item in solutions.keys():
            assert_allclose(self.dle.__dict__[
                            item], solutions[item], atol=ATOL)

    def test_compute_sequence(self):
        # Regression test for GH #839: the asset-price terms evaluate to
        # size-1 arrays, and assigning them into the scalar price slots used to
        # rely on NumPy implicitly converting an ``ndim > 0`` array to a scalar.
        # NumPy >= 2.4 raises instead, so ``compute_sequence`` must coerce the
        # values with ``.item()``.
        x0 = np.array([[5], [150], [1], [0], [0]])
        ts_length = 10
        self.dle.compute_sequence(x0, ts_length=ts_length)

        for name in ('R1_Price', 'R2_Price', 'R5_Price'):
            price = self.dle.__dict__[name]
            assert price.shape == (ts_length + 1, 1)
            assert np.all(np.isfinite(price))

        beta = 1 / 1.05
        # At t=0 the J-period risk-free prices reduce to the closed form beta**J
        assert_allclose(self.dle.R1_Price[0, 0], beta, atol=ATOL)
        assert_allclose(self.dle.R2_Price[0, 0], beta ** 2, atol=ATOL)
        assert_allclose(self.dle.R5_Price[0, 0], beta ** 5, atol=ATOL)

    def test_compute_sequence_with_pay(self):
        # The ``Pay`` branch of compute_sequence has the same size-1 assignment
        # pattern as the risk-free price terms (GH #839); make sure it runs and
        # populates the Pay_Price / Pay_Gross paths with finite values.
        x0 = np.array([[5], [150], [1], [0], [0]])
        ts_length = 10
        Pay = np.array([[1., 0., 0., 0., 0.]])
        self.dle.compute_sequence(x0, ts_length=ts_length, Pay=Pay)

        assert self.dle.Pay_Price.shape == (ts_length + 1, 1)
        assert self.dle.Pay_Gross.shape == (ts_length + 1, 1)
        assert np.all(np.isfinite(self.dle.Pay_Price))
        # Pay_Gross[0, 0] is intentionally set to nan; the rest must be finite
        assert np.all(np.isfinite(self.dle.Pay_Gross[1:]))
