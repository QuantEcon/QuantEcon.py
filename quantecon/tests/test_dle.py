"""
Tests for dle.py file
"""

import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.dle import DLE

ATOL = 1e-10

class TestDLE(unittest.TestCase):

    def setUp(self):
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

    def tearDown(self):
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


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDLE)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
