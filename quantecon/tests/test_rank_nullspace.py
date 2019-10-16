"""
Tests for rank_nullspace.py

"""
import sys
import unittest
import numpy as np
from numpy.linalg import matrix_rank as np_rank
from quantecon.rank_nullspace import rank_est, nullspace


class TestRankNullspace(unittest.TestCase):

    def setUp(self):
        self.A1 = np.eye(6)
        self.A2 = np.array([[1., 0, 0], [0., 1., 0], [1., 1., 0.]])
        self.A3 = np.zeros((3, 3))

    def tearDown(self):
        del self.A1
        del self.A2
        del self.A3

    def testRankwithNumpy(self):
        A1, A2, A3 = self.A1, self.A2, self.A3
        qe_A1 = rank_est(A1)
        qe_A2 = rank_est(A2)
        qe_A3 = rank_est(A3)

        np_A1 = np_rank(A1)
        np_A2 = np_rank(A2)
        np_A3 = np_rank(A3)


        self.assertTrue(qe_A1 == np_A1 and qe_A2 == np_A2 and qe_A3 == np_A3)


    def testNullspacewithPaper(self):
        A1, A2, A3 = self.A1, self.A2, self.A3
        ns_A1 = nullspace(A1).squeeze()
        ns_A2 = nullspace(A2).squeeze()
        ns_A3 = nullspace(A3).squeeze()

        self.assertTrue(np.allclose(ns_A1, np.array([])) and
            np.allclose(ns_A2, np.array([0, 0, 1])) and
            np.allclose(ns_A3, np.eye(3)))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRankNullspace)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)



