"""
Tests for lss.py

"""
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.lss import LinearStateSpace
from nose.tools import raises


class TestLinearStateSpace(unittest.TestCase):

    def setUp(self):
        # Initial Values
        A = .95
        C = .05
        G = 1.
        H = 1.
        V = .01
        mu_0 = .75

        self.ss_vec = []
        self.ss_vec.append(LinearStateSpace(A, C, G, mu_0=mu_0))
        self.ss_vec.append(LinearStateSpace(A, C, G, H=H, V=V, mu_0=mu_0))

    def tearDown(self):
        for ss in self.ss_vec:

            del ss

    def test_stationarity(self):
        for ss in self.ss_vec:

            vals = ss.stationary_distributions(max_iter=1000, tol=1e-9)
            ssmux, ssmuy, sssigx, sssigy = vals

            self.assertTrue(abs(ssmux - ssmuy) < 2e-8)
            if (ss.H is None) & (ss.G == 1.).all():
                self.assertTrue(abs(sssigx - sssigy) < 2e-8)
            self.assertTrue(abs(ssmux) < 2e-8)
            self.assertTrue(abs(sssigx - ss.C**2/(1 - ss.A**2)) < 2e-8)

    def test_simulate(self):
        for ss in self.ss_vec:

            sim = ss.simulate(ts_length=250)
            for arr in sim:
                self.assertTrue(len(arr[0])==250)

    def test_simulate_with_seed(self):
        expected_xval = np.array([[0.75, 0.6959564924, 0.6485540613,
                                   0.6952504141, 0.6309060605],
                                  [0.75, 0.8282543714, 0.7896838925,
                                   0.7215239351, 0.6887068618]])
        expected_yval = np.array([[0.75, 0.6959564924, 0.6485540613,
                                   0.6952504141, 0.6309060605],
                                  [0.4179363134, 0.5761084508, 2.3726112765,
                                   0.1297952303, 0.3594226235]])
        for i, ss in enumerate(self.ss_vec):

            xval, yval = ss.simulate(ts_length=5, random_state=5)
            

            assert_allclose(xval[0], expected_xval[i])
            assert_allclose(yval[0], expected_yval[i])

    def test_replicate(self):
        for ss in self.ss_vec:

            xval, yval = ss.replicate(T=100, num_reps=5000)
            if ss.H is None:
                assert_allclose(xval, yval)
            self.assertEqual(xval.size, 5000)
            self.assertLessEqual(abs(np.mean(xval)), .05)

    def test_replicate_with_seed(self):
        expected_xval = np.array([[0.0251787033, 0.1908734072, -0.1813300321,
                                   0.2305923028, -0.0412238313],
                                  [0.0989884088, 0.12561899, -0.2316570774,
                                   0.1019641968, 0.3374819642]])
        expected_yval = np.array([[0.0251787033, 0.1908734072, -0.1813300321,
                                   0.2305923028, -0.0412238313],
                                  [-1.0522559176, 0.8107821779, -1.3791302245,
                                   0.9976912727, -0.0693582509]])
        for i, ss in enumerate(self.ss_vec):

            xval, yval = ss.replicate(T=100, num_reps=5, random_state=5)
            expected_output = np.array([0.0251787033, 0.1908734072, -0.1813300321,
                                        0.2305923028, -0.0412238313])

            assert_allclose(xval[0], expected_xval[i])
            assert_allclose(yval[0], expected_yval[i])


@raises(ValueError)
def test_non_square_A():
    A = np.zeros((1, 2))
    C = np.zeros((1, 1))
    G = np.zeros((1, 1))

    LinearStateSpace(A, C, G)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinearStateSpace)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)

