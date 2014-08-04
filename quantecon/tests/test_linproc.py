"""
Filename: test_lss.py
Authors: Chase Coleman
Date: 07/24/2014

Tests for linproc.py file.  Most of this testing can be considered
covered by the numpy tests since we rely on much of their code.

"""
import sys
import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.linproc import LinearProcess


class TestLinearProcess(unittest.TestCase):

    def setUp(self):
        # Initial Values
        phi = np.array([.95, -.4, -.4])
        theta = np.zeros(3)
        sigma = .15


        self.lp = LinearProcess(phi, theta, sigma)


    def tearDown(self):
        del self.lp

    def test_simulate(self):
        lp = self.lp

        sim = lp.simulation(ts_length=250)

        self.assertTrue(sim.size==250)

    def test_impulse_response(self):
        lp = self.lp

        imp_resp = lp.impulse_response(impulse_length=75)

        self.assertTrue(imp_resp.size==75)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinearProcess)
    unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)

