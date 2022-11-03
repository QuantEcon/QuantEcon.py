"""
Tests for arma.py file.  Most of this testing can be considered
covered by the numpy tests since we rely on much of their code.

"""
import numpy as np
from numpy.testing import assert_array_equal, assert_
from quantecon.arma import ARMA


class TestARMA():
    def setup_method(self):
        # Initial Values
        phi = np.array([.95, -.4, -.4])
        theta = np.zeros(3)
        sigma = .15


        self.lp = ARMA(phi, theta, sigma)

    def teardown_method(self):
        del self.lp

    def test_simulate(self):
        lp = self.lp

        sim = lp.simulation(ts_length=250)

        assert_(sim.size == 250)

    def test_simulate_with_seed(self):
        lp = self.lp
        seed = 5
        sim0 = lp.simulation(ts_length=10, random_state=seed)
        sim1 = lp.simulation(ts_length=10, random_state=seed)

        assert_array_equal(sim0, sim1)

    def test_impulse_response(self):
        lp = self.lp

        imp_resp = lp.impulse_response(impulse_length=75)

        assert_(imp_resp.size == 75)
