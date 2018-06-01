"""
Tests for discrete_rv.py

"""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from nose.plugins.attrib import attr
from quantecon import DiscreteRV


class TestDiscreteRV(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x = np.random.rand(10)
        x /= x.sum()
        # make sure it sums to 1
        cls.x = x
        cls.drv = DiscreteRV(cls.x)

    def test_Q_updates(self):
        "discrete_rv: Q attributes updates on q change?"
        Q_init = np.copy(self.drv.Q)

        # change q, see if Q updates
        x = np.random.rand(10)
        x /= x.sum()
        self.drv.q = x
        Q_after = self.drv.Q

        # should be different
        self.assertFalse(np.allclose(Q_init, Q_after))

        # clean up: reset values
        self.drv.q = self.x

        # now we should have our original Q back
        assert_allclose(Q_init, self.drv.Q)

    def test_Q_end_1(self):
        "discrete_rv: Q sums to 1"
        assert (self.drv.Q[-1] - 1.0 < 1e-10)

    @attr("slow")
    def test_draw_lln(self):
        "discrete_rv: lln satisfied?"
        draws = self.drv.draw(1000000)
        bins = np.arange(self.drv.q.size+1)
        freqs, _ = np.histogram(draws, bins=bins, density=True)
        assert_allclose(freqs, self.drv.q, atol=1e-2)

    def test_draw_with_seed(self):
        x = np.array([0.03326189, 0.60713005, 0.84514831, 0.28493183,
                      0.12393182, 0.35308009, 0.70371579, 0.81728178,
                      0.21294538, 0.05358209])
        draws = DiscreteRV(x).draw(k=10, random_state=5)

        expected_output = np.array([1, 2, 1, 2, 1, 1, 2, 1, 1, 1])

        assert_allclose(draws, expected_output)
