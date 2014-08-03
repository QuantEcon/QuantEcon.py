"""
tests for quantecon.discrete_rv

@author : Spencer Lyon
@date : 2014-07-31

"""
from __future__ import division
from collections import Counter
import unittest
import numpy as np
from numpy.testing import assert_allclose
from nose.plugins.attrib import attr
import pandas as pd
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
        counts = pd.Series(Counter(draws))
        counts = (counts / counts.sum()).values
        assert max(np.abs(counts - self.drv.q)) < 1e-3

