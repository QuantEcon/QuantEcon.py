"""
Filename: test_asset_pricing.py
Authors: Spencer Lyon
Date: 2014-07-30

Tests for quantecon.asset_pricing module

TODO: come up with some simple examples we can check by hand for price
      methods.

"""
from __future__ import division
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.models import AssetPrices

# parameters for object
n = 5
P = 0.0125 * np.ones((n, n))
P += np.diag(0.95 - 0.0125 * np.ones(5))
s = np.array([1.05, 1.025, 1.0, 0.975, 0.95])  # state values
gamma = 2.0
bet = 0.94
zeta = 1.0
p_s = 150.0


class TestAssetPrices(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ap = AssetPrices(bet, P, s, gamma)

    def test_P_shape(self):
        shp = self.ap.P.shape
        assert shp[0] == shp[1]

    def test_n(self):
        assert self.ap.n == self.ap.P.shape[0]

    def test_P_tilde(self):
        "construct P_tilde by hand using nested for loops"
        # unpack variables and allocate memory for new P_tilde
        n, s, P, gam = (self.ap.n, self.ap.s, self.ap.P, self.ap.gamma)
        p_tilde_2 = np.empty_like(self.ap.P)

        # fill in new p_tilde by hand
        for i in range(n):
            for k in range(n):
                p_tilde_2[i, k] = P[i, k] * s[k] ** (1.0 - gam)

        assert_allclose(self.ap.P_tilde, p_tilde_2)

    def test_P_check(self):
        "construct P_check by hand using nested for loops"
        # unpack variables and allocate memory for new P_tilde
        n, s, P, gam = (self.ap.n, self.ap.s, self.ap.P, self.ap.gamma)
        p_check_2 = np.empty_like(self.ap.P)

        # fill in new p_check by hand
        for i in range(n):
            for k in range(n):
                p_check_2[i, k] = P[i, k] * s[k] ** (-gam)

        assert_allclose(self.ap.P_check, p_check_2)

    def test_tree_price_size(self):
        assert self.ap.tree_price().size == self.ap.n

    def test_consol_price_size(self):
        assert self.ap.consol_price(zeta).size == self.ap.n

    def test_call_option_size(self):
        assert self.ap.call_option(zeta, p_s)[0].size == self.ap.n

    def test_tree_price(self):
        pass

    def test_consol_price(self):
        pass

    def test_call_option_price(self):
        pass
