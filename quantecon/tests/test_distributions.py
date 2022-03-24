"""
Tests for distributions.py

"""
import numpy as np
from numpy.testing import assert_allclose, assert_
from math import sqrt
from quantecon.distributions import BetaBinomial


class TestBetaBinomial:
    def setup(self):
        self.n = 100
        self.a = 5
        self.b = 5
        self.test_obj = BetaBinomial(self.n, self.a, self.b)

    def test_init(self):
        assert_(self.test_obj.n == self.n)
        assert_(self.test_obj.a == self.a)
        assert_(self.test_obj.b == self.b)

    def test_mean(self):
        assert_(self.test_obj.mean == 50)

    def test_std(self):
        assert_(self.test_obj.std == sqrt(250))

    def test_var(self):
        assert_(self.test_obj.var == 250)

    def test_skew(self):
        assert_(self.test_obj.skew == 0)

    def test_pdf(self):
        n = 9
        a = 1
        b = 1
        test_obj = BetaBinomial(n, a, b)
        assert_allclose(test_obj.pdf(), np.full(n+1, 1/(n+1)))
