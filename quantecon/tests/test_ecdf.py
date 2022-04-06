"""
Tests for ecdf.py

"""
import numpy as np
from numpy.testing import assert_, assert_allclose
from quantecon import ECDF


class TestECDF:

    @classmethod
    def setup(cls):
        cls.obs = np.random.rand(40)  # observations defining dist
        cls.ecdf = ECDF(cls.obs)

    def test_call_high(self):
        "ecdf: x above all obs give 1.0"
        # all of self.obs <= 1 so ecdf(1.1) should be 1
        assert_allclose(self.ecdf(1.1), 1.0)

    def test_call_low(self):
        "ecdf: x below all obs give 0.0"
        # all of self.obs <= 1 so ecdf(1.1) should be 1
        assert_allclose(self.ecdf(-0.1), 0.0)

    def test_ascending(self):
        "ecdf: larger values should return F(x) at least as big"
        x = np.random.rand()
        F_1 = self.ecdf(x)
        F_2 = self.ecdf(1.1 * x)
        assert_(F_2 >= F_1)

    def test_vectorized(self):
        "ecdf: testing vectorized __call__ method"
        t = np.linspace(-1, 1, 100)
        e = self.ecdf(t)
        assert_(t.shape == e.shape)
        assert_(e.dtype == float)
        t = np.linspace(-1, 1, 100).reshape(2, 2, 25)
        e = self.ecdf(t)
        assert_(t.shape == e.shape)
        assert_(e.dtype == float)
