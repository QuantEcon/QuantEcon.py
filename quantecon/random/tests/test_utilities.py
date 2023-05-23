"""
Tests for util/random.py

Functions
---------
probvec
sample_without_replacement

"""
import numbers
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose, assert_raises,
                           assert_)
from numba import njit
from quantecon.random import probvec, sample_without_replacement, draw


# probvec #

class TestProbvec:
    def setup_method(self):
        self.m, self.k = 2, 3  # m vectors of dimension k
        seed = 1234

        self.out_parallel = probvec(self.m, self.k, random_state=seed)
        self.out_cpu = \
            probvec(self.m, self.k, random_state=seed, parallel=False)

    def test_shape(self):
        for out in [self.out_parallel, self.out_cpu]:
            assert_(out.shape == (self.m, self.k))

    def test_parallel_cpu(self):
        assert_array_equal(self.out_parallel, self.out_cpu)


# sample_without_replacement #

def test_sample_without_replacement_shape():
    assert_array_equal(sample_without_replacement(2, 0).shape, (0,))

    n, k, m = 5, 3, 4
    assert_array_equal(
        sample_without_replacement(n, k).shape,
        (k,)
    )
    assert_array_equal(
        sample_without_replacement(n, k, num_trials=m).shape,
        (m, k)
    )


def test_sample_without_replacement_uniqueness():
    n = 10
    a = sample_without_replacement(n, n)
    b = np.unique(a)
    assert_(len(b) == n)


def test_sample_without_replacement_value_error():
    # n <= 0
    assert_raises(ValueError, sample_without_replacement, 0, 2)
    assert_raises(ValueError, sample_without_replacement, -1, -1)

    # k > n
    assert_raises(ValueError, sample_without_replacement, 2, 3)


# draw #

@njit
def draw_jitted(cdf, size=None):
    return draw(cdf, size)


class TestDraw:
    def setup_method(self):
        self.pmf = np.array([0.4, 0.1, 0.5])
        self.cdf = np.cumsum(self.pmf)
        self.n = len(self.pmf)
        self.draw_funcs = [draw, draw_jitted]

    def test_return_types(self):
        for func in self.draw_funcs:
            out = func(self.cdf)
            assert_(isinstance(out, numbers.Integral))

        size = 10
        for func in self.draw_funcs:
            out = func(self.cdf, size)
            assert_(out.shape == (size,))

    def test_return_values(self):
        for func in self.draw_funcs:
            out = func(self.cdf)
            assert_(out in range(self.n))

        size = 10
        for func in self.draw_funcs:
            out = func(self.cdf, size)
            assert_(np.isin(out, range(self.n)).all())

    def test_lln(self):
        size = 1000000
        for func in self.draw_funcs:
            out = func(self.cdf, size)
            hist, bin_edges = np.histogram(out, bins=self.n, density=True)
            pmf_computed = hist * np.diff(bin_edges)
            atol = 1e-2
            assert_allclose(pmf_computed, self.pmf, atol=atol)


@njit
def draw_jitted_w_o_size(n):
    cdf = np.linspace(1/n, 1, n)
    return draw(cdf)


def test_draw_jitted_w_o_size():
    n = 3
    assert_(draw_jitted_w_o_size(n) in range(n))
