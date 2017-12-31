"""
Tests for util/combinatorics.py

"""
import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_
import scipy.special
from quantecon.util.combinatorics import (
    next_k_array, k_array_rank, k_array_rank_jit
)


class TestKArray:
    def setUp(self):
        self.k_arrays = np.array(
            [[0, 1, 2],
             [0, 1, 3],
             [0, 2, 3],
             [1, 2, 3],
             [0, 1, 4],
             [0, 2, 4],
             [1, 2, 4],
             [0, 3, 4],
             [1, 3, 4],
             [2, 3, 4],
             [0, 1, 5],
             [0, 2, 5],
             [1, 2, 5],
             [0, 3, 5],
             [1, 3, 5],
             [2, 3, 5],
             [0, 4, 5],
             [1, 4, 5],
             [2, 4, 5],
             [3, 4, 5]]
        )
        self.L, self.k = self.k_arrays.shape

    def test_next_k_array(self):
        k_arrays_computed = np.empty((self.L, self.k), dtype=int)
        k_arrays_computed[0] = np.arange(self.k)
        for i in range(1, self.L):
            k_arrays_computed[i] = k_arrays_computed[i-1]
            next_k_array(k_arrays_computed[i])
        assert_array_equal(k_arrays_computed, self.k_arrays)

    def test_k_array_rank(self):
        for i in range(self.L):
            eq_(k_array_rank(self.k_arrays[i]), i)

    def test_k_array_rank_jit(self):
        for i in range(self.L):
            eq_(k_array_rank_jit(self.k_arrays[i]), i)


def test_k_array_rank_arbitrary_precision():
    n, k = 100, 50
    a = np.arange(n-k, n)
    eq_(k_array_rank(a), scipy.special.comb(n, k, exact=True)-1)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
