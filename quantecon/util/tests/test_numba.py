"""
Tests for Numba support utilities

"""
import numpy as np
from numpy.testing import assert_array_equal, assert_
from numba import jit
from quantecon.util.numba import _numba_linalg_solve, comb_jit


@jit(nopython=True)
def numba_linalg_solve_orig(a, b):
    return np.linalg.solve(a, b)


class TestNumbaLinalgSolve:
    def setup(self):
        self.dtypes = [np.float32, np.float64]
        self.a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
        self.b_1dim = np.array([2, 4, -1])
        self.b_2dim = np.array([[2, 3], [4, 1], [-1, 0]])
        self.a_singular = np.array([[0, 1, 2], [3, 4, 5], [3, 5, 7]])

    def test_b_1dim(self):
        for dtype in self.dtypes:
            a = np.asfortranarray(self.a, dtype=dtype)
            b = np.asfortranarray(self.b_1dim, dtype=dtype)
            sol_orig = numba_linalg_solve_orig(a, b)
            r = _numba_linalg_solve(a, b)
            assert_(r == 0)
            assert_array_equal(b, sol_orig)

    def test_b_2dim(self):
        for dtype in self.dtypes:
            a = np.asfortranarray(self.a, dtype=dtype)
            b = np.asfortranarray(self.b_2dim, dtype=dtype)
            sol_orig = numba_linalg_solve_orig(a, b)
            r = _numba_linalg_solve(a, b)
            assert_(r == 0)
            assert_array_equal(b, sol_orig)

    def test_singular_a(self):
        for b in [self.b_1dim, self.b_2dim]:
            for dtype in self.dtypes:
                a = np.asfortranarray(self.a_singular, dtype=dtype)
                b = np.asfortranarray(b, dtype=dtype)
                r = _numba_linalg_solve(a, b)
                assert_(r != 0)


class TestCombJit:
    def setup(self):
        self.MAX_INTP = np.iinfo(np.intp).max

    def test_comb(self):
        N, k = 10, 3
        N_choose_k = 120
        assert_(comb_jit(N, k) == N_choose_k)

    def test_comb_zeros(self):
        assert_(comb_jit(2, 3) == 0)
        assert_(comb_jit(-1, 3) == 0)
        assert_(comb_jit(2, -1) == 0)

        assert_(comb_jit(self.MAX_INTP, 2) == 0)

        N = np.intp(self.MAX_INTP**0.5 * 2**0.5) + 1
        assert_(comb_jit(N, 2) == 0)

    def test_max_intp(self):
        assert_(comb_jit(self.MAX_INTP, 0) == 1)
        assert_(comb_jit(self.MAX_INTP, 1) == self.MAX_INTP)
