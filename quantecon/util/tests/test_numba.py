"""
Tests for Numba support utilities

"""
import numpy as np
from numpy.testing import assert_array_equal
from numba import jit
from nose.tools import eq_, ok_
from quantecon.util.numba import _numba_linalg_solve


@jit(nopython=True)
def numba_linalg_solve_orig(a, b):
    return np.linalg.solve(a, b)


class TestNumbaLinalgSolve:
    def setUp(self):
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
            eq_(r, 0)
            assert_array_equal(b, sol_orig)

    def test_b_2dim(self):
        for dtype in self.dtypes:
            a = np.asfortranarray(self.a, dtype=dtype)
            b = np.asfortranarray(self.b_2dim, dtype=dtype)
            sol_orig = numba_linalg_solve_orig(a, b)
            r = _numba_linalg_solve(a, b)
            eq_(r, 0)
            assert_array_equal(b, sol_orig)

    def test_singular_a(self):
        for b in [self.b_1dim, self.b_2dim]:
            for dtype in self.dtypes:
                a = np.asfortranarray(self.a_singular, dtype=dtype)
                b = np.asfortranarray(b, dtype=dtype)
                r = _numba_linalg_solve(a, b)
                ok_(r != 0)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
