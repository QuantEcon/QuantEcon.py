"""
Tests for pivoting.py

"""
import numpy as np
from numpy.testing import assert_array_equal

from quantecon.optimize.pivoting import _pivoting


def _pivoting_reference(tableau, pivot_col, pivot_row):
    # Direct-division reference implementation
    tableau = tableau.copy()
    tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
    for i in range(tableau.shape[0]):
        if i != pivot_row:
            tableau[i, :] -= tableau[pivot_row, :] * tableau[i, pivot_col]
    tableau[:, pivot_col] = 0
    tableau[pivot_row, pivot_col] = 1
    return tableau


class TestPivoting:
    def test_equals_direct_division_reference(self):
        rng = np.random.default_rng(0)
        nrows, ncols = 10, 22
        pivot_col, pivot_row = 2, 1
        tableau_0 = rng.random((nrows, ncols)) + 0.5
        desired = _pivoting_reference(tableau_0, pivot_col, pivot_row)
        tableau = tableau_0.copy()
        _pivoting(tableau, pivot_col, pivot_row)
        assert_array_equal(tableau, desired)

    def test_pivot_column_exact_unit_vector(self):
        rng = np.random.default_rng(1)
        nrows, ncols = 6, 14
        pivot_col, pivot_row = 3, 4
        tableau = rng.random((nrows, ncols)) + 0.5
        _pivoting(tableau, pivot_col, pivot_row)
        e = np.zeros(nrows)
        e[pivot_row] = 1
        assert_array_equal(tableau[:, pivot_col], e)

    def test_subnormal_pivot_stays_finite(self):
        # 1/p overflows for a subnormal pivot, while direct division
        # stays finite
        p = np.float64(1e-309)
        tableau = np.array([[p, p, 0.], [p, 2*p, p]])
        _pivoting(tableau, 0, 0)
        assert_array_equal(tableau, np.array([[1., 1., 0.], [0., p, p]]))
