"""
Tests for linear equation systems' utilities

"""
import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import raises
from quantecon.util import make_tableau, standardize_lp_problem


class TestLESUtil:
    def test_only_inequalities(self):
        # http://mat.gsia.cmu.edu/classes/QUANT/NOTES/chap7.pdf
        c = np.array([1., 1.])
        A_ub = np.array([[2., 1.],
                         [1., 2.]])
        b_ub = np.array([4., 3.])

        tableau = standardize_lp_problem(c, A_ub=A_ub, b_ub=b_ub)
        solution = np.array([[2., 1., 1., 0., 4.],
                             [1., 2., 0., 1., 3.],
                             [1., 1., 0., 0., 0.]])

        assert_array_equal(tableau, solution)

    def test_only_equalities(self):
        c = np.array([1., 1.])
        A_eq = np.array([[2., 1.],
                         [1., 2.]])
        b_eq = np.array([4., 3.])
        tableau = standardize_lp_problem(c, A_eq=A_eq, b_eq=b_eq)
        solution = np.array([[2., 1., 4.],
                             [1., 2., 3.],
                             [1., 1., 0.]])

        assert_array_equal(tableau, solution)

    def test_make_tableau(self):
        c = np.array([1., 2.])
        A_ub = np.array([[3., 4.],
                         [5., 6.]])
        b_ub = np.array([7., 8.])
        A_eq = np.array([[9., 10.],
                         [11., 12.]])
        b_eq = np.array([13., 14.])
        tableau = make_tableau(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                               b_eq=b_eq)

        solution = np.array([[3., 4., 1., 0., 1., 0., 0., 0., 7.],
                             [5., 6., 0., 1., 0., 1., 0., 0., 8.],
                             [9., 10., 0., 0., 0., 0., 1., 0., 13.],
                             [11., 12., 0., 0., 0., 0., 0., 1., 14.],
                             [1., 2., 0., 0., 0., 0., 0., 0., 0.],
                             [-28, -32., -1., -1., 0., 0., 0., 0., -42.]])

        assert_array_equal(tableau, solution)

    @raises(ValueError)
    def test_no_constraints(self):
        c = np.array([1, 1])

        standardize_lp_problem(c)

    @raises(ValueError)
    def test_improper_ub(self):
        c = np.array([1., 2.])
        A_ub = np.array([[3., 4.],
                         [5., 6.]])
        b_ub = np.array([0.])

        standardize_lp_problem(c, A_ub, b_ub)

    @raises(ValueError)
    def test_improper_eq(self):
        c = np.array([1., 2.])
        A_eq = np.array([[3., 4.],
                         [5., 6.]])
        b_eq = np.array([0.])

        standardize_lp_problem(c, A_eq=A_eq, b_eq=b_eq)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
