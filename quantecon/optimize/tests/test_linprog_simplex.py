"""
Tests for linprog_simplex

"""
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal

from quantecon.optimize import linprog_simplex


# Assert functions from scipy/optimize/tests/test_linprog.py

def _assert_infeasible(res):
    # res: linprog result object
    assert_(not res.success, "incorrectly reported success")
    assert_equal(res.status, 2, "failed to report infeasible status")


def _assert_unbounded(res):
    # res: linprog result object
    assert_(not res.success, "incorrectly reported success")
    assert_equal(res.status, 3, "failed to report unbounded status")


def _assert_success(res, c, b_ub=np.array([]), b_eq=np.array([]),
                    desired_fun=None, desired_x=None, rtol=1e-15, atol=1e-15):
    if not res.success:
        msg = "linprog status {0}".format(res.status)
        raise AssertionError(msg)

    assert_equal(res.status, 0)
    if desired_fun is not None:
        assert_allclose(res.fun, desired_fun,
                        err_msg="converged to an unexpected objective value",
                        rtol=rtol, atol=atol)
    if desired_x is not None:
        assert_allclose(res.x, desired_x,
                        err_msg="converged to an unexpected solution",
                        rtol=rtol, atol=atol)

    fun_p = c @ res.x
    fun_d = np.concatenate([b_ub, b_eq]) @ res.lambd
    assert_allclose(fun_p, fun_d, rtol=rtol, atol=atol)
    assert_allclose(res.fun, fun_d, rtol=rtol, atol=atol)


class TestLinprogSimplexScipy:
    # Test cases from scipy/optimize/tests/test_linprog.py

    def test_infeasible(self):
        # Test linprog response to an infeasible problem
        c = np.array([-1, -1]) * (-1)
        A_ub = [[1, 0],
                [0, 1],
                [-1, -1]]
        b_ub = [2, 2, -5]
        c, A_ub, b_ub = map(np.asarray, [c, A_ub, b_ub])
        res = linprog_simplex(c, A_ub=A_ub, b_ub=b_ub)
        _assert_infeasible(res)

    def test_unbounded(self):
        # Test linprog response to an unbounded problem
        c = np.array([1, 1])
        A_ub = [[-1, 1],
                [-1, -1]]
        b_ub = [-1, -2]
        c, A_ub, b_ub = map(np.asarray, [c, A_ub, b_ub])
        res = linprog_simplex(c, A_ub=A_ub, b_ub=b_ub)
        _assert_unbounded(res)

    def test_nontrivial_problem(self):
        # Test linprog for a problem involving all constraint types,
        # negative resource limits, and rounding issues.
        c = np.array([-1, 8, 4, -6]) * (-1)
        A_ub = [[-7, -7, 6, 9],
                [1, -1, -3, 0],
                [10, -10, -7, 7],
                [6, -1, 3, 4]]
        b_ub = [-3, 6, -6, 6]
        A_eq = [[-10, 1, 1, -8]]
        b_eq = [-4]
        c, A_ub, b_ub, A_eq, b_eq = \
            map(np.asarray, [c, A_ub, b_ub, A_eq, b_eq])
        desired_fun = 7083 / 1391 * (-1)
        desired_x = [101 / 1391, 1462 / 1391, 0, 752 / 1391]
        res = linprog_simplex(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        _assert_success(res, c, b_ub=b_ub, b_eq=b_eq, desired_fun=desired_fun,
                        desired_x=desired_x)

    def test_network_flow(self):
        # A network flow problem with supply and demand at nodes
        # and with costs along directed edges.
        # https://www.princeton.edu/~rvdb/542/lectures/lec10.pdf
        c = np.array([2, 4, 9, 11, 4, 3, 8, 7, 0, 15, 16, 18]) * (-1)
        n, p = -1, 1
        A_eq = [
            [n, n, p, 0, p, 0, 0, 0, 0, p, 0, 0],
            [p, 0, 0, p, 0, p, 0, 0, 0, 0, 0, 0],
            [0, 0, n, n, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, p, p, 0, 0, p, 0],
            [0, 0, 0, 0, n, n, n, 0, p, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, n, n, 0, 0, p],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, n, n, n]]
        b_eq = [0, 19, -16, 33, 0, 0, -36]
        c, A_eq, b_eq = map(np.asarray, [c, A_eq, b_eq])
        desired_fun = -755
        res = linprog_simplex(c, A_eq=A_eq, b_eq=b_eq)
        _assert_success(res, c, b_eq=b_eq, desired_fun=desired_fun)

    def test_basic_artificial_vars(self):
        # Test if linprog succeeds when at the end of Phase 1 some artificial
        # variables remain basic, and the row in T corresponding to the
        # artificial variables is not all zero.
        c = np.array([-0.1, -0.07, 0.004, 0.004, 0.004, 0.004]) * (-1)
        A_ub = np.array([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0],
                         [0, -1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0],
                         [1.0, 1.0, 0, 0, 0, 0]])
        b_ub = np.array([3.0, 3.0, 3.0, 3.0, 20.0])
        A_eq = np.array([[1.0, 0, -1, 1, -1, 1], [0, -1.0, -1, 1, -1, 1]])
        b_eq = np.array([0, 0])
        res = linprog_simplex(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        _assert_success(res, c, b_ub=b_ub, b_eq=b_eq, desired_fun=0,
                        desired_x=np.zeros_like(c))

    def test_bug_5400(self):
        # https://github.com/scipy/scipy/issues/5400
        f = 1 / 9
        g = -1e4
        h = -3.1
        A_ub = np.array([
            [1, -2.99, 0, 0, -3, 0, 0, 0, -1, -1, 0, -1, -1, 1, 1, 0, 0, 0, 0],
            [1, 0, -2.9, h, 0, -3, 0, -1, 0, 0, -1, 0, -1, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, h, 0, 0, -3, -1, -1, 0, -1, -1, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 1.99, -1, -1, 0, 0, 0, -1, f, f, 0, 0, 0, g, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, -1, -1, 0, 0, 0, -1, f, f, 0, g, 0, 0, 0, 0],
            [0, -1, 1.9, 2.1, 0, 0, 0, f, -1, -1, 0, 0, 0, 0, 0, g, 0, 0, 0],
            [0, 0, 0, 0, -1, 2, -1, 0, 0, 0, f, -1, f, 0, 0, 0, g, 0, 0],
            [0, -1, -1, 2.1, 0, 0, 0, f, f, -1, 0, 0, 0, 0, 0, 0, 0, g, 0],
            [0, 0, 0, 0, -1, -1, 2, 0, 0, 0, f, f, -1, 0, 0, 0, 0, 0, g]])

        b_ub = np.array([
            0.0, 0, 0, 100, 100, 100, 100, 100, 100, 900, 900, 900, 900, 900,
            900, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        c = np.array([-1.0, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 0, 0, 0, 0, 0, 0]) * (-1)

        desired_fun = 106.63507541835018

        res = linprog_simplex(c, A_ub=A_ub, b_ub=b_ub)
        _assert_success(res, c, b_ub=b_ub, desired_fun=desired_fun,
                        rtol=1e-8, atol=1e-8)

    def test_issue_8174_stackoverflow(self):
        # Test supplementary example from issue 8174.
        # https://github.com/scipy/scipy/issues/8174
        # https://stackoverflow.com/questions/47717012/
        c = np.array([1, 0, 0, 0, 0, 0, 0]) * (-1)
        A_ub = -np.identity(7)
        b_ub = np.full(7, -2)
        A_eq = np.array([
            [1, 1, 1, 1, 1, 1, 0],
            [0.3, 1.3, 0.9, 0, 0, 0, -1],
            [0.3, 0, 0, 0, 0, 0, -2/3],
            [0, 0.65, 0, 0, 0, 0, -1/15],
            [0, 0, 0.3, 0, 0, 0, -1/15]
        ])
        b_eq = np.array([100, 0, 0, 0, 0])

        res = linprog_simplex(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        _assert_success(res, c, b_ub=b_ub, b_eq=b_eq)

    def test_linprog_cyclic_recovery(self):
        # Test linprogs recovery from cycling using the Klee-Minty problem
        # Klee-Minty  https://www.math.ubc.ca/~israel/m340/kleemin3.pdf
        c = np.array([100, 10, 1])
        A_ub = [[1, 0, 0],
                [20, 1, 0],
                [200, 20, 1]]
        b_ub = [1, 100, 10000]
        c, A_ub, b_ub = map(np.asarray, [c, A_ub, b_ub])
        desired_x = [0, 0, 10000]
        res = linprog_simplex(c, A_ub=A_ub, b_ub=b_ub)
        _assert_success(res, c, b_ub=b_ub, desired_x=desired_x)

    def test_linprog_cyclic_bland(self):
        # Test the effect of Bland's rule on a cycling problem
        c = np.array([-10, 57, 9, 24.]) * (-1)
        A_ub = np.array([[0.5, -5.5, -2.5, 9],
                         [0.5, -1.5, -0.5, 1],
                         [1, 0, 0, 0]])
        b_ub = np.array([0, 0, 1])
        desired_x = [1, 0, 1, 0]
        res = linprog_simplex(c, A_ub=A_ub, b_ub=b_ub)
        _assert_success(res, c, b_ub=b_ub, desired_x=desired_x)

    def test_linprog_cyclic_bland_bug_8561(self):
        # Test that pivot row is chosen correctly when using Bland's rule
        c = np.array([7, 0, -4, 1.5, 1.5]) * (-1)
        A_ub = np.array([
            [4, 5.5, 1.5, 1.0, -3.5],
            [1, -2.5, -2, 2.5, 0.5],
            [3, -0.5, 4, -12.5, -7],
            [-1, 4.5, 2, -3.5, -2],
            [5.5, 2, -4.5, -1, 9.5]])
        b_ub = np.array([0, 0, 0, 0, 1])
        # desired_x = [0, 0, 19, 16/3, 29/3]
        res = linprog_simplex(c, A_ub=A_ub, b_ub=b_ub)
        _assert_success(res, c, b_ub=b_ub)

    def test_bug_8662(self):
        # linprog simplex used to report inncorrect optimal results
        # https://github.com/scipy/scipy/issues/8662
        c = np.array([-10, 10, 6, 3]) * (-1)
        A_ub = [[8, -8, -4, 6],
                [-8, 8, 4, -6],
                [-4, 4, 8, -4],
                [3, -3, -3, -10]]
        b_ub = [9, -9, -9, -4]
        c, A_ub, b_ub = map(np.asarray, [c, A_ub, b_ub])
        desired_fun = -36.0000000000
        res = linprog_simplex(c, A_ub=A_ub, b_ub=b_ub)
        _assert_success(res, c, b_ub=b_ub, desired_fun=desired_fun)


class TestLinprogSimplex:
    def test_phase_1_bug_725(self):
        # Identified a bug in Phase 1
        # https://github.com/QuantEcon/QuantEcon.py/issues/725
        c = np.array([-4.09555556, 4.59044444])
        A_ub = np.array([[1, 0.1], [-1, -0.1], [1, 1]])
        b_ub = np.array([9.1, -0.1, 0.1])
        desired_x = [0.1, 0]
        res = linprog_simplex(c, A_ub=A_ub, b_ub=b_ub)
        _assert_success(res, c, b_ub=b_ub, desired_x=desired_x)
