"""
Tests for simplex_solver.py

"""

import numpy as np
from numpy.testing import assert_allclose
from quantecon import linprog_simplex
from quantecon.util import make_tableau
from nose.tools import raises
from scipy.optimize import linprog
from ..util import check_random_state


class TestSimplexSolver:
    def test_aliasing_b_ub(self):
        # Adapted from Scipy
        c = np.array([1.0])
        A_ub = np.array([[1.0]])
        b_ub = np.array([3.0])

        M_ub, N = A_ub.shape
        tableau = make_tableau(c, A_ub, b_ub)

        x = linprog_simplex(tableau, N, M_ub)[0]

        solution = np.array([0.])

        assert_allclose(x, solution)

    def test_aliasing_b_eq(self):
        # Obtained from Scipy
        c = np.array([1.0])
        A_eq = np.array([[1.0]])
        b_eq = np.array([3.0])

        M_eq, N = A_eq.shape
        tableau = make_tableau(c, A_eq=A_eq, b_eq=b_eq)

        x = linprog_simplex(tableau, N, M_eq=M_eq)[0]

        solution = np.array([3.0])

        assert_allclose(x, solution)

    def test_linear_upper_bound(self):
        # Maximize a linear function subject to only linear upper bound
        # constraints.
        # http://www.dam.brown.edu/people/huiwang/classes/am121/Archive/simplex_121_c.pdf
        c = np.array([3, 2])
        A_ub = np.array([[2, 1],
                         [1, 1],
                         [1, 0]])
        b_ub = np.array([10, 8, 4])

        M_ub, N = A_ub.shape
        tableau = make_tableau(-c, A_ub, b_ub)  # Maximize

        x = linprog_simplex(tableau, N, M_ub)[0]

        solution = np.array([2., 6.])

        assert_allclose(x, solution)

    def test_mixed_constraints(self):
        # Minimize linear function subject to non-negative variables.
        # http://www.statslab.cam.ac.uk/~ff271/teaching/opt/notes/notes8.pdf
        c = np.array([6, 3])
        A_ub = np.array([[0, 3],
                         [-1, -1],
                         [-2, 1]])
        b_ub = np.array([2, -1, -1])

        M_ub, N = A_ub.shape
        tableau = make_tableau(c, A_ub, b_ub)

        x = linprog_simplex(tableau, N, M_ub)[0]

        solution = np.array([2/3, 1/3])

        assert_allclose(x, solution)

    def test_degeneracy(self):
        # http://mat.gsia.cmu.edu/classes/QUANT/NOTES/chap7.pdf
        c = np.array([2, 1])
        A_ub = np.array([[3, 1],
                         [1, -1],
                         [0, 1]])
        b_ub = np.array([6, 2, 3])

        M_ub, N = A_ub.shape
        tableau = make_tableau(-c, A_ub, b_ub)  # Maximize

        x = linprog_simplex(tableau, N, M_ub)[0]

        solution = np.array([1., 3.])

        assert_allclose(x, solution)

    def test_cyclic_recovery(self):
        # http://www.math.ubc.ca/~israel/m340/kleemin3.pdf
        c = np.array([100, 10, 1]) * -1  # Maximize
        A_ub = np.array([[1, 0, 0],
                         [20, 1, 0],
                         [200, 20, 1]])
        b_ub = np.array([1, 100, 10000])

        M_ub, N = A_ub.shape
        tableau = make_tableau(c, A_ub, b_ub)

        x = linprog_simplex(tableau, N, M_ub)[0]

        solution = np.array([0., 0., 10000.])

        assert_allclose(x, solution)

    def test_cyclic_bland(self):
        # Obtained from Scipy
        c = np.array([7, 0, -4, 1.5, 1.5])
        A_ub = np.array([[4, 5.5, 1.5, 1.0, -3.5],
                         [1, -2.5, -2, 2.5, 0.5],
                         [3, -0.5, 4, -12.5, -7],
                         [-1, 4.5, 2, -3.5, -2],
                         [5.5, 2, -4.5, -1, 9.5]])
        b_ub = np.array([0, 0, 0, 0, 1])

        M_ub, N = A_ub.shape
        tableau = make_tableau(c, A_ub, b_ub)

        x = linprog_simplex(tableau, N, M_ub)[0]

        solution = np.array([0, 0, 19, 16/3, 29/3])

        assert_allclose(x, solution)

    @raises(ValueError)
    def test_linprog_unbounded(self):
        # Obtained from Scipy
        c = np.array([1, 1]) * -1  # Maximize
        A_ub = np.array([[-1, 1],
                         [-1, -1]])
        b_ub = np.array([-1, -2])

        M_ub, N = A_ub.shape
        tableau = make_tableau(c, A_ub, b_ub)

        linprog_simplex(tableau, N, M_ub)

    @raises(ValueError)
    def test_linprog_infeasible(self):
        # Obtained from Scipy
        c = np.array([-1, -1])
        A_ub = np.array([[1, 0],
                         [0, 1],
                         [-1, -1]])
        b_ub = np.array([2, 2, -5])

        M_ub, N = A_ub.shape
        tableau = make_tableau(c, A_ub, b_ub)

        linprog_simplex(tableau, N, M_ub)

    def test_non_trivial_problem(self):
        # Obtained from Scipy
        c = np.array([-1, 8, 4, -6])
        A_ub = np.array([[-7, -7, 6, 9],
                         [1, -1, -3, 0],
                         [10, -10, -7, 7],
                         [6, -1, 3, 4]])
        b_ub = np.array([-3, 6, -6, 6])
        A_eq = np.array([[-10, 1, 1, -8]])
        b_eq = np.array([-4])

        M_ub, N = A_ub.shape
        M_eq = A_eq.shape[0]
        tableau = make_tableau(c, A_ub, b_ub, A_eq, b_eq)

        x = linprog_simplex(tableau, N, M_ub, M_eq)[0]

        solution = np.array([101 / 1391, 1462 / 1391, 0, 752 / 1391])

        assert_allclose(x, solution)

    def test_network_flow(self):
        # A network flow problem with supply and demand at nodes
        # and with costs along directed edges.
        # https://www.princeton.edu/~rvdb/542/lectures/lec10.pdf
        c = np.array([2, 4, 9, 11, 4, 3, 8, 7, 0, 15, 16, 18])
        n, p = -1, 1
        A_eq = np.array([[n, n, p, 0, p, 0, 0, 0, 0, p, 0, 0],
                         [p, 0, 0, p, 0, p, 0, 0, 0, 0, 0, 0],
                         [0, 0, n, n, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, p, p, 0, 0, p, 0],
                         [0, 0, 0, 0, n, n, n, 0, p, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, n, n, 0, 0, p],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, n, n, n]])

        b_eq = np.array([0, 19, -16, 33, 0, 0, -36])

        M_eq, N = A_eq.shape
        tableau = make_tableau(c, A_eq=A_eq, b_eq=b_eq)

        x = linprog_simplex(tableau, N, M_eq=M_eq)[0]

        solution = np.array([19., 0., 16., 0., 0., 0.,
                             0., 0., 0., 3., 33., 0.])

        assert_allclose(x, solution)

    def test_network_flow_limited_capacity(self):
        # A network flow problem with supply and demand at nodes
        # and with costs and capacities along directed edges.
        # http://blog.sommer-forst.de/2013/04/10/
        c = np.array([2, 2, 1, 3, 1])
        n, p = -1, 1
        A_eq = np.array([[n, n, 0, 0, 0],
                         [p, 0, n, n, 0],
                         [0, p, p, 0, n],
                         [0, 0, 0, p, p]])

        b_eq = np.array([-4, 0, 0, 4])

        M_eq, N = A_eq.shape
        tableau = make_tableau(c, A_eq=A_eq, b_eq=b_eq)

        x = linprog_simplex(tableau, N, M_eq=M_eq)[0]

        solution = np.array([0., 4., 0., 0., 4.])

        assert_allclose(x, solution)

    def test_wikipedia_example(self):
        # http://en.wikipedia.org/wiki/Simplex_algorithm#Example
        c = np.array([-2, -3, -4])
        A_ub = np.array([[3, 2, 1],
                         [2, 5, 3]])
        b_ub = np.array([10, 15])

        M_ub, N = A_ub.shape
        tableau = make_tableau(c, A_ub, b_ub)

        x = linprog_simplex(tableau, N, M_ub)[0]

        solution = np.array([0., 0., 5.])

        assert_allclose(x, solution)

    def test_enzo_example(self):
        # http://www.ecs.shimane-u.ac.jp/~kyoshida/lpeng.htm
        c = np.array([4, 8, 3, 0, 0, 0])
        A_eq = np.array([[2, 5, 3, -1, 0, 0],
                         [3, 2.5, 8, 0, -1, 0],
                         [8, 10, 4, 0, 0, -1]])
        b_eq = np.array([185, 155, 600])

        M_eq, N = A_eq.shape
        tableau = make_tableau(c, A_eq=A_eq, b_eq=b_eq)

        x = linprog_simplex(tableau, N, M_eq=M_eq)[0]

        solution = np.array([66.25, 0, 17.5, 0, 183.75, 0])

        assert_allclose(x, solution)

    def test_enzo_example_b(self):
        # https://github.com/scipy/scipy/pull/218
        c = np.array([2.8, 6.3, 10.8, -2.8, -6.3, -10.8])
        A_eq = np.array([[-1, -1, -1, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1],
                         [1, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 1]])
        b_eq = np.array([-0.5, 0.4, 0.3, 0.3, 0.3])

        M_eq, N = A_eq.shape
        tableau = make_tableau(c, A_eq=A_eq, b_eq=b_eq)

        x = linprog_simplex(tableau, N, M_eq=M_eq)[0]

        solution = np.array([0.3, 0.2, 0.0, 0.0, 0.1, 0.3])

        assert_allclose(x, solution)

    def test_enzo_example_c(self):
        # Obtained from Scipy
        m = 20
        c = -np.ones(m)
        tmp = 2 * np.pi * np.arange(1, m + 1) / (m + 1)
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
        b_eq = np.array([0, 0])

        M_eq, N = A_eq.shape
        tableau = make_tableau(c, A_eq=A_eq, b_eq=b_eq)

        x = linprog_simplex(tableau, N, M_eq=M_eq)[0]

        solution = np.zeros(m)

        assert_allclose(x, solution)

    @raises(ValueError)
    def test_enzo_example_c_unbounded(self):
        # Obtained from Scipy
        m = 50
        c = -np.ones(m)
        tmp = 2 * np.pi * np.arange(m) / (m + 1)
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
        b_eq = np.array([0., 0.])

        M_eq, N = A_eq.shape
        tableau = make_tableau(c, A_eq=A_eq, b_eq=b_eq)

        linprog_simplex(tableau, N, M_eq=M_eq)

    @raises(ValueError)
    def test_enzo_example_c_infeasible(self):
        # Obtained from Scipy
        m = 50
        c = -np.ones(m)
        tmp = 2 * np.pi * np.arange(m) / (m + 1)
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
        b_eq = np.array([1., 1.])

        M_eq, N = A_eq.shape
        tableau = make_tableau(c, A_eq=A_eq, b_eq=b_eq)

        linprog_simplex(tableau, N, M_eq=M_eq)

    def test_basic_artificial_vars(self):
        # Obtained from Scipy
        # Test if linprog succeeds when at the end of Phase 1 some artificial
        # variables remain basic, and the row in T corresponding to the
        # artificial variables is not all zero.
        c = np.array([-0.1, -0.07, 0.004, 0.004, 0.004, 0.004])
        A_ub = np.array([[1.0, 0, 0, 0, 0, 0],
                         [-1.0, 0, 0, 0, 0, 0],
                         [0, -1.0, 0, 0, 0, 0],
                         [0, 1.0, 0, 0, 0, 0],
                         [1.0, 1.0, 0, 0, 0, 0]])
        b_ub = np.array([3.0, 3.0, 3.0, 3.0, 20.0])
        A_eq = np.array([[1.0, 0, -1, 1, -1, 1], [0, -1.0, -1, 1, -1, 1]])
        b_eq = np.array([0, 0])

        M_ub, N = A_ub.shape
        M_eq = A_eq.shape[0]
        tableau = make_tableau(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

        x = linprog_simplex(tableau, N, M_ub=M_ub, M_eq=M_eq)[0]

        solution = np.zeros_like(c)

        assert_allclose(x, solution)

    def test_zero_row_2(self):
        # Obtained from Scipy
        c = np.array([1, 2, 3])
        A_eq = np.array([[0, 0, 0],
                         [1, 1, 1],
                         [0, 0, 0]])
        b_eq = np.array([0, 3, 0])

        M_eq, N = A_eq.shape
        tableau = make_tableau(c, A_eq=A_eq, b_eq=b_eq)

        x = linprog_simplex(tableau, N, M_eq=M_eq)[0]

        solution = np.array([3., 0., 0.])

        assert_allclose(x, solution)

    def test_simple_input(self):
        # Obtained from Scipy
        A_eq = np.array([[0, -7]])
        b_eq = np.array([-6])
        c = np.array([1, 5])

        M_eq, N = A_eq.shape
        tableau = make_tableau(c, A_eq=A_eq, b_eq=b_eq)

        x = linprog_simplex(tableau, N, M_eq=M_eq)[0]

        solution = np.array([0., 6./7])

        assert_allclose(x, solution)

    def test_match_scipy(self):
        # Has a solution by the inequality version of Farkas' Lemma
        random_state = check_random_state(0)

        for i in range(100):
            c = -np.ones(100)
            A_ub = 0.5 + random_state.rand(100, 100)
            b_ub = random_state.rand(100)

            M_ub, N = A_ub.shape
            tableau = make_tableau(c, A_ub, b_ub)

            sol = linprog_simplex(tableau, N, M_ub)[0]
            sci = linprog(c, A_ub, b_ub, method='interior-point')

            assert_allclose(sol, sci.x, atol=1e-07)

    def test_scipy_simplex_fails_lb(self):
        # https://github.com/scipy/scipy/issues/8174
        c = np.array([1,0,0,0,0,0,0])
        A_ub = np.identity(7)*(-1)
        b_ub = np.array([-2, -2, -2, -2, -2, -2, -2])
        A_eq = np.array([[1,1,1,1,1,1,0],
                         [0.3,1.3,0.9,0,0,0,-1],
                         [0.3,0,0,0,0,0,-2/3],
                         [0,0.65,0,0,0,0,-1/15],
                         [0,0,0.3,0,0,0,-1/15]])
        b_eq = np.array([100, 0, 0, 0, 0])

        M_ub, N = A_ub.shape
        M_eq = A_eq.shape[0]
        tableau = make_tableau(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

        x = linprog_simplex(tableau, N, M_ub=M_ub, M_eq=M_eq)[0]

        solution = np.array([43.33333333, 2., 4.33333333, 46.33333333,
                             2., 2., 19.5])

        assert_allclose(x, solution)

    def test_scipy_simplex_fails_ub(self):
        # https://github.com/scipy/scipy/issues/8174
        A_ub = np.array([[ 22714.,    1008.,   13380.,   -2713.5,  -1116. ],
                         [ -4986.,   -1092.,  -31220.,   17386.5,    684. ],
                         [ -4986.,       0.,       0.,   -2713.5,      0. ],
                         [ 22714.,       0.,       0.,   17386.5,      0. ],
                         [ 1.,       0.,       0.,   0.,      0. ],
                         [ 0.,       1.,       0.,   0.,      0. ],
                         [ 0.,       0.,       1.,   0.,      0. ],
                         [ 0.,       0.,       0.,   1.,      0. ],
                         [ 0.,       0.,       0.,   0.,      1. ]])

        A_ub = np.array(A_ub)
        b_ub = np.zeros(A_ub.shape[0])
        b_ub[4:] = 1.
        c = -np.ones(A_ub.shape[1])

        M_ub, N = A_ub.shape
        tableau = make_tableau(c, A_ub, b_ub)

        x = linprog_simplex(tableau, N, M_ub)[0]

        solution = np.array([0., 1., 0., 0., 1.])

        assert_allclose(x, solution, atol=1e-2)

    def test_multiple_basic_solutions(self):
        # Note: there are multiple basic solutions to this LP problem
        c = np.array([-1., -1., -1., 0., 0.])
        A_eq = np.array([[0., 1., 1., 1., 0.],
                         [1., 1., 1., 0., 1.],
                         [0., 1., 1., 0., 0.]])
        b_eq = np.array([3., 6., 1.])

        M_eq, N = A_eq.shape
        tableau = make_tableau(c, A_eq=A_eq, b_eq=b_eq)

        x = linprog_simplex(tableau, N, M_eq=M_eq)[0]

        solution = np.array([5., 1., 0., 2., 0.])

        assert_allclose(x, solution)

    def test_cycling(self):
        # From reference [4]
        c = np.array([0., 0., 0., -.75, 20., -0.5, 6])
        A_eq = np.array([[1., 0., 0., 0.25, -8., -1., 9.],
                         [0., 1., 0., .5, -12., -.5, 3],
                         [0., 0., 1., 0., 0., 1., 0.]])
        b_eq = np.array([0., 0., 1.])

        M_eq, N = A_eq.shape
        tableau = make_tableau(c, A_eq=A_eq, b_eq=b_eq)

        x = linprog_simplex(tableau, N, M_eq=M_eq, tie_breaking_rule=1)[0]

        solution = np.array([0.75, 0.  , 0.  , 1.  , 0.  , 1.  , 0.  ])

        assert_allclose(x, solution)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
