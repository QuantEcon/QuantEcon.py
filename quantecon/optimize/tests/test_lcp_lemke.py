"""
Tests for lcp_lemke

"""
import numpy as np
from numpy.testing import (
    assert_, assert_allclose, assert_equal, assert_raises
)

from quantecon.optimize import lcp_lemke, PivOptions
from quantecon.optimize.lcp_lemke import _lcp_lemke_tbl
from quantecon.game_theory import NormalFormGame, Player, lemke_howson


def _assert_ray_termination(res):
    # res: lcp result object
    assert_(not res.success, "incorrectly reported success")
    assert_equal(res.status, 2, "failed to report ray termination status")


def _assert_success(res, M, q, desired_z=None, rtol=1e-15, atol=1e-15):
    if not res.success:
        msg = "lcp_lemke status {0}".format(res.status)
        raise AssertionError(msg)

    assert_equal(res.status, 0)
    if desired_z is not None:
        assert_allclose(res.z, desired_z,
                        err_msg="converged to an unexpected solution",
                        rtol=rtol, atol=atol)

    assert_((res.z >= -atol).all())

    w = M @ res.z + q
    assert_((w >= -atol).all())

    assert_allclose(w * res.z, np.zeros_like(res.z), rtol=rtol, atol=atol)


class TestLCPLemke:
    def test_Murty_Ex_2_8(self):
        M = [[1, -1, -1, -1],
             [-1, 1, -1, -1],
             [1, 1, 2, 0],
             [1, 1, 0, 2]]
        q = [3, 5, -9, -5]
        M, q = map(np.asarray, [M, q])
        res = lcp_lemke(M, q)
        _assert_success(res, M, q)

    def test_Murty_Ex_2_9(self):
        M = [[-1, 0, -3],
             [1, -2, -5],
             [-2, -1, -2]]
        q = [-3, -2, -1]
        M, q = map(np.asarray, [M, q])
        res = lcp_lemke(M, q)
        _assert_ray_termination(res)

    def test_Kostreva_Ex_1(self):
        # Cycling without careful tie breaking
        M = [[1, 2, 0],
             [0, 1, 2],
             [2, 0, 1]]
        q = [-1, -1, -1]
        M, q = map(np.asarray, [M, q])
        res = lcp_lemke(M, q)
        _assert_success(res, M, q)

    def test_Kostreva_Ex_2(self):
        # Cycling without careful tie breaking
        M = [[1, -1, 3],
             [2, -1, 3],
             [-1, -2, 0]]
        q = [-1, -1, -1]
        M, q = map(np.asarray, [M, q])
        res = lcp_lemke(M, q)
        _assert_ray_termination(res)

    def test_Murty_Ex_2_11(self):
        M = [[-1.5, 2],
             [-4, 4]]
        q = [-5, 17]
        d = [5., 16.]
        M, q, d = map(np.asarray, [M, q, d])
        res = lcp_lemke(M, q, d=d)
        _assert_ray_termination(res)

        res = lcp_lemke(M, q, d=np.ones_like(d))
        _assert_success(res, M, q, atol=1e-13)

    def test_bimatrix_game(self):
        A = [[3, 3],
             [2, 5],
             [0, 6]]
        B = [[3, 2, 3],
             [2, 6, 1]]
        A, B = map(np.asarray, [A, B])
        m, n = A.shape
        I = np.cumsum([0, m, n, m, m, n, n])
        M = np.zeros((3*m+3*n, 3*m+3*n))
        M[I[0]:I[1], I[1]:I[2]] = -A + A.max()
        M[I[0]:I[1], I[2]:I[3]], M[I[0]:I[1], I[3]:I[4]] = 1, -1
        M[I[1]:I[2], I[0]:I[1]] = -B + B.max()
        M[I[1]:I[2], I[4]:I[5]], M[I[1]:I[2], I[5]:I[6]] = 1, -1
        M[I[2]:I[3], I[0]:I[1]], M[I[3]:I[4], I[0]:I[1]] = -1, 1
        M[I[4]:I[5], I[1]:I[2]], M[I[5]:I[6], I[1]:I[2]] = -1, 1
        q = np.zeros(3*m+3*n)
        q[I[2]:I[3]], q[I[3]:I[4]] = 1, -1
        q[I[4]:I[5]], q[I[5]:I[6]] = 1, -1

        res = lcp_lemke(M, q)
        _assert_success(res, M, q)

    def test_bug_768(self):
        n = 10
        seed = 0
        rng = np.random.RandomState(seed)
        A = rng.standard_normal((n, n))
        M = A.T @ A + np.eye(n)
        q = rng.standard_normal(n)
        res = lcp_lemke(M, q)
        _assert_success(res, M, q)


def test_lcp_lemke_nonpositive_d():
    M = np.array([[2., 1.], [1., 2.]])
    q = np.array([-1., -1.])
    assert_raises(ValueError, lcp_lemke, M, q, np.array([1., 0.]))
    assert_raises(ValueError, lcp_lemke, M, q, np.array([1., -1.]))
    # Also in the trivial case, where d is not used
    q_trivial = np.array([1., 1.])
    assert_raises(ValueError, lcp_lemke, M, q_trivial, np.array([1., 0.]))


def test_lcp_lemke_workspaces():
    M = np.array([[2., 1.], [1., 2.]])
    q = np.array([-1., -1.])
    n = 2
    res = lcp_lemke(M, q, tableau=np.empty((n, 2*n+2)),
                    basis=np.empty(n, dtype=int), z=np.empty(n))
    assert_(res.success)
    assert_allclose(M @ res.z + q, np.zeros(n) + (M @ res.z + q).clip(0))
    assert_allclose(res.z @ (M @ res.z + q), 0, atol=1e-12)


def test_lcp_lemke_numerical_breakdown():
    # Entries of order 1e14: the lexicographic tie breaking fails within
    # `tol_ratio_diff`. This used to be reported as ray termination, and
    # pivoting through the tie leads to a wrong "solution" reported as
    # success; it is a numerical breakdown
    s = 2e14
    M = np.array([[-1., -4., 1.],
                  [-5*s, s, 3*s],
                  [-5*s, 3*s, s]])
    q = -np.ones(3)
    res = lcp_lemke(M, q)
    assert_(not res.success)
    assert_equal(res.status, 3)


def test_lcp_lemke_initial_ratio_test_anchored():
    # Chained near-ties in the initial ratios q / d: the second is within
    # `tol_ratio_diff` of the first, the third within the tolerance of the
    # second but not of the first. Ties are measured against the minimum,
    # so the artificial variable enters at row 1, not at row 2 (chaining).
    # With max_iter=1 only the initial pivot is performed.
    n, tol = 3, 1e-13
    M = np.eye(n)
    q = np.array([-3., -3. + 0.9*tol, -3. + 1.8*tol])
    basis = np.empty(n, dtype=int)
    res = lcp_lemke(M, q, basis=basis, max_iter=1)
    assert_equal(res.status, 1)
    assert_equal(basis[1], 2*n)  # Artificial variable

    # Decreasing then increasing: the second ratio is the minimum, the
    # third is within the tolerance of the first but not of the minimum
    q = np.array([-3., -3. - 0.9*tol, -3. + 0.9*tol])
    res = lcp_lemke(M, q, basis=basis, max_iter=1)
    assert_equal(res.status, 1)
    assert_equal(basis[1], 2*n)  # Artificial variable


def _symmetric_game_tableau(A, buf=None):
    # Scheme II tableau [I | A | 1] for the symmetric game (A, A.T) with
    # positive payoff matrix A, optionally as a view into `buf`
    n = A.shape[0]
    tableau = np.empty((n, 2*n+1)) if buf is None else buf[:n, :2*n+1]
    tableau[:, :n] = np.eye(n)
    tableau[:, n:2*n] = A
    tableau[:, 2*n] = 1
    basis = np.arange(n)
    return tableau, basis


def _lemke_path(A, s, max_iter=10**6, buf=None):
    # Lemke path of the symmetric game (A, A.T) from the pure strategy s
    n = A.shape[0]
    tableau, basis = _symmetric_game_tableau(A, buf)
    argmins = np.empty(n, dtype=int)
    stop_vars = np.array([s, n+s])
    piv_options = PivOptions()
    status, num_iter = _lcp_lemke_tbl(
        tableau, basis, n+s, stop_vars, max_iter, argmins,
        piv_options.tol_piv, piv_options.tol_ratio_diff
    )
    y = np.zeros(n)
    for i in range(n):
        if n <= basis[i] < 2*n:
            y[basis[i]-n] = tableau[i, -1]
    return status, num_iter, y


def _assert_symmetric_nash(A, rho, atol=1e-12):
    payoffs = A @ rho
    assert_((rho >= -atol).all())
    assert_allclose(rho.sum(), 1)
    # Every action in the support is a best response to rho
    assert_(np.all(payoffs[rho > atol] >= payoffs.max() - atol))


class TestLCPLemkeTblSchemeII:
    def test_rock_paper_scissors(self):
        # Unique symmetric equilibrium (1/3, 1/3, 1/3); no pure strategy
        # is an equilibrium, so the path needs more than one step
        A = np.array([[2, 1, 3],
                      [3, 2, 1],
                      [1, 3, 2]], dtype=float)
        n = A.shape[0]
        for s in range(n):
            status, num_iter, y = _lemke_path(A, s)
            assert_equal(status, 0)
            assert_(num_iter > 1)
            assert_allclose(y / y.sum(), np.ones(n) / n)

            # Iteration limit
            status, num_iter, _ = _lemke_path(A, s, max_iter=1)
            assert_equal(status, 1)
            assert_equal(num_iter, 1)

    def test_pure_equilibrium_one_step(self):
        # Action 0 is a strict symmetric equilibrium: the path from it
        # ends after a single pivot (slack 0 leaves)
        A = np.array([[3, 1],
                      [1, 2]], dtype=float)
        status, num_iter, y = _lemke_path(A, 0)
        assert_equal(status, 0)
        assert_equal(num_iter, 1)
        assert_allclose(y / y.sum(), [1, 0])

    def test_random_games_vs_lemke_howson(self):
        # The Lemke path from s coincides with the Lemke-Howson path of
        # the imitation game (A, I) with initial pivot s, projected onto
        # the imitator's strategy
        n = 6
        rng = np.random.RandomState(1234)
        for trial in range(5):
            A = rng.uniform(1, 2, size=(n, n))
            g = NormalFormGame((Player(A), Player(np.eye(n))))
            for s in range(n):
                status, _, y = _lemke_path(A, s)
                assert_equal(status, 0)
                rho = y / y.sum()
                _assert_symmetric_nash(A, rho)
                NE = lemke_howson(g, init_pivot=s)
                assert_allclose(rho, NE[1], atol=1e-12)

    def test_view_of_larger_buffer(self):
        # The driver works on a non-contiguous view, as used by callers
        # that keep a growing tableau in a preallocated buffer
        n = 5
        rng = np.random.RandomState(0)
        A = rng.uniform(1, 2, size=(n, n))
        buf = np.empty((n+3, 2*n+4))
        for s in range(n):
            status, num_iter, y = _lemke_path(A, s)
            status_v, num_iter_v, y_v = _lemke_path(A, s, buf=buf)
            assert_equal(status_v, status)
            assert_equal(num_iter_v, num_iter)
            assert_allclose(y_v, y)
