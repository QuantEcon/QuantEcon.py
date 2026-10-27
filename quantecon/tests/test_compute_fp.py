"""
Tests for compute_fp.py

References
----------

https://www.math.ucdavis.edu/~hunter/book/ch3.pdf

TODO: add multivariate case

"""
import warnings
import numpy as np
from numpy.testing import (
    assert_, assert_allclose, assert_array_equal, assert_raises
)
from quantecon import compute_fixed_point
import quantecon._compute_fp
from quantecon._compute_fp import (
    _compute_fixed_point_ig, _initialize_tableau_ig, _update_distances,
    _newest_is_equilibrium, _TOL_PIV
)
from quantecon.optimize.pivoting import _lex_min_ratio_test, TOL_RATIO_DIFF


class TestFPLogisticEquation():

    @classmethod
    def setup_method(cls):
        cls.mu_1 = 0.2  # 0 is unique fixed point forall x_0 \in [0, 1]

        # (4mu - 1)/(4mu) is a fixed point forall x_0 \in [0, 1]
        cls.mu_2 = 0.3

        # starting points on (0, 1)
        cls.unit_inverval = [0.1, 0.3, 0.6, 0.9]

        # arguments for compute_fixed_point
        cls.kwargs = {"error_tol": 1e-5, "max_iter": 200, "verbose": 0}

    def T(self, x, mu):
        return 4.0 * mu * x * (1.0 - x)

    def test_contraction_1(self):
        "compute_fp: convergence inside interval of convergence"
        f = lambda x: self.T(x, self.mu_1)
        for i in self.unit_inverval:
            # should have fixed point of 0.0
            assert_(abs(compute_fixed_point(f, i, **self.kwargs)) < 1e-4)

    def test_not_contraction_2(self):
        "compute_fp: no convergence outside interval of convergence"
        f = lambda x: self.T(x, self.mu_2)
        for i in self.unit_inverval:
            # This shouldn't converge to 0.0
            assert_(not (abs(compute_fixed_point(f, i, **self.kwargs)) < 1e-4))

    def test_contraction_2(self):
        "compute_fp: convergence inside interval of convergence"
        f = lambda x: self.T(x, self.mu_2)
        fp = (4 * self.mu_2 - 1) / (4 * self.mu_2)
        for i in self.unit_inverval:
            # This should converge to fp
            assert_(abs(compute_fixed_point(f, i, **self.kwargs)-fp) < 1e-4)

    def test_not_contraction_1(self):
        "compute_fp: no convergence outside interval of convergence"
        f = lambda x: self.T(x, self.mu_1)
        fp = (4 * self.mu_1 - 1) / (4 * self.mu_1)
        for i in self.unit_inverval:
            # This should not converge  (b/c unique fp is 0.0)
            assert_(not (abs(compute_fixed_point(f, i, **self.kwargs)-fp)
                    < 1e-4))

    def test_imitation_game_method(self):
        "compute_fp: Test imitation game method"
        method = 'imitation_game'
        error_tol = self.kwargs['error_tol']

        for mu in [self.mu_1, self.mu_2]:
            for i in self.unit_inverval:
                fp_computed = compute_fixed_point(self.T, i, method=method,
                                                  mu=mu, **self.kwargs)
                assert_(
                    abs(self.T(fp_computed, mu=mu) - fp_computed) <= error_tol
                )

            # numpy array input
            i = np.asarray(self.unit_inverval)
            fp_computed = compute_fixed_point(self.T, i, method=method, mu=mu,
                                              **self.kwargs)
            assert_(
                abs(self.T(fp_computed, mu=mu) - fp_computed).max() <=
                error_tol
            )


class TestComputeFPContraction():
    def setup_method(self):
        self.coeff = 0.5
        self.methods = ['iteration', 'imitation_game']

    def f(self, x):
        return self.coeff * x

    def test_num_iter_one(self):
        init = 1.
        error_tol = self.coeff

        for method in self.methods:
            fp_computed = compute_fixed_point(self.f, init,
                                              error_tol=error_tol,
                                              method=method)
            assert_(fp_computed <= error_tol * 2)

    def test_num_iter_large(self):
        init = 1.
        buff_size = 2**8  # buff_size in 'imitation_game'
        max_iter = buff_size + 2
        error_tol = self.coeff**max_iter

        for method in self.methods:
            fp_computed = compute_fixed_point(self.f, init,
                                              error_tol=error_tol,
                                              max_iter=max_iter, method=method,
                                              print_skip=max_iter)
            assert_(fp_computed <= error_tol * 2)

    def test_2d_input(self):
        error_tol = self.coeff**4

        for method in self.methods:
            init = np.array([[-1, 0.5], [-1/3, 0.1]])
            fp_computed = compute_fixed_point(self.f, init,
                                              error_tol=error_tol,
                                              method=method)
            assert_((fp_computed <= error_tol * 2).all())


def test_raises_value_error_nonpositive_max_iter():
    f = lambda x: 0.5*x
    init = 1.
    max_iter = 0
    assert_raises(ValueError, compute_fixed_point, f, init, max_iter=max_iter)


def test_imitation_game_large_scale():
    # Values of order 1e7: before the payoffs were scaled, the pivoting
    # on the imitation game broke down numerically (squared distances of
    # order 1e14 next to the identity block of the tableau). The map,
    # a nearest-recorded-point lookup, has no fixed point, so the
    # routine must run to max_iter and report non-convergence only
    X = np.array([-43904436.689917795, -58130594.69589446,
                  -51017515.692906134])
    Y = np.array([-58130594.69589446, 73372551.04835357,
                  45264022.10520259])
    max_iter = 10
    evaluations = []

    def T(x):
        evaluations.append(x)
        return Y[np.argmin(np.abs(X - x))]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        x = compute_fixed_point(T, X[0], error_tol=1e-3, max_iter=max_iter,
                                verbose=1, method='imitation_game')
    messages = [str(wi.message) for wi in w]
    assert_(not any('breakdown' in msg for msg in messages))
    assert_(any('max_iter' in msg for msg in messages))
    assert_(len(evaluations) == max_iter)
    assert_(np.isfinite(x))


def test_imitation_game_tiny_scale():
    # Squared distances below the smallest normal float: the scale of the
    # payoffs is clamped so that its reciprocal does not overflow (which
    # produced NaN payoffs and a spurious numerical breakdown), and the
    # contraction converges as at any other scale
    evaluations = []

    def T(x):
        evaluations.append(x)
        return 0.5 * x

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        x = compute_fixed_point(T, 1e-155, error_tol=1e-160, max_iter=100,
                                verbose=1, method='imitation_game')
    assert_(len(w) == 0, [str(wi.message) for wi in w])
    assert_(abs(T(x) - x) <= 1e-160)
    # Same path as plain iteration: 17 steps from 1e-155 to a point with
    # residual 1e-155 * 2**-17 <= 1e-160, one evaluation of T per step
    assert_(len(evaluations) - 1 == 17)


def test_imitation_game_far_start():
    # Iterates far apart before they cluster: the payoffs are the squared
    # distances divided by their maximum, so the pivots that the Lemke
    # path needs near the fixed point are of order (error / largest
    # distance)**2, here about 1e-11, and were excluded by the shared
    # pivot tolerance 1e-10, which stalled the iteration at a residual of
    # 3e-3 (the iterate repeated itself) although the map is a continuous
    # self-map of [-1000, 1000] with fixed point 0
    evaluations = []

    def T(x):
        evaluations.append(x)
        return -1000 * np.sin(1.1 * x / 1000)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        x = compute_fixed_point(T, 300., error_tol=1e-3, max_iter=100,
                                verbose=1, method='imitation_game')
    assert_(len(w) == 0, [str(wi.message) for wi in w])
    assert_(abs(T(x) - x) <= 1e-3)
    assert_(len(evaluations) - 1 <= 20)  # 11 iterations


def test_imitation_game_tight_tolerance():
    # Absolute tolerance 1e-6 from a start at distance of order 1e2 from
    # the fixed point, on a map that is not a contraction in the sup norm:
    # the recorded points sit at every intermediate scale, and the path
    # needs pivots between 1e-14 and the shared pivot tolerance 1e-10,
    # with which the iteration stalled at every coordinate scale
    theta = 0.7
    R = 0.95 * np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    c = np.array([1., -2.])
    x_star = np.linalg.solve(np.eye(2) - R, c)
    evaluations = []

    def T(x):
        evaluations.append(x)
        return R @ x + c

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        x = compute_fixed_point(T, np.array([100., 100.]), error_tol=1e-6,
                                max_iter=300, verbose=1,
                                method='imitation_game')
    assert_(len(w) == 0, [str(wi.message) for wi in w])
    assert_(np.max(np.abs(T(x) - x)) <= 1e-6)
    assert_allclose(x, x_star, atol=1e-5)
    assert_(len(evaluations) - 1 <= 150)  # 111 iterations


def test_imitation_game_driver_failure_warnings(monkeypatch):
    # A nonzero status of the pivoting driver stops the iteration with a
    # warning naming the cause, and the last iterate is returned
    T = lambda x: 4 * x * (1 - x)  # Chaotic: pivoting always needed
    for status, keyword in [(3, 'numerical breakdown'),
                            (2, 'numerical breakdown'),
                            (1, 'pivot limit')]:
        evaluations = []

        def _failing_tbl(*args, **kwargs):
            return status, 0

        def T_counting(x):
            evaluations.append(x)
            return T(x)

        monkeypatch.setattr(quantecon._compute_fp, '_lcp_lemke_tbl',
                            _failing_tbl)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            x = compute_fixed_point(T_counting, 0.3, error_tol=0,
                                    max_iter=10, verbose=1,
                                    method='imitation_game')
        messages = [str(wi.message) for wi in w]
        assert_(any(keyword in msg for msg in messages), messages)
        assert_(not any('max_iter' in msg for msg in messages))
        assert_(len(evaluations) == 2)  # Stopped at the first pivoting
        assert_(x == T(0.3))


def test_initialize_tableau_ig_layout():
    # [I | shifted payoffs | 1] with the slacks basic: the mover's payoff
    # from point i against point j is -||X[i] - Y[j]||^2, shifted column
    # by column so that each column has minimum 1
    rng = np.random.RandomState(0)
    m, n = 4, 3
    X, Y = rng.standard_normal((m, n)), rng.standard_normal((m, n))
    D = np.empty((m, m))
    for k in range(m):  # One point at a time, as in the iteration
        _update_distances(X, Y, k, D)
    assert_allclose(D, ((X[:, np.newaxis, :] - Y[np.newaxis, :, :])**2)
                    .sum(axis=-1))

    tableau = np.empty((m, 2*m+1))
    basis = np.empty(m, dtype=int)
    for scale in [1., D.max()]:
        _initialize_tableau_ig(D, scale, tableau, basis)
        assert_array_equal(tableau[:, :m], np.eye(m))
        assert_allclose(tableau[:, m:2*m], (D.max(axis=0) - D) / scale + 1)
        assert_array_equal(tableau[:, 2*m], np.ones(m))
        assert_array_equal(basis, np.arange(m))
    # With the maximum as the scale, the payoff entries lie in [1, 2]
    assert_((tableau[:, m:2*m] >= 1).all() and (tableau[:, m:2*m] <= 2).all())

    # Subnormal distances: with the scale clamped at the smallest normal
    # float, as in the iteration, the entries are finite and in [1, 2]
    D_sub = D * 1e-310
    assert_(D_sub.max() < np.finfo(np.float64).tiny)
    scale = max(D_sub.max(), np.finfo(np.float64).tiny)
    _initialize_tableau_ig(D_sub, scale, tableau, basis)
    payoffs = tableau[:, m:2*m]
    assert_(np.isfinite(payoffs).all())
    assert_((payoffs >= 1).all() and (payoffs <= 2).all())
    assert_allclose(payoffs, (D_sub.max(axis=0) - D_sub) / scale + 1)


def test_update_distances_scalar_points():
    X = np.array([0., 1., 3.])
    Y = np.array([1., 1., 0.])
    D = np.empty((3, 3))
    for k in range(3):
        _update_distances(X, Y, k, D)
    assert_array_equal(D, (X[:, np.newaxis] - Y[np.newaxis, :])**2)


def test_imitation_game_buffer_growth(monkeypatch):
    # The buffers for the points, the distances and the tableau grow by
    # doubling; the iterates must not depend on the initial buffer size.
    # The chaotic logistic map never converges, so the iteration runs to
    # max_iter and goes through several doublings from a tiny buffer
    T = lambda x: 4 * x * (1 - x)
    kwargs = dict(error_tol=0, max_iter=60, verbose=0,
                  method='imitation_game')
    x_ref = compute_fixed_point(T, 0.3, **kwargs)
    monkeypatch.setattr(quantecon._compute_fp, '_BUFF_SIZE', 3)
    x_small = compute_fixed_point(T, 0.3, **kwargs)
    assert_array_equal(x_small, x_ref)


def test_imitation_game_one_oracle_call_per_iteration():
    # The map is evaluated exactly once per iteration, by the oracle,
    # which also supplies the error used in the convergence test
    calls = []

    def oracle(x):
        calls.append(x)
        y = 0.5 * x
        return y, abs(y - x)

    x, converged, iterate = _compute_fixed_point_ig(
        oracle, 1., error_tol=0.5**10, max_iter=50, verbose=0, print_skip=1
    )
    assert_(converged)
    assert_(len(calls) == iterate)
    assert_(abs(0.5 * x - x) <= 0.5**10)


def test_newest_is_equilibrium_matches_ratio_test():
    # The fast path decides the first ratio test of the Lemke path from
    # the newest point exactly as the pivoting would, including exact
    # ties (decided toward the largest row index) and near ties within
    # the tolerance
    rng = np.random.RandomState(0)
    m = 6
    tableau = np.empty((m, 2*m+1))
    basis = np.empty(m, dtype=int)
    argmins = np.empty(m, dtype=int)
    work = np.empty(m)
    num_fast = 0
    for trial in range(400):
        if trial % 2 == 0:  # Continuous distances
            D = rng.uniform(0, 10, size=(m, m)) ** 2
        else:  # Discrete distances: many exact ties
            D = rng.randint(0, 3, size=(m, m)).astype(float)
        if trial % 4 == 1:  # Near ties with the newest point
            D[rng.randint(m-1), m-1] = D[m-1, m-1] * (1 + 1e-16)
        if trial % 8 == 3:  # Large scale
            D *= 1e14
        scale = D.max() if D.max() > 0 else 1.
        _initialize_tableau_ig(D, scale, tableau, basis)
        found, row, resolved = _lex_min_ratio_test(
            tableau, 2*m-1, 0, argmins, _TOL_PIV, TOL_RATIO_DIFF
        )
        assert_(found and resolved)
        fast = _newest_is_equilibrium(D[:, m-1], scale, work,
                                      _TOL_PIV, TOL_RATIO_DIFF)
        assert_(fast == (row == m-1), (trial, fast, row))
        num_fast += fast
    assert_(0 < num_fast < 400)  # Both outcomes exercised


def test_imitation_game_fast_path_identical(monkeypatch):
    # With the fast path disabled, the pivoting takes the same single
    # step and produces the same iterates to the last bit
    maps = [(lambda x: 0.5 * x, 1., 0.5**20),
            (lambda x: 4 * 0.3 * x * (1 - x), 0.4, 1e-10),
            (lambda x: 4 * x * (1 - x), 0.3, 0.)]  # Chaotic, max_iter
    results = []
    for disabled in [False, True]:
        if disabled:
            monkeypatch.setattr(quantecon._compute_fp,
                                '_newest_is_equilibrium',
                                lambda *args: False)
        results.append([
            compute_fixed_point(T, x0, error_tol=tol, max_iter=60,
                                verbose=0, method='imitation_game')
            for T, x0, tol in maps
        ])
    for x_fast, x_piv in zip(*results):
        assert_array_equal(x_fast, x_piv)


def test_imitation_game_contraction_never_pivots(monkeypatch):
    # On a contraction with factor 1/2 the newest point is always the
    # equilibrium (McLennan and Tourky, Proposition 4.3): the tableau is
    # never formed and the method reduces to iteration
    def _no_pivoting(*args, **kwargs):
        raise AssertionError('pivoting should not be needed')

    monkeypatch.setattr(quantecon._compute_fp, '_lcp_lemke_tbl',
                        _no_pivoting)
    T = lambda x: 0.5 * x + 1
    x = compute_fixed_point(T, np.array([5., -3.]), error_tol=1e-10,
                            max_iter=100, verbose=0,
                            method='imitation_game')
    assert_allclose(x, [2., 2.], atol=1e-9)
