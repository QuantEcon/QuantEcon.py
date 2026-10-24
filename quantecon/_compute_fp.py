"""
Compute an approximate fixed point of a given operator T, starting from
specified initial condition v.

"""
import time
import warnings
import numpy as np
from numba import jit, types
from numba.extending import overload
from .optimize.lcp_lemke import _lcp_lemke_tbl
from .optimize.pivoting import TOL_PIV, TOL_RATIO_DIFF


def _print_after_skip(skip, it=None, dist=None, etime=None):
    if it is None:
        # print initial header
        msg = "{i:<13}{d:<15}{t:<17}".format(i="Iteration",
                                             d="Distance",
                                             t="Elapsed (seconds)")
        print(msg)
        print("-" * len(msg))

        return

    if it % skip == 0:
        if etime is None:
            print("After {it} iterations dist is {d}".format(it=it, d=dist))

        else:
            # leave 4 spaces between columns if we have %3.3e in d, t
            msg = "{i:<13}{d:<15.3e}{t:<18.3e}"
            print(msg.format(i=it, d=dist, t=etime))

    return


_convergence_msg = 'Converged in {iterate} steps'
_non_convergence_msg = \
    'max_iter attained before convergence in compute_fixed_point'
_breakdown_msg = \
    'numerical breakdown in the imitation game in compute_fixed_point'
_pivot_limit_msg = \
    'pivot limit reached in the imitation game in compute_fixed_point'

# Initial length of the buffers for the computed sequences of x and y in
# the imitation game method; doubled, up to max_iter, when exceeded
_BUFF_SIZE = 2**8


def compute_fixed_point(T, v, error_tol=1e-3, max_iter=50, verbose=2,
                        print_skip=5, method='iteration', *args, **kwargs):
    r"""
    Computes and returns an approximate fixed point of the function `T`.

    The default method `'iteration'` simply iterates the function given
    an initial condition `v` and returns :math:`T^k v` when the
    condition :math:`\lVert T^k v - T^{k-1} v\rVert \leq
    \mathrm{error\_tol}` is satisfied or the number of iterations
    :math:`k` reaches `max_iter`. Provided that `T` is a contraction
    mapping or similar, :math:`T^k v` will be an approximation to the
    fixed point.

    The method `'imitation_game'` uses the "imitation game algorithm"
    developed by McLennan and Tourky [1]_, which internally constructs
    a sequence of two-player games called imitation games and utilizes
    their Nash equilibria, computed by the Lemke paths algorithm (the
    Lemke-Howson algorithm on the imitator's polytope). It finds an
    approximate fixed point of `T`, a point
    :math:`v^*` such that :math:`\lVert T(v) - v\rVert \leq
    \mathrm{error\_tol}`, provided `T` is a function that satisfies the
    assumptions of Brouwer's fixed point theorem, i.e., a continuous
    function that maps a compact and convex set to itself.

    Parameters
    ----------
    T : callable
        A callable object (e.g., function) that acts on v
    v : object
        An object such that T(v) is defined; modified in place if
        `method='iteration'` and `v` is an array
    error_tol : scalar(float), optional(default=1e-3)
        Error tolerance
    max_iter : scalar(int), optional(default=50)
        Maximum number of iterations
    verbose : scalar(int), optional(default=2)
        Level of feedback (0 for no output, 1 for warnings only, 2 for
        warning and residual error reports during iteration)
    print_skip : scalar(int), optional(default=5)
        How many iterations to apply between print messages (effective
        only when `verbose=2`)
    method : str, optional(default='iteration')
        str in {'iteration', 'imitation_game'}. Method of computing
        an approximate fixed point
    args, kwargs :
        Other arguments and keyword arguments that are passed directly
        to  the function T each time it is called

    Returns
    -------
    v : object
        The approximate fixed point

    References
    ----------
    .. [1] A. McLennan and R. Tourky, "From Imitation Games to
       Kakutani," 2006.

    """
    if max_iter < 1:
        raise ValueError('max_iter must be a positive integer')

    if verbose not in (0, 1, 2):
        raise ValueError('verbose should be 0, 1 or 2')

    if method not in ['iteration', 'imitation_game']:
        raise ValueError('invalid method')

    if method == 'imitation_game':
        def oracle(x):
            y = T(x, *args, **kwargs)
            return y, np.max(np.abs(y - x))

        v_star, converged, iterate = \
            _compute_fixed_point_ig(oracle, v, error_tol, max_iter, verbose,
                                    print_skip)
        return v_star

    # method == 'iteration'
    iterate = 0

    if verbose == 2:
        start_time = time.time()
        _print_after_skip(print_skip, it=None)

    while True:
        new_v = T(v, *args, **kwargs)
        iterate += 1
        error = np.max(np.abs(new_v - v))

        try:
            v[:] = new_v
        except TypeError:
            v = new_v

        if error <= error_tol or iterate >= max_iter:
            break

        if verbose == 2:
            etime = time.time() - start_time
            _print_after_skip(print_skip, iterate, error, etime)

    if verbose == 2:
        etime = time.time() - start_time
        print_skip = 1
        _print_after_skip(print_skip, iterate, error, etime)
    if verbose >= 1:
        if error > error_tol:
            warnings.warn(_non_convergence_msg, RuntimeWarning)
        elif verbose == 2:
            print(_convergence_msg.format(iterate=iterate))

    return v


def _compute_fixed_point_ig(oracle, v, error_tol, max_iter, verbose,
                            print_skip):
    """
    Implement the imitation game algorithm by McLennan and Tourky (2006)
    for computing an approximate fixed point of a map `T`, given through
    `oracle`.

    Parameters
    ----------
    oracle : callable
        A callable with signature `oracle(x)` returning the pair `(y,
        error)`, where `y = T(x)` and `error` is the scalar measure of
        the distance of `x` from being a fixed point against which
        `error_tol` is compared (e.g., `np.max(np.abs(y - x))`). It is
        called once per iteration, so that quantities shared by `y` and
        `error` need to be computed only once.

    error_tol : scalar(float)
        Error tolerance: the iteration stops at the first `x` with
        `error <= error_tol`.

    For the other parameters, see Parameters in compute_fixed_point.

    Returns
    -------
    x_new : scalar(float) or ndarray(float)
        Approximate fixed point.

    converged : bool
        Whether the routine has converged.

    iterate : scalar(int)
        Number of iterations.

    """
    if verbose == 2:
        start_time = time.time()
        _print_after_skip(print_skip, it=None)

    x_new = v
    y_new, error = oracle(x_new)
    iterate = 1
    converged = error <= error_tol

    if converged or iterate >= max_iter:
        if verbose == 2:
            etime = time.time() - start_time
            print_skip = 1
            _print_after_skip(print_skip, iterate, error, etime)
        if verbose >= 1:
            if not converged:
                warnings.warn(_non_convergence_msg, RuntimeWarning)
            elif verbose == 2:
                print(_convergence_msg.format(iterate=iterate))
        return x_new, converged, iterate

    if verbose == 2:
        etime = time.time() - start_time
        _print_after_skip(print_skip, iterate, error, etime)

    # Length of the arrays to store the computed sequences of x and y.
    # If exceeded, reset to min(max_iter, buff_size*2).
    buff_size = min(max_iter, _BUFF_SIZE)

    shape = (buff_size,) + np.asarray(x_new).shape
    X, Y = np.empty(shape), np.empty(shape)
    X[0], Y[0] = x_new, y_new
    x_new = Y[0]

    # Squared distances D[i, j] = ||X[i] - Y[j]||^2, extended by one row
    # and one column per new point, and their running maximum, by which
    # the payoffs of the imitation game are scaled
    D = np.empty((buff_size, buff_size))
    d_max = _update_distances(X, Y, 0, D)

    tableau = np.empty((buff_size, buff_size*2+1))
    basis = np.empty(buff_size, dtype=int)
    argmins = np.empty(buff_size, dtype=int)
    z = np.empty(buff_size)
    stop_vars = np.empty(2, dtype=int)
    max_piv = 10**6  # Max number of pivoting steps in _lcp_lemke_tbl
    status = 0

    while True:
        y_new, error = oracle(x_new)
        iterate += 1
        converged = error <= error_tol

        if converged or iterate >= max_iter:
            break

        if verbose == 2:
            etime = time.time() - start_time
            _print_after_skip(print_skip, iterate, error, etime)

        try:
            X[iterate-1] = x_new
            Y[iterate-1] = y_new
        except IndexError:
            buff_size = min(max_iter, buff_size*2)
            shape = (buff_size,) + X.shape[1:]
            X_tmp, Y_tmp = X, Y
            X, Y = np.empty(shape), np.empty(shape)
            X[:X_tmp.shape[0]], Y[:Y_tmp.shape[0]] = X_tmp, Y_tmp
            X[iterate-1], Y[iterate-1] = x_new, y_new

            D_tmp = D
            D = np.empty((buff_size, buff_size))
            D[:iterate-1, :iterate-1] = D_tmp[:iterate-1, :iterate-1]

            tableau = np.empty((buff_size, buff_size*2+1))
            basis = np.empty(buff_size, dtype=int)
            argmins = np.empty(buff_size, dtype=int)
            z = np.empty(buff_size)

        m = iterate
        d_max = max(d_max, _update_distances(X, Y, m-1, D))
        scale = d_max if d_max > 0 else 1.

        if _newest_is_equilibrium(D[:m, m-1], scale, z[:m],
                                  TOL_PIV, TOL_RATIO_DIFF):
            # Fast path: the Lemke path from the newest point ends after
            # its first pivot, with rho = delta_{m-1}, so that the next
            # iterate is the image of the newest point
            x_new = Y[m-1].copy()
            continue

        tableau_curr, basis_curr = tableau[:m, :2*m+1], basis[:m]
        _initialize_tableau_ig(D[:m, :m], scale, tableau_curr, basis_curr)
        # Lemke path from the pure strategy of the newest point (index
        # m-1): its variable enters, and the path ends when either
        # variable of its complementary pair leaves
        stop_vars[0], stop_vars[1] = m-1, 2*m-1
        status, _ = _lcp_lemke_tbl(
            tableau_curr, basis_curr, 2*m-1, stop_vars, max_piv, argmins,
            TOL_PIV, TOL_RATIO_DIFF
        )
        if status != 0:  # The tableau does not hold an equilibrium
            break
        rho = _get_mixed_action(tableau_curr, basis_curr, z[:m])

        if Y.ndim <= 2:
            x_new = rho @ Y[:m]
        else:
            shape_Y = Y.shape
            Y_2d = Y.reshape(shape_Y[0], np.prod(shape_Y[1:]))
            x_new = (rho @ Y_2d[:m]).reshape(shape_Y[1:])

    if verbose == 2:
        etime = time.time() - start_time
        print_skip = 1
        _print_after_skip(print_skip, iterate, error, etime)
    if verbose >= 1:
        if status != 0:
            msg = _breakdown_msg if status in (2, 3) else _pivot_limit_msg
            warnings.warn(msg, RuntimeWarning)
        elif not converged:
            warnings.warn(_non_convergence_msg, RuntimeWarning)
        elif verbose == 2:
            print(_convergence_msg.format(iterate=iterate))

    return x_new, converged, iterate


@jit(nopython=True)
def _update_distances(X, Y, k, D):
    """
    Given sequences `X` and `Y` of ndarrays, store in place the squared
    distances between the point `k` and the points `0, ..., k`, in row
    and column `k` of `D`: `D[k, j] = ||X[k] - Y[j]||^2` for `j <= k`
    and `D[i, k] = ||X[i] - Y[k]||^2` for `i < k`.

    Parameters
    ----------
    X, Y : ndarray(float)
        Arrays of the same shape (m, n), with m > k.

    k : scalar(int)
        Index of the new point.

    D : ndarray(float, ndim=2)
        Array of shape (m, m) holding the squared distances in
        `D[:k, :k]`. Modified in place.

    Returns
    -------
    d_max : scalar(float)
        Maximum of the new entries.

    """
    d_max = 0.
    for j in range(k+1):
        D[k, j] = _square_sum(X[k] - Y[j])
        if D[k, j] > d_max:
            d_max = D[k, j]
    for i in range(k):
        D[i, k] = _square_sum(X[i] - Y[k])
        if D[i, k] > d_max:
            d_max = D[i, k]

    return d_max


@jit(nopython=True, cache=True)
def _newest_is_equilibrium(d, scale, work, tol_piv, tol_ratio_diff):
    """
    Determine whether the pure strategy of the newest point, the last
    of m points, is an equilibrium of the imitation game, by deciding
    the first ratio test of the Lemke path from it without forming the
    tableau: with `d` the column of the squared distances from the m
    points to the image of the newest point, the test enters that
    point's payoff column and selects the row of the nearest point,
    ties going to the largest index. The payoffs are computed from `d`
    by the same operations as in `_initialize_tableau_ig`, and the test
    as in `_lex_min_ratio_test` with a right-hand side of ones, so that
    the decision coincides with that of the pivoting.

    Parameters
    ----------
    d : ndarray(float, ndim=1)
        Array of shape (m,), the last column of the squared distances.

    scale : scalar(float)
        Positive number by which the payoffs are divided.

    work : ndarray(float, ndim=1)
        Workspace array of shape (m,). Modified in place.

    tol_piv : scalar(float)
        Pivot tolerance below which a number is considered to be
        nonpositive.

    tol_ratio_diff : scalar(float)
        Tolerance to determine a tie between ratio values.

    Returns
    -------
    bool
        True if the first ratio test would select the row of the newest
        point, in which case the pivoting would end there with the pure
        strategy of the newest point as the equilibrium.

    """
    m = d.shape[0]
    inv_scale = 1 / scale

    # Payoff column, as in `_initialize_tableau_ig`
    min_ = 0.
    for i in range(m):
        work[i] = -d[i] * inv_scale
        if work[i] < min_:
            min_ = work[i]
    for i in range(m):
        work[i] -= min_
        work[i] += 1

    # Minimum ratio test with right-hand side 1, as in
    # `_min_ratio_test_no_tie_breaking`
    ratio_min = np.inf
    for i in range(m):
        denom = work[i]
        if denom <= tol_piv:
            continue
        ratio = 1. / denom
        if ratio < ratio_min:
            ratio_min = ratio

    # The lexicographic tie breaking selects the largest row index among
    # the rows within `tol_ratio_diff` of the minimum ratio, so the
    # newest point's row is selected if and only if it is one of them
    denom = work[m-1]
    if denom <= tol_piv:
        return False
    return 1. / denom <= ratio_min + tol_ratio_diff


@jit(nopython=True, cache=True)
def _initialize_tableau_ig(D, scale, tableau, basis):
    """
    Given the squared distances `D` between the points of the sequences
    `X` and `Y`, initialize the tableau and basis arrays in place for
    the Lemke path on the imitator's polytope of the "geometric"
    imitation game as defined in McLennan and Tourky (2006), to be
    passed to `_lcp_lemke_tbl`.

    The game has m pure strategies, one per point. Row i of the tableau
    corresponds to the mover's pure strategy i, whose payoff against the
    imitator's pure strategy j is `-D[i, j]`. The payoffs are divided by
    `scale` and shifted column by column so that the minimum of each
    column is 1; with `scale` the maximum of `D`, the entries lie in
    [1, 2] whatever the magnitude of the distances, which keeps the
    pivoting tolerances meaningful. Neither transformation changes the
    mover's best responses, hence the Lemke path.

    * Columns 0, ..., m-1 are those of the slack variables, forming the
      identity matrix: the slack of row i is the mover's payoff loss
      from strategy i, zero if and only if i is a best response.

    * Columns m, ..., 2m-1 are those of the imitator's pure strategies,
      holding the shifted payoffs.

    * Column 2m contains the values of the basic variables, initially
      1.

    * `basis` contains the basic variables, initially the slacks 0, ...,
      m-1.

    Parameters
    ----------
    D : ndarray(float, ndim=2)
        Array of shape (m, m) of squared distances, `D[i, j] = ||X[i] -
        Y[j]||^2`.

    scale : scalar(float)
        Positive number by which the payoffs are divided.

    tableau : ndarray(float, ndim=2)
        Array to be used to store the tableau, of shape (m, 2m+1).
        Modified in place.

    basis : ndarray(int, ndim=1)
        Array to be used to store the basis, of shape (m,). Modified in
        place.

    Returns
    -------
    tableau : ndarray(float, ndim=2)
        View to `tableau`.

    basis : ndarray(int, ndim=1)
        View to `basis`.

    """
    m = D.shape[0]
    min_ = np.zeros(m)
    inv_scale = 1 / scale

    for i in range(m):
        # Slack variables
        for j in range(m):
            if j == i:
                tableau[i, j] = 1
            else:
                tableau[i, j] = 0
        # Payoff variables
        for j in range(m):
            tableau[i, m+j] = -D[i, j] * inv_scale
            if tableau[i, m+j] < min_[j]:
                min_[j] = tableau[i, m+j]
        # Right hand side
        tableau[i, 2*m] = 1
    # Shift the payoff values
    for i in range(m):
        for j in range(m):
            tableau[i, m+j] -= min_[j]
            tableau[i, m+j] += 1

    for i in range(m):
        basis[i] = i

    return tableau, basis


@jit(nopython=True, cache=True)
def _get_mixed_action(tableau, basis, out):
    """
    From `tableau` and `basis` of the imitation game, extract the
    imitator's basic variables and store the corresponding normalized
    mixed action in `out`, in place.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        Array of shape (m, 2m+1) containing the terminal tableau.

    basis : ndarray(int, ndim=1)
        Array of shape (m,) containing the terminal basis.

    out : ndarray(float, ndim=1)
        Array of shape (m,) to store the mixed action. Modified in
        place.

    Returns
    -------
    out : ndarray(float, ndim=1)
        View to `out`.

    """
    m = basis.shape[0]

    out[:] = 0
    sum_ = 0.
    for i in range(m):
        k = basis[i]
        if m <= k < 2*m:
            out[k-m] = tableau[i, -1]
            sum_ += tableau[i, -1]
    if sum_ != 0:
        out /= sum_

    return out


def _square_sum(a):  # pragma: no cover
    pass


@overload(_square_sum, jit_options={'cache': True})
def _square_sum_ol(a):
    if isinstance(a, types.Number):
        return lambda a: a**2
    elif isinstance(a, types.Array):
        return _square_sum_array


def _square_sum_array(a):  # pragma: no cover
    sum_ = 0
    for x in a.flat:
        sum_ += x**2
    return sum_
