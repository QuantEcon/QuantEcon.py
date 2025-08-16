"""
Compute an approximate fixed point of a given operator T, starting from
specified initial condition v.

"""
import time
import warnings
import numpy as np
from numba import jit, types
from numba.extending import overload
from .game_theory.lemke_howson import _lemke_howson_tbl, _get_mixed_actions


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


def _is_approx_fp(T, v, error_tol, *args, **kwargs):
    error = np.max(np.abs(T(v, *args, **kwargs) - v))
    return error <= error_tol


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
    their Nash equilibria, computed by the Lemke-Howson algorithm
    routine. It finds an approximate fixed point of `T`, a point
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
        `method='iteration' and `v` is an array
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
        is_approx_fp = \
            lambda v: _is_approx_fp(T, v, error_tol, *args, **kwargs)
        v_star, converged, iterate = \
             _compute_fixed_point_ig(T, v, max_iter, verbose, print_skip,
                                     is_approx_fp, *args, **kwargs)
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


def _compute_fixed_point_ig(T, v, max_iter, verbose, print_skip, is_approx_fp,
                            *args, **kwargs):
    """
    Implement the imitation game algorithm by McLennan and Tourky (2006)
    for computing an approximate fixed point of `T`.

    Parameters
    ----------
    is_approx_fp : callable
        A callable with signature `is_approx_fp(v)` which determines
        whether `v` is an approximate fixed point with a bool return
        value (i.e., True or False)

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
    y_new = T(x_new, *args, **kwargs)
    iterate = 1
    converged = is_approx_fp(x_new)

    if converged or iterate >= max_iter:
        if verbose == 2:
            error = np.max(np.abs(y_new - x_new))
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
        error = np.max(np.abs(y_new - x_new))
        etime = time.time() - start_time
        _print_after_skip(print_skip, iterate, error, etime)

    # Length of the arrays to store the computed sequences of x and y.
    # If exceeded, reset to min(max_iter, buff_size*2).
    buff_size = 2**8
    buff_size = min(max_iter, buff_size)

    shape = (buff_size,) + np.asarray(x_new).shape
    X, Y = np.empty(shape), np.empty(shape)
    X[0], Y[0] = x_new, y_new
    x_new = Y[0]

    tableaux = tuple(np.empty((buff_size, buff_size*2+1)) for i in range(2))
    bases = tuple(np.empty(buff_size, dtype=int) for i in range(2))
    max_piv = 10**6  # Max number of pivoting steps in _lemke_howson_tbl

    while True:
        y_new = T(x_new, *args, **kwargs)
        iterate += 1
        converged = is_approx_fp(x_new)

        if converged or iterate >= max_iter:
            break

        if verbose == 2:
            error = np.max(np.abs(y_new - x_new))
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

            tableaux = tuple(np.empty((buff_size, buff_size*2+1))
                             for i in range(2))
            bases = tuple(np.empty(buff_size, dtype=int) for i in range(2))

        m = iterate
        tableaux_curr = tuple(tableau[:m, :2*m+1] for tableau in tableaux)
        bases_curr = tuple(basis[:m] for basis in bases)
        _initialize_tableaux_ig(X[:m], Y[:m], tableaux_curr, bases_curr)
        converged, num_iter = _lemke_howson_tbl(
            tableaux_curr, bases_curr, init_pivot=m-1, max_iter=max_piv
        )
        _, rho = _get_mixed_actions(tableaux_curr, bases_curr)

        if Y.ndim <= 2:
            x_new = rho.dot(Y[:m])
        else:
            shape_Y = Y.shape
            Y_2d = Y.reshape(shape_Y[0], np.prod(shape_Y[1:]))
            x_new = rho.dot(Y_2d[:m]).reshape(shape_Y[1:])

    if verbose == 2:
        error = np.max(np.abs(y_new - x_new))
        etime = time.time() - start_time
        print_skip = 1
        _print_after_skip(print_skip, iterate, error, etime)
    if verbose >= 1:
        if not converged:
            warnings.warn(_non_convergence_msg, RuntimeWarning)
        elif verbose == 2:
            print(_convergence_msg.format(iterate=iterate))

    return x_new, converged, iterate


@jit(nopython=True)
def _initialize_tableaux_ig(X, Y, tableaux, bases):
    """
    Given sequences `X` and `Y` of ndarrays, initialize the tableau and
    basis arrays in place for the "geometric" imitation game as defined
    in McLennan and Tourky (2006), to be passed to `_lemke_howson_tbl`.

    Parameters
    ----------
    X, Y : ndarray(float)
        Arrays of the same shape (m, n).

    tableaux : tuple(ndarray(float, ndim=2))
        Tuple of two arrays to be used to store the tableaux, of shape
        (2m, 2m). Modified in place.

    bases : tuple(ndarray(int, ndim=1))
        Tuple of two arrays to be used to store the bases, of shape
        (m,). Modified in place.

    Returns
    -------
    tableaux : tuple(ndarray(float, ndim=2))
        View to `tableaux`.

    bases : tuple(ndarray(int, ndim=1))
        View to `bases`.

    """
    m = X.shape[0]
    min_ = np.zeros(m)

    # Mover
    for i in range(m):
        for j in range(2*m):
            if j == i or j == i + m:
                tableaux[0][i, j] = 1
            else:
                tableaux[0][i, j] = 0
        # Right hand side
        tableaux[0][i, 2*m] = 1

    # Imitator
    for i in range(m):
        # Slack variables
        for j in range(m):
            if j == i:
                tableaux[1][i, j] = 1
            else:
                tableaux[1][i, j] = 0
        # Payoff variables
        for j in range(m):
            d = X[i] - Y[j]
            tableaux[1][i, m+j] = _square_sum(d) * (-1)
            if tableaux[1][i, m+j] < min_[j]:
                min_[j] = tableaux[1][i, m+j]
        # Right hand side
        tableaux[1][i, 2*m] = 1
    # Shift the payoff values
    for i in range(m):
        for j in range(m):
            tableaux[1][i, m+j] -= min_[j]
            tableaux[1][i, m+j] += 1

    for pl, start in enumerate([m, 0]):
        for i in range(m):
            bases[pl][i] = start + i

    return tableaux, bases


def _square_sum(a):  # pragma: no cover
    pass


@overload(_square_sum, jit_options={'cache':True})
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
