"""
Filename: compute_fp.py
Authors: Thomas Sargent, John Stachurski, Daisuke Oyama

Compute an approximate fixed point of a given operator T, starting from
specified initial condition v.

"""
import time
import warnings
import numpy as np
from numba import jit, generated_jit, types
from .game_theory.lemke_howson import lemke_howson_tbl, get_mixed_actions


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
_non_convergence_msg = 'max_iter attained in compute_fixed_point'


def compute_fixed_point(T, v, error_tol=1e-3, max_iter=50, verbose=2,
                        print_skip=5, method='iteration', *args, **kwargs):
    """
    Computes and returns an approximate fixed point of the function `T`.

    The default method `'iteration'` simply iterates the function given
    an initial condition `v` and returns :math:`T^k v` when the
    condition :math:`\lVert T^k v - T^{k-1} v\rVert \leq
    \mathrm{error_tol}` is satisfied or the number of iterations
    :math:`k` reaches `max_iter`. Provided that `T` is a contraction
    mapping or similar, :math:`T^k v` will be an approximation to the
    fixed point.

    The method `'imitation_game'` uses the "imitation game algorithm"
    developed by McLennan and Tourky [1]_, which internally constructs
    a sequence of two-player games called imitation games and utilizes
    their Nash equilibria, computed by the Lemke-Howson algorithm
    routine. It finds an approximate fixed point of `T`, a point
    :math:`v^*` such that :math:`\lVert T(v) - v\rVert \leq
    \mathrm{error_tol}`, provided `T` is a function that satisfies the
    assumptions of Brouwer's fixed point theorm, i.e., a continuous
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
    method : str in {'iteration', 'imitation_game'},
             optional(default='iteration')
        Method of computing an approximate fixed point
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
    if verbose not in (0, 1, 2):
        raise ValueError('verbose should be 0, 1 or 2')

    if method not in ['iteration', 'imitation_game']:
        raise ValueError('invalid method')

    if method == 'imitation_game':
        return _compute_fixed_point_ig(T, v, error_tol, max_iter, verbose,
                                       print_skip, *args, **kwargs)

    # method == 'iteration'
    iterate = 0
    error = error_tol + 1

    if verbose == 2:
        start_time = time.time()
        _print_after_skip(print_skip, it=None)

    while iterate < max_iter and error > error_tol:
        new_v = T(v, *args, **kwargs)
        iterate += 1
        error = np.max(np.abs(new_v - v))

        if verbose == 2:
            etime = time.time() - start_time
            _print_after_skip(print_skip, iterate, error, etime)

        try:
            v[:] = new_v
        except TypeError:
            v = new_v

    if verbose >= 1:
        if iterate == max_iter:
            warnings.warn(_non_convergence_msg, RuntimeWarning)
        elif verbose == 2:
            print(_convergence_msg.format(iterate=iterate))

    return v


def _compute_fixed_point_ig(T, v, error_tol, max_iter, verbose, print_skip,
                            *args, **kwargs):
    """
    Implement the imitation game algorithm by McLennan and Tourky (2006)
    for computing an approximate fixed point of `T`.

    Parameters
    ----------
    See Parameters in compute_fixed_point.

    Returns
    -------
    x_new : scalar(float) or ndarray(float)
        Approximate fixed point.

    """
    if verbose == 2:
        start_time = time.time()
        _print_after_skip(print_skip, it=None)

    x_new = v
    y_new = T(x_new, *args, **kwargs)
    iterate = 1
    error = np.max(np.abs(y_new - x_new))

    if verbose == 2:
        etime = time.time() - start_time
        _print_after_skip(print_skip, iterate, error, etime)

    if error <= error_tol or iterate >= max_iter:
        if verbose >= 1:
            if iterate == max_iter:
                warnings.warn(_non_convergence_msg, RuntimeWarning)
            elif verbose == 2:
                print(_convergence_msg.format(iterate=iterate))
        return x_new

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
    max_piv = 10**6  # Max number of pivoting steps in lemke_howson_tbl

    while True:
        y_new = T(x_new, *args, **kwargs)
        iterate += 1
        error = np.max(np.abs(y_new - x_new))

        if verbose == 2:
            etime = time.time() - start_time
            _print_after_skip(print_skip, iterate, error, etime)

        if error <= error_tol or iterate >= max_iter:
            break

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
        converged, num_iter = lemke_howson_tbl(
            tableaux_curr, bases_curr, init_pivot=m-1, max_iter=max_piv
        )
        _, rho = get_mixed_actions(tableaux_curr, bases_curr)

        if Y.ndim <= 2:
            x_new = rho.dot(Y[:m])
        else:
            shape_Y = Y.shape
            Y_2d = Y.reshape(shape_Y[0], np.prod(shape_Y[1:]))
            x_new = rho.dot(Y_2d[:m]).reshape(shape_Y[1:])

    if verbose >= 1:
        if iterate == max_iter:
            warnings.warn(_non_convergence_msg, RuntimeWarning)
        elif verbose == 2:
            print(_convergence_msg.format(iterate=iterate))

    return x_new


@jit(nopython=True, cache=True)
def _initialize_tableaux_ig(X, Y, tableaux, bases):
    """
    Given sequences `X` and `Y` of ndarrays, initialize the tableau and
    basis arrays in place for the "geometric" imitation game as defined
    in McLennan and Tourky (2006), to be passed to `lemke_howson_tbl`.

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


@generated_jit(nopython=True, cache=True)
def _square_sum(a):
    if isinstance(a, types.Number):
        return lambda a: a**2
    elif isinstance(a, types.Array):
        return _square_sum_array


def _square_sum_array(a):
    sum_ = 0
    for x in a.flat:
        sum_ += x**2
    return sum_
