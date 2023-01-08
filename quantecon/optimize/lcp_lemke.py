"""
Contain a Numba-jitted linear complementarity problem (LCP) solver based
on Lemke's algorithm.

"""
from collections import namedtuple
import numpy as np
from numba import jit
from .pivoting import _pivoting, _lex_min_ratio_test
from .linprog_simplex import PivOptions, FEA_TOL, TOL_PIV, TOL_RATIO_DIFF


LCPResult = namedtuple(
    'LCPResult', ['z', 'success', 'status', 'num_iter']
)
LCPResult.__doc__ = \
    'namedtuple containing the result from `lcp_lemke`.'


# Delete useless docstring for fields of namedtuple
for field in LCPResult._fields:
    getattr(LCPResult, field).__doc__ = ''


@jit(nopython=True, cache=True)
def lcp_lemke(M, q, d=None, max_iter=10**6, piv_options=PivOptions(),
              tableau=None, basis=None, z=None):
    """
    Solve the linear complementarity problem::

        z >= 0
        M @ z + q >= 0
        z @ (M @ z + q) = 0

    by Lemke's algorithm (with the lexicographic pivoting rule).

    Parameters
    ----------
    M : ndarray(float, ndim=2)
        ndarray of shape (n, n).

    q : ndarray(float, ndim=1)
        ndarray of shape (n,).

    d : ndarray(float, ndim=1), optional
        Covering vector, ndarray of shape (n,). Must be strictly
        positive. If None, default to vector of ones.

    max_iter : int, optional(default=10**6)
        Maximum number of iteration to perform.

    piv_options : PivOptions, optional
        PivOptions namedtuple to set the following tolerance values:

            fea_tol : float
                Feasibility tolerance (default={FEA_TOL}). (Not used in
                this function.)

            tol_piv : float
                Pivot tolerance (default={TOL_PIV}).

            tol_ratio_diff : float
                Tolerance used in the the lexicographic pivoting
                (default={TOL_RATIO_DIFF}).

    tableau : ndarray(float, ndim=2), optional
        Temporary ndarray of shape (n, 2*n+2) to store the tableau.
        Modified in place.

    basis : ndarray(int, ndim=1), optional
        Temporary ndarray of shape (n,) to store the basic variables.
        Modified in place.

    z : ndarray(float, ndim=1), optional
        Output ndarray of shape (n,) to store the solution.

    Returns
    -------
    res : LCPResult
        namedtuple consisting of the fields:

            z : ndarray(float, ndim=1)
                ndarray of shape (n,) containing the solution.

            success : bool
                True if the algorithm succeeded in finding a solution.

            status : int
                An integer representing the exit status of the result::

                    0 : Solution found successfully
                    1 : Iteration limit reached
                    2 : Secondary ray termination

            num_iter : int
                The number of iterations performed.

    Examples
    --------
    >>> M = np.array([[1, 0, 0], [2, 1, 0], [2, 2, 1]])
    >>> q = np.array([-8, -12, -14])
    >>> res = lcp_lemke(M, q)
    >>> res.success
    True
    >>> res.z
    array([8., 0., 0.])
    >>> w = M @ res.z + q
    >>> w
    array([0., 4., 2.])
    >>> res.z @ w
    0.0

    References
    ----------
    * K. G. Murty, Linear Complementarity, Linear and Nonlinear
      Programming, 1988.

    """
    n = M.shape[0]

    success = False
    status = 1
    num_iter = 0

    if z is None:
        z = np.empty(n)

    if (q >= 0).all():  # Trivial case
        z[:] = 0
        success = True
        status = 0
        return LCPResult(z, success, status, num_iter)

    if d is None:
        d = np.ones(n)
    if tableau is None:
        tableau = np.empty((n, 2*n+2))
    if basis is None:
        basis = np.empty(n, dtype=np.int_)

    _initialize_tableau(M, q, d, tableau, basis)

    art_var = 2*n  # Artificial variable
    pivcol = art_var

    # Equivalent to lex_min_ratio_test
    pivrow = 0
    ratio_min = q[0] / d[0]
    for i in range(1, n):
        ratio = q[i] / d[i]
        if ratio <= ratio_min + piv_options.tol_ratio_diff:
            pivrow = i
            ratio = ratio_min

    _pivoting(tableau, pivcol, pivrow)
    basis[pivrow], pivcol = pivcol, pivrow + n
    num_iter += 1

    # Array to store row indices in lex_min_ratio_test
    argmins = np.empty(n, dtype=np.int_)

    while num_iter < max_iter:
        pivrow_found, pivrow = _lex_min_ratio_test(
            tableau, pivcol, 0, argmins,
            piv_options.tol_piv, piv_options.tol_ratio_diff
        )

        if not pivrow_found:  # Ray termination
            success = False
            status = 2
            break

        _pivoting(tableau, pivcol, pivrow)
        basis[pivrow], leaving_var = pivcol, basis[pivrow]
        num_iter += 1

        if leaving_var == art_var:  # Solution found
            success = True
            status = 0
            break
        elif leaving_var < n:
            pivcol = leaving_var + n
        else:
            pivcol = leaving_var - n

    _get_solution(tableau, basis, z)

    return LCPResult(z, success, status, num_iter)


lcp_lemke.__doc__ = lcp_lemke.__doc__.format(
    FEA_TOL=FEA_TOL, TOL_PIV=TOL_PIV, TOL_RATIO_DIFF=TOL_RATIO_DIFF
)


@jit(nopython=True, cache=True)
def _initialize_tableau(M, q, d, tableau, basis):
    """
    Initialize the `tableau` and `basis` arrays in place.

    With covering vector d and artificial variable z0, the LCP is
    written as

        q = w - M z - d z0

    where the variables are ordered as (w, z, z0). Thus,
    `tableaus[:, :n]` stores I, `tableaus[:, n:2*n]` stores -M,
    `tableaus[:, 2*n]` stores -d, and `tableaus[:, -1]` stores q, while
    `basis` stores 0, ..., n-1 (variables w).

    Parameters
    ----------
    M : ndarray(float, ndim=2)
        ndarray of shape (n, n).

    q : ndarray(float, ndim=1)
        ndarray of shape (n,).

    d : ndarray(float, ndim=1)
        ndarray of shape (n,).

    tableau : ndarray(float, ndim=2)
        Empty ndarray of shape (n, 2*n+2) to store the tableau.
        Modified in place.

    basis : ndarray(int, ndim=1)
        Empty ndarray of shape (n,) to store the basic variables.
        Modified in place.

    Returns
    -------
    tableau : ndarray(float, ndim=2)
        View to `tableau`.

    basis : ndarray(int, ndim=1)
        View to `basis`.

    """
    n = M.shape[0]

    tableau[:n, :n] = 0
    for i in range(n):
        tableau[i, i] = 1

    for i in range(n):
        for j in range(n):
            tableau[i, n+j] = -M[i, j]

    for i in range(n):
        tableau[i, 2*n] = -d[i]

    for i in range(n):
        tableau[i, -1] = q[i]

    for i in range(n):
        basis[i] = i

    return tableau, basis


@jit(nopython=True, cache=True)
def _get_solution(tableau, basis, z):
    """
    Fetch the solution from `tableau` and `basis`.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        ndarray of shape (n, 2*n+2) containing the terminal tableau.

    basis : ndarray(int, ndim=1)
        ndarray of shape (n,) containing the terminal basis.

    z : ndarray(float, ndim=1)
        Empty ndarray of shape (n,) to store the solution. Modified in
        place.

    Returns
    -------
    z : ndarray(float, ndim=1)
        View to `z`.

    """
    n = z.size

    z[:] = 0
    for i in range(n):
        if n <= basis[i] < 2*n:
            z[basis[i]-n] = tableau[i, -1]

    return z
