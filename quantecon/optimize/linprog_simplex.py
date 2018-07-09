"""
Contain a linear programming solver routine based on the Simplex Method.

"""
from collections import namedtuple
import numpy as np
from numba import jit
from .pivoting import _pivoting, _lex_min_ratio_test


FEA_TOL = 1e-6


SimplexResult = namedtuple(
    'SimplexResult', ['x', 'lambd', 'fun', 'success', 'status', 'num_iter']
)


@jit(nopython=True, cache=True)
def linprog_simplex(c, A_ub=np.empty((0, 0)), b_ub=np.empty((0,)),
                    A_eq=np.empty((0, 0)), b_eq=np.empty((0,)), max_iter=10**6,
                    tableau=None, basis=None, x=None, lambd=None):
    n, m, k = c.shape[0], A_ub.shape[0], A_eq.shape[0]
    L = m + k

    if tableau is None:
        tableau = np.empty((L+1, n+m+L+1))
    if basis is None:
        basis = np.empty(L, dtype=np.int_)
    if x is None:
        x = np.empty(n)
    if lambd is None:
        lambd = np.empty(L)

    num_iter = 0
    fun = -np.inf

    b_signs = np.empty(L, dtype=np.bool_)
    for i in range(m):
        b_signs[i] = True if b_ub[i] >= 0 else False
    for i in range(k):
        b_signs[m+i] = True if b_eq[i] >= 0 else False

    # Construct initial tableau for Phase 1
    _initialize_tableau(A_ub, b_ub, A_eq, b_eq, tableau, basis)

    # Phase 1
    success, status, num_iter_1 = \
        solve_tableau(tableau, basis, max_iter, skip_aux=False)
    num_iter += num_iter_1
    if not success:  # max_iter exceeded
        return SimplexResult(x, lambd, fun, success, status, num_iter)
    if tableau[-1, -1] > FEA_TOL:  # Infeasible
        success = False
        status = 2
        return SimplexResult(x, lambd, fun, success, status, num_iter)

    # Modify the criterion row for Phase 2
    _set_criterion_row(c, basis, tableau)

    # Phase 2
    success, status, num_iter_2 = \
        solve_tableau(tableau, basis, max_iter-num_iter, skip_aux=True)
    num_iter += num_iter_2
    fun = get_solution(tableau, basis, x, lambd, b_signs)

    return SimplexResult(x, lambd, fun, success, status, num_iter)


@jit(nopython=True, cache=True)
def _initialize_tableau(A_ub, b_ub, A_eq, b_eq, tableau, basis):
    m, k = A_ub.shape[0], A_eq.shape[0]
    L = m + k
    n = tableau.shape[1] - (m+L+1)

    for i in range(m):
        for j in range(n):
            tableau[i, j] = A_ub[i, j]
    for i in range(k):
        for j in range(n):
            tableau[m+i, j] = A_eq[i, j]

    tableau[:L, n:-1] = 0

    for i in range(m):
        tableau[i, -1] = b_ub[i]
        if tableau[i, -1] < 0:
            for j in range(n):
                tableau[i, j] *= -1
            tableau[i, n+i] = -1
            tableau[i, -1] *= -1
        else:
            tableau[i, n+i] = 1
        tableau[i, n+m+i] = 1
    for i in range(k):
        tableau[m+i, -1] = b_eq[i]
        if tableau[m+i, -1] < 0:
            for j in range(n):
                tableau[m+i, j] *= -1
            tableau[m+i, -1] *= -1
        tableau[m+i, n+m+m+i] = 1

    tableau[-1, :] = 0
    for i in range(L):
        for j in range(n+m):
            tableau[-1, j] += tableau[i, j]
        tableau[-1, -1] += tableau[i, -1]

    for i in range(L):
        basis[i] = n+m+i

    return tableau, basis


@jit(nopython=True, cache=True)
def _set_criterion_row(c, basis, tableau):
    n = c.shape[0]
    L = basis.shape[0]

    for j in range(n):
        tableau[-1, j] = c[j]
    tableau[-1, n:] = 0

    for i in range(L):
        multiplier = tableau[-1, basis[i]]
        for j in range(tableau.shape[1]):
            tableau[-1, j] -= tableau[i, j] * multiplier

    return tableau


@jit(nopython=True, cache=True)
def solve_tableau(tableau, basis, max_iter=10**6, skip_aux=True):
    """
    Perform the simplex algorithm on a given tableau in canonical form.

    Used to solve a linear program in the following form:

        maximize:     c @ x

        subject to:   A_ub @ x <= b_ub
                      A_eq @ x == b_eq
                      x >= 0

    where A_ub is of shape (m, n) and A_eq is of shape (k, n). Thus,
    `tableau` is of shape (L+1, n+m+L+1), where L=m+k, and

    * `tableau[np.arange(L), :][:, basis]` must be an identity matrix,
      and
    * the elements of `tableau[:-1, -1]` must be nonnegative.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        ndarray of shape (L+1, n+m+L+1) containing the tableau. Modified
        in place.

    basis : ndarray(int, ndim=1)
        ndarray of shape (L,) containing the basic variables. Modified
        in place.

    max_iter : scalar(int), optional(default=10**6)
        Maximum number of pivoting steps.

    skip_aux : bool, optional(default=True)
        Whether to skip the coefficients of the auxiliary (or
        artificial) variables in pivot column selection.

    """
    L = tableau.shape[0] - 1

    # Array to store row indices in lex_min_ratio_test
    argmins = np.empty(L, dtype=np.int_)

    success = False
    status = 1
    num_iter = 0

    while num_iter < max_iter:
        num_iter += 1

        pivcol_found, pivcol = _pivot_col(tableau, skip_aux)

        if not pivcol_found:  # Optimal
            success = True
            status = 0
            break

        aux_start = tableau.shape[1] - L - 1
        pivrow_found, pivrow = _lex_min_ratio_test(tableau[:-1, :], pivcol,
                                                   aux_start, argmins)

        if not pivrow_found:  # Unbounded
            success = False
            status = 3
            break

        _pivoting(tableau, pivcol, pivrow)
        basis[pivrow] = pivcol

    return success, status, num_iter


@jit(nopython=True, cache=True)
def _pivot_col(tableau, skip_aux):
    L = tableau.shape[0] - 1
    criterion_row_stop = tableau.shape[1] - 1
    if skip_aux:
        criterion_row_stop -= L

    found = False
    pivcol = -1
    coeff = FEA_TOL
    for j in range(criterion_row_stop):
        if tableau[-1, j] > coeff:
            coeff = tableau[-1, j]
            pivcol = j
            found = True

    return found, pivcol


@jit(nopython=True, cache=True)
def get_solution(tableau, basis, x, lambd, b_signs):
    n, L = x.size, lambd.size
    aux_start = tableau.shape[1] - L - 1

    x[:] = 0
    for i in range(L):
        if basis[i] < n:
            x[basis[i]] = tableau[i, -1]
    for j in range(L):
        lambd[j] = tableau[-1, aux_start+j]
        if lambd[j] != 0 and b_signs[j]:
            lambd[j] *= -1
    fun = tableau[-1, -1] * (-1)

    return fun
