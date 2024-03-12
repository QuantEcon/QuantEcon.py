"""
Contain a Numba-jitted linear programming solver based on the Simplex
Method.

"""
from collections import namedtuple
import numpy as np
from numba import jit
from .pivoting import _pivoting, _lex_min_ratio_test


SimplexResult = namedtuple(
    'SimplexResult', ['x', 'lambd', 'fun', 'success', 'status', 'num_iter']
)
SimplexResult.__doc__ = \
    'namedtuple containing the result from `linprog_simplex`.'

FEA_TOL = 1e-6
TOL_PIV = 1e-7
TOL_RATIO_DIFF = 1e-13

PivOptions = namedtuple(
    'PivOptions', ['fea_tol', 'tol_piv', 'tol_ratio_diff']
)
PivOptions.__new__.__defaults__ = (FEA_TOL, TOL_PIV, TOL_RATIO_DIFF)
PivOptions.__doc__ = 'namedtuple to hold tolerance values for pivoting.'


# Delete useless docstring for fields of namedtuple
def _del_field_docstring(nt):
    for field in nt._fields:
        getattr(nt, field).__doc__ = ''


for _nt in [SimplexResult, PivOptions]:
    _del_field_docstring(_nt)


@jit(nopython=True, cache=True)
def linprog_simplex(c, A_ub=np.empty((0, 0)), b_ub=np.empty((0,)),
                    A_eq=np.empty((0, 0)), b_eq=np.empty((0,)),
                    max_iter=10**6, piv_options=PivOptions(),
                    tableau=None, basis=None, x=None, lambd=None):
    """
    Solve a linear program in the following form by the simplex
    algorithm (with the lexicographic pivoting rule):

    maximize::

        c @ x

    subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
               x >= 0

    Parameters
    ----------
    c : ndarray(float, ndim=1)
        ndarray of shape (n,).

    A_ub : ndarray(float, ndim=2), optional
        ndarray of shape (m, n).

    b_ub : ndarray(float, ndim=1), optional
        ndarray of shape (m,).

    A_eq : ndarray(float, ndim=2), optional
        ndarray of shape (k, n).

    b_eq : ndarray(float, ndim=1), optional
        ndarray of shape (k,).

    max_iter : int, optional(default=10**6)
        Maximum number of iteration to perform.

    piv_options : PivOptions, optional
        PivOptions namedtuple to set the following tolerance values:

            fea_tol : float
                Feasibility tolerance (default={FEA_TOL}).

            tol_piv : float
                Pivot tolerance (default={TOL_PIV}).

            tol_ratio_diff : float
                Tolerance used in the the lexicographic pivoting
                (default={TOL_RATIO_DIFF}).

    tableau : ndarray(float, ndim=2), optional
        Temporary ndarray of shape (L+1, n+m+L+1) to store the tableau,
        where L=m+k. Modified in place.

    basis : ndarray(int, ndim=1), optional
        Temporary ndarray of shape (L,) to store the basic variables.
        Modified in place.

    x : ndarray(float, ndim=1), optional
        Output ndarray of shape (n,) to store the primal solution.

    lambd : ndarray(float, ndim=1), optional
        Output ndarray of shape (L,) to store the dual solution.

    Returns
    -------
    res : SimplexResult
        namedtuple consisting of the fields:

            x : ndarray(float, ndim=1)
                ndarray of shape (n,) containing the primal solution.

            lambd : ndarray(float, ndim=1)
                ndarray of shape (L,) containing the dual solution.

            fun : float
                Value of the objective function.

            success : bool
                True if the algorithm succeeded in finding an optimal
                solution.

            status : int
                An integer representing the exit status of the
                optimization::

                    0 : Optimization terminated successfully
                    1 : Iteration limit reached
                    2 : Problem appears to be infeasible
                    3 : Problem apperas to be unbounded

            num_iter : int
                The number of iterations performed.

    References
    ----------
    * K. C. Border, "The Gauss–Jordan and Simplex Algorithms" 2004.

    """
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
        solve_phase_1(tableau, basis, max_iter, piv_options=piv_options)
    num_iter += num_iter_1
    if not success:
        return SimplexResult(x, lambd, fun, success, status, num_iter)

    # Modify the criterion row for Phase 2
    _set_criterion_row(c, basis, tableau)

    # Phase 2
    success, status, num_iter_2 = \
        solve_tableau(tableau, basis, max_iter-num_iter, skip_aux=True,
                      piv_options=piv_options)
    num_iter += num_iter_2
    fun = get_solution(tableau, basis, x, lambd, b_signs)

    return SimplexResult(x, lambd, fun, success, status, num_iter)


linprog_simplex.__doc__ = linprog_simplex.__doc__.format(
    FEA_TOL=FEA_TOL, TOL_PIV=TOL_PIV, TOL_RATIO_DIFF=TOL_RATIO_DIFF
)


@jit(nopython=True, cache=True)
def _initialize_tableau(A_ub, b_ub, A_eq, b_eq, tableau, basis):
    """
    Initialize the `tableau` and `basis` arrays in place for Phase 1.

    Suppose that the original linear program has the following form:

    maximize::

        c @ x

    subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
               x >= 0

    Let s be a vector of slack variables converting the inequality
    constraint to an equality constraint so that the problem turns to be
    the standard form:

    maximize::

        c @ x

    subject to::

        A_ub @ x + s == b_ub
        A_eq @ x     == b_eq
        x, s         >= 0

    Then, let (z1, z2) be a vector of artificial variables for Phase 1.
    We solve the following LP:

    maximize::

        -(1 @ z1 + 1 @ z2)

    subject to::

        A_ub @ x + s + z1 == b_ub
        A_eq @ x + z2     == b_eq
        x, s, z1, z2      >= 0

    The tableau needs to be of shape (L+1, n+m+L+1), where L=m+k.

    Parameters
    ----------
    A_ub : ndarray(float, ndim=2)
        ndarray of shape (m, n).

    b_ub : ndarray(float, ndim=1)
        ndarray of shape (m,).

    A_eq : ndarray(float, ndim=2)
        ndarray of shape (k, n).

    b_eq : ndarray(float, ndim=1)
        ndarray of shape (k,).

    tableau : ndarray(float, ndim=2)
        Empty ndarray of shape (L+1, n+m+L+1) to store the tableau.
        Modified in place.

    basis : ndarray(int, ndim=1)
        Empty ndarray of shape (L,) to store the basic variables.
        Modified in place.

    Returns
    -------
    tableau : ndarray(float, ndim=2)
        View to `tableau`.

    basis : ndarray(int, ndim=1)
        View to `basis`.

    """
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
    """
    Modify the criterion row of the tableau for Phase 2.

    Parameters
    ----------
    c : ndarray(float, ndim=1)
        ndarray of shape (n,).

    basis : ndarray(int, ndim=1)
        ndarray of shape (L,) containing the basis obtained by Phase 1.

    tableau : ndarray(float, ndim=2)
        ndarray of shape (L+1, n+m+L+1) containing the tableau obtained
        by Phase 1. Modified in place.

    Returns
    -------
    tableau : ndarray(float, ndim=2)
        View to `tableau`.

    """
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
def solve_tableau(tableau, basis, max_iter=10**6, skip_aux=True,
                  piv_options=PivOptions()):
    """
    Perform the simplex algorithm on a given tableau in canonical form.

    Used to solve a linear program in the following form:

    maximize::

        c @ x

    subject to::

        A_ub @ x <= b_ub
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

    max_iter : int, optional(default=10**6)
        Maximum number of pivoting steps.

    skip_aux : bool, optional(default=True)
        Whether to skip the coefficients of the auxiliary (or
        artificial) variables in pivot column selection.

    piv_options : PivOptions, optional
        PivOptions namedtuple to set the tolerance values.

    Returns
    -------
    success : bool
        True if the algorithm succeeded in finding an optimal solution.

    status : int
        An integer representing the exit status of the optimization.

    num_iter : int
        The number of iterations performed.

    """
    L = tableau.shape[0] - 1

    # Array to store row indices in lex_min_ratio_test
    argmins = np.empty(L, dtype=np.int_)

    success = False
    status = 1
    num_iter = 0

    while num_iter < max_iter:
        num_iter += 1

        pivcol_found, pivcol = _pivot_col(tableau, skip_aux, piv_options)

        if not pivcol_found:  # Optimal
            success = True
            status = 0
            break

        aux_start = tableau.shape[1] - L - 1
        pivrow_found, pivrow = _lex_min_ratio_test(
            tableau[:-1, :], pivcol, aux_start, argmins,
            piv_options.tol_piv, piv_options.tol_ratio_diff
        )

        if not pivrow_found:  # Unbounded
            success = False
            status = 3
            break

        _pivoting(tableau, pivcol, pivrow)
        basis[pivrow] = pivcol

    return success, status, num_iter


@jit(nopython=True, cache=True)
def solve_phase_1(tableau, basis, max_iter=10**6, piv_options=PivOptions()):
    """
    Perform the simplex algorithm for Phase 1 on a given tableau in
    canonical form, by calling `solve_tableau` with `skip_aux=False`.

    Parameters
    ----------
    See `solve_tableau`.

    Returns
    -------
    See `solve_tableau`.

    """
    L = tableau.shape[0] - 1
    nm = tableau.shape[1] - (L+1)  # n + m

    success, status, num_iter_1 = \
        solve_tableau(tableau, basis, max_iter, skip_aux=False,
                      piv_options=piv_options)

    if not success:  # max_iter exceeded
        return success, status, num_iter_1
    if tableau[-1, -1] > piv_options.fea_tol:  # Infeasible
        success = False
        status = 2
        return success, status, num_iter_1

    # Check artifial variables have been eliminated
    tol_piv = piv_options.tol_piv
    for i in range(L):
        if basis[i] >= nm:  # Artifial variable not eliminated
            for j in range(nm):
                if tableau[i, j] < -tol_piv or \
                   tableau[i, j] > tol_piv:  # Treated nonzero
                    _pivoting(tableau, j, i)
                    basis[i] = j
                    num_iter_1 += 1
                    break

    return success, status, num_iter_1


@jit(nopython=True, cache=True)
def _pivot_col(tableau, skip_aux, piv_options):
    """
    Choose the column containing the pivot element: the column
    containing the maximum positive element in the last row of the
    tableau.

    `skip_aux` should be True in phase 1, and False in phase 2.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        ndarray of shape (L+1, n+m+L+1) containing the tableau.

    skip_aux : bool
        Whether to skip the coefficients of the auxiliary (or
        artificial) variables in pivot column selection.

    piv_options : PivOptions, optional
        PivOptions namedtuple to set the tolerance values.

    Returns
    -------
    found : bool
        True iff there is a positive element in the last row of the
        tableau (and then pivoting should be conducted).

    pivcol : int
        The index of column containing the pivot element. (-1 if `found
        == False`.)

    """
    L = tableau.shape[0] - 1
    criterion_row_stop = tableau.shape[1] - 1
    if skip_aux:
        criterion_row_stop -= L

    found = False
    pivcol = -1
    coeff = piv_options.fea_tol
    for j in range(criterion_row_stop):
        if tableau[-1, j] > coeff:
            coeff = tableau[-1, j]
            pivcol = j
            found = True

    return found, pivcol


@jit(nopython=True, cache=True)
def get_solution(tableau, basis, x, lambd, b_signs):
    """
    Fetch the optimal solution and value from an optimal tableau.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        ndarray of shape (L+1, n+m+L+1) containing the optimal tableau,
        where L=m+k.

    basis : ndarray(int, ndim=1)
        ndarray of shape (L,) containing the optimal basis.

    x : ndarray(float, ndim=1)
        Empty ndarray of shape (n,) to store the primal solution.
        Modified in place.

    lambd : ndarray(float, ndim=1)
        Empty ndarray of shape (L,) to store the dual solution. Modified
        in place.

    b_signs : ndarray(bool, ndim=1)
        ndarray of shape (L,) whose i-th element is True iff the i-th
        element of the vector (b_ub, b_eq) is positive.

    Returns
    -------
    fun : float
        The optimal value.

    """
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
