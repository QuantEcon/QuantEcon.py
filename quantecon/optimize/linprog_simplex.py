"""
Implements a linear programming solver using the "Simplex" method.

"""

import numpy as np
from numba import jit
from collections import namedtuple
from ..util import pivot_operation, min_ratio_test, lex_min_ratio_test


results = namedtuple('results', 'fun sol sol_dual status')

@jit(nopython=True, cache=True)
def linprog_simplex(tableau, N, M_ub=0, M_eq=0, tie_breaking_rule=0,
                    max_iter=10000, tol_piv=1e-9, fea_tol=1e-6,
                    tol_ratio_diff=1e-15):
    """
    .. highlight:: none

    Solve the following LP problem using the Simplex method:

    Maximize: :math:`c^{T}x`

    Subject to:

    .. math::

        A_{ub}x ≤ b_{ub}

        A_{eq}x = b_{eq}

        x ≥ 0

    Jit-compiled in `nopython` mode.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        2-D array containing the standardized LP problem in detached
        coefficients form augmented with the artificial variables, the
        infeasibility form and with nonnegative constant terms. The
        infeasibility form is assumed to be placed in the last row, and the
        objective function in the second last row.

    N : scalar(int)
        Number of control variables.

    M_ub : scalar(int)
        Number of inequality constraints.

    M_eq : scalar(int)
        Number of equality constraints.

    tie_breaking_rule : scalar(int), optional(default=0)
        An integer representing the rule used to break ties:
        ::

            0 : Bland's rule
            1 : Lexicographic rule

    max_iter : scalar(int), optional(default=10000)
        Maximum number of pivot operation for each phase.

    tol_piv : scalar(float), optional(default=1e-9)
        Tolerance for treating an element as nonpositive during the
        minimum-ratio test.

    fea_tol : scalar(float), optional(default=1e-6)
        Tolerance for treating an element as nonpositive when choosing the
        pivot column.

    tol_ratio_diff : scalar(float), optional(default=1e-15)
        Tolerance for comparing candidate minimum ratios.

    Returns
    ----------
    results : namedtuple
        A namedtuple containing the following items:
        ::

            "fun" : The value of the maximized objective.
            "sol" : If `status` is 0, a basic solution to the linear
                    programming problem.
            "sol_dual" : If `status` is 0, a basic solution to the dual linear
                         programming problem.
            "status" : An integer representing the exit status of the
                       optimization:
                       0 : Optimization terminated successfully
                       1 : Iteration limit reached

    Examples
    --------
    >>> M_ub = 3
    >>> N = 2
    >>> tableau = np.array([[2., 1., 1., 0., 0., 1., 0., 0., 0., 10.],
    ...                     [1., 1., 0., 1., 0., 0., 1., 0., 0., 8.],
    ...                     [1., 0., 0., 0., 1., 0., 0., 1., 0., 4.],
    ...                     [3., 2., 0., 0., 0., 0., 0., 0., 1., 0.],
    ...                     [-4., -2., -1., -1., -1., 0., 0., 0., 0., -22.]])
    >>> linprog_simplex(tableau, N, M_ub)
    results(fun=-0.0, sol=array([2., 6.]), sol_dual=array([-0., -0., -0.]),
            status=0)

    References
    ----------

    .. [1] Dantzig, George B., "Linear programming and extensions". Rand
           Corporation Research Study Princeton Univ. Press, Princeton, NJ,
           1963

    .. [2] Bland, Robert G. (May 1977). "New finite pivoting tie_breaking_rules
           for the simplex method". Mathematics of Operations Research.
           2 (2): 103–107.

    .. [3] Pan, Ping-Qi., "Linear Programming Computation". Springer,
           Berlin (2014).

    .. [4] https://www.whitman.edu/Documents/Academics/Mathematics/lewis.pdf

    .. [5] http://mat.gsia.cmu.edu/classes/QUANT/NOTES/chap7.pdf

    .. [6] http://slahaie.net/docs/lpdual.pdf

    """
    M = M_ub + M_eq

    tableau[M, 0:N] = -tableau[M, 0:N] # Maximize

    basis = np.empty(M, dtype=np.int64)
    for i in range(M):  # Warning: the artificial variables are used as a basis
        basis[i] = N + M_ub + i

    # Phase I
    status = simplex_algorithm(tableau, basis, M, tie_breaking_rule,
                               max_iter, tol_piv, fea_tol, tol_ratio_diff)

    if status == 1:
        return results(0., np.empty(1), np.empty(1), status)

    if abs(tableau[-1, -1]) > fea_tol:
        raise ValueError("The problem appears to be infeasible")

    # Update `tableau`
    updated_tableau = tableau[:-1, tableau[-1, :] <= fea_tol]

    # Phase II
    status = simplex_algorithm(updated_tableau, basis, M, tie_breaking_rule,
                               max_iter, tol_piv, fea_tol, tol_ratio_diff)

    tableau[:-1, tableau[-1, :] <= fea_tol] = updated_tableau[:, :]

    # Find solution
    sol = np.zeros(N, dtype=np.float64)
    sol_dual = np.zeros(M, dtype=np.float64)

    fun = get_solution(tableau, basis, sol, sol_dual)

    return results(fun, sol, sol_dual, status)


@jit(nopython=True, cache=True)
def simplex_algorithm(tableau, basis, M, tie_breaking_rule, max_iter,
                      tol_piv, fea_tol, tol_ratio_diff):
    """
    .. highlight:: none

    Execute the simplex algorithm on `tableau` using `tie_breaking_rule`.
    Jit-compiled in `nopython` mode.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        2-D array to be modified inplace which contains the LP problem in
        detached coefficients form. The objective is assumed to be placed in
        the last row.

    basis : ndarray(int, ndim=1)
        1-D array containing the indices of the basic variables in `tableau`.

    M : scalar(int)
        Total number of constraints in the LP problem.

    tie_breaking_rule : scalar(int)
        Rule used to break ties when choosing pivot elements.
        ::

            0 : Bland's rule
            1 : Lexicographic rule

    max_iter : scalar(int), optional(default=10000)
        Maximum number of pivot operation for each phase.

    tol_piv : scalar(float)
        Tolerance for treating an element as nonpositive during the
        minimum-ratio test.

    fea_tol : scalar(float)
        Tolerance for treating an element as nonpositive when choosing the
        pivot column.

    tol_ratio_diff : scalar(float)
        Tolerance for comparing candidate minimum ratios.

    Returns
    ----------
    status : scalar(int)
        An integer representing the exit status of the optimization:
        ::

            0 : Optimization terminated successfully
            1 : Iteration limit reached

    """
    pivot_col = _choose_pivot_col(tableau, basis, tie_breaking_rule, fea_tol)

    argmins = np.arange(0, M)

    for num_iter in range(max_iter):
        if pivot_col == -1:
            return 0

        pivot_row, num_argmins = _choose_pivot_row(tableau, pivot_col,
                                                   argmins, M,
                                                   tie_breaking_rule, tol_piv,
                                                   tol_ratio_diff)

        # Check if there is no lower bound
        if num_argmins == 0:
            raise ValueError("The problem appears to be unbounded.")

        pivot_operation(tableau, (pivot_row, pivot_col))

        # Update `basis`
        basis[pivot_row] = pivot_col

        # Update `pivot_col`
        pivot_col = _choose_pivot_col(tableau, basis, tie_breaking_rule,
                                      fea_tol)

    return 1


@jit(nopython=True, cache=True)
def _choose_pivot_col(tableau, basis, tie_breaking_rule, fea_tol):
    """
    Choose the column index of the pivot element in `tableau` using
    `tie_breaking_rule`. Jit-compiled in `nopython` mode.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        2-D array which contains the LP problem in detached coefficients form.

    basis : ndarray(int, ndim=1)
        1-D array containing the indices of the basic variables in `tableau`.

    tie_breaking_rule : scalar(int)
        An integer representing the rule used to break ties:
        0 : Bland's rule
        1 : Lexicographic rule

    fea_tol : scalar(float)
        Tolerance for treating an element as nonpositive.

    Returns
    ----------
    idx : scalar(int)
        The index of the variable with a negative coefficient and the lowest
        column index. If all variables have nonnegative coefficients, return
        -1.

    """
    if tie_breaking_rule == 0:  # Bland's tie_breaking_rule
        for idx in range(tableau.shape[1]-1):
            if tableau[-1, idx] < -fea_tol and (idx != basis).all():
                return idx

    if tie_breaking_rule == 1:  # Lexicographic rule
        idx = tableau[-1, :-1].argmin()
        if tableau[-1, idx] < -fea_tol:
            return idx

    return -1


@jit(nopython=True, cache=True)
def _choose_pivot_row(tableau, pivot_col, argmins, M, tie_breaking_rule,
                      tol_piv, tol_ratio_diff):
    """
    Choose the row index of the pivot element in `tableau` using
    `tie_breaking_rule`. Jit-compiled in `nopython` mode.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        2-D array which contains the LP problem in detached coefficients form.

    pivot_col : scalar(int)
        The index of the chosen pivot column.

    argmins : ndarray(int, ndim=1)
        Array containing the indices of the candidate rows. Modified in
        place to store the indices of minimizing rows.

    M : scalar(int)
        Total number of constraints in the LP problem.

    tie_breaking_rule : scalar(int)
        An integer representing the rule used to break ties:
        0 : Bland's rule
        1 : Lexicographic rule

    tol_piv : scalar(float)
        Tolerance for treating an element as nonpositive during the
        minimum-ratio test.

    tol_ratio_diff : scalar(float)
        Tolerance for comparing candidate minimum ratios.

    Returns
    ----------
    pivot_row : scalar(int)
        The index of the chosen pivot row.

    num_argmins : scalar(int)
        Number of minimizing rows.

    """
    if tie_breaking_rule == 0:  # Bland's tie_breaking_rule
        num_argmins = min_ratio_test(tableau, pivot_col, -1, argmins, M,
                                     tol_piv, tol_ratio_diff)
        pivot_row = argmins[:num_argmins].min()

        # Restore `argmins`
        for i in range(M):
            argmins[i] = i

        return pivot_row, num_argmins

    if tie_breaking_rule == 1:  # Lexicographic rule
        return lex_min_ratio_test(tableau, pivot_col, 0, tableau.shape[1],
                                  argmins, M, tol_piv, tol_ratio_diff)

    else:
        return -1, -1


@jit(nopython=True, cache=True)
def get_solution(tableau, basis, sol, sol_dual):
    """
    Find a basic solution to the LP problem in `tableau`. Jit-compiled in
    `nopython` mode.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        2-D array which contains the LP problem in detached coefficients form.

    basis : ndarray(int, ndim=1)
        1-D array containing the indices of the basic variables in `tableau`.

    sol : ndarray(float, ndim=1)
        1-D array to be filled with the solution values.

    sol_dual : ndarray(float, ndim=1)
        1-D array to be filled with the dual solution values.

    Returns
    ----------
    fun : scalar(float)
        The value of the maximized objective function

    """
    N, M = sol.size, sol_dual.size

    for i in range(M):
        if basis[i] < N:
            sol[basis[i]] = tableau[i, -1]

    for j in range(M):
        sol_dual[j] = tableau[-2, N+j]

    fun = tableau[-2, -1]

    return fun
