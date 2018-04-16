"""
Implements a linear programming solver using the "Simplex" method.

"""

import numpy as np
from numba import jit
from .util import pivot_operation, min_ratio_test,


unbounded_obj = "The problem appears to be unbounded."


@jit(nopython=True, cache=True)
def simplex_algorithm(tableau, basis, M, maxiter=10000, tol_npos=1e-10,
                      tol_ratio_diff=1e-15):
    """
    Execute the simplex algorithm on `tableau`. Jit-compiled in
    `nopython` mode.

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

    maxiter : scalar(int), optional(default=10000)
        Maximum number of pivot operation for each phase.

    tol_npos : scalar(float), optional(default=1e-10)
        Tolerance for treating an element as nonpositive.

    tol_ratio_diff : scalar(float), optional(default=1e-15)
        Tolerance for comparing candidate minimum ratios.

    Return
    ----------
    status : scalar(int)
        An integer representing the exit status of the optimization:
        0 : Optimization terminated successfully
        1 : Iteration limit reached

    """
    pivot_col = _choose_pivot_col(tableau, basis, tol_npos)

    argmins = np.arange(0, M)

    for num_iter in range(maxiter):
        if pivot_col == -1:
            return 0

        pivot_row, num_argmins = _choose_pivot_row(tableau, pivot_col, argmins,
                                                   M, tol_npos, tol_ratio_diff)

        # Check if there is no lower bound
        if num_argmins == 0:
            raise ValueError(unbounded_obj)

        pivot_operation(tableau, (pivot_row, pivot_col))

        # Update `basis`
        basis[pivot_row] = pivot_col

        # Update `pivot_col`
        pivot_col = _choose_pivot_col(tableau, basis, tol_npos)

    return 1


@jit(nopython=True, cache=True)
def _choose_pivot_col(tableau, basis, tol_npos=1e-10):
    """
    Choose the column index of the pivot element in `tableau`. Jit-compiled
    in `nopython` mode.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        2-D array which contains the LP problem in detached coefficients form.

    basis : ndarray(int, ndim=1)
        1-D array containing the indices of the basic variables in `tableau`.

    tol_npos : scalar(float), optional(default=1e-10)
        Tolerance for treating an element as nonpositive.

    Return
    ----------
    idx : scalar(int)
        The index of the variable with a negative coefficient and the lowest
        column index. If all variables have nonnegative coefficients, return
        -1.

    """
    for idx in range(tableau.shape[1]-1):
        if tableau[-1, idx] < -tol_npos and (idx != basis).all():
            return idx


@jit(nopython=True, cache=True)
def _choose_pivot_row(tableau, pivot_col, argmins, M, tol_npos=1e-10,
                      tol_ratio_diff=1e-15):
    """
    Choose the row index of the pivot element in `tableau`.
    Jit-compiled in `nopython` mode.

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

    tol_npos : scalar(float), optional(default=1e-10)
        Tolerance for treating an element as nonpositive.

    tol_ratio_diff : scalar(float), optional(default=1e-15)
        Tolerance for comparing candidate minimum ratios.

    Return
    ----------
    pivot_row : scalar(int)
        The index of the chosen pivot row.

    num_argmins : scalar(int)
        Number of minimizing rows.

    """
    num_argmins = min_ratio_test(tableau, pivot_col, -1, argmins, M,
                                 tol_npos, tol_ratio_diff)
    pivot_row = argmins[:num_argmins].min()

    # Restore `argmins`
    for i in range(M):
        argmins[i] = i

    return pivot_row, num_argmins
