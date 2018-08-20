"""
Useful routines for manipulating linear equation systems through pivoting.

"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def make_tableau(c, A_ub=np.array([[]]).T, b_ub=np.array([[]]),
                 A_eq=np.array([[]]).T, b_eq=np.array([[]])):
    """
    Create a tableau for an LP problem given an objective function and
    constraints by transforming the problem to its standard form, making,
    constants nonnegative, augmenting it with artificial variables and with an
    infeasibility form. Jit-compiled in `nopython` mode.

    Parameters
    ----------
    c : ndarray(float, ndim=1)
        1-D array containing the coefficients of the `N` variables of the
        linear objective function to be minimized.

    A_ub : ndarray(float, ndim=2)
        2-D array containing the coefficients of the left hand side of the
        `M_ub` inequality constraints in `N` variables.

    b_ub : ndarray(float, ndim=1)
        1-D array containing the values of the right hand side of the `M_ub`
        inequality constraints.

    A_eq : ndarray(float, ndim=2)
        2-D array containing the coefficients of the left hand side of the
        `M_eq` equality constraints in `N` variables.

    b_eq : ndarray(float, ndim=1)
        1-D array containing the values of the right hand side of the `M_eq`
        equality constraints.

    Return
    ----------
    tableau : ndarray(float, ndim=2)
        2-D array containing the standardized LP problem in detached
        coefficients form augmented with the artificial variables, the
        infeasibility form and with nonnegative constant terms.

    """
    M_ub, N_ub = A_ub.shape
    M_eq, N_eq = A_eq.shape

    M = M_ub + M_eq
    N = max(N_ub, N_eq)

    tableau = np.zeros((M+2, N+M_ub+M+1))

    standardize_lp_problem(c, A_ub, b_ub, A_eq, b_eq, tableau)

    # Make constraints nonnegative
    for i in range(M):
        if tableau[i, -1] < 0:
            tableau[i, :] = -tableau[i, :]

    # Add artificial variables
    for (row_idx, col_idx) in zip(range(M), range(N+M_ub, N+M_ub+M)):
        tableau[row_idx, col_idx] = 1.

    # Add infeasability form
    for row_idx in range(M):
        for col_idx in range(N+M_ub):
            tableau[-1, col_idx] -= tableau[row_idx, col_idx]

    for row_idx in range(M):
        tableau[-1, -1] -= tableau[row_idx, -1]

    return tableau


@jit(nopython=True, cache=True)
def standardize_lp_problem(c, A_ub=np.array([[]]).T, b_ub=np.array([[]]),
                           A_eq=np.array([[]]).T, b_eq=np.array([[]]),
                           tableau=None):
    """
    Standardize an LP problem of the following form:

    Objective:   c.T @ x
    Subject to:  A_ub @ x ≤ b_ub
                 A_eq @ x = b_eq

    Jit-compiled in `nopython` mode.

    Parameters
    ----------
    c : ndarray(float, ndim=1)
        1-D array containing the coefficients of the `N` variables of the
        linear objective function to be minimized.

    A_ub : ndarray(float, ndim=2)
        2-D array containing the coefficients of the left hand side of the
        `M_ub` inequality constraints in `N` variables.

    b_ub : ndarray(float, ndim=1)
        1-D array containing the values of the right hand side of the `M_ub`
        inequality constraints.

    A_eq : ndarray(float, ndim=2)
        2-D array containing the coefficients of the left hand side of the
        `M_eq` equality constraints in `N` variables.

    b_eq : ndarray(float, ndim=1)
        1-D array containing the values of the right hand side of the `M_eq`
        equality constraints.

    tableau : ndarray(float, ndim=2), optional(default=None)
        2-D array to be modified inplace which will contain the standardized
        LP problem in detached coefficients form. If there are any, the
        inequality constrains are stacked on top of the equality constraints.
        The constant terms are placed in the last column. The objective is
        placed in row `M_ub+M_eq`.

    Return
    ----------
    tableau : ndarray(float, ndim=2)
        View of `tableau`.

    """
    M_ub, N_ub = A_ub.shape
    M_eq, N_eq = A_eq.shape
    M = M_ub + M_eq
    N = max(N_ub, N_eq)

    if M_ub != b_ub.size:
        raise ValueError("Inequality constraints are not properly specified.")

    if M_eq != b_eq.size:
        raise ValueError("Equality constraints are not properly specified.")

    if M_ub > 0 and N > 0:  # At least inequality constraints
        if tableau is None:
            tableau = np.zeros((M+1, N+M_ub+1))

        # Place the inequality contraints in the tableau
        tableau[0:M_ub, -1] = b_ub
        tableau[0:M_ub, 0:N] = A_ub

        if M > M_ub: # Both type of constraints
            # Place equality constraints in the tableau
            tableau[M_ub:M, -1] = b_eq
            tableau[M_ub:M, 0:N] = A_eq

        # Add the slack variables
        for (row_idx, col_idx) in zip(range(M_ub), range(N, N+M_ub)):
            # Make diagonal elements equal to one for slack variables part
            tableau[row_idx, col_idx] = 1.

        # Standardize the objective function
        tableau[M, 0:N] = c

        return tableau

    elif M_eq > 0 and N > 0:  # Only equality constraints
        if tableau is None:
            tableau = np.zeros((M+1, N+1))

        # Place the equality constraints in the tableau
        tableau[0:M, -1] = b_eq
        tableau[0:M, 0:N] = A_eq

        # Standardize the objective function
        tableau[M, 0:N] = c

        return tableau

    else:
        raise ValueError("At least one type of constraints must be specified.")


@jit(nopython=True, cache=True)
def pivot_operation(tableau, pivot):
    """
    Perform a pivoting step. Modify `tableau` in place.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        Array containing the tableau.

    pivot : tuple(int)
        Tuple containing the row and column index of the pivot element.

    Returns
    -------
    tableau : ndarray(float, ndim=2)
        View of `tableau`.

    """
    nrows, ncols = tableau.shape

    pivot_row, pivot_col = pivot

    pivot_elt = tableau[pivot]
    for j in range(ncols):
        tableau[pivot_row, j] /= pivot_elt

    for i in range(nrows):
        if i == pivot_row:
            continue
        multiplier = tableau[i, pivot_col]
        if multiplier == 0:
            continue
        for j in range(ncols):
            tableau[i, j] -= tableau[pivot_row, j] * multiplier

    return tableau


@jit(nopython=True, cache=True)
def min_ratio_test(tableau, pivot_col, test_col, argmins, num_candidates,
                   tol_piv=1e-10, tol_ratio_diff=1e-15):
    """
    Perform the minimum ratio test, without tie breaking, for the
    candidate rows in `argmins[:num_candidates]`. Return the number
    `num_argmins` of the rows minimizing the ratio and store thier
    indices in `argmins[:num_argmins]`.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        Array containing the tableau.

    pivot_col : scalar(int)
        Index of the column of the pivot element.

    test_col : scalar(int)
        Index of the column used in the test.

    argmins : ndarray(int, ndim=1)
        Array containing the indices of the candidate rows. Modified in
        place to store the indices of minimizing rows.

    num_candidates : scalar(int)
        Number of candidate rows in `argmins`.

    tol_piv : scalar(float), optional(default=1e-10)
        Tolerance for treating an element of the pivot column as nonpositive.

    tol_ratio_diff : scalar(float), optional(default=1e-15)
        Tolerance for comparing candidate minimum ratios.

    Returns
    -------
    num_argmins : scalar(int)
        Number of minimizing rows.

    """
    ratio_min = np.inf
    num_argmins = 0

    for k in range(num_candidates):
        i = argmins[k]
        if tableau[i, pivot_col] <= tol_piv:  # Treated as nonpositive
            continue
        ratio = tableau[i, test_col] / tableau[i, pivot_col]
        if ratio > ratio_min + tol_ratio_diff:  # Ratio large for i
            continue
        elif ratio < ratio_min - tol_ratio_diff:  # Ratio smaller for i
            ratio_min = ratio
            num_argmins = 1
        else:  # Ratio equal
            num_argmins += 1
        argmins[num_argmins-1] = i

    return num_argmins


@jit(nopython=True, cache=True)
def lex_min_ratio_test(tableau, pivot_col, start, end, argmins, nrows,
                       tol_piv=1e-10, tol_ratio_diff=1e-15):
    """
    Perform the lexico-minimum ratio test.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        Array containing the tableau.

    pivot_col : scalar(int)
        Index of the column of the pivot element.

    start : scalar(int)
        Index on which to start the tie breaking test.

    end : scalar(int)
        Index on which to end the tie breaking test.

    argmins : ndarray(int, ndim=1)
        Empty array used to store the row indices. Its length must be no
        smaller than the number of the rows of `tableau`.

    nrows : scalar(int)
        Number of candidate rows for the lexico-minimum ratio test.

    tol_piv : scalar(float), optional(default=1e-10)
        Tolerance for treating an element of the pivot column as nonpositive.

    tol_ratio_diff : scalar(float), optional(default=1e-15)
        Tolerance for comparing candidate minimum ratios.

    Returns
    -------
    row_min : scalar(int)
        Index of the row with the lexico-minimum ratio.

    """
    num_candidates = nrows

    # Initialize `argmins`
    for i in range(nrows):
        argmins[i] = i

    num_argmins = min_ratio_test(tableau, pivot_col, -1, argmins,
                                 num_candidates, tol_piv, tol_ratio_diff)
    if num_argmins == 1:
        return argmins[0], num_argmins

    for j in range(start, end):
        if j == pivot_col:
            continue
        num_argmins = min_ratio_test(tableau, pivot_col, j, argmins,
                                     num_argmins, tol_piv, tol_ratio_diff)
        if num_argmins == 1:
            break
    return argmins[0], num_argmins
