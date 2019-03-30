"""
Contain pivoting routines commonly used in the Simplex Algorithm and
Lemke-Howson Algorithm routines.

"""
import numpy as np
from numba import jit


TOL_PIV = 1e-10
TOL_RATIO_DIFF = 1e-15


@jit(nopython=True, cache=True)
def _pivoting(tableau, pivot_col, pivot_row):
    """
    Perform a pivoting step. Modify `tableau` in place.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        Array containing the tableau.

    pivot_col : scalar(int)
        Pivot column index.

    pivot_row : scalar(int)
        Pivot row index.

    Returns
    -------
    tableau : ndarray(float, ndim=2)
        View to `tableau`.

    """
    nrows, ncols = tableau.shape

    pivot_elt = tableau[pivot_row, pivot_col]
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
def _min_ratio_test_no_tie_breaking(tableau, pivot, test_col,
                                    argmins, num_candidates,
                                    tol_piv, tol_ratio_diff):
    """
    Perform the minimum ratio test, without tie breaking, for the
    candidate rows in `argmins[:num_candidates]`. Return the number
    `num_argmins` of the rows minimizing the ratio and store thier
    indices in `argmins[:num_argmins]`.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        Array containing the tableau.

    pivot : scalar(int)
        Pivot.

    test_col : scalar(int)
        Index of the column used in the test.

    argmins : ndarray(int, ndim=1)
        Array containing the indices of the candidate rows. Modified in
        place to store the indices of minimizing rows.

    num_candidates : scalar(int)
        Number of candidate rows in `argmins`.

    tol_piv : scalar(float)
        Pivot tolerance below which a number is considered to be
        nonpositive.

    tol_ratio_diff : scalar(float)
        Tolerance to determine a tie between ratio values.

    Returns
    -------
    num_argmins : scalar(int)
        Number of minimizing rows.

    """
    ratio_min = np.inf
    num_argmins = 0

    for k in range(num_candidates):
        i = argmins[k]
        if tableau[i, pivot] <= tol_piv:  # Treated as nonpositive
            continue
        ratio = tableau[i, test_col] / tableau[i, pivot]
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
def _lex_min_ratio_test(tableau, pivot, slack_start, argmins,
                        tol_piv=TOL_PIV, tol_ratio_diff=TOL_RATIO_DIFF):
    """
    Perform the lexico-minimum ratio test.

    Parameters
    ----------
    tableau : ndarray(float, ndim=2)
        Array containing the tableau.

    pivot : scalar(int)
        Pivot.

    slack_start : scalar(int)
        First index for the slack variables.

    argmins : ndarray(int, ndim=1)
        Empty array used to store the row indices. Its length must be no
        smaller than the number of the rows of `tableau`.

    tol_piv : scalar(float), optional
        Pivot tolerance below which a number is considered to be
        nonpositive. Default value is {TOL_PIV}.

    tol_ratio_diff : scalar(float), optional
        Tolerance to determine a tie between ratio values. Default value
        is {TOL_RATIO_DIFF}.

    Returns
    -------
    found : bool
        False if there is no positive entry in the pivot column.

    row_min : scalar(int)
        Index of the row with the lexico-minimum ratio.

    """
    nrows = tableau.shape[0]
    num_candidates = nrows

    found = False

    # Initialize `argmins`
    for i in range(nrows):
        argmins[i] = i

    num_argmins = _min_ratio_test_no_tie_breaking(
        tableau, pivot, -1, argmins, num_candidates, tol_piv, tol_ratio_diff
    )
    if num_argmins == 1:
        found = True
    elif num_argmins >= 2:
        for j in range(slack_start, slack_start+nrows):
            if j == pivot:
                continue
            num_argmins = _min_ratio_test_no_tie_breaking(
                tableau, pivot, j, argmins, num_argmins,
                tol_piv, tol_ratio_diff
            )
            if num_argmins == 1:
                found = True
                break
    return found, argmins[0]


_lex_min_ratio_test.__doc__ = _lex_min_ratio_test.__doc__.format(
    TOL_PIV=TOL_PIV, TOL_RATIO_DIFF=TOL_RATIO_DIFF
)
