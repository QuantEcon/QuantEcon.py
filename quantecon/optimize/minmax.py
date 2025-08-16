"""
Contain a minmax problem solver routine.

"""
import numpy as np
from numba import jit
from .linprog_simplex import solve_tableau, PivOptions
from .pivoting import _pivoting


@jit(nopython=True, cache=True)
def minmax(A, max_iter=10**6, piv_options=PivOptions()):
    r"""
    Given an m x n matrix `A`, return the value :math:`v^*` of the
    minmax problem:

    .. math::

        v^* = \max_{x \in \Delta_m} \min_{y \in \Delta_n} x^T A y
            = \min_{y \in \Delta_n}\max_{x \in \Delta_m} x^T A y

    and the optimal solutions :math:`x^* \in \Delta_m` and
    :math:`y^* \in \Delta_n`: :math:`v^* = x^{*T} A y^*`, where
    :math:`\Delta_k = \{z \in \mathbb{R}^k_+ \mid z_1 + \cdots + z_k =
    1\}`, :math:`k = m, n`.

    This routine is jit-compiled by Numba, using
    `optimize.linprog_simplex` routines.

    Parameters
    ----------
    A : ndarray(float, ndim=2)
        ndarray of shape (m, n).

    max_iter : int, optional(default=10**6)
        Maximum number of iteration in the linear programming solver.

    piv_options : PivOptions, optional
        PivOptions namedtuple to set tolerance values used in the linear
        programming solver.

    Returns
    -------
    v : float
        Value :math:`v^*` of the minmax problem.

    x : ndarray(float, ndim=1)
        Optimal solution :math:`x^*`, of shape (m,).

    y : ndarray(float, ndim=1)
        Optimal solution :math:`y^*`, of shape (n,).

    """
    m, n = A.shape

    min_ = A.min()
    const = 0.
    if min_ <= 0:
        const = min_ * (-1) + 1

    tableau = np.zeros((m+2, n+1+m+1))

    for i in range(m):
        for j in range(n):
            tableau[i, j] = A[i, j] + const
        tableau[i, n] = -1
        tableau[i, n+1+i] = 1

    tableau[-2, :n] = 1
    tableau[-2, -1] = 1
    tableau[-1, n] = -1

    # Phase 1
    pivcol = 0

    pivrow = 0
    max_ = tableau[0, pivcol]
    for i in range(1, m):
        if tableau[i, pivcol] > max_:
            pivrow = i
            max_ = tableau[i, pivcol]

    _pivoting(tableau, n, pivrow)
    _pivoting(tableau, pivcol, m)

    basis = np.arange(n+1, n+1+m+1)
    basis[pivrow] = n
    basis[-1] = 0

    # Phase 2
    solve_tableau(tableau, basis, max_iter-2, skip_aux=False,
                  piv_options=piv_options)

    # Obtain solution
    x = np.empty(m)
    y = np.zeros(n)

    for i in range(m+1):
        if basis[i] < n:
            y[basis[i]] = tableau[i, -1]

    for j in range(m):
        x[j] = tableau[-1, n+1+j]
        if x[j] != 0:
            x[j] *= -1

    v = tableau[-1, -1] - const

    return v, x, y
