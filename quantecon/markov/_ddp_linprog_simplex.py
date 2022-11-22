import numpy as np
from numba import jit
from .utilities import _find_indices
from ..optimize.linprog_simplex import solve_tableau, PivOptions
from ..optimize.pivoting import _pivoting


@jit(nopython=True, cache=True)
def ddp_linprog_simplex(R, Q, beta, a_indices, a_indptr, sigma,
                        max_iter=10**6, piv_options=PivOptions(),
                        tableau=None, basis=None, v=None):
    r"""
    Numba jit complied function to solve a discrete dynamic program via
    linear programming, using `optimize.linprog_simplex` routines. The
    problem has to be represented in state-action pair form with 1-dim
    reward ndarray `R` of shape (n,), 2-dim transition probability
    ndarray `Q` of shapce (L, n), and disount factor `beta`, where n is
    the number of states and L is the number of feasible state-action
    pairs.

    The approach exploits the fact that the optimal value function is
    the smallest function that satisfies :math:`v \geq T v`, where
    :math:`T` is the Bellman operator, and hence it is a (unique)
    solution to the linear program:

    minimize::

        \sum_{s \in S} v(s)

    subject to ::

        v(s) \geq r(s, a) + \beta \sum_{s' \in S} q(s'|s, a) v(s')
                  \quad ((s, a) \in \mathit{SA}).

    This function solves its dual problem:

    maximize::

        \sum_{(s, a) \in \mathit{SA}} r(s, a) y(s, a)

    subject to::

        \sum_{a: (s', a) \in \mathit{SA}} y(s', a) -
            \sum_{(s, a) \in \mathit{SA}} \beta q(s'|s, a) y(s, a) = 1
            \quad (s' \in S),

        y(s, a) \geq 0 \quad ((s, a) \in \mathit{SA}),

    where the optimal value function is obtained as an optimal dual
    solution and an optimal policy as an optimal basis.

    Parameters
    ----------
    R : ndarray(float, ndim=1)
        Reward ndarray, of shape (n,).

    Q : ndarray(float, ndim=2)
        Transition probability ndarray, of shape (L, n).

    beta : scalar(float)
        Discount factor. Must be in [0, 1).

    a_indices : ndarray(int, ndim=1)
        Action index ndarray, of shape (L,).

    a_indptr : ndarray(int, ndim=1)
        Action index pointer ndarray, of shape (n+1,).

    sigma : ndarray(int, ndim=1)
        ndarray containing the initial feasible policy, of shape (n,).
        To be modified in place to store the output optimal policy.

    max_iter : int, optional(default=10**6)
        Maximum number of iteration in the linear programming solver.

    piv_options : PivOptions, optional
        PivOptions namedtuple to set tolerance values used in the linear
        programming solver.

    tableau : ndarray(float, ndim=2), optional
        Temporary ndarray of shape (n+1, L+n+1) to store the tableau.
        Modified in place.

    basis : ndarray(int, ndim=1), optional
        Temporary ndarray of shape (n,) to store the basic variables.
        Modified in place.

    v : ndarray(float, ndim=1), optional
        Output ndarray of shape (n,) to store the optimal value
        function. Modified in place.

    Returns
    -------
    success : bool
        True if the algorithm succeeded in finding an optimal solution.

    num_iter : int
        The number of iterations performed.

    v : ndarray(float, ndim=1)
        Optimal value function (view to `v` if supplied).

    sigma : ndarray(int, ndim=1)
        Optimal policy (view to `sigma`).

    """
    L, n = Q.shape

    if tableau is None:
        tableau = np.empty((n+1, L+n+1))
    if basis is None:
        basis = np.empty(n, dtype=np.int_)
    if v is None:
        v = np.empty(n)

    _initialize_tableau(R, Q, beta, a_indptr, tableau)
    _find_indices(a_indices, a_indptr, sigma, out=basis)

    # Phase 1
    for i in range(n):
        _pivoting(tableau, basis[i], i)

    # Phase 2
    success, status, num_iter = \
        solve_tableau(tableau, basis, max_iter-n, skip_aux=True,
                      piv_options=piv_options)

    # Obtain solution
    for i in range(n):
        v[i] = tableau[-1, L+i] * (-1)

    for i in range(n):
        sigma[i] = a_indices[basis[i]]

    return success, num_iter+n, v, sigma


@jit(nopython=True, cache=True)
def _initialize_tableau(R, Q, beta, a_indptr, tableau):
    """
    Initialize the `tableau` array.

    Parameters
    ----------
    R : ndarray(float, ndim=1)
        Reward ndarray, of shape (n,).

    Q : ndarray(float, ndim=2)
        Transition probability ndarray, of shape (L, n).

    beta : scalar(float)
        Discount factor. Must be in [0, 1).

    a_indptr : ndarray(int, ndim=1)
        Action index pointer ndarray, of shape (n+1,).

    tableau : ndarray(float, ndim=2)
        Empty ndarray of shape (n+1, L+n+1) to store the tableau.
        Modified in place.

    Returns
    -------
    tableau : ndarray(float, ndim=2)
        View to `tableau`.

    """
    L, n = Q.shape

    for j in range(L):
        for i in range(n):
            tableau[i, j] = Q[j, i] * (-beta)

    for i in range(n):
        for j in range(a_indptr[i], a_indptr[i+1]):
            tableau[i, j] += 1

    tableau[:n, L:-1] = 0

    for i in range(n):
        tableau[i, L+i] = 1
        tableau[i, -1] = 1

    for j in range(L):
        tableau[-1, j] = R[j]

    tableau[-1, L:] = 0

    return tableau
