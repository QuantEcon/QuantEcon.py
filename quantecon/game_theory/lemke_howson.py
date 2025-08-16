"""
Compute mixed Nash equilibria of a 2-player normal form game by the
Lemke-Howson algorithm.

"""
import numbers
import numpy as np
from numba import jit
from .utilities import NashResult
from ..optimize.pivoting import _pivoting, _lex_min_ratio_test


def lemke_howson(g, init_pivot=0, max_iter=10**6, capping=None,
                 full_output=False):
    """
    Find one mixed-action Nash equilibrium of a 2-player normal form
    game by the Lemke-Howson algorithm [2]_, implemented with
    "complementary pivoting" (see, e.g., von Stengel [3]_ for details).

    Parameters
    ----------
    g : NormalFormGame
        NormalFormGame instance with 2 players.

    init_pivot : scalar(int), optional(default=0)
        Initial pivot, an integer k such that 0 <= k < m+n, where
        integers 0, ..., m-1 and m, ..., m+n-1 correspond to the actions
        of players 0 and 1, respectively.

    max_iter : scalar(int), optional(default=10**6)
        Maximum number of pivoting steps.

    capping : scalar(int), optional(default=None)
        If supplied, the routine is executed with the heuristics
        proposed by Codenotti et al. [1]_; see Notes below for details.

    full_output : bool, optional(default=False)
        If False, only the computed Nash equilibrium is returned. If
        True, the return value is `(NE, res)`, where `NE` is the Nash
        equilibrium and `res` is a `NashResult` object.

    Returns
    -------
    NE : tuple(ndarray(float, ndim=1))
        Tuple of computed Nash equilibrium mixed actions.

    res : NashResult
        Object containing information about the computation. Returned
        only when `full_output` is True. See `NashResult` for details.

    Examples
    --------
    Consider the following game from von Stengel [3]_:

    >>> np.set_printoptions(precision=4)  # Reduce the digits printed
    >>> bimatrix = [[(3, 3), (3, 2)],
    ...             [(2, 2), (5, 6)],
    ...             [(0, 3), (6, 1)]]
    >>> g = NormalFormGame(bimatrix)

    Obtain a Nash equilibrium of this game by `lemke_howson` with player
    0's action 1 (out of the three actions 0, 1, and 2) as the initial
    pivot:

    >>> lemke_howson(g, init_pivot=1)
    (array([ 0.    ,  0.3333,  0.6667]), array([ 0.3333,  0.6667]))
    >>> g.is_nash(_)
    True

    Additional information is returned if `full_output` is set True:

    >>> NE, res = lemke_howson(g, init_pivot=1, full_output=True)
    >>> res.converged  # Whether the routine has converged
    True
    >>> res.num_iter  # Number of pivoting steps performed
    4

    Notes
    -----
    * This routine is implemented with floating point arithmetic and
      thus is subject to numerical instability.

    * If `capping` is set to a positive integer, the routine is executed
      with the heuristics proposed by [1]_:

      * For k = `init_pivot`, `init_pivot` + 1, ..., `init_pivot` +
        (m+n-2), (modulo m+n), the Lemke-Howson algorithm is executed
        with k as the initial pivot and `capping` as the maximum number
        of pivoting steps. If the algorithm converges during this loop,
        then the Nash equilibrium found is returned.

      * Otherwise, the Lemke-Howson algorithm is executed with
        `init_pivot` + (m+n-1) (modulo m+n) as the initial pivot, with a
        limit `max_iter` on the total number of pivoting steps.

      Accoding to the simulation results for *uniformly random games*,
      for medium- to large-size games this heuristics outperforms the
      basic Lemke-Howson algorithm with a fixed initial pivot, where
      [1]_ suggests that `capping` be set to 10.

    References
    ----------
    .. [1] B. Codenotti, S. De Rossi, and M. Pagan, "An Experimental
       Analysis of Lemke-Howson Algorithm," arXiv:0811.3247, 2008.

    .. [2] C. E. Lemke and J. T. Howson, "Equilibrium Points of Bimatrix
       Games," Journal of the Society for Industrial and Applied
       Mathematics (1964), 413-423.

    .. [3] B. von Stengel, "Equilibrium Computation for Two-Player Games
       in Strategic and Extensive Form," Chapter 3, N. Nisan, T.
       Roughgarden, E. Tardos, and V. Vazirani eds., Algorithmic Game
       Theory, 2007.

    """
    try:
        N = g.N
    except AttributeError:
        raise TypeError('g must be a 2-player NormalFormGame')
    if N != 2:
        raise NotImplementedError('Implemented only for 2-player games')

    payoff_matrices = g.payoff_arrays
    nums_actions = g.nums_actions
    total_num = sum(nums_actions)

    msg = '`init_pivot` must be an integer k' + \
          'such that 0 <= k < {0}'.format(total_num)

    if not isinstance(init_pivot, numbers.Integral):
        raise TypeError(msg)

    if not (0 <= init_pivot < total_num):
        raise ValueError(msg)

    if capping is None:
        capping = max_iter

    tableaux = tuple(
        np.empty((nums_actions[1-i], total_num+1)) for i in range(N)
    )
    bases = tuple(np.empty(nums_actions[1-i], dtype=int) for i in range(N))

    converged, num_iter, init_pivot_used = \
        _lemke_howson_capping(payoff_matrices, tableaux, bases, init_pivot,
                              max_iter, capping)
    NE = _get_mixed_actions(tableaux, bases)

    if not full_output:
        return NE

    res = NashResult(NE=NE,
                     converged=converged,
                     num_iter=num_iter,
                     max_iter=max_iter,
                     init=init_pivot_used)

    return NE, res


@jit(nopython=True, cache=True)
def _lemke_howson_capping(payoff_matrices, tableaux, bases, init_pivot,
                          max_iter, capping):
    """
    Execute the Lemke-Howson algorithm with the heuristics proposed by
    Codenotti et al.

    Parameters
    ----------
    payoff_matrices : tuple(ndarray(ndim=2))
        Tuple of two arrays representing payoff matrices, of shape
        (m, n) and (n, m), respectively.

    tableaux : tuple(ndarray(float, ndim=2))
        Tuple of two arrays to be used to store the tableaux, of shape
        (n, m+n+1) and (m, m+n+1), respectively. Modified in place.

    bases : tuple(ndarray(int, ndim=1))
        Tuple of two arrays to be used to store the bases, of shape (n,)
        and (m,), respectively. Modified in place.

    init_pivot : scalar(int)
        Integer k such that 0 <= k < m+n.

    max_iter : scalar(int)
        Maximum number of pivoting steps.

    capping : scalar(int)
        Value for capping. If set equal to `max_iter`, then the routine
        is equivalent to the standard Lemke-Howson algorithm.

    """
    m, n = tableaux[1].shape[0], tableaux[0].shape[0]
    init_pivot_curr = init_pivot
    max_iter_curr = max_iter
    total_num_iter = 0

    for k in range(m+n-1):
        capping_curr = min(max_iter_curr, capping)

        _initialize_tableaux(payoff_matrices, tableaux, bases)
        converged, num_iter = \
            _lemke_howson_tbl(tableaux, bases, init_pivot_curr, capping_curr)

        total_num_iter += num_iter

        if converged or total_num_iter >= max_iter:
            return converged, total_num_iter, init_pivot_curr

        init_pivot_curr += 1
        if init_pivot_curr >= m + n:
            init_pivot_curr -= m + n
        max_iter_curr -= num_iter

    _initialize_tableaux(payoff_matrices, tableaux, bases)
    converged, num_iter = \
        _lemke_howson_tbl(tableaux, bases, init_pivot_curr, max_iter_curr)
    total_num_iter += num_iter

    return converged, total_num_iter, init_pivot_curr


@jit(nopython=True, cache=True)
def _initialize_tableaux(payoff_matrices, tableaux, bases):
    """
    Given a tuple of payoff matrices, initialize the tableau and basis
    arrays in place.

    For each player `i`, if `payoff_matrices[i].min()` is non-positive,
    then stored in the tableau are payoff values incremented by
    `abs(payoff_matrices[i].min()) + 1` (to ensure for the tableau not
    to have a negative entry or a column identically zero).

    Suppose that the players 0 and 1 have m and n actions, respectively.

    * `tableaux[0]` has n rows and m+n+1 columns, where columns 0, ...,
      m-1 and m, ..., m+n-1 correspond to the non-slack and slack
      variables, respectively.

    * `tableaux[1]` has m rows and m+n+1 columns, where columns 0, ...,
      m-1 and m, ..., m+n-1 correspond to the slack and non-slack
      variables, respectively.

    * In each `tableaux[i]`, column m+n contains the values of the basic
      variables (which are initially 1).

    * `bases[0]` and `bases[1]` contain basic variable indices, which
      are initially m, ..., m+n-1 and 0, ..., m-1, respectively.

    Parameters
    ----------
    payoff_matrices : tuple(ndarray(ndim=2))
        Tuple of two arrays representing payoff matrices, of shape
        (m, n) and (n, m), respectively.

    tableaux : tuple(ndarray(float, ndim=2))
        Tuple of two arrays to be used to store the tableaux, of shape
        (n, m+n+1) and (m, m+n+1), respectively. Modified in place.

    bases : tuple(ndarray(int, ndim=1))
        Tuple of two arrays to be used to store the bases, of shape (n,)
        and (m,), respectively. Modified in place.

    Returns
    -------
    tableaux : tuple(ndarray(float, ndim=2))
        View to `tableaux`.

    bases : tuple(ndarray(int, ndim=1))
        View to `bases`.

    Examples
    --------
    >>> A = np.array([[3, 3], [2, 5], [0, 6]])
    >>> B = np.array([[3, 2, 3], [2, 6, 1]])
    >>> m, n = A.shape
    >>> tableaux = (np.empty((n, m+n+1)), np.empty((m, m+n+1)))
    >>> bases = (np.empty(n, dtype=int), np.empty(m, dtype=int))
    >>> tableaux, bases = _initialize_tableaux((A, B), tableaux, bases)
    >>> tableaux[0]
    array([[ 3.,  2.,  3.,  1.,  0.,  1.],
           [ 2.,  6.,  1.,  0.,  1.,  1.]])
    >>> tableaux[1]
    array([[ 1.,  0.,  0.,  4.,  4.,  1.],
           [ 0.,  1.,  0.,  3.,  6.,  1.],
           [ 0.,  0.,  1.,  1.,  7.,  1.]])
    >>> bases
    (array([3, 4]), array([0, 1, 2]))

    """
    nums_actions = payoff_matrices[0].shape

    consts = np.zeros(2)  # To be added to payoffs if min <= 0
    for pl in range(2):
        min_ = payoff_matrices[pl].min()
        if min_ <= 0:
            consts[pl] = min_ * (-1) + 1

    for pl, (py_start, sl_start) in enumerate(zip((0, nums_actions[0]),
                                                  (nums_actions[0], 0))):
        for i in range(nums_actions[1-pl]):
            for j in range(nums_actions[pl]):
                tableaux[pl][i, py_start+j] = \
                    payoff_matrices[1-pl][i, j] + consts[1-pl]
            for j in range(nums_actions[1-pl]):
                if j == i:
                    tableaux[pl][i, sl_start+j] = 1
                else:
                    tableaux[pl][i, sl_start+j] = 0
            tableaux[pl][i, -1] = 1

        for i in range(nums_actions[1-pl]):
            bases[pl][i] = sl_start + i

    return tableaux, bases


@jit(nopython=True, cache=True)
def _lemke_howson_tbl(tableaux, bases, init_pivot, max_iter):
    """
    Main body of the Lemke-Howson algorithm implementation.

    Perform the complementary pivoting. Modify `tablaux` and `bases` in
    place.

    Parameters
    ----------
    tableaux : tuple(ndarray(float, ndim=2))
        Tuple of two arrays containing the tableaux, of shape (n, m+n+1)
        and (m, m+n+1), respectively. Modified in place.

    bases : tuple(ndarray(int, ndim=1))
        Tuple of two arrays containing the bases, of shape (n,) and
        (m,), respectively. Modified in place.

    init_pivot : scalar(int)
        Integer k such that 0 <= k < m+n.

    max_iter : scalar(int)
        Maximum number of pivoting steps.

    Returns
    -------
    converged : bool
        Whether the pivoting terminated before `max_iter` was reached.

    num_iter : scalar(int)
        Number of pivoting steps performed.

    Examples
    --------
    >>> np.set_printoptions(precision=4)  # Reduce the digits printed
    >>> A = np.array([[3, 3], [2, 5], [0, 6]])
    >>> B = np.array([[3, 2, 3], [2, 6, 1]])
    >>> m, n = A.shape
    >>> tableaux = (np.empty((n, m+n+1)), np.empty((m, m+n+1)))
    >>> bases = (np.empty(n, dtype=int), np.empty(m, dtype=int))
    >>> tableaux, bases = _initialize_tableaux((A, B), tableaux, bases)
    >>> _lemke_howson_tbl(tableaux, bases, 1, 10)
    (True, 4)
    >>> tableaux[0]
    array([[ 0.875 ,  0.    ,  1.    ,  0.375 , -0.125 ,  0.25  ],
           [ 0.1875,  1.    ,  0.    , -0.0625,  0.1875,  0.125 ]])
    >>> tableaux[1]
    array([[ 1.    , -1.6   ,  0.8   ,  0.    ,  0.    ,  0.2   ],
           [ 0.    ,  0.4667, -0.4   ,  1.    ,  0.    ,  0.0667],
           [ 0.    , -0.0667,  0.2   ,  0.    ,  1.    ,  0.1333]])
    >>> bases
    (array([2, 1]), array([0, 3, 4]))

    The outputs indicate that in the Nash equilibrium obtained, player
    0's mixed action plays actions 2 and 1 with positive weights 0.25
    and 0.125, while player 1's mixed action plays actions 0 and 1
    (labeled as 3 and 4) with positive weights 0.0667 and 0.1333.

    """
    init_player = 0
    for k in bases[0]:
        if k == init_pivot:
            init_player = 1
            break
    pls = [init_player, 1 - init_player]

    pivot = init_pivot

    m, n = tableaux[1].shape[0], tableaux[0].shape[0]
    slack_starts = (m, 0)

    # Array to store row indices in lex_min_ratio_test
    argmins = np.empty(max(m, n), dtype=np.int_)

    converged = False
    num_iter = 0

    while True:
        for pl in pls:
            # Determine the leaving variable
            _, row_min = _lex_min_ratio_test(tableaux[pl], pivot,
                                             slack_starts[pl], argmins)

            # Pivoting step: modify tableau in place
            _pivoting(tableaux[pl], pivot, row_min)

            # Update the basic variables and the pivot
            bases[pl][row_min], pivot = pivot, bases[pl][row_min]

            num_iter += 1

            if pivot == init_pivot:
                converged = True
                break
            if num_iter >= max_iter:
                break
        else:
            continue
        break

    return converged, num_iter


@jit(nopython=True, cache=True)
def _get_mixed_actions(tableaux, bases):
    """
    From `tableaux` and `bases`, extract non-slack basic variables and
    return a tuple of the corresponding, normalized mixed actions.

    Parameters
    ----------
    tableaux : tuple(ndarray(float, ndim=2))
        Tuple of two arrays containing the tableaux, of shape (n, m+n+1)
        and (m, m+n+1), respectively.

    bases : tuple(ndarray(int, ndim=1))
        Tuple of two arrays containing the bases, of shape (n,) and
        (m,), respectively.

    Returns
    -------
    tuple(ndarray(float, ndim=1))
        Tuple of mixed actions as given by the non-slack basic variables
        in the tableaux.

    """
    nums_actions = tableaux[1].shape[0], tableaux[0].shape[0]
    num = nums_actions[0] + nums_actions[1]
    out = np.zeros(num)

    for pl, (start, stop) in enumerate(zip((0, nums_actions[0]),
                                           (nums_actions[0], num))):
        sum_ = 0.
        for i in range(nums_actions[1-pl]):
            k = bases[pl][i]
            if start <= k < stop:
                out[k] = tableaux[pl][i, -1]
                sum_ += tableaux[pl][i, -1]
        if sum_ != 0:
            out[start:stop] /= sum_

    return out[:nums_actions[0]], out[nums_actions[0]:]
