"""
Author: Daisuke Oyama

Compute all mixed Nash equilibria of a 2-player (non-degenerate) normal
form game by support enumeration.

References
----------
B. von Stengel, "Equilibrium Computation for Two-Player Games in
Strategic and Extensive Form," Chapter 3, N. Nisan, T. Roughgarden, E.
Tardos, and V. Vazirani eds., Algorithmic Game Theory, 2007.

"""
from distutils.version import LooseVersion
import numpy as np
import numba
from numba import jit


least_numba_version = LooseVersion('0.28')
is_numba_required_installed = True
if LooseVersion(numba.__version__) < least_numba_version:
    is_numba_required_installed = False
nopython = is_numba_required_installed

EPS = np.finfo(float).eps


def support_enumeration(g):
    """
    Compute mixed-action Nash equilibria with equal support size for a
    2-player normal form game by support enumeration. For a
    non-degenerate game input, these are all Nash equilibria.

    The algorithm checks all the equal-size support pairs; if the
    players have the same number n of actions, there are 2n choose n
    minus 1 such pairs. This should thus be used only for small games.

    Parameters
    ----------
    g : NormalFormGame
        NormalFormGame instance with 2 players.

    Returns
    -------
    list(tuple(ndarray(float, ndim=1)))
        List containing tuples of Nash equilibrium mixed actions.

    Notes
    -----
    This routine is jit-complied if Numba version 0.28 or above is
    installed.

    """
    return list(support_enumeration_gen(g))


def support_enumeration_gen(g):
    """
    Generator version of `support_enumeration`.

    Parameters
    ----------
    g : NormalFormGame
        NormalFormGame instance with 2 players.

    Yields
    -------
    tuple(ndarray(float, ndim=1))
        Tuple of Nash equilibrium mixed actions.

    """
    try:
        N = g.N
    except:
        raise TypeError('input must be a 2-player NormalFormGame')
    if N != 2:
        raise NotImplementedError('Implemented only for 2-player games')
    return _support_enumeration_gen(g.players[0].payoff_array,
                                    g.players[1].payoff_array)


@jit(nopython=nopython)  # cache=True raises _pickle.PicklingError
def _support_enumeration_gen(payoff_matrix0, payoff_matrix1):
    """
    Main body of `support_enumeration_gen`.

    Parameters
    ----------
    payoff_matrix0 : ndarray(float, ndim=2)
        Payoff matrix for player 0, of shape (m, n)

    payoff_matrix1 : ndarray(float, ndim=2)
        Payoff matrix for player 1, of shape (n, m)

    Yields
    ------
    out : tuple(ndarray(float, ndim=1))
        Tuple of Nash equilibrium mixed actions, of lengths m and n,
        respectively.

    """
    nums_actions = payoff_matrix0.shape[0], payoff_matrix1.shape[0]
    n_min = min(nums_actions)

    for k in range(1, n_min+1):
        supps = (np.arange(k), np.empty(k, np.int_))
        actions = (np.empty(k), np.empty(k))
        A = np.empty((k+1, k+1))
        A[:-1, -1] = -1
        A[-1, :-1] = 1
        A[-1, -1] = 0
        b = np.zeros(k+1)
        b[-1] = 1
        while supps[0][-1] < nums_actions[0]:
            supps[1][:] = np.arange(k)
            while supps[1][-1] < nums_actions[1]:
                if _indiff_mixed_action(payoff_matrix0, supps[0], supps[1],
                                        A, b, actions[1]):
                    if _indiff_mixed_action(payoff_matrix1, supps[1], supps[0],
                                            A, b, actions[0]):
                        out = (np.zeros(nums_actions[0]),
                               np.zeros(nums_actions[1]))
                        for p, (supp, action) in enumerate(zip(supps,
                                                               actions)):
                            out[p][supp] = action
                        yield out
                next_k_array(supps[1])
            next_k_array(supps[0])


@jit(nopython=nopython)
def _indiff_mixed_action(payoff_matrix, own_supp, opp_supp, A, b, out):
    """
    Given a player's payoff matrix `payoff_matrix`, an array `own_supp`
    of this player's actions, and an array `opp_supp` of the opponent's
    actions, each of length k, compute the opponent's mixed action whose
    support equals `opp_supp` and for which the player is indifferent
    among the actions in `own_supp`, if any such exists. Return `True`
    if such a mixed action exists and actions in `own_supp` are indeed
    best responses to it, in which case the outcome is stored in `out`;
    `False` otherwise. Arrays `A` and `b` are used in intermediate
    steps.

    Parameters
    ----------
    payoff_matrix : ndarray(ndim=2)
        The player's payoff matrix, of shape (m, n).

    own_supp : ndarray(int, ndim=1)
        Array containing the player's action indices, of length k.

    opp_supp : ndarray(int, ndim=1)
        Array containing the opponent's action indices, of length k.

    A : ndarray(float, ndim=2)
        Array used in intermediate steps, of shape (k+1, k+1). The
        following values must be assigned in advance: `A[:-1, -1] = -1`,
        `A[-1, :-1] = 1`, and `A[-1, -1] = 0`.

    b : ndarray(float, ndim=1)
        Array used in intermediate steps, of shape (k+1,). The following
        values must be assigned in advance `b[:-1] = 0` and `b[-1] = 1`.

    out : ndarray(float, ndim=1)
        Array of length k to store the k nonzero values of the desired
        mixed action.

    Returns
    -------
    bool
        `True` if a desired mixed action exists and `False` otherwise.

    """
    m = payoff_matrix.shape[0]
    k = len(own_supp)

    A[:-1, :-1] = payoff_matrix[own_supp, :][:, opp_supp]
    if is_singular(A):
        return False

    sol = np.linalg.solve(A, b)
    if (sol[:-1] <= 0).any():
        return False
    out[:] = sol[:-1]
    val = sol[-1]

    if k == m:
        return True

    own_supp_flags = np.zeros(m, np.bool_)
    own_supp_flags[own_supp] = True

    for i in range(m):
        if not own_supp_flags[i]:
            payoff = 0
            for j in range(k):
                payoff += payoff_matrix[i, opp_supp[j]] * out[j]
            if payoff > val:
                return False
    return True


@jit(nopython=True, cache=True)
def next_k_combination(x):
    """
    Find the next k-combination, as described by an integer in binary
    representation with the k set bits, by "Gosper's hack".

    Copy-paste from en.wikipedia.org/wiki/Combinatorial_number_system

    Parameters
    ----------
    x : int
        Integer with k set bits.

    Returns
    -------
    int
        Smallest integer > x with k set bits.

    """
    u = x & -x
    v = u + x
    return v + (((v ^ x) // u) >> 2)


@jit(nopython=True, cache=True)
def next_k_array(a):
    """
    Given an array `a` of k distinct nonnegative integers, return the
    next k-array in lexicographic ordering of the descending sequences
    of the elements. `a` is modified in place.

    Parameters
    ----------
    a : ndarray(int, ndim=1)
        Array of length k.

    Returns
    -------
    a : ndarray(int, ndim=1)
        View of `a`.

    Examples
    --------
    Enumerate all the subsets with k elements of the set {0, ..., n-1}.

    >>> n, k = 4, 2
    >>> a = np.arange(k)
    >>> while a[-1] < n:
    ...     print(a)
    ...     a = next_k_array(a)
    ...
    [0 1]
    [0 2]
    [1 2]
    [0 3]
    [1 3]
    [2 3]

    """
    k = len(a)
    if k == 0:
        return a

    x = 0
    for i in range(k):
        x += (1 << a[i])

    x = next_k_combination(x)

    pos = 0
    for i in range(k):
        while x & 1 == 0:
            x = x >> 1
            pos += 1
        a[i] = pos
        x = x >> 1
        pos += 1

    return a


if is_numba_required_installed:
    @jit(nopython=True, cache=True)
    def is_singular(a):
        s = numba.targets.linalg._compute_singular_values(a)
        if s[-1] <= s[0] * EPS:
            return True
        else:
            return False
else:
    def is_singular(a):
        s = np.linalg.svd(a, compute_uv=False)
        if s[-1] <= s[0] * EPS:
            return True
        else:
            return False

_is_singular_docstr = \
"""
Determine whether matrix `a` is numerically singular, by checking
its singular values.

Parameters
----------
a : ndarray(float, ndim=2)
    2-dimensional array of floats.

Returns
-------
bool
    Whether `a` is numerically singular.

"""

is_singular.__doc__ = _is_singular_docstr
