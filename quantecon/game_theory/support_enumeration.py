"""
Compute all mixed Nash equilibria of a 2-player (non-degenerate) normal
form game by support enumeration.

References
----------
B. von Stengel, "Equilibrium Computation for Two-Player Games in
Strategic and Extensive Form," Chapter 3, N. Nisan, T. Roughgarden, E.
Tardos, and V. Vazirani eds., Algorithmic Game Theory, 2007.

"""
import numpy as np
from numba import jit
from ..util.numba import _numba_linalg_solve
from ..util.combinatorics import next_k_array


def support_enumeration(g):
    """
    Compute mixed-action Nash equilibria with equal support size for a
    2-player normal form game by support enumeration. For a
    non-degenerate game input, these are all the Nash equilibria.

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
    ------
    tuple(ndarray(float, ndim=1))
        Tuple of Nash equilibrium mixed actions.

    """
    try:
        N = g.N
    except AttributeError:
        raise TypeError('input must be a 2-player NormalFormGame')
    if N != 2:
        raise NotImplementedError('Implemented only for 2-player games')
    return _support_enumeration_gen(g.payoff_arrays)


@jit(nopython=True)  # cache=True raises _pickle.PicklingError
def _support_enumeration_gen(payoff_matrices):
    """
    Main body of `support_enumeration_gen`.

    Parameters
    ----------
    payoff_matrices : tuple(ndarray(float, ndim=2))
        Tuple of payoff matrices, of shapes (m, n) and (n, m),
        respectively.

    Yields
    ------
    out : tuple(ndarray(float, ndim=1))
        Tuple of Nash equilibrium mixed actions, of lengths m and n,
        respectively.

    """
    nums_actions = payoff_matrices[0].shape
    n_min = min(nums_actions)

    for k in range(1, n_min+1):
        supps = (np.arange(0, k, 1, np.int_), np.empty(k, np.int_))
        actions = (np.empty(k+1), np.empty(k+1))
        A = np.empty((k+1, k+1))

        while supps[0][-1] < nums_actions[0]:
            supps[1][:] = np.arange(k)
            while supps[1][-1] < nums_actions[1]:
                if _indiff_mixed_action(
                    payoff_matrices[0], supps[0], supps[1], A, actions[1]
                ):
                    if _indiff_mixed_action(
                        payoff_matrices[1], supps[1], supps[0], A, actions[0]
                    ):
                        out = (np.zeros(nums_actions[0]),
                               np.zeros(nums_actions[1]))
                        for p, (supp, action) in enumerate(zip(supps,
                                                               actions)):
                            out[p][supp] = action[:-1]
                        yield out
                next_k_array(supps[1])
            next_k_array(supps[0])


@jit(nopython=True, cache=True)
def _indiff_mixed_action(payoff_matrix, own_supp, opp_supp, A, out):
    """
    Given a player's payoff matrix `payoff_matrix`, an array `own_supp`
    of this player's actions, and an array `opp_supp` of the opponent's
    actions, each of length k, compute the opponent's mixed action whose
    support equals `opp_supp` and for which the player is indifferent
    among the actions in `own_supp`, if any such exists. Return `True`
    if such a mixed action exists and actions in `own_supp` are indeed
    best responses to it, in which case the outcome is stored in `out`;
    `False` otherwise. Array `A` is used in intermediate steps.

    Parameters
    ----------
    payoff_matrix : ndarray(ndim=2)
        The player's payoff matrix, of shape (m, n).

    own_supp : ndarray(int, ndim=1)
        Array containing the player's action indices, of length k.

    opp_supp : ndarray(int, ndim=1)
        Array containing the opponent's action indices, of length k.

    A : ndarray(float, ndim=2)
        Array used in intermediate steps, of shape (k+1, k+1).

    out : ndarray(float, ndim=1)
        Array of length k+1 to store the k nonzero values of the desired
        mixed action in `out[:-1]` (and the payoff value in `out[-1]`).

    Returns
    -------
    bool
        `True` if a desired mixed action exists and `False` otherwise.

    """
    m = payoff_matrix.shape[0]
    k = len(own_supp)

    for i in range(k):
        for j in range(k):
            A[j, i] = payoff_matrix[own_supp[i], opp_supp[j]]  # transpose
    A[:-1, -1] = 1
    A[-1, :-1] = -1
    A[-1, -1] = 0
    out[:-1] = 0
    out[-1] = 1

    r = _numba_linalg_solve(A, out)
    if r != 0:  # A: singular
        return False
    for i in range(k):
        if out[i] <= 0:
            return False
    val = out[-1]

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
