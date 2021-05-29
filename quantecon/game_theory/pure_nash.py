"""
Methods for computing pure Nash equilibria of a normal form game.
(For now, only brute force method is supported)

"""

import numpy as np


def pure_nash_brute(g, tol=None):
    """
    Find all pure Nash equilibria of a normal form game by brute force.

    Parameters
    ----------
    g : NormalFormGame
    tol : scalar(float), optional(default=None)
        Tolerance level used in determining best responses. If None,
        default to the value of the `tol` attribute of `g`.

    Returns
    -------
    NEs : list(tuple(int))
        List of tuples of Nash equilibrium pure actions.
        If no pure Nash equilibrium is found, return empty list.

    Examples
    --------
    Consider the "Prisoners' Dilemma" game:

    >>> PD_bimatrix = [[(1, 1), (-2, 3)],
    ...                [(3, -2), (0, 0)]]
    >>> g_PD = NormalFormGame(PD_bimatrix)
    >>> pure_nash_brute(g_PD)
    [(1, 1)]

    If we consider the "Matching Pennies" game, which has no pure nash
    equilibrium:

    >>> MP_bimatrix = [[(1, -1), (-1, 1)],
    ...                [(-1, 1), (1, -1)]]
    >>> g_MP = NormalFormGame(MP_bimatrix)
    >>> pure_nash_brute(g_MP)
    []

    """
    return list(pure_nash_brute_gen(g, tol=tol))


def pure_nash_brute_gen(g, tol=None):
    """
    Generator version of `pure_nash_brute`.

    Parameters
    ----------
    g : NormalFormGame
    tol : scalar(float), optional(default=None)
        Tolerance level used in determining best responses. If None,
        default to the value of the `tol` attribute of `g`.

    Yields
    ------
    out : tuple(int)
        Tuple of Nash equilibrium pure actions.

    """
    for a in np.ndindex(*g.nums_actions):
        if g.is_nash(a, tol=tol):
            yield a
