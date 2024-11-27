"""
Compute a Nash Equilibrium of a Polymatrix Game.

Examples
--------

Find a Nash Equilibrium of a Matching Pennies Game.
(This NE is unique so this works reliably.)

>>> matrices = {
...     (0, 1): [[1., -1.], [-1., 1.]],
...     (1, 0): [[-1., 1.], [1., -1.]]
... }
>>> polymg = PolymatrixGame(matrices)
>>> result = polym_lcp_solver(polymg, full_output=True)
>>> print(result[0])
(array([0.5, 0.5]), array([0.5, 0.5]))
>>> print(result[1])
        NE: (array([0.5, 0.5]), array([0.5, 0.5]))
 converged: True
      init: {0: 0, 1: 0}
  max_iter: -1
  num_iter: 4

"""

import numpy as np
from .polymatrix_game import PolymatrixGame
from ..optimize.pivoting import _pivoting, _lex_min_ratio_test
from ..optimize.lcp_lemke import _get_solution
from .utilities import NashResult
from typing import Sequence, Union, Optional
from numpy.typing import NDArray


def polym_lcp_solver(
        polymg: PolymatrixGame,
        starting_player_actions: Optional[Sequence[int]] = None,
        max_iter: int = -1,
        full_output: bool = False,
) -> Union[tuple[NDArray], NashResult]:
    """
    Finds a Nash Equilbrium of a Polymatrix Game.

    Uses Howson's algorithm which utilises
    linear complementarity [1]_.

    Parameters
    ----------
    polymg : PolymatrixGame
        Polymatrix game to solve.

    starting_player_actions : Sequence[int], optional
        Pure actions for each player at which the algorithm begins.
        Defaults to each player playing their first action.

    max_iter : int, optional
        Maximum number of iterations of the complementary
        pivoting before giving up. Howson proves that with enough
        iterations, it will reach a Nash Equilibrium.
        Defaults to never giving up.

    full_output : bool, optional
        When True, adds information about the run to the output
        actions and puts them in a NashResult. Defaults to False.

    Returns
    -------
    tuple(ndarray(float, ndim=1)) or NashResult
        The mixed actions at termination, a Nash Equilibrium if
        not stopped early by reaching `max_iter`. If `full_output`,
        then the number of iterations, whether it has converged,
        and the initial conditions of the algorithm are included
        in the returned `NashResult` alongside the actions.

    References
    ----------
    .. [1] Howson, Joseph T. “Equilibria of Polymatrix Games.”
       Management Science 18, no. 5 (1972): 312–18.
       http://www.jstor.org/stable/2634798.

    """
    LOW_AVOIDER = 2.0
    # makes all of the costs greater than 0
    positive_cost_maker = polymg.range_of_payoffs()[1] + LOW_AVOIDER
    # Construct the LCP like Howson:
    M = np.vstack([
        np.hstack([
            np.zeros(
                (polymg.nums_actions[player], polymg.nums_actions[player])
            ) if p2 == player
            else (positive_cost_maker - polymg.polymatrix[(player, p2)])
            for p2 in range(polymg.N)
        ] + [
            -np.outer(np.ones(
                polymg.nums_actions[player]), np.eye(polymg.N)[player])
        ])
        for player in range(polymg.N)
    ] + [
        np.hstack([
            np.hstack([
                np.outer(np.eye(polymg.N)[player], np.ones(
                    polymg.nums_actions[player]))
                for player in range(polymg.N)
            ]
            ),
            np.zeros((polymg.N, polymg.N))
        ])
    ]
    )
    total_actions = sum(polymg.nums_actions)
    q = np.hstack([np.zeros(total_actions), -np.ones(polymg.N)])

    n = np.shape(M)[0]
    tableau = np.hstack([
        np.eye(n),
        -M,
        np.reshape(q, (-1, 1))
    ])

    basis = np.array(range(n))
    z = np.empty(n)

    if starting_player_actions is None:
        starting_player_actions = {
            player: 0
            for player in range(polymg.N)
        }
    else:
        valid_start = (
            len(starting_player_actions) == polymg.N and
            all(a < max_a for a, max_a in zip(
                starting_player_actions, polymg.nums_actions))
        )
        assert valid_start, "Invalid starting pure actions."

    for player in range(polymg.N):
        row = sum(polymg.nums_actions) + player
        col = n + sum(polymg.nums_actions[:player]) + \
            starting_player_actions[player]
        _pivoting(tableau, col, row)
        # These pivots do not count as iters
        basis[row] = col

    num_iter = 0
    converging = True

    # Array to store row indices in lex_min_ratio_test
    argmins = np.empty(n + polymg.N, dtype=np.int_)
    p = 0
    retro = False
    while p < polymg.N and converging:
        finishing_v = sum(polymg.nums_actions) + n + p
        finishing_x = n + \
            sum(polymg.nums_actions[:p]) + starting_player_actions[p]
        finishing_y = finishing_x - n

        pivcol = (
            finishing_v if not retro
            else finishing_x if finishing_y in basis
            else finishing_y
        )

        retro = False

        while True:

            if num_iter == max_iter:
                converging = False
                break
            num_iter += 1

            _, pivrow = _lex_min_ratio_test(
                tableau, pivcol, 0, argmins,
            )

            _pivoting(tableau, pivcol, pivrow)
            basis[pivrow], leaving_var = pivcol, basis[pivrow]

            if leaving_var == finishing_x or leaving_var == finishing_y:
                p += 1
                break
            elif leaving_var == finishing_v:
                # print("entering the backtracking case")
                p -= 1
                retro = True
                break
            elif leaving_var < n:
                pivcol = leaving_var + n
            else:
                pivcol = leaving_var - n

    combined_solution = _get_solution(tableau, basis, z)

    # might not actually be Nash Equilibrium if we hit max iter
    NE = tuple(
        combined_solution[
            sum(polymg.nums_actions[:player]):
            sum(polymg.nums_actions[:player + 1])
        ]
        for player in range(polymg.N)
    )

    if not full_output:
        return NE

    res = NashResult(NE=NE,
                     converged=converging,
                     num_iter=num_iter,
                     max_iter=max_iter,
                     init=starting_player_actions)

    return NE, res
