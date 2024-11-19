import numpy as np
from .polymatrix_game import PolymatrixGame
from ..optimize.pivoting import _pivoting, _lex_min_ratio_test
from ..optimize.lcp_lemke import _get_solution


def polym_lcp_solver(polymg: PolymatrixGame):
    """
    Finds a Nash Equilbrium of a polymatrix game
    using Howson's algorithm described in
    https://www.jstor.org/stable/2634798
    which utilises linear complimentarity.

    Args:
        polymg (PolymatrixGame): Polymatrix game to solve.

    Returns:
        Probability distribution across actions for each player
        at Nash Equilibrium.
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

    starting_player_actions = {
        player: 0
        for player in range(polymg.N)
    }

    for player in range(polymg.N):
        row = sum(polymg.nums_actions) + player
        col = n + sum(polymg.nums_actions[:player]) + \
            starting_player_actions[player]
        _pivoting(tableau, col, row)
        basis[row] = col

    # Array to store row indices in lex_min_ratio_test
    argmins = np.empty(n + polymg.N, dtype=np.int_)
    p = 0
    retro = False
    while p < polymg.N:
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

            _, pivrow = _lex_min_ratio_test(
                tableau, pivcol, 0, argmins,
            )

            _pivoting(tableau, pivcol, pivrow)
            basis[pivrow], leaving_var = pivcol, basis[pivrow]

            if leaving_var == finishing_x or leaving_var == finishing_y:
                p += 1
                break
            elif leaving_var == finishing_v:
                print("entering the backtracking case")
                p -= 1
                retro = True
                break
            elif leaving_var < n:
                pivcol = leaving_var + n
            else:
                pivcol = leaving_var - n

    combined_solution = _get_solution(tableau, basis, z)

    eq_strategies = [
        combined_solution[
            sum(polymg.nums_actions[:player]):
            sum(polymg.nums_actions[:player+1])
            ]
        for player in range(polymg.N)
    ]

    return eq_strategies
