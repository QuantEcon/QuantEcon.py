from normal_form_game import NormalFormGame, Player
import numpy as np
from itertools import product
from ..optimize.pivoting import _pivoting, _lex_min_ratio_test
from ..optimize.lcp_lemke import _get_solution


def hh_payoff_player(
        nf: NormalFormGame,
        my_player_number,
        my_action_number,
        is_polymatrix=False
):
    """
    hh stands for head-to-head.
    Approximates the payoffs to a player when they play
    an action with values at the actions of other players
    that try to sum to the payoff.
    Precise when the game can be represented with a polymatrix.

    Args:
        my_player_number (_type_): The number of the player making the action.
        my_action_number (_type_): The number of the action.
        is_polymatrix :
            Is the game represented by this normal form
            actually a polymatrix game.

    Returns:
        _type_:
            Dictionary giving an approximate component
            of the payoff at each action for each other player.
    """
    action_combinations = product(*(
        [range(nf.nums_actions[p]) if p != my_player_number else [my_action_number]
            for p in range(nf.N)]
    ))
    my_player: Player = nf.players[my_player_number]
    my_payoffs = my_player.payoff_array
    hh_actions_and_payoffs = np.vstack([
        np.hstack(
            [
                np.eye(nf.nums_actions[p])[action_combination[p]]
                for p in range(nf.N) if p != my_player_number
            ] + [
                my_payoffs[
                    action_combination[my_player_number:]
                    + action_combination[:my_player_number]
                ]
            ]
        )
        for action_combination in action_combinations
    ])
    hh_actions = hh_actions_and_payoffs[:, :-1]
    combined_payoffs = hh_actions_and_payoffs[:, -1]

    # different ways to solve the simultaneous equations
    if is_polymatrix:
        # this does not go much faster and
        # could also be used for games with no actual polymatrix
        hh_payoffs_array, residuals, _, _ = np.linalg.lstsq(
            hh_actions, combined_payoffs, rcond=None
        )
        assert not residuals, "The game is not actually a polymatrix game"
    else:
        hh_payoffs_array = np.dot(np.linalg.pinv(hh_actions), combined_payoffs)

    payoff_labels = [
        (p, a)
        for p in range(nf.N) if p != my_player_number
        for a in range(nf.nums_actions[p])
    ]

    payoffs = {label: payoff for label, payoff in zip(
        payoff_labels, hh_payoffs_array)}

    return payoffs


class PolymatrixGame:

    def __str__(self) -> str:
        str_builder = ""
        for k, v in self.polymatrix.items():
            str_builder += str(k) + ":\n"
            str_builder += str(v) + "\n\n"
        return str_builder

    def __init__(self, number_of_players, nums_actions, polymatrix) -> None:
        self.N = number_of_players
        self.nums_actions = nums_actions
        self.polymatrix = polymatrix

    @classmethod
    def from_nf(cls, nf: NormalFormGame, is_polymatrix=False):
        """
        Creates a Polymatrix approximation to a
        Normal Form Game. Precise if possible.

        Args:
            nf (NormalFormGame): Normal Form Game to approximate.
            is_polymatrix :
                Is the game represented by the normal form
                actually a polymatrix game.

        Returns:
            _type_: New Polymatrix Game.
        """
        polymatrix_builder = {
            (p1, p2): np.full((nf.nums_actions[p1], nf.nums_actions[p2]), -np.inf)
            for p1 in range(nf.N)
            for p2 in range(nf.N)
            if p1 != p2
        }
        for p1 in range(nf.N):
            for a1 in range(nf.nums_actions[p1]):
                payoffs = hh_payoff_player(
                    nf, p1, a1, is_polymatrix=is_polymatrix)
                for ((p2, a2), payoff) in payoffs.items():
                    polymatrix_builder[(p1, p2)][a1][a2] = payoff

        return cls(nf.N, nf.nums_actions, polymatrix_builder)

    def range_of_payoffs(self):
        """
        The lowest and highest components of payoff from
        head to head games.

        Returns:
            _type_: Tuple of minimum and maximum.
        """
        min_p = min([np.min(M) for M in self.polymatrix.values()])
        max_p = max([np.max(M) for M in self.polymatrix.values()])
        return (min_p, max_p)


def polym_lcp_solver(polym: PolymatrixGame):
    LOW_AVOIDER = 2.0
    # makes all of the costs greater than 0
    positive_cost_maker = polym.range_of_payoffs()[1] + LOW_AVOIDER
    # Construct the LCP like Howson:
    M = np.vstack([
        np.hstack([
            np.zeros((polym.nums_actions[player], polym.nums_actions[player])) if p2 == player
            else (positive_cost_maker - polym.polymatrix[(player, p2)])
            for p2 in range(polym.N)
        ] + [
            -np.outer(np.ones(polym.nums_actions[player]), np.eye(polym.N)[player])
        ])
        for player in range(polym.N)
    ] + [
        np.hstack([
            np.hstack([
                np.outer(np.eye(polym.N)[player], np.ones(
                    polym.nums_actions[player]))
                for player in range(polym.N)
            ]
            ),
            np.zeros((polym.N, polym.N))
        ])
    ]
    )
    total_actions = sum(polym.nums_actions)
    q = np.hstack([np.zeros(total_actions), -np.ones(polym.N)])

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
        for player in range(polym.N)
    }

    for player in range(polym.N):
        row = sum(polym.nums_actions) + player
        col = n + sum(polym.nums_actions[:player]) + \
            starting_player_actions[player]
        _pivoting(tableau, col, row)
        basis[row] = col

    # Array to store row indices in lex_min_ratio_test
    argmins = np.empty(n + polym.N, dtype=np.int_)
    p = 0
    retro = False
    while p < polym.N:
        finishing_v = sum(polym.nums_actions) + n + p
        finishing_x = n + \
            sum(polym.nums_actions[:p]) + starting_player_actions[p]
        finishing_y = finishing_x - n

        pivcol = finishing_v if not retro else finishing_x if finishing_y in basis else finishing_y

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
            sum(polym.nums_actions[:player]) : sum(polym.nums_actions[:player+1])
            ]
        for player in range(polym.N)
    ]

    return eq_strategies
