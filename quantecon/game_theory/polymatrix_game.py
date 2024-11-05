import numpy as np
from itertools import product
from .normal_form_game import NormalFormGame, Player
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
