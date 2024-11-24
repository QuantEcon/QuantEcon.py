""""
Tools for working with games in Polymatrix form.

In a Polymatrix Game, the payoff to a player is the sum
of their payoffs from bimatrix games against each player.
i.e. If two opponents deviate, the change in payoff
is the sum of the changes in payoff of each deviation.

"""

import numpy as np
from itertools import product
from collections.abc import Sequence, Mapping
from typing import Any, TypeAlias, Self
from .normal_form_game import NormalFormGame, Player

# Matrix with rows and columns.
Matrix: TypeAlias = Any


def hh_payoff_player(
        nfg: NormalFormGame,
        my_player_number: int,
        my_action_number: int,
        is_polymatrix: bool = False
) -> Mapping[tuple[int, int], float]:
    """
    Head-to-head payoff components.

    hh stands for head-to-head.
    Approximates the payoffs to a player when they play
    an action with values at the actions of other players
    that try to sum to the payoff.
    Precise when the game can be represented with a polymatrix.

    Parameters
    ----------
    nfg : NormalFormGame
        The game.

    my_player_number : int
        The number of the player making the action.

    my_action_number : int
        The number of our player's action.

    is_polymatrix : bool, optional
        Is the game represented by this normal form
        actually a polymatrix game. Defaults to False.

    Returns
    -------
    Mapping[tuple[int, int], float]
        Dictionary giving a (approximate) component
        of the payoff at each other player and their action.
    """
    action_combinations = product(*(
        [
            range(nfg.nums_actions[p]) if p != my_player_number
            else [my_action_number]
            for p in range(nfg.N)
        ]
    ))
    my_player: Player = nfg.players[my_player_number]
    my_payoffs = my_player.payoff_array
    hh_actions_and_payoffs = np.vstack([
        np.hstack(
            [
                np.eye(nfg.nums_actions[p])[action_combination[p]]
                for p in range(nfg.N) if p != my_player_number
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
        assert np.allclose(
            residuals, 0), "The game is not actually a polymatrix game."
    else:
        hh_payoffs_array = np.dot(
            np.linalg.pinv(hh_actions), combined_payoffs)

    payoff_labels = [
        (p, a)
        for p in range(nfg.N) if p != my_player_number
        for a in range(nfg.nums_actions[p])
    ]

    payoffs = {
        label: payoff
        for label, payoff in zip(payoff_labels, hh_payoffs_array)}

    return payoffs


class PolymatrixGame:
    """
    Polymatrix Game.

    `polymatrix[(a, b)]` is a 2D matrix, the payoff matrix of `a`
    in the bimatrix game between `a` and `b`;
    player number `a` is the row player
    and player number `b` is the column player.

    Attributes
    ----------
    number_of_players : int
        Number of players.

    nums_actions : Sequence[int]
        The number of actions available to each player.

    polymatrix : Mapping[ tuple[int, int], Matrix ]
        Maps each pair of player numbers to a matrix.

    """

    def __str__(self) -> str:
        str_builder = "Polymatrix Game:\n"
        for k, v in self.polymatrix.items():
            str_builder += str(k) + ":\n"
            str_builder += str(v) + "\n\n"
        return str_builder.rstrip()

    def __init__(
            self,
            number_of_players: int,
            nums_actions: Sequence[int],
            polymatrix: Mapping[
                tuple[int, int],
                Matrix
            ]
    ) -> None:
        self.N = number_of_players
        self.nums_actions = nums_actions
        self.polymatrix = polymatrix

    @classmethod
    def from_nf(
        cls,
        nf: NormalFormGame,
        is_polymatrix: bool = True
    ) -> Self:
        """
        Creates a Polymatrix from a Normal Form Game.

        Precise if possible; many Normal Form Games are not representable
        precisely with a Polymatrix.
        With payoffs (not costs).

        Parameters
        ----------
        nf : NormalFormGame
            Normal Form Game to convert.

        is_polymatrix : bool, optional
            Is the Normal Form Game precisely convertible to a
            Polymatrix Game. By default True

        Returns
        -------
        Self
            The Polymatrix Game.
        """
        polymatrix_builder = {
            (p1, p2): np.full(
                (nf.nums_actions[p1], nf.nums_actions[p2]), -np.inf)
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

    def get_player(self, player_idx: int) -> Player:
        """
        Calculates the payoff function of a player.

        Parameters
        ----------
        player_idx : int
            Player number who we want to extract.

        Returns
        -------
        Player
            Player object which has the player's payoff function.
        """
        self.N
        opps = tuple(range(player_idx+1, self.N)) + tuple(range(player_idx))
        newaxes = np.full((self.N-1, self.N-1), np.newaxis)
        np.fill_diagonal(newaxes, slice(None))
        payoff_array = sum([
            self.polymatrix[(player_idx, opps[j])][:, *newaxes[j]]
            for j in range(self.N-1)
        ])
        return Player(payoff_array)

    def to_nfg(self) -> NormalFormGame:
        """
        Creates a Normal Form Game from the Polymatrix Game.

        Returns
        -------
        NormalFormGame
            The game in Normal Form.
        """
        nfg = NormalFormGame([self.get_player(i) for i in range(self.N)])

        return nfg

    def range_of_payoffs(self) -> tuple[float, float]:
        """
        The lowest and highest components of payoff from
        head to head games.

        Returns
        -------
        tuple[float, float]
            Tuple of minimum and maximum.
        """
        min_p = min([np.min(M) for M in self.polymatrix.values()])
        max_p = max([np.max(M) for M in self.polymatrix.values()])
        return (min_p, max_p)
