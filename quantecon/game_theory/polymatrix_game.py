""""
Tools for working with games in Polymatrix form.

In a Polymatrix Game, the payoff to a player is the sum
of their payoffs from bimatrix games against each player.
i.e. If two opponents deviate, the change in payoff
is the sum of the changes in payoff of each deviation.

Examples
--------

Turn a Matching Pennies Normal Form Game into a Polymatrix Game.

>>> matching_pennies_bimatrix = [
...     [(1, -1), (-1, 1)], [(-1, 1), (1, -1)]]
>>> nfg = NormalFormGame(matching_pennies_bimatrix)
>>> polymg = PolymatrixGame.from_nf(nfg)
>>> print(polymg)
2-player PolymatrixGame with payoff matrices:
(0, 1):
[[ 1. -1.]
 [-1.  1.]]
<BLANKLINE>
(1, 0):
[[-1.  1.]
 [ 1. -1.]]

(An example of a multiplayer game is not given because
then the polymatrix representation would not be unique and therefore
could not be reliably quoted for this doctest.)

"""

import numpy as np
from itertools import product
from math import isqrt

from collections.abc import Sequence, Mapping, Iterable
# from typing import TypeAlias, Self
from numpy.typing import NDArray

from .normal_form_game import NormalFormGame, Player, _nums_actions2string


def hh_payoff_player(
        nfg: NormalFormGame,
        my_player_number: int,
        my_action_number: int,
        is_polymatrix: bool = True
) -> dict[tuple[int, int], float]:
    """
    Head-to-head payoff components.

    hh stands for head-to-head.
    Calculates the payoffs to a player when they play
    an action with values at the actions of other players
    that try to sum to the payoff.
    Precise when the game can be represented with a polymatrix;
    otherwise, an approximation using least squares on the
    payoffs at every action combination.
    If an approximation, it may drastically change the
    game, use with caution.

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
        actually a polymatrix game. Defaults to True.

    Returns
    -------
    dict[tuple[int, int], float]
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

    # Could use pinv, but this is clearer.
    hh_payoffs_array, _, _, _ = np.linalg.lstsq(
        hh_actions,
        combined_payoffs,
        rcond=None
    )
    if is_polymatrix:
        was_polymatrix = np.allclose(
            hh_actions @ hh_payoffs_array,
            combined_payoffs
        )
        assert was_polymatrix, "The game is not actually a polymatrix game."

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
    N : scalar(int)
        Number of players.

    nums_actions : tuple(int)
        The number of actions available to each player.

    polymatrix : dict[tuple(int), ndarray(float, ndim=2)]
        Maps each pair of player numbers to a matrix.

    """
    def __repr__(self):
        s = '<{nums_actions} {N}-player PolymatrixGame>'
        return s.format(nums_actions=_nums_actions2string(self.nums_actions),
                        N=self.N)

    def __str__(self) -> str:
        str_builder = (
            f"{self.N}-player PolymatrixGame with payoff matrices:\n"
        )
        for k, v in self.polymatrix.items():
            str_builder += str(k) + ":\n"
            str_builder += str(v) + "\n\n"
        return str_builder.rstrip()

    def __init__(
            self,
            polymatrix: Mapping[
                tuple[int, int],
                Sequence[Sequence[float]]
            ],
            nums_actions: Iterable[int] = None
    ) -> None:
        """_summary_

        Parameters
        ----------
        polymatrix : Mapping[ tuple[int, int], Sequence[Sequence[float]] ]
            Polymatrix. Numbers of players and actions can be
            inferred from this if `nums_actions` is left None.
            This inferrence uses the number of actions they have
            against the next player.
            Actions with unspecified payoff are given
            payoff of `-np.inf`.
        nums_actions : Iterable[int], optional
            If desired, nums_actions can be set so that unspecified
            matchups in the polymatrix will be filled with matrices
            of 0s (while unspecified actions give payoff
            of `-np.inf`).
        """
        if nums_actions is None:
            self.N = (isqrt(4*len(polymatrix)+1) + 1) // 2
            self.nums_actions = tuple(
                np.shape(polymatrix[(p1, (p1 + 1) % self.N)])[0]
                for p1 in range(self.N)
            )
        else:
            self.N = len(nums_actions)
            self.nums_actions = tuple(nums_actions)
        matchups = [
            (p1, p2)
            for p1 in range(self.N)
            for p2 in range(self.N)
            if p1 != p2
        ]
        self.polymatrix: dict[tuple[int, int], NDArray] = {}
        for (p1, p2) in matchups:
            rows = self.nums_actions[p1]
            cols = self.nums_actions[p2]
            incoming = np.asarray(polymatrix.get(
                (p1, p2),
                np.zeros((rows, cols))
            ))
            matrix_builder = np.full((rows, cols), -np.inf)
            matrix_builder[:incoming.shape[0],
                           :incoming.shape[1]] = incoming[:rows, :cols]
            self.polymatrix[(p1, p2)] = matrix_builder

    @classmethod
    def from_nf(
        cls,
        nf: NormalFormGame,
        is_polymatrix: bool = True
        # ) -> Self:
    ):
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

        return cls(polymatrix_builder)

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
        N = self.N
        opps = tuple(range(player_idx+1, N)) + tuple(range(player_idx))
        newaxes = np.full((N-1, N), np.newaxis)
        newaxes[:, 0] = slice(None)
        newaxes[range(N-1), range(1, N)] = slice(None)
        payoff_array = sum([
            self.polymatrix[(player_idx, opps[j])][tuple(newaxes[j])]
            for j in range(N-1)
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
