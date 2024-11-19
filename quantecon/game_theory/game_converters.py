"""
Functions for converting between ways of storing games.
"""

from .normal_form_game import NormalFormGame
from itertools import product


def qe_nfg_from_gam_file(filename: str) -> NormalFormGame:
    """
    Makes a QuantEcon Normal Form Game from a gam file.
    Gam files are described by GameTracer.
    http://dags.stanford.edu/Games/gametracer.html

    Args:
        filename: path to gam file.

    Returns:
        NormalFormGame:
            The QuantEcon Normal Form Game
            described by the gam file.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        combined = [
            token
            for line in lines
            for token in line.split()
        ]

        i = iter(combined)
        players = int(next(i))
        actions = [int(next(i)) for _ in range(players)]

        nfg = NormalFormGame(actions)

        entries = [
                {
                    tuple(reversed(action_combination)): float(next(i))
                    for action_combination in product(
                        *[range(a) for a in actions])
                }
                for _ in range(players)
            ]

        for action_combination in product(*[range(a) for a in actions]):
            nfg[action_combination] = tuple(
                entries[p][action_combination] for p in range(players)
            )

    return nfg
